"""Calibrated decision rules for duplicate and case-identity triage.

SAFETY RULES:
1. NEVER merge if courts conflict (`same_court == 0` and not valid parent link).
2. NEVER merge if case names completely disagree (`title_exact == 0` and low token similarity),
   even if text hash matches (which occurs on shared reporter/OCR pages holding multiple decisions).
3. NEVER merge if docket numbers conflict (`doc_conflict == 1`).
4. A page-span length is not a count of distinct decisions on a page.
   - Use explicit page-level decision counts to distinguish shared_page_ambiguity from citation_data_error.
   - If that count is unavailable, route contradictory shared-citation records to citation_topology_unverified.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from triage.constants import (
    LABEL_CITATION_ERROR,
    LABEL_CITATION_TOPOLOGY_UNVERIFIED,
    LABEL_INSUFFICIENT,
    LABEL_PARALLEL_SPLIT,
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    LABEL_RELATED_DECISION,
    LABEL_SHARED_PAGE,
    LABEL_UNRELATED,
)
from triage.features import extract_enriched_features


RECURRING_PROSECUTOR_PREFIXES = (
    "state v.",
    "united states v.",
    "people v.",
    "com. v.",
    "commonwealth v.",
    "u.s. v.",
    "city of ",
    "in re ",
    "in the matter of ",
)


def _is_recurring_generic_caption(title: str) -> bool:
    if not title:
        return False
    lower = title.lower().strip()
    return any(lower.startswith(prefix) for prefix in RECURRING_PROSECUTOR_PREFIXES)


def classify_pair_rule(row: Dict[str, Any]) -> Tuple[str, float, List[str], List[str]]:
    """Evaluate a candidate pair using an ordered, evidence-backed decision list.

    Returns:
        (label, confidence, supporting_evidence, contradicting_evidence)
    """
    f = extract_enriched_features(row)
    low = row.get("cluster_low", {})
    high = row.get("cluster_high", {})

    supporting: List[str] = []
    contradicting: List[str] = []

    title_low = (low.get("case_name") or "").lower()
    title_high = (high.get("case_name") or "").lower()

    # -------------------------------------------------------------------------
    # RULE 1: Shared-citation topology must preempt docket-conflict fallback.
    # Page-span length is not a count of distinct decisions on a physical page.
    # Use an explicit page-level count when available; otherwise abstain into a
    # citation-specific review category instead of guessing between two causes.
    # -------------------------------------------------------------------------
    if f["shared_cits"] >= 1:
        if (
            (f["shared_sha1"] or f["shared_text_sha"])
            and f["same_court"]
            and f["title_exact"]
            and not f["doc_conflict"]
        ):
            supporting.append("Identical source hash or normalized text with matching court and case title")
            return LABEL_PROBABLE_DUPLICATE, 0.99, supporting, contradicting

        same_decision_evidence = (
            not f["doc_conflict"]
            and f["same_court"]
            and f["min_ham"] is not None
            and f["min_ham"] <= 12
            and f["len_ratio"] is not None
            and f["len_ratio"] >= 0.65
            and (f["title_exact"] or f["token_jaccard"] >= 0.5)
        )
        if same_decision_evidence:
            supporting.append(
                "Matching court, case identity, and near-identical text across shared reporter citations"
            )
            return LABEL_PROBABLE_VERSION, 0.90, supporting, contradicting

        distinct_page_decisions = f["distinct_decisions_on_shared_page"]
        unrelated_captions = (
            not f["title_exact"]
            and f["min_ham"] is not None
            and f["min_ham"] > 20
        )
        conflicting_metadata = f["doc_conflict"] or not f["same_court"] or unrelated_captions

        if distinct_page_decisions is not None and distinct_page_decisions >= 2:
            supporting.append(
                f"Reporter-page evidence identifies {distinct_page_decisions} distinct decisions on the shared page"
            )
            if f["doc_conflict"]:
                contradicting.append("Conflicting docket numbers distinguish separate proceedings")
            if not f["same_court"]:
                contradicting.append("Different courts assigned to clusters")
            if not f["title_exact"]:
                contradicting.append("Case captions do not match exactly")
            return LABEL_SHARED_PAGE, 0.95, supporting, contradicting

        if distinct_page_decisions == 1 and conflicting_metadata:
            supporting.append(
                "Page-level evidence identifies one decision, but the records carry conflicting case metadata for the shared citation"
            )
            if f["doc_conflict"]:
                contradicting.append("Conflicting docket numbers")
            if not f["same_court"]:
                contradicting.append("Different courts assigned to clusters")
            if unrelated_captions:
                contradicting.append("Case captions and opinion text are substantially different")
            return LABEL_CITATION_ERROR, 0.88, supporting, contradicting

        supporting.append(
            "The records share a citation, but page-span length does not establish how many decisions appear on that page"
        )
        if f["doc_conflict"]:
            contradicting.append("Conflicting docket numbers")
        if not f["same_court"]:
            contradicting.append("Different courts assigned to clusters")
        if not f["title_exact"]:
            contradicting.append("Case captions do not match exactly")
        return LABEL_CITATION_TOPOLOGY_UNVERIFIED, 0.90, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 2: Conflicting Dockets or Stock Boilerplate Text (Trap Avoidance)
    # -------------------------------------------------------------------------
    if f["doc_conflict"]:
        supporting.append("Conflicting docket numbers establish records are separate legal proceedings")
        if f["shared_text_sha"] or f["min_ham"] == 0 or f["is_stock"]:
            contradicting.append("Identical normalized text or stock boilerplate order phrasing")
        return LABEL_UNRELATED, 0.92, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 4: Different Courts with No Parent Link (Must Stay Separate)
    # -------------------------------------------------------------------------
    if not f["same_court"] and not f["parent_link"]:
        if f["shared_cits"] >= 1:
            supporting.append("Clusters in completely different jurisdictions share a citation string")
            return LABEL_CITATION_ERROR, 0.88, supporting, contradicting
        supporting.append("Different courts and jurisdictions with no appellate parent relationship")
        return LABEL_UNRELATED, 0.90, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 5: Completely Different Case Names (Title Token Mismatch)
    # -------------------------------------------------------------------------
    if not f["title_exact"] and f["shared_cits"] == 0 and not f["same_doc_num"]:
        supporting.append(f"Distinct case captions ('{low.get('case_name')}' vs '{high.get('case_name')}') with no shared docket or citation")
        return LABEL_UNRELATED, 0.90, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 6: Positive Exact Duplicate (Identical File Hash or Exact Ingestion)
    # Require matching court and matching title!
    # -------------------------------------------------------------------------
    if (f["shared_sha1"] or f["shared_text_sha"]) and f["same_court"] and f["title_exact"] and not f["doc_conflict"]:
        supporting.append("Identical source file SHA-1 or normalized text hash with matching court and title")
        return LABEL_PROBABLE_DUPLICATE, 0.99, supporting, contradicting

    # Near-exact duplicate text with consistent docket and court
    if (
        f["title_exact"]
        and f["same_court"]
        and f["same_doc_num"]
        and not f["doc_conflict"]
        and (f["delta_days"] is None or f["delta_days"] <= 5)
        and (f["len_ratio"] is None or f["len_ratio"] >= 0.85)
        and (f["min_ham"] is not None and f["min_ham"] <= 5)
    ):
        supporting.append(
            f"Same court, docket number, exact title, and near-identical text (Hamming distance: {f['min_ham']})"
        )
        return LABEL_PROBABLE_DUPLICATE, 0.92, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 7: Recurring Generic Case Name Trap (e.g. State v. Smith, People v. Garner)
    # -------------------------------------------------------------------------
    is_generic = _is_recurring_generic_caption(title_low) or _is_recurring_generic_caption(title_high)
    if is_generic and f["shared_cits"] == 0:
        if (
            (not f["same_doc_num"] or f["doc_conflict"])
            and (f["delta_days"] is None or f["delta_days"] > 45)
        ):
            supporting.append(
                f"Recurring institutional case name with different dockets and filing dates {f['delta_days']} days apart represents distinct proceedings"
            )
            contradicting.append("Similar case caption")
            return LABEL_UNRELATED, 0.90, supporting, contradicting

    # Distinct defendant parties check (Smith, E. vs Smith, M.)
    if " v. " in title_low and " v. " in title_high:
        p1_low, p2_low = title_low.split(" v. ", 1)
        p1_high, p2_high = title_high.split(" v. ", 1)
        if p2_low != p2_high and f["shared_cits"] == 0 and not f["doc_conflict"]:
            if ("," in p2_low and "," in p2_high) or len(p2_low) != len(p2_high):
                supporting.append(f"Distinct defendant parties ('{p2_low.strip()}' vs '{p2_high.strip()}')")
                return LABEL_UNRELATED, 0.88, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 8: Procedural Order vs. Merits Opinion (Length ratio disparity)
    # -------------------------------------------------------------------------
    if f["len_ratio"] is not None and f["len_ratio"] < 0.25:
        if f["same_doc_num"] or f["title_exact"]:
            supporting.append(
                f"Text length ratio ({f['len_ratio']:.2f}) indicates distinct records (e.g., separate judgment/order vs explanatory opinion) on same docket"
            )
            contradicting.append("Shared title or docket number")
            return LABEL_RELATED_DECISION, 0.88, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 9: Multi-stage Litigation / Appellate Chain / Filing Date Gap
    # -------------------------------------------------------------------------
    if f["parent_link"] or (f["delta_days"] is not None and f["delta_days"] > 45):
        if (f["same_doc_num"] or f["title_exact"]) and not is_generic:
            supporting.append(
                f"Separate stages or rulings in the same litigation (filing date gap: {f['delta_days']} days, parent court link: {f['parent_link']})"
            )
            contradicting.append("Shared party names / case caption")
            return LABEL_RELATED_DECISION, 0.85, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 10: Parallel Split Cluster (Split across reporters or opinions)
    # -------------------------------------------------------------------------
    if (
        f["disjoint_rep"]
        and f["title_exact"]
        and f["same_court"]
        and (f["delta_days"] is None or f["delta_days"] <= 14)
        and (f["len_ratio"] is None or f["len_ratio"] >= 0.40)
        and not f["doc_conflict"]
    ):
        supporting.append("One decision split across parallel reporter citations with matching court, title, and filing date")
        if f["min_ham"] is not None and f["min_ham"] > 20:
            contradicting.append(f"Text differences across versions (Hamming distance: {f['min_ham']})")
        return LABEL_PARALLEL_SPLIT, 0.86, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 11: Probable Version (Slip vs. Published, Revised Text)
    # -------------------------------------------------------------------------
    if (
        f["title_exact"]
        and (f["same_court"] or f["parent_link"])
        and (f["delta_days"] is None or f["delta_days"] <= 30)
        and (f["len_ratio"] is not None and f["len_ratio"] >= 0.65)
        and (f["min_ham"] is not None and f["min_ham"] <= 18)
        and not f["doc_conflict"]
    ):
        supporting.append(
            f"Same judicial decision in alternate versions (Hamming distance: {f['min_ham']}, length ratio: {f['len_ratio']:.2f})"
        )
        return LABEL_PROBABLE_VERSION, 0.84, supporting, contradicting

    # Excerpt Jaccard high match
    if f["has_both_excerpts"] and f["token_jaccard"] >= 0.75 and f["same_court"] and f["title_exact"] and not f["doc_conflict"]:
        supporting.append(f"High excerpt similarity (token Jaccard: {f['token_jaccard']:.2f})")
        return LABEL_PROBABLE_VERSION, 0.82, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 12: Abstain / Insufficient Evidence (Safety First)
    # -------------------------------------------------------------------------
    supporting.append("Evidence is conflicting or insufficient to safely recommend consolidation or separation")
    if f["min_ham"] is not None:
        contradicting.append(f"Hamming distance is {f['min_ham']}")
    return LABEL_INSUFFICIENT, 0.50, supporting, contradicting
