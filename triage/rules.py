"""Calibrated decision rules for duplicate and case-identity triage.

SAFETY RULES:
1. NEVER merge if courts conflict (`same_court == 0` and not valid parent link).
2. NEVER merge if case names completely disagree (`title_exact == 0` and low token similarity),
   even if text hash matches (which occurs on shared reporter/OCR pages holding multiple decisions).
3. NEVER merge if docket numbers conflict (`doc_conflict == 1`).
4. If reporter page span >= 2 and shared citations >= 1:
   - Only consider probable_version if court matches, title matches, docket does not conflict, and text is near-identical.
   - Otherwise, it is shared_page_ambiguity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from triage.constants import (
    LABEL_CITATION_ERROR,
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


GOVERNMENT_PARTIES = (
    "state",
    "the state",
    "people",
    "the people",
    "united states",
    "commonwealth",
    "com",
    "com.",
    "u.s",
    "u.s.",
)


def _is_surname_only_government_caption(title: str) -> bool:
    """True for captions like "State v. Smith" or "Brown v. State".

    One side is the government and the other is identified by a single name, so many
    unrelated defendants share the exact same caption. A matching caption is therefore
    no evidence that two records come from the same litigation.
    """
    if not title:
        return False
    lower = title.lower().strip()

    def _name_words(part: str) -> List[str]:
        # Drop court/district designations ("ca2/6") and initials ("j.", "i.f.").
        words = [w for w in re.split(r"[\s,]+", part) if w and not re.search(r"\d", w)]
        return [w for w in (re.sub(r"[^a-z]", "", w) for w in words) if len(w) >= 2]

    # "In re I.F.", "In re Smith": a matter caption with at most one real name word.
    for prefix in ("in re ", "in the matter of "):
        if lower.startswith(prefix):
            return len(_name_words(lower[len(prefix):])) <= 1

    if " v. " not in lower:
        return False
    left, right = (part.strip(" .,") for part in lower.split(" v. ", 1))

    def _is_gov(part: str) -> bool:
        return part in GOVERNMENT_PARTIES or any(part.startswith(g + " of ") for g in GOVERNMENT_PARTIES)

    def _single_name(part: str) -> bool:
        return len(_name_words(part)) <= 1

    return (_is_gov(left) and _single_name(right)) or (_is_gov(right) and _single_name(left))


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
    # RULE 1: SCOTUS / CAFC Cert Denial Defect (Track 4 Defect)
    # -------------------------------------------------------------------------
    if f["is_scotus_cafc_cert_bug"]:
        supporting.append(
            "Different cert denials printed on the same U.S. Reports orders page (known CourtListener cafc misassignment defect)"
        )
        contradicting.append("Conflicting court assignments (scotus vs cafc) due to data extraction error")
        return LABEL_SHARED_PAGE, 0.95, supporting, contradicting

    generic_surname_caption = _is_surname_only_government_caption(title_low) or _is_surname_only_government_caption(
        title_high
    )

    # -------------------------------------------------------------------------
    # RULE 1b: Companion Instruments (Role 3)
    # Same parties, but the records are different instruments or different
    # parties' motions issued in the same litigation. These must never merge.
    # -------------------------------------------------------------------------
    if f["role_conflict"] and f["same_court"]:
        supporting.append(f["role_conflict_reason"])
        contradicting.append("Identical party list, court and filing date")
        return LABEL_RELATED_DECISION, 0.90, supporting, contradicting

    if (
        f["is_j_vs_op"]
        and f["same_court"]
        and (f["same_doc_num"] or f["title_exact"])
        and not f["doc_conflict"]
        and (f["delta_days"] is None or f["delta_days"] <= 7)
    ):
        supporting.append(f["j_vs_op_reason"])
        contradicting.append("Same docket, court and filing date")
        return LABEL_RELATED_DECISION, 0.88, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 2a: Same Parties and Court, Different Docket Numbers (Role 3)
    # A distinctive caption repeated under a different docket number in the same
    # court is a companion appeal or a later stage of the same litigation.
    # Generic "State v. Smith" style captions are excluded: there the shared
    # caption says nothing about shared litigation.
    # -------------------------------------------------------------------------
    if f["doc_conflict"] and f["same_court"] and f["title_exact"] and not generic_surname_caption:
        supporting.append(
            "Same parties and court under different docket numbers: separate proceedings in the same "
            f"litigation (companion appeal or later stage; filing date gap: {f['delta_days']} days)"
        )
        contradicting.append("Identical case caption")
        if f["min_ham"] is not None and f["min_ham"] <= 3:
            contradicting.append(
                f"Near-identical text (Hamming distance: {f['min_ham']}); check whether the docket numbers differ only in formatting"
            )
        return LABEL_RELATED_DECISION, 0.88, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 2: Conflicting Dockets or Stock Boilerplate Text (Trap Avoidance)
    # -------------------------------------------------------------------------
    if f["doc_conflict"]:
        supporting.append("Conflicting docket numbers establish records are separate legal proceedings")
        if f["shared_text_sha"] or f["min_ham"] == 0 or f["is_stock"]:
            contradicting.append("Identical normalized text or stock boilerplate order phrasing")
        return LABEL_UNRELATED, 0.92, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 3: Shared Page with Different Courts or Different Case Titles
    # -------------------------------------------------------------------------
    if f["max_page_span"] >= 2 and f["shared_cits"] >= 1:
        # If courts differ or titles differ, it is definitively shared page ambiguity
        if not f["same_court"] or not f["title_exact"]:
            supporting.append(
                f"Shared reporter citation page carries multiple distinct decisions (cluster span: {f['max_page_span']})"
            )
            if not f["same_court"]:
                contradicting.append("Different courts assigned to clusters")
            if not f["title_exact"]:
                contradicting.append(f"Distinct case names ('{low.get('case_name')}' vs '{high.get('case_name')}')")
            return LABEL_SHARED_PAGE, 0.94, supporting, contradicting

        # If court and title match, check if it's actually alternate versions
        if (
            f["same_court"]
            and f["title_exact"]
            and not f["doc_conflict"]
            and f["min_ham"] is not None
            and f["min_ham"] <= 15
            and f["len_ratio"] is not None
            and f["len_ratio"] >= 0.65
        ):
            supporting.append(
                f"Same decision in alternate versions sharing reporter page (Hamming distance: {f['min_ham']}, length ratio: {f['len_ratio']:.2f})"
            )
            return LABEL_PROBABLE_VERSION, 0.88, supporting, contradicting

        supporting.append(
            f"Shared reporter citation page carries multiple distinct decisions (cluster span: {f['max_page_span']})"
        )
        return LABEL_SHARED_PAGE, 0.90, supporting, contradicting

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

    # Generic surname-only caption with materially different texts and nothing else
    # tying the records together (no shared docket, citation or parent court).
    if (
        generic_surname_caption
        and f["shared_cits"] == 0
        and not f["same_doc_num"]
        and not f["parent_link"]
        and f["len_ratio"] is not None
        and f["len_ratio"] < 0.25
    ):
        supporting.append(
            f"Recurring government-v-surname caption with materially different texts (length ratio: {f['len_ratio']:.2f}) "
            "and no shared docket or citation: distinct defendants"
        )
        contradicting.append("Identical caption, court and filing date")
        return LABEL_UNRELATED, 0.85, supporting, contradicting

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

    # Separate rulings entered on different days in one docket (Role 3): the text
    # differs beyond the version threshold, so these are two orders, not two copies.
    if (
        f["same_doc_num"]
        and f["same_court"]
        and f["title_exact"]
        and not generic_surname_caption
        and f["delta_days"] is not None
        and f["delta_days"] >= 1
        and f["min_ham"] is not None
        and f["min_ham"] > 18
    ):
        supporting.append(
            f"Same docket, but filed {f['delta_days']} day(s) apart with materially different text "
            f"(Hamming distance: {f['min_ham']}): separate rulings in one case"
        )
        contradicting.append("Same docket number, court and caption")
        return LABEL_RELATED_DECISION, 0.88, supporting, contradicting

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
    # Excerpts are only the first few hundred characters (mostly caption headers), so
    # their similarity must not override a full-text fingerprint that says they differ.
    if (
        f["has_both_excerpts"]
        and f["token_jaccard"] >= 0.75
        and f["same_court"]
        and f["title_exact"]
        and not f["doc_conflict"]
        and (f["min_ham"] is None or f["min_ham"] <= 18)
    ):
        supporting.append(f"High excerpt similarity (token Jaccard: {f['token_jaccard']:.2f})")
        return LABEL_PROBABLE_VERSION, 0.82, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 12: Citation Data Error
    # -------------------------------------------------------------------------
    if f["shared_cits"] >= 1 and not f["title_exact"] and not f["same_court"] and f["max_page_span"] <= 1:
        supporting.append("Clusters share a citation but have completely different case titles and courts without a page span")
        contradicting.append("Shared citation string")
        return LABEL_CITATION_ERROR, 0.80, supporting, contradicting

    # -------------------------------------------------------------------------
    # RULE 13: Abstain / Insufficient Evidence (Safety First)
    # -------------------------------------------------------------------------
    supporting.append("Evidence is conflicting or insufficient to safely recommend consolidation or separation")
    if f["min_ham"] is not None:
        contradicting.append(f"Hamming distance is {f['min_ham']}")
    return LABEL_INSUFFICIENT, 0.50, supporting, contradicting
