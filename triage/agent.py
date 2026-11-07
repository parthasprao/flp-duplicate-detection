"""Tier 2: Falsification-First LLM Agent for case-identity adjudication and conflict review.

Implements the official NYU Law Hackathon prompt recipes:
1. Falsification check: Attempts to break the merge hypothesis before confirming.
2. Version vs Duplicate: Distinguishes re-scraped identical files from slip/revised versions.
3. Adjudication: Analyzes AI disagreements on candidate pairs.

Supports OpenRouter (tiered between GLM Flash for high-volume repetitive cases
and Nemotron Ultra for complex cases), Gemini, and a zero-dependency deterministic Offline Analyzer.
"""

from __future__ import annotations

import json
import os
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
    MERGE_LABELS,
    VALID_LABELS,
)
from triage.features import extract_enriched_features

DEFAULT_OPENROUTER_MODEL_FAST = "z-ai/glm-5.3-flash"
DEFAULT_OPENROUTER_MODEL_COMPLEX = "nvidia/nemotron-4-340b-instruct"

FALSIFICATION_SYSTEM_PROMPT = """You are an expert legal data engineer helping the Free Law Project / CourtListener decide whether two court decision records are the same judicial decision.

SAFETY MANDATE:
Wrongly merging two distinct decisions silently corrupts legal research and citation counts. A false merge is MUCH worse than leaving an unresolved duplicate.
Always try to FALSIFY the merge hypothesis first.

LABEL SET (choose exactly one):
probable_duplicate      Same decision, ingested twice (identical file or identical text re-scraped)
probable_version        Same decision, but different version (slip vs published, revised/corrected text)
parallel_split_cluster  One decision split across parallel reporters or majority/dissent records
related_decision        Legally related but DISTINCT (same litigation at different stages, rehearing, remand)
shared_page_ambiguity   A reporter page or table holding several different decisions; NOT the same decision
citation_data_error     Records are distinct; a citation attached to one or both is erroneous
citation_topology_unverified  Records share a citation, but available evidence cannot establish whether the physical page contains multiple decisions or the citation was attached incorrectly; require human/source review
unrelated               Different decisions with superficial metadata overlap
insufficient_evidence   Evidence cannot settle it; ABSTAIN rather than guess

Output valid JSON matching this schema:
{
  "label": "<one of the 9 labels>",
  "confidence": <float 0.0 to 1.0>,
  "falsification_finding": "<one concrete fact that breaks the merge, or 'None found after checking court, docket, dates, page occupancy, and text'>",
  "supporting_evidence": ["<evidence 1>", "<evidence 2>"],
  "contradicting_evidence": ["<contradiction 1>"],
  "maintainer_summary": "<two concise sentences explaining recommendation to a CourtListener maintainer>"
}
"""


def _build_pair_prompt(row: Dict[str, Any]) -> str:
    """Format pair metadata and excerpts for the agent."""
    low = row.get("cluster_low", {})
    high = row.get("cluster_high", {})
    feat = row.get("features", {})
    exc_low = row.get("excerpt_low", "") or "[No excerpt available]"
    exc_high = row.get("excerpt_high", "") or "[No excerpt available]"

    return f"""CANDIDATE PAIR: {row.get('pair_id')}

RECORD LOW:
  Cluster ID: {low.get('cluster_id')}
  Case Name: {low.get('case_name')}
  Court: {low.get('court', {}).get('short_name') or low.get('court_id')}
  Date Filed: {low.get('date_filed')}
  Docket: {low.get('docket', {}).get('docket_number')}
  Citations: {json.dumps(low.get('citations', []))}
  Excerpt: {exc_low[:600]}

RECORD HIGH:
  Cluster ID: {high.get('cluster_id')}
  Case Name: {high.get('case_name')}
  Court: {high.get('court', {}).get('short_name') or high.get('court_id')}
  Date Filed: {high.get('date_filed')}
  Docket: {high.get('docket', {}).get('docket_number')}
  Citations: {json.dumps(high.get('citations', []))}
  Excerpt: {exc_high[:600]}

PAIR FEATURES:
  Docket conflict: {feat.get('docket_numbers_conflict')}
  Same docket number: {feat.get('same_docket_number')}
  Same court: {feat.get('same_court')}
  Filing date delta: {feat.get('filing_date_delta_days')} days
  Shared citations: {feat.get('shared_citations')}
  Reporter pages spanned by each cluster: {max(feat.get('min_page_cluster_span', 0) or 0, feat.get('max_page_cluster_span', 0) or 0)}
  Distinct decisions verified on the shared page: {feat.get('distinct_decisions_on_shared_page')}
  Hamming distance: {feat.get('min_hamming')}
  Text length ratio: {feat.get('text_length_ratio')}
  Shared SHA-1: {feat.get('shared_sha1')}
"""


def _has_same_decision_evidence(features: Dict[str, Any]) -> bool:
    """Return whether shared records have enough identity evidence to avoid topology review."""
    exact_ingestion = (
        (features["shared_sha1"] or features["shared_text_sha"])
        and features["same_court"]
        and features["title_exact"]
        and not features["doc_conflict"]
    )
    near_identity = (
        not features["doc_conflict"]
        and features["same_court"]
        and features["min_ham"] is not None
        and features["min_ham"] <= 12
        and features["len_ratio"] is not None
        and features["len_ratio"] >= 0.65
        and (features["title_exact"] or features["token_jaccard"] >= 0.5)
    )
    return bool(exact_ingestion or near_identity)


def is_complex_pair(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Determine whether a candidate pair requires deep reasoning (Nemotron) or simple check (GLM Flash)."""
    f = extract_enriched_features(row)

    # 1. Shared citation without page-occupancy evidence
    if (
        f["shared_cits"] >= 1
        and f["distinct_decisions_on_shared_page"] is None
        and not _has_same_decision_evidence(f)
    ):
        reason = "Shared citation lacks page-occupancy evidence; page span alone cannot settle topology"
        if f["doc_conflict"]:
            reason += " and docket numbers conflict"
        if f["len_ratio"] is not None and f["len_ratio"] < 0.35:
            reason += "; severe text length disparity also needs review"
        return True, reason

    # 2. Docket conflicts
    if f["doc_conflict"]:
        return True, "Conflicting docket numbers detected"

    if f["distinct_decisions_on_shared_page"] is not None and f["distinct_decisions_on_shared_page"] >= 2:
        return True, f"Page-level evidence confirms {f['distinct_decisions_on_shared_page']} distinct decisions"

    if f["shared_cits"] >= 1 and f["distinct_decisions_on_shared_page"] == 1 and (
        f["doc_conflict"] or not f["same_court"] or not f["title_exact"]
    ):
        return True, "One decision is verified on the page but the shared citation conflicts with case metadata"

    # 3. Severe text length disparity
    if f["len_ratio"] is not None and f["len_ratio"] < 0.35:
        return True, f"Severe text length disparity ({f['len_ratio']:.2f}) indicates order vs substantive opinion"

    # 4. Large filing date delta
    if f["delta_days"] is not None and f["delta_days"] > 45:
        return True, f"Filing date gap of {f['delta_days']} days indicates successive stages"

    # 5. Court mismatch without standard appellate progression
    if not f["same_court"] and not f["parent_link"]:
        return True, "Different courts with no clear appellate link"

    # 6. Specific known edge patterns (e.g. SCOTUS cert denial CAFC bug)
    if f["is_scotus_cafc_cert_bug"]:
        return True, "Known SCOTUS cert denial CAFC volume bug requires careful disambiguation"

    return False, "Standard clean candidate pair suitable for fast inference"


def call_openrouter(
    prompt: str,
    model: str,
    api_key: str,
    timeout: int = 20,
) -> Optional[Dict[str, Any]]:
    """Send structured prompt to OpenRouter API and return parsed JSON result."""
    try:
        import requests

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/freelawproject/courtlistener",
            "X-Title": "CourtListener Duplicate Detection & Triage",
        }
        payload = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": FALSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        if not content:
            return None

        # Clean markdown codeblocks if model wrapped output in ```json
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "label" in parsed:
            return parsed
        return None
    except Exception:
        return None


def run_offline_falsification(row: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic Devil's Advocate agent that runs offline without API keys."""
    f = extract_enriched_features(row)
    low = row.get("cluster_low", {})
    high = row.get("cluster_high", {})
    reasons_to_falsify: List[str] = []

    # 1. Check for docket conflicts
    if f["doc_conflict"]:
        reasons_to_falsify.append(
            f"Conflicting docket numbers ({low.get('docket', {}).get('docket_number')} vs {high.get('docket', {}).get('docket_number')})"
        )

    # 2. Check for court conflict
    if not f["same_court"] and not f["parent_link"]:
        reasons_to_falsify.append(
            f"Different courts ({low.get('court_id')} vs {high.get('court_id')}) with no appellate link"
        )

    # 3. Check citation topology using page-level evidence, not cluster span.
    page_case_count = f["distinct_decisions_on_shared_page"]
    conflicting_metadata = f["doc_conflict"] or not f["same_court"] or not f["title_exact"]
    if f["shared_cits"] >= 1 and page_case_count is None and not _has_same_decision_evidence(f):
        reasons_to_falsify.append(
            "Shared citation has no verified page-occupancy count; cannot distinguish a multi-decision page from a citation attachment error"
        )
    elif f["shared_cits"] >= 1 and page_case_count == 1 and conflicting_metadata:
        reasons_to_falsify.append(
            "Page-level evidence shows one decision while the records have conflicting metadata for the shared citation"
        )
    elif f["shared_cits"] >= 1 and page_case_count is not None and page_case_count >= 2 and conflicting_metadata:
        reasons_to_falsify.append(
            f"Page-level evidence confirms {page_case_count} distinct decisions on the cited page"
        )

    # 4. Check for length disparity
    if f["len_ratio"] is not None and f["len_ratio"] < 0.25:
        reasons_to_falsify.append(
            f"Severe length disparity (ratio: {f['len_ratio']:.2f}) indicates an order/judgment vs substantive opinion"
        )

    # 5. Check for large date gap
    if f["delta_days"] is not None and f["delta_days"] > 45:
        reasons_to_falsify.append(
            f"Filing date gap of {f['delta_days']} days indicates successive stages rather than identical decision"
        )

    # Determine outcome
    if reasons_to_falsify:
        falsified_label = LABEL_UNRELATED
        if f["same_doc_num"] or f["title_exact"]:
            falsified_label = LABEL_RELATED_DECISION
        if f["shared_cits"] >= 1 and page_case_count is None and conflicting_metadata:
            falsified_label = LABEL_CITATION_TOPOLOGY_UNVERIFIED
        elif f["shared_cits"] >= 1 and page_case_count == 1 and conflicting_metadata:
            falsified_label = LABEL_CITATION_ERROR
        elif f["shared_cits"] >= 1 and page_case_count is not None and page_case_count >= 2:
            falsified_label = LABEL_SHARED_PAGE

        return {
            "label": falsified_label,
            "confidence": 0.88,
            "falsification_finding": "; ".join(reasons_to_falsify),
            "supporting_evidence": reasons_to_falsify,
            "contradicting_evidence": ["Passed initial candidate blocking filter"],
            "maintainer_summary": f"Falsification check broke merge hypothesis: {reasons_to_falsify[0]}. Records must stay separate.",
            "tier": "offline",
            "model_used": "offline_analyzer",
        }

    # If no falsification reasons found, positive merge is supported
    merge_type = LABEL_PROBABLE_DUPLICATE if (f["shared_sha1"] or f["shared_text_sha"]) else LABEL_PROBABLE_VERSION
    return {
        "label": merge_type,
        "confidence": 0.90,
        "falsification_finding": "No falsifying contradictions found after checking dockets, courts, dates, and text",
        "supporting_evidence": [
            "Consistent court and docket numbers",
            f"Hamming distance is {f['min_ham']}",
            f"Text length ratio is {f['len_ratio']}",
        ],
        "contradicting_evidence": [],
        "maintainer_summary": f"Falsification verified: records describe the same decision ({merge_type}) with no conflicting metadata.",
        "tier": "offline",
        "model_used": "offline_analyzer",
    }


def analyze_with_agent(row: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Tier 2 falsification analysis using configured provider or offline fallback.

    Automatically handles OpenRouter tiered routing:
    - High-volume simple candidate pairs -> GLM 5.3 Flash
    - High-risk / complex pairs or low-confidence fast returns -> Nemotron Ultra
    - Gracefully falls back to offline analyzer on missing keys or network failure.
    """
    provider = os.environ.get("TRIAGE_LLM", "offline").lower()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    # If OPENROUTER_API_KEY is present and provider wasn't explicitly set to offline/gemini, default to openrouter
    if openrouter_key and provider not in {"offline", "gemini"}:
        provider = "openrouter"

    # 1. OpenRouter Tiered Routing
    if provider == "openrouter":
        if not openrouter_key:
            return run_offline_falsification(row)

        fast_model = os.environ.get("OPENROUTER_MODEL_FAST", DEFAULT_OPENROUTER_MODEL_FAST)
        complex_model = os.environ.get("OPENROUTER_MODEL_COMPLEX", DEFAULT_OPENROUTER_MODEL_COMPLEX)

        complex_flag, reason = is_complex_pair(row)
        prompt = _build_pair_prompt(row)

        # Direct complex routing for high-risk pairs
        if complex_flag:
            res = call_openrouter(prompt, model=complex_model, api_key=openrouter_key)
            if res and res.get("label") in VALID_LABELS:
                res["tier"] = "complex"
                res["model_used"] = complex_model
                return res
        else:
            # Fast inference path
            res = call_openrouter(prompt, model=fast_model, api_key=openrouter_key)
            # Accept if valid and confident
            if (
                res
                and res.get("label") in VALID_LABELS
                and float(res.get("confidence", 0.0)) >= 0.80
                and res.get("label") != LABEL_INSUFFICIENT
            ):
                res["tier"] = "fast"
                res["model_used"] = fast_model
                return res

            # Escalation ladder: if fast model was unconfident, abstained, or failed
            esc_res = call_openrouter(prompt, model=complex_model, api_key=openrouter_key)
            if esc_res and esc_res.get("label") in VALID_LABELS:
                esc_res["tier"] = "complex"
                esc_res["model_used"] = complex_model
                return esc_res

        # If OpenRouter failed completely, fall back to offline
        return run_offline_falsification(row)

    # 2. Legacy Gemini integration
    if provider == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            return run_offline_falsification(row)
        try:
            import requests

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": f"{FALSIFICATION_SYSTEM_PROMPT}\n\n{_build_pair_prompt(row)}"}]}],
                "generationConfig": {"response_mime_type": "application/json"},
            }
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text)
                parsed["tier"] = "fast"
                parsed["model_used"] = "gemini-2.5-flash"
                return parsed
        except Exception:
            return run_offline_falsification(row)

    # 3. Default: Offline Deterministic Analyzer
    return run_offline_falsification(row)
