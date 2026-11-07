"""Feature extraction and engineering for duplicate and case-identity triage."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 1]


def compute_text_jaccard(text1: str, text2: str) -> float:
    """Compute token-level Jaccard similarity between two excerpts."""
    if not text1 or not text2:
        return 0.0
    tokens1 = set(_tokenize(text1))
    tokens2 = set(_tokenize(text2))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def compute_char_ngram_jaccard(text1: str, text2: str, n: int = 5) -> float:
    """Compute character n-gram Jaccard similarity for near-duplicate text detection."""
    if not text1 or not text2:
        return 0.0
    clean1 = re.sub(r"\s+", " ", text1.lower()).strip()
    clean2 = re.sub(r"\s+", " ", text2.lower()).strip()
    if len(clean1) < n or len(clean2) < n:
        return 0.0

    ngrams1 = {clean1[i : i + n] for i in range(len(clean1) - n + 1)}
    ngrams2 = {clean2[i : i + n] for i in range(len(clean2) - n + 1)}
    if not ngrams1 or not ngrams2:
        return 0.0
    return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)


def is_stock_postconviction_or_summary(text: str) -> bool:
    """Detect boilerplate stock order text (e.g. summary post-conviction affirmance)."""
    if not text:
        return False
    lower = text.lower()
    stock_phrases = [
        "rule 29.15",
        "rule 24.035",
        "judgment affirmed. rule 84.16(b)",
        "memorandum for the information of the parties",
        "motion for post-conviction relief",
        "an extended opinion would have no precedential value",
    ]
    return any(p in lower for p in stock_phrases)


def detect_judgment_vs_opinion(
    low: Any,
    high: Any,
    exc_low: str = "",
    exc_high: str = "",
) -> Tuple[bool, str]:
    """Detect if one cluster represents a judgment/order and the other an opinion."""
    if isinstance(low, str) and isinstance(high, str):
        all_text = f"{low.lower()} {high.lower()}"
    else:
        low_d = low if isinstance(low, dict) else {}
        high_d = high if isinstance(high, dict) else {}
        parts = [
            (low_d.get("case_name") or "").lower(),
            (high_d.get("case_name") or "").lower(),
            (low_d.get("case_name_full") or "").lower(),
            (high_d.get("case_name_full") or "").lower(),
            (exc_low or "")[:200].lower(),
            (exc_high or "")[:200].lower(),
        ]
        all_text = " ".join(parts)
    has_judgment = "judgment" in all_text or "order" in all_text
    has_opinion = "opinion" in all_text or "memorandum" in all_text
    if has_judgment and has_opinion:
        return True, "One record appears to be a separate judgment/order while the other is an explanatory opinion"
    return False, ""


def extract_enriched_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and derive all operational features for a pair."""
    feat = row.get("features", {})
    low = row.get("cluster_low", {})
    high = row.get("cluster_high", {})
    exc_low = row.get("excerpt_low", "") or ""
    exc_high = row.get("excerpt_high", "") or ""

    # Base features from CourtListener package
    len_ratio = feat.get("text_length_ratio")
    min_ham = feat.get("min_hamming")
    same_doc_num = bool(feat.get("same_docket_number"))
    doc_conflict = bool(feat.get("docket_numbers_conflict"))
    same_court = bool(feat.get("same_court"))
    parent_link = bool(feat.get("court_parent_link"))
    same_year = bool(feat.get("same_filing_year"))
    delta_days = feat.get("filing_date_delta_days")
    shared_cits = feat.get("shared_citations", 0) or 0
    span_min = feat.get("min_page_cluster_span", 0) or 0
    span_max = feat.get("max_page_cluster_span", 0) or 0
    max_page_span = max(span_min, span_max)
    shared_sha1 = bool(feat.get("shared_sha1"))
    shared_text_sha = bool(feat.get("shared_text_sha256"))
    disjoint_rep = bool(feat.get("disjoint_reporters"))
    title_exact = bool(feat.get("title_exact") or feat.get("title_sorted_exact"))

    # Excerpt similarity
    has_both_excerpts = bool(exc_low and exc_high)
    token_jaccard = compute_text_jaccard(exc_low, exc_high) if has_both_excerpts else 0.0
    char_jaccard = compute_char_ngram_jaccard(exc_low, exc_high, n=5) if has_both_excerpts else 0.0

    # Boilerplate / stock check
    low_is_stock = is_stock_postconviction_or_summary(exc_low)
    high_is_stock = is_stock_postconviction_or_summary(exc_high)

    # Judgment vs opinion
    is_j_vs_op, j_vs_op_reason = detect_judgment_vs_opinion(low, high, exc_low, exc_high)

    # SCOTUS cert denial bug (CourtListener data quality defect)
    court_low = (low.get("court_id") or "").lower()
    court_high = (high.get("court_id") or "").lower()
    is_scotus_cafc_cert_bug = (
        (court_low == "scotus" and court_high == "cafc") or
        (court_low == "cafc" and court_high == "scotus")
    ) and shared_cits > 0

    return {
        "len_ratio": len_ratio,
        "min_ham": min_ham,
        "same_doc_num": same_doc_num,
        "doc_conflict": doc_conflict,
        "same_court": same_court,
        "parent_link": parent_link,
        "same_year": same_year,
        "delta_days": delta_days,
        "shared_cits": shared_cits,
        "max_page_span": max_page_span,
        "shared_sha1": shared_sha1,
        "shared_text_sha": shared_text_sha,
        "disjoint_rep": disjoint_rep,
        "title_exact": title_exact,
        "has_both_excerpts": has_both_excerpts,
        "token_jaccard": token_jaccard,
        "char_jaccard": char_jaccard,
        "is_stock": low_is_stock or high_is_stock,
        "is_j_vs_op": is_j_vs_op,
        "j_vs_op_reason": j_vs_op_reason,
        "is_scotus_cafc_cert_bug": is_scotus_cafc_cert_bug,
    }
