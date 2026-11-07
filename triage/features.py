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


# A formal decree's caption block carries a heading line such as "JUDGMENT" or
# "JUDGMENT RENDERED JUNE 30, 2016" (Texas courts of appeals use this layout).
_JUDGMENT_HEADING = re.compile(r"^\s*JUDGMENT(?:\s+RENDERED\b[^\n]*)?\s*$", re.MULTILINE)
_OPINION_HEADING = re.compile(r"^\s*(?:MEMORANDUM\s+)?OPINION\b[^\n]*$", re.MULTILINE)
_HEADER_WINDOW = 800


def normalize_docket_number(docket_str: str) -> str:
    """Normalize a docket number string to a canonical comparable format.

    - Strips procedural prefixes ('Civil Action No.', 'Civil Action', 'Case No.', 'No.', 'Appeal No.', etc.)
    - Normalizes Unicode dashes ('—', '–') to standard hyphens ('-')
    - Strips trailing judge initials and status tokens ('(JDB)', '(ESH)', 'Rel', 'Slip Op')
    - Normalizes state district prefixes ('3D14-3142' -> '14-3142' -> '2014-3142')
    - Normalizes 2-digit vs 4-digit year-sequence pairs ('08-1963' / '2008-1963', '93-200' / '1993-0200')
    """
    if not docket_str or not isinstance(docket_str, str):
        return ""
    s = docket_str.strip()
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", s)
    s = re.sub(r"\s*\([A-Za-z0-9\s\.\-]+\)\s*$", "", s)
    s = re.sub(r"\s+(?:rel|slip\s+op|order)\b", "", s, flags=re.IGNORECASE)
    s = s.strip(" .,;:")
    s = re.sub(
        r"^(?:civil\s+action\s+no\.?|civil\s+action|case\s+no\.?|appeal\s+no\.?|docket\s+no\.?|nos?\.?)\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"^(\d+)d(?=\d{2,4}-\d+)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\bno\.?\s*", "", s, flags=re.IGNORECASE)

    m = re.match(r"^(\d{1,2})-(\d{2,4})-(\d+)$", s)
    if m:
        dist, yr, seq = m.group(1), m.group(2), m.group(3)
        if len(yr) == 2:
            yr_int = int(yr)
            yr = str(1900 + yr_int if yr_int >= 50 else 2000 + yr_int)
        return f"{dist}-{yr}-{int(seq)}"

    m = re.match(r"^(\d{2,4})-(\d+)$", s)
    if m:
        yr, seq = m.group(1), m.group(2)
        if len(yr) == 2:
            yr_int = int(yr)
            yr = str(1900 + yr_int if yr_int >= 50 else 2000 + yr_int)
        return f"{yr}-{int(seq)}"

    return s.lower().strip()


def extract_docket_tokens(cluster: Dict[str, Any]) -> Set[str]:
    """Extract all normalized docket tokens and core representations from a cluster dict."""
    tokens: Set[str] = set()
    if not isinstance(cluster, dict):
        return tokens
    candidates = []
    d = cluster.get("docket")
    if isinstance(d, dict):
        if d.get("docket_number"):
            candidates.append(str(d["docket_number"]))
        if d.get("docket_number_core"):
            core = str(d["docket_number_core"]).strip()
            candidates.append(core)
            if len(core) == 8 and core.isdigit():
                yy = int(core[:2])
                yr = 1900 + yy if yy >= 50 else 2000 + yy
                seq = int(core[2:])
                tokens.add(f"{yr}-{seq}")
    if cluster.get("docket_number"):
        candidates.append(str(cluster["docket_number"]))
    if cluster.get("docket_number_core"):
        candidates.append(str(cluster["docket_number_core"]))

    for cand in candidates:
        norm = normalize_docket_number(cand)
        if norm and any(c.isdigit() for c in norm):
            tokens.add(norm)
        for sub in re.split(r"[,;]+", cand):
            sub_norm = normalize_docket_number(sub)
            if sub_norm and any(c.isdigit() for c in sub_norm):
                tokens.add(sub_norm)
            for tok in sub.strip().split():
                if any(c.isdigit() for c in tok):
                    tok_norm = normalize_docket_number(tok)
                    if tok_norm and any(c.isdigit() for c in tok_norm):
                        tokens.add(tok_norm)
    return tokens


def get_reporter_family(cit_key: str) -> str:
    """Extract canonical reporter family from citation key (e.g. '915|F2D|469' -> 'F')."""
    if not cit_key:
        return ""
    parts = cit_key.split("|")
    rep = parts[1].upper() if len(parts) >= 2 else cit_key.upper()
    return re.sub(r"(?:2D|3D|4TH|5TH|NS)$", "", rep).strip()


def _instrument_type(text: str) -> str:
    """Classify an excerpt's header as 'judgment', 'opinion' or '' (unknown)."""
    head = (text or "")[:_HEADER_WINDOW]
    if not head.strip():
        return ""
    if _JUDGMENT_HEADING.search(head):
        return "judgment"
    if _OPINION_HEADING.search(head):
        return "opinion"
    return "other"


def detect_judgment_vs_opinion(
    low: Any,
    high: Any,
    exc_low: str = "",
    exc_high: str = "",
) -> Tuple[bool, str]:
    """Detect a companion-instrument pair: one record is the formal judgment, the other the opinion.

    Each side is classified separately from its own header, so the pair only counts
    as a companion when exactly one side carries a stand-alone JUDGMENT heading and
    the other side is an opinion (or some other non-judgment document).
    """
    if isinstance(low, str) and isinstance(high, str):
        text_low, text_high = low, high
    else:
        text_low, text_high = exc_low or "", exc_high or ""

    kind_low = _instrument_type(text_low)
    kind_high = _instrument_type(text_high)
    if not kind_low or not kind_high:
        return False, ""
    if (kind_low == "judgment") != (kind_high == "judgment"):
        return True, (
            "One record is the court's separate formal judgment and the other is the "
            "explanatory opinion (companion instruments issued together, not one document twice)"
        )
    return False, ""


# Procedural role designations that appear in full captions.
_ROLE_WORDS = re.compile(
    r"\b(movant|appellant|appellee|petitioner|respondent|intervenor|cross)\b", re.IGNORECASE
)


def detect_caption_role_conflict(low: Dict[str, Any], high: Dict[str, Any]) -> Tuple[bool, str]:
    """Same parties, but the full captions assign a procedural role to a different party.

    Example shape: "A v. B, DEFENDANT-MOVANT, AND C" vs "A v. B, AND C, DEFENDANT-MOVANT".
    The word sets match but the movant/appellant designation moved, which means two
    separate motions or appeals by different parties, not one decision twice.
    """
    low_d = low or {}
    high_d = high or {}
    full_low = (low_d.get("case_name_full") or "").strip()
    full_high = (high_d.get("case_name_full") or "").strip()
    if not full_low or not full_high:
        return False, ""
    if not (_ROLE_WORDS.search(full_low) and _ROLE_WORDS.search(full_high)):
        return False, ""
    toks_low = _tokenize(full_low)
    toks_high = _tokenize(full_high)
    if toks_low == toks_high or sorted(toks_low) != sorted(toks_high):
        return False, ""
    return True, (
        "Full captions name the same parties but assign the procedural role "
        "(movant/appellant/petitioner) to a different party: separate motions or appeals"
    )


def extract_enriched_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and derive all operational features for a pair."""
    feat = row.get("features") or {}
    low = row.get("cluster_low") or {}
    high = row.get("cluster_high") or {}
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

    # Normalized core docket matching
    tokens_low = extract_docket_tokens(low)
    tokens_high = extract_docket_tokens(high)
    if tokens_low and tokens_high and (tokens_low & tokens_high):
        same_doc_num = True
        doc_conflict = False

    # True disjoint reporters verification
    if disjoint_rep and shared_cits == 0:
        cits_l = [c.get("citation_key") or "" for c in (low.get("citations") or []) if isinstance(c, dict)]
        cits_h = [c.get("citation_key") or "" for c in (high.get("citations") or []) if isinstance(c, dict)]
        reps_l = {get_reporter_family(c) for c in cits_l if c}
        reps_h = {get_reporter_family(c) for c in cits_h if c}
        if reps_l and reps_h and (reps_l & reps_h):
            disjoint_rep = False

    # Excerpt similarity
    has_both_excerpts = bool(exc_low and exc_high)
    token_jaccard = compute_text_jaccard(exc_low, exc_high) if has_both_excerpts else 0.0
    char_jaccard = compute_char_ngram_jaccard(exc_low, exc_high, n=5) if has_both_excerpts else 0.0

    # Boilerplate / stock check
    low_is_stock = is_stock_postconviction_or_summary(exc_low)
    high_is_stock = is_stock_postconviction_or_summary(exc_high)

    # Judgment vs opinion
    is_j_vs_op, j_vs_op_reason = detect_judgment_vs_opinion(low, high, exc_low, exc_high)
    role_conflict, role_conflict_reason = detect_caption_role_conflict(low, high)

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
        "role_conflict": role_conflict,
        "role_conflict_reason": role_conflict_reason,
        "is_scotus_cafc_cert_bug": is_scotus_cafc_cert_bug,
    }
