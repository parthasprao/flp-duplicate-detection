"""I/O utilities for loading CourtListener candidates, silver labels, excerpts, and exporting predictions.

Includes strict anti-leakage guardrails to prevent accidental or unauthorized access to held-out `test` split rows.
"""

from __future__ import annotations

import csv
import functools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from triage.constants import VALID_LABELS

ALLOWED_DEV_SPLITS: Set[str] = {"training", "validation"}


class TestSetLockError(PermissionError):
    """Raised when an operation attempts to access or evaluate the held-out test split without explicit unlock."""

    __test__ = False


def find_default_data_dir() -> Path:
    """Locate the default hackathon data directory."""
    env_dir = os.environ.get("TRIAGE_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    candidates = [
        Path("nyu-law-hackathon-2026-duplicates/student-packages/duplicates/data"),
        Path("../nyu-law-hackathon-2026-duplicates/student-packages/duplicates/data"),
        Path("data"),
    ]
    for c in candidates:
        if c.exists() and (c / "candidates.jsonl").exists():
            return c
    # Fallback to current directory
    return Path(".")


@functools.lru_cache(maxsize=4)
def load_excerpts(excerpts_path: Union[str, Path]) -> Dict[int, str]:
    """Load cluster excerpts into a map of cluster_id -> excerpt text."""
    path = Path(excerpts_path)
    if not path.exists():
        return {}

    excerpts: Dict[int, str] = {}
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = row.get("cluster_id")
                exc = row.get("excerpt", "")
                if cid is not None and exc:
                    excerpts[int(cid)] = exc.strip()
    elif path.suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("cluster_id")
                exc = row.get("excerpt", "")
                if cid is not None and exc:
                    try:
                        excerpts[int(cid)] = exc.strip()
                    except ValueError:
                        pass
    return excerpts


def _normalize_flp_version_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an opinion-level row from flp_version_pairs into the canonical pair structure."""
    dup_op_id = row.get("duplicate_opinion_id")
    main_op_id = row.get("main_opinion_id")
    pair_id = str(row.get("pair_id") or f"flp-{dup_op_id}-{main_op_id}")

    exc_low = (row.get("duplicate_opinion_excerpt") or "").strip()
    exc_high = (row.get("main_opinion_excerpt") or "").strip()
    len_low = len(exc_low)
    len_high = len(exc_high)
    len_ratio = (
        (min(len_low, len_high) / max(len_low, len_high))
        if (len_low > 0 and len_high > 0)
        else (1.0 if row.get("same_page_count") else 0.85)
    )

    dup_doc = row.get("duplicate_docket_number")
    main_doc = row.get("main_docket_number")
    same_doc = 1 if (dup_doc and main_doc and str(dup_doc).strip() == str(main_doc).strip()) else 0
    same_court = 1 if (row.get("duplicate_court_id") == row.get("main_court_id")) else 0
    same_title = 1 if (row.get("duplicate_case_name") == row.get("main_case_name")) else 0
    sim = row.get("excerpt_similarity")
    est_hamming = int(round((1.0 - float(sim)) * 32)) if sim is not None else (0 if row.get("same_sha1") else 6)

    row["pair_id"] = pair_id
    row["excerpt_low"] = exc_low
    row["excerpt_high"] = exc_high
    row["features"] = {
        "cluster_id_low": row.get("duplicate_cluster_id") or dup_op_id,
        "cluster_id_high": row.get("main_cluster_id") or main_op_id,
        "same_docket_id": 1 if row.get("same_cluster") else same_doc,
        "same_docket_number": 1 if row.get("same_cluster") else same_doc,
        "docket_numbers_conflict": 0,
        "same_court": same_court,
        "court_parent_link": 0,
        "title_exact": same_title,
        "title_sorted_exact": same_title,
        "same_filing_year": 1 if row.get("duplicate_date_filed") == row.get("main_date_filed") else 0,
        "filing_date_delta_days": row.get("created_gap_days") or 0,
        "shared_citations": 1 if (row.get("duplicate_citations") and row.get("duplicate_citations") == row.get("main_citations")) else 0,
        "disjoint_reporters": 0,
        "min_page_cluster_span": 1,
        "max_page_cluster_span": 1,
        "same_precedential": 1 if row.get("duplicate_precedential_status") == row.get("main_precedential_status") else 0,
        "same_source": 1 if row.get("duplicate_source") == row.get("main_source") else 0,
        "shared_sha1": 1 if row.get("same_sha1") else 0,
        "shared_text_sha256": 1 if (row.get("same_sha1") or sim == 1.0) else 0,
        "min_hamming": est_hamming,
        "text_length_ratio": len_ratio,
        "same_download_url": bool(row.get("same_download_url")),
    }
    row["cluster_low"] = {
        "cluster_id": row.get("duplicate_cluster_id") or dup_op_id,
        "case_name": row.get("duplicate_case_name") or "",
        "court_id": row.get("duplicate_court_id") or "",
        "date_filed": row.get("duplicate_date_filed"),
        "source": row.get("duplicate_source"),
        "precedential_status": row.get("duplicate_precedential_status"),
        "docket": {"docket_number": dup_doc},
        "court": {"short_name": row.get("duplicate_court")},
        "citations": [{"citation_key": c} for c in (row.get("duplicate_citations") or [])],
    }
    row["cluster_high"] = {
        "cluster_id": row.get("main_cluster_id") or main_op_id,
        "case_name": row.get("main_case_name") or "",
        "court_id": row.get("main_court_id") or "",
        "date_filed": row.get("main_date_filed"),
        "source": row.get("main_source"),
        "precedential_status": row.get("main_precedential_status"),
        "docket": {"docket_number": main_doc},
        "court": {"short_name": row.get("main_court")},
        "citations": [{"citation_key": c} for c in (row.get("main_citations") or [])],
    }
    return row


def _normalize_row_from_json(row: Dict[str, Any], excerpts: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """Ensure consistent structure for JSONL candidate/silver/FLP rows."""
    if "duplicate_opinion_id" in row and "features" not in row:
        return _normalize_flp_version_row(row)

    pair_id = str(row.get("pair_id", ""))
    features = row.get("features", {})
    low = row.get("cluster_low", {})
    high = row.get("cluster_high", {})

    cid_low = features.get("cluster_id_low") or low.get("cluster_id")
    cid_high = features.get("cluster_id_high") or high.get("cluster_id")

    if not pair_id and cid_low and cid_high:
        pair_id = f"{cid_low}-{cid_high}"

    exc_low = row.get("excerpt_low", "")
    exc_high = row.get("excerpt_high", "")

    if excerpts:
        if not exc_low and cid_low and int(cid_low) in excerpts:
            exc_low = excerpts[int(cid_low)]
        if not exc_high and cid_high and int(cid_high) in excerpts:
            exc_high = excerpts[int(cid_high)]

    row["pair_id"] = pair_id
    row["excerpt_low"] = exc_low
    row["excerpt_high"] = exc_high
    row["features"] = features
    row["cluster_low"] = low
    row["cluster_high"] = high
    return row


def _normalize_row_from_csv(row: Dict[str, Any], excerpts: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """Convert flattened CSV row (feat_*, low_*, high_*) into canonical nested dictionary."""
    pair_id = row.get("pair_id", "")
    features: Dict[str, Any] = {}
    cluster_low: Dict[str, Any] = {}
    cluster_high: Dict[str, Any] = {}
    top_level: Dict[str, Any] = {}

    for k, v in row.items():
        if v == "" or v is None:
            parsed_v: Any = None
        elif v == "True":
            parsed_v = True
        elif v == "False":
            parsed_v = False
        else:
            try:
                if "." in v:
                    parsed_v = float(v)
                else:
                    parsed_v = int(v)
            except ValueError:
                parsed_v = v

        if k.startswith("feat_"):
            features[k[5:]] = parsed_v
        elif k.startswith("low_"):
            cluster_low[k[4:]] = parsed_v
        elif k.startswith("high_"):
            cluster_high[k[5:]] = parsed_v
        else:
            top_level[k] = parsed_v

    if "duplicate_opinion_id" in top_level and not features:
        return _normalize_flp_version_row(top_level)

    cid_low = features.get("cluster_id_low") or cluster_low.get("cluster_id")
    cid_high = features.get("cluster_id_high") or cluster_high.get("cluster_id")

    if not pair_id and cid_low and cid_high:
        pair_id = f"{cid_low}-{cid_high}"

    exc_low = top_level.get("excerpt_low", "") or ""
    exc_high = top_level.get("excerpt_high", "") or ""

    if excerpts:
        if not exc_low and cid_low and int(cid_low) in excerpts:
            exc_low = excerpts[int(cid_low)]
        if not exc_high and cid_high and int(cid_high) in excerpts:
            exc_high = excerpts[int(cid_high)]

    top_level["pair_id"] = pair_id
    top_level["features"] = features
    top_level["cluster_low"] = cluster_low
    top_level["cluster_high"] = cluster_high
    top_level["excerpt_low"] = exc_low
    top_level["excerpt_high"] = exc_high
    return top_level


def _audit_log_test_access(path: Path, split: Optional[str], caller_reason: str) -> None:
    """Append an entry to outputs/.test_access_audit.log whenever test split is unlocked."""
    audit_path = Path("outputs/.test_access_audit.log")
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file": str(path),
            "split": split,
            "reason": caller_reason,
        }
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def load_dataset(
    path: Union[str, Path],
    excerpts_path: Optional[Union[str, Path]] = None,
    split: Optional[Union[str, Iterable[str]]] = None,
    allow_test_split: bool = False,
) -> List[Dict[str, Any]]:
    """Load a dataset from JSONL or CSV and normalize it into canonical format.

    Args:
        path: Path to .jsonl or .csv dataset.
        excerpts_path: Optional path to excerpts.jsonl/.csv.
        split: Optional split filter ('training', 'validation', 'train_val', or an iterable of splits).
        allow_test_split: Must be True to explicitly load 'test' split rows when filtering by split='test'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    excerpts: Optional[Dict[int, str]] = None
    if excerpts_path:
        excerpts = load_excerpts(excerpts_path)
    else:
        same_dir_excerpts = p.parent / "excerpts.jsonl"
        if same_dir_excerpts.exists():
            excerpts = load_excerpts(same_dir_excerpts)

    records: List[Dict[str, Any]] = []
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                records.append(_normalize_row_from_json(row, excerpts))
    elif p.suffix == ".csv":
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(_normalize_row_from_csv(row, excerpts))
    else:
        raise ValueError(f"Unsupported format: {p.suffix}. Expected .jsonl or .csv")

    if split is not None:
        if isinstance(split, str):
            if split == "train_val":
                allowed_splits = {"training", "validation"}
            else:
                allowed_splits = {split}
        else:
            allowed_splits = set(split)

        if "test" in allowed_splits and not allow_test_split:
            raise TestSetLockError(
                "TEST SET LOCKED: Accessing split='test' is forbidden during iterative development and agent tuning. "
                "Use split='training' or split='validation'. For a final end-of-project check, pass allow_test_split=True "
                "or --unlock-test-set on the CLI."
            )
        if "test" in allowed_splits and allow_test_split:
            _audit_log_test_access(p, str(split), "load_dataset(allow_test_split=True)")

        records = [r for r in records if r.get("split") in allowed_splits]

    return records


def load_development_splits(
    path: Union[str, Path],
    splits: Iterable[str] = ("training", "validation"),
) -> List[Dict[str, Any]]:
    """Safely load ONLY development splits (training and/or validation), hard-blocking test rows."""
    requested = set(splits)
    if "test" in requested or not requested.issubset(ALLOWED_DEV_SPLITS):
        raise TestSetLockError(
            f"load_development_splits only permits {ALLOWED_DEV_SPLITS}, got {requested}."
        )
    return load_dataset(path, split=requested, allow_test_split=False)


def write_predictions_jsonl(predictions: List[Dict[str, Any]], out_path: Union[str, Path]) -> None:
    """Save predictions in official JSONL submission format."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in predictions:
            label = p.get("label")
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label {label} in pair {p.get('pair_id')}")

            record = {
                "pair_id": p["pair_id"],
                "label": label,
                "confidence": round(float(p.get("confidence", 0.0)), 4),
                "evidence": {
                    "supporting": p.get("evidence", {}).get("supporting", []),
                    "contradicting": p.get("evidence", {}).get("contradicting", []),
                },
            }
            if "priority" in p:
                record["priority"] = p["priority"]
            if "routing_reason" in p:
                record["routing_reason"] = p["routing_reason"]
            f.write(json.dumps(record) + "\n")


def write_predictions_csv(predictions: List[Dict[str, Any]], out_path: Union[str, Path]) -> None:
    """Save predictions in official CSV format."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pair_id", "label", "confidence", "supporting_evidence", "contradicting_evidence"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in predictions:
            label = p.get("label")
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label {label} in pair {p.get('pair_id')}")
            sup = "; ".join(p.get("evidence", {}).get("supporting", []))
            con = "; ".join(p.get("evidence", {}).get("contradicting", []))
            writer.writerow({
                "pair_id": p["pair_id"],
                "label": label,
                "confidence": round(float(p.get("confidence", 0.0)), 4),
                "supporting_evidence": sup,
                "contradicting_evidence": con,
            })
