"""I/O utilities for loading CourtListener candidates, silver labels, excerpts, and exporting predictions."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from triage.constants import VALID_LABELS


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


def _normalize_row_from_json(row: Dict[str, Any], excerpts: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """Ensure consistent structure for JSONL candidate/silver rows."""
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


def load_dataset(
    path: Union[str, Path],
    excerpts_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Load a dataset from JSONL or CSV and normalize it into canonical format."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    excerpts: Optional[Dict[int, str]] = None
    if excerpts_path:
        excerpts = load_excerpts(excerpts_path)
    else:
        # Check if excerpts.jsonl lives in the same folder
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

    return records


def write_predictions_jsonl(predictions: List[Dict[str, Any]], out_path: Union[str, Path]) -> None:
    """Save predictions in official JSONL submission format."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in predictions:
            # Validate label
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
