"""Export utilities for Hackathon submissions, Maintainer Decision Logs, and CourtListener Dry-Run Payloads."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from triage.constants import (
    CLUSTER_REDIRECTION_MAP,
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    MERGE_LABELS,
)


def export_submission(
    predictions: List[Dict[str, Any]],
    out_path: Union[str, Path],
    format: str = "jsonl",
) -> None:
    """Export predictions in the official hackathon submission format."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl" or out.suffix == ".jsonl":
        with open(out, "w", encoding="utf-8") as f:
            for p in predictions:
                rec = {
                    "pair_id": p["pair_id"],
                    "label": p["label"],
                    "confidence": round(float(p.get("confidence", 0.5)), 4),
                    "evidence": {
                        "supporting": p.get("evidence", {}).get("supporting", []),
                        "contradicting": p.get("evidence", {}).get("contradicting", []),
                    },
                }
                if "priority" in p:
                    rec["priority"] = p["priority"]
                if "routing_reason" in p:
                    rec["routing_reason"] = p["routing_reason"]
                if "component_id" in p:
                    rec["component_id"] = p["component_id"]
                f.write(json.dumps(rec) + "\n")
    elif format == "csv" or out.suffix == ".csv":
        fieldnames = ["pair_id", "label", "confidence", "supporting_evidence", "contradicting_evidence"]
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in predictions:
                writer.writerow({
                    "pair_id": p["pair_id"],
                    "label": p["label"],
                    "confidence": round(float(p.get("confidence", 0.5)), 4),
                    "supporting_evidence": "; ".join(p.get("evidence", {}).get("supporting", [])),
                    "contradicting_evidence": "; ".join(p.get("evidence", {}).get("contradicting", [])),
                })


def export_decision_log(
    decisions: List[Dict[str, Any]],
    out_path: Union[str, Path],
) -> None:
    """Log reviewer decisions to CSV for CourtListener maintainer audit."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "pair_id",
        "decision",
        "keep_cluster_id",
        "delete_cluster_id",
        "reviewer",
        "review_mode",
        "reason",
        "evidence_cited",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in decisions:
            writer.writerow({
                "timestamp": d.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "pair_id": d["pair_id"],
                "decision": d["decision"],
                "keep_cluster_id": d.get("keep_cluster_id", ""),
                "delete_cluster_id": d.get("delete_cluster_id", ""),
                "reviewer": d.get("reviewer", "human_maintainer"),
                "review_mode": d.get("review_mode", "co_pilot"),
                "reason": d.get("reason", ""),
                "evidence_cited": d.get("evidence_cited", ""),
            })


def export_courtlistener_dry_run_payloads(
    decisions_or_predictions: List[Dict[str, Any]],
    out_path: Union[str, Path],
) -> None:
    """Generate dry-run Django ORM payloads for CourtListener maintainers.

    Adheres strictly to the safety constraint:
    Output is purely proposed specs for ClusterRedirection and Opinion.main_version;
    NEVER executes real database migrations.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payloads: List[Dict[str, Any]] = []

    for item in decisions_or_predictions:
        label = item.get("label") or item.get("decision")
        if label not in MERGE_LABELS and label not in {"merge", "version"}:
            continue

        pair_id = item["pair_id"]
        parts = pair_id.split("-")
        if len(parts) != 2:
            continue

        try:
            cid_low, cid_high = int(parts[0]), int(parts[1])
        except ValueError:
            continue

        # In CourtListener, maintainers typically keep the earlier/more complete cluster
        target_cluster = item.get("keep_cluster_id") or cid_low
        deleted_cluster = item.get("delete_cluster_id") or cid_high
        if target_cluster == deleted_cluster:
            deleted_cluster = cid_high if target_cluster == cid_low else cid_low

        merge_type = LABEL_PROBABLE_VERSION if label == "version" else (label if label in MERGE_LABELS else LABEL_PROBABLE_DUPLICATE)
        redir_info = CLUSTER_REDIRECTION_MAP.get(merge_type, CLUSTER_REDIRECTION_MAP[LABEL_PROBABLE_DUPLICATE])

        payload = {
            "dry_run": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pair_id": pair_id,
            "operation": "ClusterRedirection.create_from_clusters",
            "django_command_spec": {
                "deleted_cluster_id": int(deleted_cluster),
                "target_cluster_id": int(target_cluster),
                "reason_code": redir_info["reason_code"],
                "reason_name": redir_info["reason_str"],
                "courtlistener_model": "cl.search.models.ClusterRedirection",
            },
            "audit_evidence": item.get("evidence", {}),
            "safety_note": "Dry-run only. Requires data maintainer sign-off before executing delete_duplicates or merge_opinion_versions.",
        }
        payloads.append(payload)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2)
