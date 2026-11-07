"""Tier 3: Priority Routing for the Maintainer Review Queue.

Assigns queue priority and actionable routing reasons:
- CRITICAL: Transitive graph contradictions and high-risk false merge traps.
- HIGH: Proposed database repairs (ClusterRedirection candidates) awaiting human confirmation.
- MEDIUM: Ambiguous pairs, rule-vs-silver disagreements, or insufficient evidence.
- LOW: Confident keep-separate pairs (unrelated/shared page) for background spot-checks.
"""

from __future__ import annotations

from typing import Any, Dict

from triage.constants import (
    LABEL_INSUFFICIENT,
    MERGE_LABELS,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
)


def assign_queue_priority(pair_prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Assign priority tier and routing reason to a prediction record."""
    pred = dict(pair_prediction)
    label = pred.get("label")
    conf = float(pred.get("confidence", 0.5))
    contradictions = pred.get("evidence", {}).get("contradicting", [])

    # If already set by graph contradiction check, preserve CRITICAL
    if pred.get("priority") == PRIORITY_CRITICAL:
        return pred

    # 1. Any contradictions on a merge candidate -> CRITICAL
    if label in MERGE_LABELS and contradictions:
        pred["priority"] = PRIORITY_CRITICAL
        pred["routing_reason"] = f"Proposed merge has contradicting evidence: {contradictions[0]}"
        return pred

    # 2. Clean merge proposals -> HIGH priority review queue
    if label in MERGE_LABELS:
        pred["priority"] = PRIORITY_HIGH
        pred["routing_reason"] = f"Actionable consolidation proposed ({label}, confidence: {conf:.2f})"
        return pred

    # 3. Insufficient evidence / low confidence -> MEDIUM
    if label == LABEL_INSUFFICIENT or conf < 0.70:
        pred["priority"] = PRIORITY_MEDIUM
        pred["routing_reason"] = "Confidence below threshold or ambiguous evidence; requires human review"
        return pred

    # 4. Confident keep-separate -> LOW
    pred["priority"] = PRIORITY_LOW
    pred["routing_reason"] = f"Confident keep-separate ({label}, confidence: {conf:.2f})"
    return pred
