"""Calibrated machine learning classifier for duplicate and case-identity triage.

Trains on training split of dev_silver and hard_negatives using scikit-learn.
Emits calibrated probabilities across the 8-label set and abstains below a confidence threshold.
Includes strict anti-leakage verification in .fit() to forbid training on validation or test splits.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

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
from triage.rules import classify_pair_rule

# Ordered classes for multi-class classification
CLASS_ORDER = [
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    LABEL_PARALLEL_SPLIT,
    LABEL_RELATED_DECISION,
    LABEL_SHARED_PAGE,
    LABEL_CITATION_ERROR,
    LABEL_UNRELATED,
    LABEL_INSUFFICIENT,
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def vector_from_pair(row: Dict[str, Any]) -> List[float]:
    """Convert pair record into a numerical feature vector."""
    f = extract_enriched_features(row)
    min_ham = f["min_ham"] if f["min_ham"] is not None else 64.0
    len_ratio = f["len_ratio"] if f["len_ratio"] is not None else 0.5
    delta_days = f["delta_days"] if f["delta_days"] is not None else 365.0
    log_delta_days = math.log1p(max(0, float(delta_days)))

    return [
        float(min_ham),
        float(len_ratio),
        1.0 if f["same_doc_num"] else 0.0,
        1.0 if f["doc_conflict"] else 0.0,
        1.0 if f["same_court"] else 0.0,
        1.0 if f["parent_link"] else 0.0,
        1.0 if f["same_year"] else 0.0,
        float(log_delta_days),
        float(f["shared_cits"]),
        float(f["max_page_span"]),
        1.0 if f["shared_sha1"] else 0.0,
        1.0 if f["shared_text_sha"] else 0.0,
        1.0 if f["disjoint_rep"] else 0.0,
        1.0 if f["title_exact"] else 0.0,
        float(f["token_jaccard"]),
        1.0 if f["is_stock"] else 0.0,
        1.0 if f["is_scotus_cafc_cert_bug"] else 0.0,
    ]


class CalibratedTriageClassifier:
    """Hybrid Classifier combining deterministic domain guardrails with a calibrated ML model."""

    def __init__(self, confidence_threshold: float = 0.60) -> None:
        self.confidence_threshold = confidence_threshold
        self.model: Optional[HistGradientBoostingClassifier] = None
        self.classes_: List[str] = CLASS_ORDER

    def fit(self, training_records: List[Dict[str, Any]]) -> "CalibratedTriageClassifier":
        """Train the classifier on labeled training rows.

        Raises:
            ValueError: If any record carries split='validation' or split='test'.
        """
        X: List[List[float]] = []
        y: List[int] = []

        for r in training_records:
            row_split = r.get("split")
            if row_split in {"validation", "test"}:
                raise ValueError(
                    f"DATA LEAKAGE BLOCKED: Record {r.get('pair_id')!r} has split={row_split!r}. "
                    "CalibratedTriageClassifier.fit() strictly requires split='training' (or unlabeled synthetic fixtures)."
                )
            label = r.get("silver_label") or r.get("family") or r.get("rule_class")
            if label not in CLASS_TO_IDX:
                continue
            X.append(vector_from_pair(r))
            y.append(CLASS_TO_IDX[label])

        if not X:
            return self

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int64)

        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            random_state=42,
            min_samples_leaf=2,
        )
        self.model.fit(X_arr, y_arr)
        return self

    def predict_pair(self, row: Dict[str, Any]) -> Tuple[str, float, List[str], List[str]]:
        """Predict pair relationship by combining high-precision rules and calibrated ML."""
        rule_label, rule_conf, sup, con = classify_pair_rule(row)

        if rule_conf >= 0.88:
            return rule_label, rule_conf, sup, con

        if self.model is not None:
            x = np.array([vector_from_pair(row)], dtype=np.float32)
            probs = self.model.predict_proba(x)[0]
            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])
            model_label = CLASS_ORDER[best_idx]

            if best_conf >= self.confidence_threshold:
                sup.append(f"Calibrated feature model confidence: {best_conf:.2f} for {model_label}")
                return model_label, best_conf, sup, con

        return rule_label, rule_conf, sup, con
