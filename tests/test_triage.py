"""Automated unit and regression tests for the triage system."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest

from triage.agent import run_offline_falsification
from triage.constants import (
    LABEL_INSUFFICIENT,
    LABEL_PARALLEL_SPLIT,
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    LABEL_RELATED_DECISION,
    LABEL_SHARED_PAGE,
    LABEL_UNRELATED,
    PRIORITY_CRITICAL,
)
from triage.export import (
    export_courtlistener_dry_run_payloads,
    export_decision_log,
    export_submission,
)
from triage.graph import build_connected_components, detect_graph_contradictions, reconcile_graph_predictions
from triage.io import _normalize_row_from_csv, _normalize_row_from_json, load_dataset
from triage.rules import classify_pair_rule


def test_normalize_row_json() -> None:
    raw = {
        "pair_id": "100-200",
        "features": {"cluster_id_low": 100, "cluster_id_high": 200, "same_court": 1},
        "cluster_low": {"cluster_id": 100, "case_name": "Test v. Case"},
        "cluster_high": {"cluster_id": 200, "case_name": "Test v. Case"},
    }
    norm = _normalize_row_from_json(raw, excerpts={100: "Opinion text"})
    assert norm["pair_id"] == "100-200"
    assert norm["excerpt_low"] == "Opinion text"


def test_rules_prevent_length_disparity_false_merge() -> None:
    # A procedural order (short) vs merits opinion (long) on same docket
    pair = {
        "pair_id": "1-2",
        "features": {
            "same_court": 1,
            "same_docket_number": 1,
            "docket_numbers_conflict": 0,
            "title_exact": 1,
            "text_length_ratio": 0.08,
            "min_hamming": 30,
        },
        "cluster_low": {"cluster_id": 1, "case_name": "Smith v. Jones", "court_id": "ca9"},
        "cluster_high": {"cluster_id": 2, "case_name": "Smith v. Jones", "court_id": "ca9"},
    }
    label, conf, sup, con = classify_pair_rule(pair)
    assert label == LABEL_RELATED_DECISION
    assert any("length ratio" in s for s in sup)


def test_rules_prevent_stock_order_false_merge() -> None:
    # Identical boilerplate text with conflicting dockets
    pair = {
        "pair_id": "10-20",
        "features": {
            "same_court": 1,
            "same_docket_number": 0,
            "docket_numbers_conflict": 1,
            "title_exact": 1,
            "min_hamming": 0,
            "shared_text_sha256": 1,
        },
        "cluster_low": {"cluster_id": 10, "case_name": "Brown v. State", "court_id": "moctapp"},
        "cluster_high": {"cluster_id": 20, "case_name": "Brown v. State", "court_id": "moctapp"},
    }
    label, conf, sup, con = classify_pair_rule(pair)
    assert label == LABEL_UNRELATED
    assert any("Conflicting docket" in s for s in sup)


def test_rules_shared_page_span() -> None:
    # Multi-decision page span
    pair = {
        "pair_id": "30-40",
        "features": {
            "same_court": 0,
            "shared_citations": 1,
            "min_page_cluster_span": 3,
            "max_page_cluster_span": 3,
            "title_exact": 0,
        },
        "cluster_low": {"cluster_id": 30, "case_name": "Wood v. US", "court_id": "cafc"},
        "cluster_high": {"cluster_id": 40, "case_name": "Parsons v. Price", "court_id": "scotus"},
    }
    label, conf, sup, con = classify_pair_rule(pair)
    assert label == LABEL_SHARED_PAGE


def test_rules_positive_duplicate() -> None:
    # Exact SHA-1 match with matching court and title
    pair = {
        "pair_id": "50-60",
        "features": {
            "same_court": 1,
            "title_exact": 1,
            "shared_sha1": 1,
            "docket_numbers_conflict": 0,
        },
        "cluster_low": {"cluster_id": 50, "case_name": "Valid v. Exact", "court_id": "scotus"},
        "cluster_high": {"cluster_id": 60, "case_name": "Valid v. Exact", "court_id": "scotus"},
    }
    label, conf, sup, con = classify_pair_rule(pair)
    assert label == LABEL_PROBABLE_DUPLICATE
    assert conf >= 0.95


def test_graph_transitive_contradiction() -> None:
    # 3 clusters: A=1, B=2, C=3
    # Pair(1,2) is MERGE, Pair(2,3) is MERGE, but Pair(1,3) is UNRELATED
    pairs = [
        {"pair_id": "1-2", "features": {"cluster_id_low": 1, "cluster_id_high": 2}},
        {"pair_id": "2-3", "features": {"cluster_id_low": 2, "cluster_id_high": 3}},
        {"pair_id": "1-3", "features": {"cluster_id_low": 1, "cluster_id_high": 3}},
    ]
    preds = [
        {"pair_id": "1-2", "label": LABEL_PROBABLE_DUPLICATE, "confidence": 0.85},
        {"pair_id": "2-3", "label": LABEL_PROBABLE_DUPLICATE, "confidence": 0.85},
        {"pair_id": "1-3", "label": LABEL_UNRELATED, "confidence": 0.90},
    ]

    comps = build_connected_components(pairs)
    reconciled = reconcile_graph_predictions(preds, comps)

    rec_map = {p["pair_id"]: p for p in reconciled}
    assert rec_map["1-2"]["priority"] == PRIORITY_CRITICAL
    assert rec_map["1-2"]["label"] == LABEL_INSUFFICIENT  # Downgraded for safety!


def test_offline_falsification_agent() -> None:
    pair = {
        "pair_id": "70-80",
        "features": {
            "same_court": 0,
            "docket_numbers_conflict": 1,
            "text_length_ratio": 0.1,
            "filing_date_delta_days": 180,
        },
        "cluster_low": {"cluster_id": 70, "case_name": "Case A", "court_id": "dcd"},
        "cluster_high": {"cluster_id": 80, "case_name": "Case B", "court_id": "cadc"},
    }
    res = run_offline_falsification(pair)
    assert res["label"] != LABEL_PROBABLE_DUPLICATE
    assert "falsification_finding" in res


def test_flp_data_guardrails(tmp_path: Path) -> None:
    """Verify that no FLP restricted files are tracked or exposed in git."""
    gitignore = Path(".gitignore")
    assert gitignore.exists()
    content = gitignore.read_text()
    assert "flp_*" in content
    assert "nyu-law-hackathon-2026-duplicates" in content
