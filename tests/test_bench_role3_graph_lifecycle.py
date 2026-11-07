"""Test Bench for Role 3: Litigation Lifecycle, Companion Decisions & Graph Consistency.

Run this bench with:
    pytest tests/test_bench_role3_graph_lifecycle.py tests/test_anti_leakage.py -v
And inspect full training/validation metrics with:
    python -m triage.cli bench --role 3
"""

from __future__ import annotations

import pytest

from triage.bench import run_role3_bench
from triage.constants import (
    LABEL_INSUFFICIENT,
    LABEL_PROBABLE_DUPLICATE,
    LABEL_RELATED_DECISION,
    LABEL_UNRELATED,
    MERGE_LABELS,
    PRIORITY_CRITICAL,
)
from triage.features import detect_judgment_vs_opinion
from triage.graph import (
    build_connected_components,
    detect_graph_contradictions,
    reconcile_graph_predictions,
)
from triage.io import find_default_data_dir
from triage.route import assign_queue_priority
from triage.rules import classify_pair_rule


def test_role3_companion_instrument_judgment_vs_opinion() -> None:
    """Detect separate formal Judgment vs Memorandum Opinion on the same docket and date."""
    exc_judgment = "JUDGMENT\nThe Court has considered this cause on the record and concludes that the judgment of the trial court should be affirmed."
    exc_opinion = "MEMORANDUM OPINION\nAppellant challenges the sufficiency of the evidence supporting his conviction. Having reviewed the record, we affirm."
    is_comp, reason = detect_judgment_vs_opinion(exc_judgment, exc_opinion)
    assert is_comp is True
    assert "judgment" in reason.lower()

    pair = {
        "pair_id": "4001-4002",
        "excerpt_low": exc_judgment,
        "excerpt_high": exc_opinion,
        "features": {
            "same_court": 1,
            "same_docket_number": 1,
            "docket_numbers_conflict": 0,
            "filing_date_delta_days": 0,
            "min_hamming": 29,
            "text_length_ratio": 0.14,
            "title_exact": 1,
        },
        "cluster_low": {"case_name": "Walker v. State", "court_id": "texapp", "docket": {"docket_number": "06-22-00010-CR"}},
        "cluster_high": {"case_name": "Walker v. State", "court_id": "texapp", "docket": {"docket_number": "06-22-00010-CR"}},
    }
    label, conf, sup, con = classify_pair_rule(pair)
    # Must never auto-merge as a simple duplicate when one is a 1-page judgment and the other is a full opinion
    assert label != LABEL_PROBABLE_DUPLICATE
    assert label == LABEL_RELATED_DECISION
    assert any("judgment" in s.lower() or "length ratio" in s.lower() for s in sup + con)


def test_role3_transitive_3_cycle_and_4_cycle_contradictions() -> None:
    """Verify graph engine intercepts both 3-node and 4-node transitive merge contradictions."""
    # 4-node cycle: 10==20, 20==30, 30==40 (all probable_duplicate), but 10-40 is unrelated
    records = [
        {"pair_id": "10-20", "features": {"cluster_id_low": 10, "cluster_id_high": 20}},
        {"pair_id": "20-30", "features": {"cluster_id_low": 20, "cluster_id_high": 30}},
        {"pair_id": "30-40", "features": {"cluster_id_low": 30, "cluster_id_high": 40}},
        {"pair_id": "10-40", "features": {"cluster_id_low": 10, "cluster_id_high": 40}},
    ]
    preds = [
        {"pair_id": "10-20", "label": LABEL_PROBABLE_DUPLICATE, "confidence": 0.85, "evidence": {"supporting": [], "contradicting": []}},
        {"pair_id": "20-30", "label": LABEL_PROBABLE_DUPLICATE, "confidence": 0.85, "evidence": {"supporting": [], "contradicting": []}},
        {"pair_id": "30-40", "label": LABEL_PROBABLE_DUPLICATE, "confidence": 0.85, "evidence": {"supporting": [], "contradicting": []}},
        {"pair_id": "10-40", "label": LABEL_UNRELATED, "confidence": 0.90, "evidence": {"supporting": [], "contradicting": []}},
    ]
    comps = build_connected_components(records)
    assert len(comps) == 1

    contradictions = detect_graph_contradictions(preds, comps)
    assert "10-40" in contradictions
    assert "10-20" in contradictions

    reconciled = reconcile_graph_predictions(preds, comps)
    routed = {p["pair_id"]: assign_queue_priority(p) for p in reconciled}

    # Conflicted moderate-confidence merges must be downgraded and escalated to CRITICAL
    assert routed["10-20"]["label"] == LABEL_INSUFFICIENT
    assert routed["10-20"]["priority"] == PRIORITY_CRITICAL
    assert routed["10-40"]["priority"] == PRIORITY_CRITICAL


def test_role3_multi_stage_litigation_vs_recurring_caption() -> None:
    """Verify parent/child court link or same-docket temporal gap yields related_decision rather than merge."""
    appellate_chain = {
        "pair_id": "5001-5002",
        "features": {
            "same_court": 0,
            "court_parent_link": 1,
            "same_docket_number": 0,
            "docket_numbers_conflict": 0,
            "filing_date_delta_days": 280,
            "min_hamming": 26,
            "text_length_ratio": 0.65,
            "title_exact": 1,
        },
        "cluster_low": {"case_name": "Acme Corp. v. Zenith LLC", "court_id": "nysd"},
        "cluster_high": {"case_name": "Acme Corp. v. Zenith LLC", "court_id": "ca2"},
    }
    label, conf, _, _ = classify_pair_rule(appellate_chain)
    assert label == LABEL_RELATED_DECISION
    assert label not in MERGE_LABELS
    assert conf >= 0.80


def test_role3_benchmark_runs_strictly_on_train_and_val() -> None:
    """Verify run_role3_bench executes cleanly on training (76) and validation (27) splits."""
    data_dir = find_default_data_dir()
    if not (data_dir / "dev_silver.jsonl").exists():
        pytest.skip("Local hackathon dataset not present.")

    res = run_role3_bench(data_dir=data_dir, verbose=False)
    assert res["train"]["total"] == 76
    assert res["validation"]["total"] == 27
