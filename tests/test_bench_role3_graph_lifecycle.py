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


# ---------------------------------------------------------------------------
# Synthetic fixtures for the Role 3 lifecycle rules (no real case data).
# ---------------------------------------------------------------------------

def _pair(cid_low, cid_high, name_low, name_high, feats, docket_low=None, docket_high=None, **extra):
    base = {
        "same_court": 1,
        "same_docket_number": 0,
        "docket_numbers_conflict": 0,
        "filing_date_delta_days": 0,
        "title_exact": 1,
        "shared_citations": 0,
        "cluster_id_low": cid_low,
        "cluster_id_high": cid_high,
    }
    base.update(feats)
    row = {
        "pair_id": f"{cid_low}-{cid_high}",
        "features": base,
        "cluster_low": {"cluster_id": cid_low, "case_name": name_low, "court_id": "xctapp",
                        "docket": {"docket_number": docket_low}},
        "cluster_high": {"cluster_id": cid_high, "case_name": name_high, "court_id": "xctapp",
                         "docket": {"docket_number": docket_high}},
    }
    row.update(extra)
    return row


def test_role3_companion_detection_is_per_side() -> None:
    """Only one side may carry the JUDGMENT heading; two judgments or two opinions are not companions."""
    judgment = "COURT OF APPEALS\n\n              JUDGMENT RENDERED MAY 1, 2020\n\n NO. 01-19-00001-CV\n"
    opinion = "COURT OF APPEALS\n\n NO. 01-19-00001-CV\n\n MEMORANDUM OPINION\n\n Appellant argues..."
    assert detect_judgment_vs_opinion({}, {}, judgment, opinion)[0] is True
    assert detect_judgment_vs_opinion({}, {}, judgment, judgment)[0] is False
    assert detect_judgment_vs_opinion({}, {}, opinion, opinion)[0] is False
    # Missing text on either side is not evidence of a companion pair.
    assert detect_judgment_vs_opinion({}, {}, judgment, "")[0] is False


def test_role3_companion_pair_with_moderate_length_ratio_is_related() -> None:
    """Judgment + opinion on one docket must be related_decision even when lengths are similar."""
    judgment = "COURT OF APPEALS\n\n                JUDGMENT\n\nAppellant v. State. Affirmed."
    opinion = "COURT OF APPEALS\n\n No. 06-20-00001-CR\n\n Appellant, v. The State\n\n Opinion by Justice"
    row = _pair(9001, 9002, "Doe v. State of Texas", "Doe v. State of Texas",
                {"same_docket_number": 1, "min_hamming": 21, "text_length_ratio": 0.6},
                excerpt_low=judgment, excerpt_high=opinion)
    label, conf, sup, _ = classify_pair_rule(row)
    assert label == LABEL_RELATED_DECISION
    assert any("judgment" in s.lower() for s in sup)


def test_role3_caption_role_conflict_is_related() -> None:
    """Same parties, but a different party is designated the movant: two separate motions."""
    row = _pair(9011, 9012, "Acme Group v. Roe", "Acme Group v. Roe",
                {"shared_citations": 1, "max_page_cluster_span": 3, "min_hamming": 0, "text_length_ratio": 1.0})
    row["cluster_low"]["case_name_full"] = "ACME GROUP v. JANE ROE, AND BETA CLINIC, DEFENDANT-MOVANT"
    row["cluster_high"]["case_name_full"] = "ACME GROUP v. JANE ROE, DEFENDANT-MOVANT, AND BETA CLINIC"
    label, _, _, _ = classify_pair_rule(row)
    assert label == LABEL_RELATED_DECISION
    assert label not in MERGE_LABELS


def test_role3_distinctive_caption_with_different_dockets_is_related() -> None:
    """A distinctive caption under two docket numbers in one court is the same litigation."""
    row = _pair(9021, 9022, "Railroad Co.'s Tax Assessment", "Railroad Co.'s Tax Assessment",
                {"docket_numbers_conflict": 1, "min_hamming": 0, "text_length_ratio": 1.0},
                docket_low="Appeal No. 5", docket_high="Appeal No. 7")
    label, _, _, _ = classify_pair_rule(row)
    assert label == LABEL_RELATED_DECISION


@pytest.mark.parametrize(
    "caption",
    ["State v. Smith", "Brown v. State", "Com. v. Taylor, J.", "People v. Young CA2/6", "In re I.F."],
)
def test_role3_generic_caption_with_different_dockets_stays_unrelated(caption) -> None:
    """Surname-only government captions recur across unrelated defendants; never call them related."""
    row = _pair(9031, 9032, caption, caption,
                {"docket_numbers_conflict": 1, "filing_date_delta_days": 200, "min_hamming": 30, "text_length_ratio": 0.6})
    label, _, _, _ = classify_pair_rule(row)
    assert label == LABEL_UNRELATED


def test_role3_specific_defendant_caption_is_not_generic() -> None:
    """A government caption with a full defendant name is specific enough to signal one litigation."""
    row = _pair(9041, 9042, "United States v. John Q. Public", "United States v. John Q. Public",
                {"docket_numbers_conflict": 1, "filing_date_delta_days": 300, "min_hamming": 23, "text_length_ratio": 0.9})
    label, _, _, _ = classify_pair_rule(row)
    assert label == LABEL_RELATED_DECISION


def test_role3_same_docket_different_day_different_text_is_related() -> None:
    """Two orders on one docket on different days with different text must not merge,
    even when the short excerpts (caption headers) look identical."""
    header = "UNITED STATES DISTRICT COURT\nFOR THE MIDDLE DISTRICT\n\nJOHN DOE, Plaintiff, v. TOWNSHIP, Defendant."
    row = _pair(9051, 9052, "Doe v. Township", "Doe v. Township",
                {"same_docket_number": 1, "filing_date_delta_days": 1, "min_hamming": 20, "text_length_ratio": 0.67},
                excerpt_low=header, excerpt_high=header)
    label, _, _, _ = classify_pair_rule(row)
    assert label == LABEL_RELATED_DECISION


def test_role3_hub_cluster_escalated_without_downgrade() -> None:
    """A cluster with >= 4 proposed merges is escalated to CRITICAL but keeps its merge label."""
    hub = 100
    spokes = [101, 102, 103, 104]
    records = [{"pair_id": f"{hub}-{s}", "features": {"cluster_id_low": hub, "cluster_id_high": s}} for s in spokes]
    preds = [
        {"pair_id": r["pair_id"], "label": LABEL_PROBABLE_DUPLICATE, "confidence": 0.9,
         "evidence": {"supporting": [], "contradicting": []}}
        for r in records
    ]
    comps = build_connected_components(records)
    reconciled = {p["pair_id"]: assign_queue_priority(p) for p in reconcile_graph_predictions(preds, comps)}
    for r in records:
        rec = reconciled[r["pair_id"]]
        assert rec["priority"] == PRIORITY_CRITICAL
        assert rec["label"] == LABEL_PROBABLE_DUPLICATE
        assert any("Hub cluster" in w for w in rec["evidence"]["contradicting"])

    # Three merges is below the hub threshold: no hub warning.
    small = records[:3]
    comps_small = build_connected_components(small)
    out = reconcile_graph_predictions(preds[:3], comps_small)
    assert all(not any("Hub cluster" in w for w in p["evidence"]["contradicting"]) for p in out)


def test_role3_bench_lifecycle_false_merges_do_not_regress() -> None:
    """Validation lifecycle/caption hard-negative false merges must stay at or below 2/16.

    The two remaining rows are feature-generated trap rows (no human or AI label);
    see reports/adjudications.md before tightening rules to fit them.
    """
    data_dir = find_default_data_dir()
    if not (data_dir / "hard_negatives.jsonl").exists():
        pytest.skip("Local hackathon dataset not present.")
    res = run_role3_bench(data_dir=data_dir, verbose=False)
    assert res["hn_life_val_false_merges"] <= 2
