"""Test Bench for Role 2: Citation Topology & Reporter Page Disambiguator.

Run this bench with:
    pytest tests/test_bench_role2_citations.py tests/test_anti_leakage.py -v
And inspect full training/validation metrics with:
    python -m triage.cli bench --role 2
"""

from __future__ import annotations

import pytest

from triage.bench import run_role2_bench
from triage.constants import (
    LABEL_CITATION_ERROR,
    LABEL_CITATION_TOPOLOGY_UNVERIFIED,
    LABEL_PROBABLE_VERSION,
    LABEL_SHARED_PAGE,
    MERGE_LABELS,
    PRIORITY_MEDIUM,
)
from triage.io import find_default_data_dir
from triage.features import extract_enriched_features
from triage.route import assign_queue_priority
from triage.rules import classify_pair_rule


def test_role2_shared_page_ambiguity_with_same_or_different_dockets() -> None:
    """Multiple distinct orders printed on the same reporter page (span >= 2) must not be merged."""
    shared_page_pair = {
        "pair_id": "3001-3002",
        "features": {
            "shared_citations": 1,
            "min_page_cluster_span": 4,
            "max_page_cluster_span": 4,
            "distinct_decisions_on_shared_page": 4,
            "same_court": 1,
            "same_docket_number": 0,
            "docket_numbers_conflict": 1,
            "min_hamming": 28,
            "text_length_ratio": 0.72,
            "title_exact": 0,
        },
        "cluster_low": {
            "case_name": "People v. Adams",
            "court_id": "nyappdiv",
            "docket": {"docket_number": "2018-0111"},
            "citations": [{"citation_key": "149 A.D.3d 416"}],
        },
        "cluster_high": {
            "case_name": "People v. Baker",
            "court_id": "nyappdiv",
            "docket": {"docket_number": "2018-0222"},
            "citations": [{"citation_key": "149 A.D.3d 416"}],
        },
    }
    label, conf, _, _ = classify_pair_rule(shared_page_pair)
    assert label not in MERGE_LABELS
    assert label == LABEL_SHARED_PAGE
    assert conf >= 0.85


def test_role2_single_case_page_citation_collision_is_data_error() -> None:
    """Conflicting records with one shared citation and no multi-case span are data defects."""
    pair = {
        "features": {
            "shared_citations": 1,
            "min_page_cluster_span": 1,
            "max_page_cluster_span": 1,
            "distinct_decisions_on_shared_page": 1,
            "same_court": 1,
            "docket_numbers_conflict": 1,
            "min_hamming": 28,
            "text_length_ratio": 0.7,
            "title_exact": 0,
        },
        "cluster_low": {"case_name": "People v. Adams", "citations": [{"citation_key": "149 A.D.3d 416"}]},
        "cluster_high": {"case_name": "People v. Baker", "citations": [{"citation_key": "149 A.D.3d 416"}]},
    }
    label, confidence, _, _ = classify_pair_rule(pair)
    assert label == LABEL_CITATION_ERROR
    assert label not in MERGE_LABELS
    assert confidence >= 0.85


def test_role2_page_span_alone_does_not_prove_multiple_cases_on_a_page() -> None:
    """Page-span length is not physical page occupancy; missing proof routes for review."""
    pair = {
        "features": {
            "shared_citations": None,
            "min_page_cluster_span": "4",
            "max_page_cluster_span": "4",
            "same_court": 1,
            "docket_numbers_conflict": 1,
            "min_hamming": 29,
            "text_length_ratio": 0.4,
            "title_exact": 0,
        },
        "cluster_low": {"citations": [{"citation_key": "149 A.D.3d 416"}]},
        "cluster_high": {"citations": [{"citation_key": "149 AD3d 416"}]},
    }
    enriched = extract_enriched_features(pair)
    assert enriched["shared_cits"] == 1
    assert enriched["max_page_span"] == 4
    assert enriched["distinct_decisions_on_shared_page"] is None
    label, confidence, _, _ = classify_pair_rule(pair)
    assert label == LABEL_CITATION_TOPOLOGY_UNVERIFIED
    assert label not in MERGE_LABELS
    assert confidence >= 0.90
    routed = assign_queue_priority({"label": label, "confidence": confidence})
    assert routed["priority"] == PRIORITY_MEDIUM


def test_role2_conflicting_dockets_on_shared_page_never_become_versions() -> None:
    """Docket conflict on a multi-case page blocks a merge even with matching text/caption."""
    pair = {
        "features": {
            "shared_citations": 1,
            "min_page_cluster_span": 3,
            "max_page_cluster_span": 3,
            "distinct_decisions_on_shared_page": 3,
            "same_court": 1,
            "same_docket_number": 0,
            "docket_numbers_conflict": 1,
            "min_hamming": 2,
            "text_length_ratio": 0.96,
            "title_exact": 1,
        },
        "cluster_low": {"case_name": "Matter of Green", "citations": [{"citation_key": "155 A.D.3d 900"}]},
        "cluster_high": {"case_name": "Matter of Green", "citations": [{"citation_key": "155 A.D.3d 900"}]},
    }
    label, _, _, _ = classify_pair_rule(pair)
    assert label == LABEL_SHARED_PAGE
    assert label not in MERGE_LABELS


def test_role2_shared_page_but_same_case_is_version() -> None:
    """If two records share a multi-case page citation AND have matching caption, court, and text, it is a version."""
    same_case_on_shared_page = {
        "pair_id": "3003-3004",
        "features": {
            "shared_citations": 1,
            "min_page_cluster_span": 3,
            "max_page_cluster_span": 3,
            "distinct_decisions_on_shared_page": 3,
            "same_court": 1,
            "same_docket_number": 1,
            "docket_numbers_conflict": 0,
            "min_hamming": 3,
            "text_length_ratio": 0.96,
            "title_exact": 1,
        },
        "cluster_low": {
            "case_name": "Matter of Green",
            "court_id": "nyappdiv",
            "docket": {"docket_number": "2019-0500"},
            "citations": [{"citation_key": "155 A.D.3d 900"}],
        },
        "cluster_high": {
            "case_name": "Matter of Green",
            "court_id": "nyappdiv",
            "docket": {"docket_number": "2019-0500"},
            "citations": [{"citation_key": "155 A.D.3d 900"}],
        },
    }
    label, conf, _, _ = classify_pair_rule(same_case_on_shared_page)
    assert label in MERGE_LABELS


def test_role2_scotus_cafc_certiorari_order_defect() -> None:
    """Detect the U.S. Reports Volume 543 SCOTUS / CAFC scraper misattribution bug."""
    scotus_cafc_pair = {
        "pair_id": "3005-3006",
        "features": {
            "shared_citations": 1,
            "min_page_cluster_span": 7,
            "max_page_cluster_span": 7,
            "distinct_decisions_on_shared_page": 7,
            "same_court": 0,
            "docket_numbers_conflict": 1,
            "min_hamming": 30,
            "text_length_ratio": 0.85,
        },
        "cluster_low": {
            "case_name": "Alpha Corp. v. Beta Corp.",
            "court_id": "scotus",
            "docket": {"docket_number": "04-101"},
            "citations": [{"citation_key": "543 U.S. 925"}],
        },
        "cluster_high": {
            "case_name": "Gamma Inc. v. United States",
            "court_id": "cafc",
            "docket": {"docket_number": "04-202"},
            "citations": [{"citation_key": "543 U.S. 925"}],
        },
    }
    label, conf, sup, con = classify_pair_rule(scotus_cafc_pair)
    assert label == LABEL_SHARED_PAGE
    assert label not in MERGE_LABELS
    assert conf >= 0.90


def test_role2_benchmark_runs_strictly_on_train_and_val() -> None:
    """Verify run_role2_bench executes cleanly on training and validation splits with zero hard-negative false merges."""
    data_dir = find_default_data_dir()
    if not (data_dir / "dev_silver.jsonl").exists():
        pytest.skip("Local hackathon dataset not present.")

    res = run_role2_bench(data_dir=data_dir, verbose=False)
    assert res["train"]["total"] == 76
    assert res["validation"]["total"] == 27
    # Zero false merges on citation/shared-page hard negatives in validation
    assert res["hn_cit_val_false_merges"] == 0
