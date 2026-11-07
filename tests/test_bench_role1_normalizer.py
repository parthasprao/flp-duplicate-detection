"""Test Bench for Role 1: Ingestion Normalizer & Version Classifier.

Run this bench with:
    pytest tests/test_bench_role1_normalizer.py tests/test_anti_leakage.py -v
And inspect full training/validation metrics with:
    python -m triage.cli bench --role 1
"""

from __future__ import annotations

import pytest

from triage.bench import run_role1_bench
from triage.constants import (
    LABEL_PARALLEL_SPLIT,
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    MERGE_LABELS,
)
from triage.features import extract_enriched_features
from triage.io import _normalize_flp_version_row, find_default_data_dir
from triage.rules import classify_pair_rule


def test_role1_exact_duplicate_vs_version_boundary() -> None:
    """Verify exact duplicate (min_hamming=0, shared hash) vs revised version (min_hamming=6, len_ratio=0.92)."""
    exact_dup = {
        "pair_id": "1001-1002",
        "features": {
            "min_hamming": 0,
            "text_length_ratio": 1.0,
            "same_court": 1,
            "same_docket_number": 1,
            "docket_numbers_conflict": 0,
            "shared_sha1": 1,
            "shared_text_sha256": 1,
            "filing_date_delta_days": 0,
            "shared_citations": 1,
            "disjoint_reporters": 0,
            "title_exact": 1,
        },
        "cluster_low": {"case_name": "Doe v. Roe", "court_id": "ca9", "docket": {"docket_number": "22-1001"}},
        "cluster_high": {"case_name": "Doe v. Roe", "court_id": "ca9", "docket": {"docket_number": "22-1001"}},
    }
    label_dup, conf_dup, _, _ = classify_pair_rule(exact_dup)
    assert label_dup == LABEL_PROBABLE_DUPLICATE
    assert conf_dup >= 0.90

    version_pair = {
        "pair_id": "1003-1004",
        "features": {
            "min_hamming": 6,
            "text_length_ratio": 0.92,
            "same_court": 1,
            "same_docket_number": 1,
            "docket_numbers_conflict": 0,
            "shared_sha1": 0,
            "shared_text_sha256": 0,
            "filing_date_delta_days": 5,
            "shared_citations": 1,
            "disjoint_reporters": 0,
            "title_exact": 1,
        },
        "cluster_low": {"case_name": "Doe v. Roe", "court_id": "ca9", "docket": {"docket_number": "22-1001"}},
        "cluster_high": {"case_name": "Doe v. Roe", "court_id": "ca9", "docket": {"docket_number": "22-1001"}},
    }
    label_ver, conf_ver, _, _ = classify_pair_rule(version_pair)
    assert label_ver in {LABEL_PROBABLE_VERSION, LABEL_PARALLEL_SPLIT}
    assert conf_ver >= 0.80


def test_role1_parallel_split_cluster_disjoint_reporters() -> None:
    """When two clusters have same court, docket, and text, but disjoint reporters, detect parallel_split_cluster."""
    parallel_pair = {
        "pair_id": "2001-2002",
        "features": {
            "min_hamming": 2,
            "text_length_ratio": 0.98,
            "same_court": 1,
            "same_docket_number": 1,
            "docket_numbers_conflict": 0,
            "shared_citations": 0,
            "disjoint_reporters": 1,
            "filing_date_delta_days": 0,
            "title_exact": 1,
        },
        "cluster_low": {
            "case_name": "State v. Miller",
            "court_id": "ohioctapp",
            "docket": {"docket_number": "C-180001"},
            "citations": [{"citation_key": "2019-Ohio-1234"}],
        },
        "cluster_high": {
            "case_name": "State v. Miller",
            "court_id": "ohioctapp",
            "docket": {"docket_number": "C-180001"},
            "citations": [{"citation_key": "140 N.E.3d 555"}],
        },
    }
    label, conf, _, _ = classify_pair_rule(parallel_pair)
    assert label in MERGE_LABELS
    assert conf >= 0.80


def test_role1_flp_version_row_normalization() -> None:
    """Verify opinion-level rows from flp_version_pairs.jsonl normalize cleanly into the pair schema."""
    raw_flp = {
        "duplicate_opinion_id": 5001,
        "main_opinion_id": 5002,
        "duplicate_cluster_id": 9001,
        "main_cluster_id": 9002,
        "duplicate_case_name": "United States v. Baxter",
        "main_case_name": "United States v. Baxter",
        "duplicate_court_id": "ca4",
        "main_court_id": "ca4",
        "duplicate_date_filed": "2025-11-10",
        "main_date_filed": "2025-11-10",
        "duplicate_docket_number": "24-4012",
        "main_docket_number": "24-4012",
        "duplicate_citations": ["112 F.4th 800"],
        "main_citations": ["112 F.4th 800"],
        "same_cluster": False,
        "same_sha1": False,
        "same_page_count": True,
        "excerpt_similarity": 0.97,
        "duplicate_opinion_excerpt": "Appeal from the United States District Court...",
        "main_opinion_excerpt": "Appeal from the United States District Court...",
    }
    norm = _normalize_flp_version_row(raw_flp)
    assert norm["pair_id"] == "flp-5001-5002"
    assert norm["features"]["same_court"] == 1
    assert norm["features"]["same_docket_number"] == 1
    assert norm["features"]["min_hamming"] <= 2

    label, conf, _, _ = classify_pair_rule(norm)
    assert label in MERGE_LABELS
    assert conf >= 0.85


def test_role1_benchmark_runs_strictly_on_train_and_val() -> None:
    """Verify run_role1_bench executes cleanly and evaluates strictly on training (76) and validation (27)."""
    data_dir = find_default_data_dir()
    if not (data_dir / "dev_silver.jsonl").exists():
        pytest.skip("Local hackathon dataset not present.")

    res = run_role1_bench(data_dir=data_dir, verbose=False)
    assert res["train"]["total"] == 76
    assert res["validation"]["total"] == 27
    assert res["train"]["merge_precision"] >= 0.80
    if res["flp_val_recall"] is not None:
        assert res["flp_val_recall"] >= 0.85
