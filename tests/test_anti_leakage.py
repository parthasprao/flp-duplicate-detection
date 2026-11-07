"""Strict anti-cheating, anti-leakage, and anti-overfitting test suite.

This test suite enforces the non-negotiable guardrails for human and agentic developers:
1. Held-out `test` split records cannot be loaded via `load_development_splits` or `load_dataset(split='test')`
   without explicit unlock.
2. `CalibratedTriageClassifier.fit()` rejects any record with `split in {'validation', 'test'}`.
3. No `triage/*.py` module may contain hardcoded `pair_id`, `pair_uid`, or `cluster_id` lookup tables.
4. No code, test, or agent prompt file may reference any `test`-split `pair_uid` or `pair_id`.
5. The generalization gap (`train_accuracy - validation_accuracy`) must remain below 25 percentage points.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Set, Tuple

import pytest

from triage.bench import run_all_benches
from triage.io import (
    TestSetLockError,
    find_default_data_dir,
    load_dataset,
    load_development_splits,
)
from triage.model import CalibratedTriageClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent

# Static list of held-out test-split pair_uids so anti-leakage checks run even without local raw data
KNOWN_TEST_PAIR_UIDS: Set[str] = {
    # dev_silver test split (25)
    "dev-004", "dev-009", "dev-015", "dev-017", "dev-021",
    "dev-022", "dev-023", "dev-032", "dev-036", "dev-042",
    "dev-045", "dev-062", "dev-064", "dev-077", "dev-078",
    "dev-082", "dev-088", "dev-093", "dev-095", "dev-100",
    "dev-107", "dev-111", "dev-114", "dev-123", "dev-128",
    # adjudicate_me test split (2)
    "adj-010", "adj-011",
    # hard_negatives test split (23)
    "hn-silver-003", "hn-silver-007", "hn-silver-012", "hn-silver-015", "hn-silver-018",
    "hn-silver-027", "hn-silver-030", "hn-silver-045", "hn-silver-047", "hn-silver-056",
    "hn-silver-057", "hn-silver-060", "hn-silver-065", "hn-silver-068", "hn-silver-070",
    "hn-silver-080", "hn-silver-084", "hn-silver-086", "hn-silver-095", "hn-trap-100",
    "hn-trap-108", "hn-trap-115", "hn-trap-127",
}


def _collect_dynamic_test_ids() -> Tuple[Set[str], Set[str]]:
    """If the hackathon dataset directory is present, collect all split=='test' pair_uids and pair_ids."""
    uids = set(KNOWN_TEST_PAIR_UIDS)
    pids: Set[str] = set()
    data_dir = find_default_data_dir()
    for fname in ("dev_silver.jsonl", "hard_negatives.jsonl", "adjudicate_me.jsonl"):
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("split") == "test":
                        if row.get("pair_uid"):
                            uids.add(str(row["pair_uid"]))
                        if row.get("pair_id"):
                            pids.add(str(row["pair_id"]))
    return uids, pids


def test_load_dataset_blocks_test_split_by_default(tmp_path: Path) -> None:
    """load_dataset(..., split='test') and load_development_splits(..., splits=('test',)) must raise TestSetLockError."""
    dummy_file = tmp_path / "dummy.jsonl"
    dummy_file.write_text(
        json.dumps({"pair_id": "1-2", "split": "training", "features": {}}) + "\n"
        + json.dumps({"pair_id": "3-4", "split": "validation", "features": {}}) + "\n"
        + json.dumps({"pair_id": "5-6", "split": "test", "features": {}}) + "\n",
        encoding="utf-8",
    )

    # Training and validation load cleanly
    train_rows = load_development_splits(dummy_file, splits=("training",))
    val_rows = load_development_splits(dummy_file, splits=("validation",))
    both_rows = load_dataset(dummy_file, split="train_val")
    assert len(train_rows) == 1
    assert len(val_rows) == 1
    assert len(both_rows) == 2

    # Test split is strictly blocked
    with pytest.raises(TestSetLockError, match="TEST SET LOCKED"):
        load_dataset(dummy_file, split="test")

    with pytest.raises(TestSetLockError):
        load_development_splits(dummy_file, splits=("test",))

    with pytest.raises(TestSetLockError):
        load_development_splits(dummy_file, splits=("training", "test"))


def test_classifier_fit_rejects_validation_and_test_splits() -> None:
    """CalibratedTriageClassifier.fit() must raise ValueError if passed validation or test records."""
    clf = CalibratedTriageClassifier()

    val_record = {
        "pair_id": "100-200",
        "split": "validation",
        "silver_label": "probable_duplicate",
        "features": {"min_hamming": 0, "text_length_ratio": 1.0, "same_court": 1},
    }
    test_record = {
        "pair_id": "300-400",
        "split": "test",
        "silver_label": "unrelated",
        "features": {"min_hamming": 30, "text_length_ratio": 0.4, "same_court": 0},
    }

    with pytest.raises(ValueError, match="DATA LEAKAGE BLOCKED"):
        clf.fit([val_record])

    with pytest.raises(ValueError, match="DATA LEAKAGE BLOCKED"):
        clf.fit([test_record])


def test_cli_evaluate_blocks_test_split_without_unlock(tmp_path: Path) -> None:
    """Running `python -m triage.cli evaluate --split test` without `--unlock-test-set` must fail."""
    preds_file = tmp_path / "preds.jsonl"
    gold_file = tmp_path / "gold.jsonl"
    preds_file.write_text(
        json.dumps({
            "pair_id": "10-20",
            "label": "unrelated",
            "confidence": 0.9,
            "evidence": {"supporting": [], "contradicting": []},
        }) + "\n",
        encoding="utf-8",
    )
    gold_file.write_text(
        json.dumps({
            "pair_id": "10-20",
            "split": "test",
            "silver_label": "unrelated",
            "features": {},
        }) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "triage.cli",
            "evaluate",
            str(preds_file),
            "--gold",
            str(gold_file),
            "--split",
            "test",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "TestSetLockError" in proc.stderr or "TEST SET LOCKED" in proc.stderr


def test_no_hardcoded_pair_ids_in_triage_library() -> None:
    """Scan all `triage/*.py` modules to forbid hardcoded pair_uid or cluster-pair lookup tables."""
    triage_dir = REPO_ROOT / "triage"
    forbidden_uid_pattern = re.compile(r"""['"](?:dev|cand|hn-silver|hn-trap|adj)-\d{3}['"]""")
    forbidden_pair_id_pattern = re.compile(r"""['"]\d{5,9}-\d{5,9}['"]""")

    violations = []
    for py_file in sorted(triage_dir.glob("*.py")):
        content = py_file.read_text(encoding="utf-8")
        for m in forbidden_uid_pattern.finditer(content):
            violations.append(f"{py_file.name}: hardcoded pair_uid {m.group(0)}")
        for m in forbidden_pair_id_pattern.finditer(content):
            violations.append(f"{py_file.name}: hardcoded pair_id {m.group(0)}")

    assert not violations, (
        "ANTI-OVERFITTING VIOLATION: Hardcoded pair IDs or UIDs found in triage library:\n"
        + "\n".join(violations)
    )


def test_no_test_split_examples_referenced_in_code_tests_or_agent_prompts() -> None:
    """Ensure zero test-split pair_uids or pair_ids are referenced in code, unit tests, or agent prompts."""
    test_uids, test_pids = _collect_dynamic_test_ids()

    files_to_scan = []
    files_to_scan.extend((REPO_ROOT / "triage").glob("*.py"))
    files_to_scan.extend((REPO_ROOT / "app").glob("*.py"))
    for tf in (REPO_ROOT / "tests").glob("*.py"):
        if tf.name != "test_anti_leakage.py":
            files_to_scan.append(tf)
    files_to_scan.extend(REPO_ROOT.glob("AGENT_PROMPT_*.md"))
    for doc_name in ("AGENTS.md", "CLAUDE.md"):
        doc_path = REPO_ROOT / doc_name
        if doc_path.exists():
            files_to_scan.append(doc_path)

    leaks = []
    for fpath in sorted(files_to_scan):
        text = fpath.read_text(encoding="utf-8")
        for uid in sorted(test_uids):
            if re.search(rf"\b{re.escape(uid)}\b", text):
                leaks.append(f"{fpath.relative_to(REPO_ROOT)} references held-out test UID '{uid}'")
        for pid in sorted(test_pids):
            if re.search(rf"\b{re.escape(pid)}\b", text):
                leaks.append(f"{fpath.relative_to(REPO_ROOT)} references held-out test pair_id '{pid}'")

    assert not leaks, (
        "TEST SET CONTAMINATION DETECTED! Held-out test split IDs must never appear in code, tests, or agent prompts:\n"
        + "\n".join(leaks)
    )


def test_generalization_gap_guardrail() -> None:
    """When local dev_silver.jsonl is available, ensure train_acc - val_acc is within healthy bounds (< 0.25)."""
    data_dir = find_default_data_dir()
    if not (data_dir / "dev_silver.jsonl").exists():
        pytest.skip("Local hackathon dataset not present; skipping live generalization gap check.")

    results = run_all_benches(data_dir=data_dir, verbose=False)
    gen_gap = results["generalization_gap"]
    assert gen_gap < 0.25, (
        f"OVERFITTING DETECTED: Training accuracy exceeds validation accuracy by {gen_gap:.1%} "
        "(max allowed gap is 25.0%). Generalize your rules/features instead of memorizing training rows."
    )
