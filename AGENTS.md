# Agentic Coding Guardrails & Evaluation Protocol (`AGENTS.md`)

**STOP AND READ BEFORE WRITING ANY CODE OR RUNNING ANY EVALUATION.**

This repository implements the **CourtListener Duplicate Detection & Case-Identity Triage System** for the Free Law Project (FLP). Both human teammates and AI coding agents must strictly follow these scientific integrity, anti-leakage, and data-governance rules.

---

## 1. Non-Negotiable Anti-Cheating & Anti-Overfitting Rules

### Rule 1: The `test` Split is Strictly Off-Limits (LOCKED)
- Every labeled dataset in `nyu-law-hackathon-2026-duplicates/student-packages/duplicates/data/` (`dev_silver.jsonl`, `hard_negatives.jsonl`, `adjudicate_me.jsonl`, `flp_version_pairs.jsonl`) contains a `"split"` field with values `"training"`, `"validation"`, or `"test"`.
- **NEVER** inspect, `grep`, print, load, train on, or tune rules/thresholds against rows where `split == "test"`.
- Programmatic enforcement:
  - Always load development data using `triage.io.load_development_splits(path, splits=("training", "validation"))` or `load_dataset(path, split="training" | "validation")`.
  - Calling `load_dataset(..., split="test")` or `python -m triage.cli evaluate --split test` raises `TestSetLockError` by default.
  - **DO NOT** pass `allow_test_split=True` or `--unlock-test-set` during development or agent sessions. Any unlock is logged to `outputs/.test_access_audit.log`.

### Rule 2: No Lookup Tables, Pair IDs, or Case-Specific Memorization
- **NEVER** hardcode `pair_id` (e.g., `"138713-7383705"`), `pair_uid` (e.g., `"dev-019"`, `"cand-0001"`), `cluster_id`, or case-specific party names into `triage/*.py`.
- `tests/test_anti_leakage.py` scans all `.py` files in `triage/` and fails immediately if any pair ID, UID, or lookup dictionary is detected.
- Every feature in `triage/features.py` and rule in `triage/rules.py` must express a **generalizable legal or structural pattern** (e.g., stripping procedural prefixes from docket strings, checking reporter page cluster spans, comparing opinion vs. judgment headers) rather than memorizing individual rows.

### Rule 3: Train/Validation Split Discipline & Generalization Gap
- **`training` split** (`n=76` in `dev_silver`, `n=76` in `hard_negatives`): Used for fitting `CalibratedTriageClassifier.fit()` and discovering structural failure patterns. Passing `validation` or `test` records to `.fit()` raises `ValueError("DATA LEAKAGE BLOCKED")`.
- **`validation` split** (`n=27` in `dev_silver`, `n=28` in `hard_negatives`): Used to verify that your rules and features generalize to unseen pairs.
- **Generalization Gate**:
  - Before accepting any code change, run `python -m triage.cli bench --role all`.
  - A change is **valid** only if it improves (or maintains) **`validation`** metrics without degrading hard-negative safety (`0` false merges is the target).
  - If `train_accuracy - validation_accuracy >= 25.0%`, `tests/test_anti_leakage.py::test_generalization_gap_guardrail` will fail your build for overfitting.

### Rule 4: Free Law Project (FLP) Data Terms
- **NEVER** commit raw hackathon data files (`nyu-law-hackathon-2026-duplicates/` or any `flp_version_pairs*` data file) to git.
- Before every `git commit`, run `git status` and ensure no dataset files are staged.

---

## 2. How to Run the Test Benches & Verify Your Work

Always use the project virtual environment (`.venv/bin/pytest` and `.venv/bin/python3`):

### A. Run the Full Pytest Suite (Including Anti-Leakage & All Role Benches)
```bash
.venv/bin/pytest -v
```

### B. Run Role-Specific Pytest Benches
- **Role 1 (Ingestion Normalizer & Version Classifier)**:
  ```bash
  .venv/bin/pytest tests/test_bench_role1_normalizer.py tests/test_anti_leakage.py -v
  .venv/bin/python3 -m triage.cli bench --role 1
  ```
- **Role 2 (Citation Topology & Reporter Page Disambiguator)**:
  ```bash
  .venv/bin/pytest tests/test_bench_role2_citations.py tests/test_anti_leakage.py -v
  .venv/bin/python3 -m triage.cli bench --role 2
  ```
- **Role 3 (Litigation Lifecycle, Companion Decisions & Graph Consistency)**:
  ```bash
  .venv/bin/pytest tests/test_bench_role3_graph_lifecycle.py tests/test_anti_leakage.py -v
  .venv/bin/python3 -m triage.cli bench --role 3
  ```

### C. Run the Whole-Project Generalization & Safety Audit
```bash
.venv/bin/python3 -m triage.cli bench --role all
```
This prints side-by-side `training` vs. `validation` accuracy, Macro-F1, per-class F1, generalization gap (`Train - Val`), and Hard-Negative false-merge counts while keeping the `test` split locked.
