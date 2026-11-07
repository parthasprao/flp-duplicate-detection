# Agent Instructions: Ingestion Normalizer & Version Classifier (Role 1)

**Role:** Ingestion Normalizer & Version Disambiguator  
**Repository:** `flp-duplicate-detection`  
**Target Audience:** Free Law Project (FLP) / CourtListener Data Maintainers  
**Safety Mandate:** Wrongly merging distinct judicial decisions is catastrophic for citation networks and legal research. False merges must be prevented. However, superficial scraper formatting noise must not cause true duplicates or versions to be erroneously labeled as `unrelated`.

---

## 0. Mandatory Anti-Cheating & Anti-Overfitting Guardrails

Before writing any code, read [`AGENTS.md`](./AGENTS.md). You are strictly bound by the following rules (enforced automatically by `tests/test_anti_leakage.py`):
1. **THE `test` SPLIT IS LOCKED AND OFF-LIMITS**: Never inspect, filter for, load, or evaluate on rows with `split == "test"`. In Python, always use `triage.io.load_development_splits(path, splits=("training", "validation"))`. Never pass `allow_test_split=True` or `--unlock-test-set`.
2. **NO HARDCODED IDs OR CASE LOOKUPS**: Never hardcode `pair_id`, `pair_uid` (`dev-*`, `cand-*`, `hn-*`, `adj-*`), `cluster_id`, or case party names in `triage/*.py`. Every normalization rule must be a generalizable regex or structural transformation.
3. **TRAIN VS. VALIDATION GENERALIZATION**: Fit models only on `split == "training"`. Verify every rule or feature change on **both** `training` and `validation` splits using `python -m triage.cli bench --role 1`. Reject any change that improves `training` while hurting `validation` or increasing hard-negative false merges.

---

## 1. Context & Architecture

CourtListener ingests opinions from dozens of sources (scrapers, CAP/Harvard, RECAP, state feeds).
Our system operates a **3-Tier Ingestion Cascade**:
- **Tier 1:** Deterministic domain guardrails and calibrated feature scoring (`training`-fitted).
- **Tier 2:** Falsification Devil's Advocate agent.
- **Tier 3:** Priority review queue (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) with dry-run `ClusterRedirection` payloads.

### Your Module Ownership
You own the **docket normalization and version classification engine**:
- Primary files to inspect and modify:
  - `triage/features.py` (Feature extraction, docket normalization, text similarity)
  - `triage/rules.py` (Rules 6, 8, 9: `probable_duplicate`, `probable_version`, `parallel_split_cluster`)
  - `tests/test_bench_role1_normalizer.py` (Role 1 dedicated test bench)

---

## 2. Key Edge Cases & Failing Examples (`training` & `validation` Splits Only)

Currently, our pipeline suffers from two major error categories in the development splits of `dev_silver.jsonl`:

### Problem A: Scraper Formatting Noise Triggering False `docket_numbers_conflict`
CourtListener scrapers extract docket strings with varying noise, punctuation, judge initials, and court prefixes. The baseline feature `docket_numbers_conflict = 1` fires on strings that describe the exact same case, causing Rule 2 to prematurely classify them as `unrelated`!

**Concrete Failing Development Pairs (`training` & `validation` only):**
1. **`dev-019` `[training]`** (*Saldana v. Federal Bureau of Prisons*, D.D.C.):
   - Record Low: `Civil Action 08-1963 (JDB)` (Date: 2010-06-04)
   - Record High: `Civil Action No. 2008-1963` (Date: 2010-06-04)
   - Reality: Exact same case, `min_hamming = 2`, `text_length_ratio = 0.98`.
   - Current Bug: `docket_numbers_conflict = 1` falsely triggered `unrelated`. Gold Label: `probable_version`.
2. **`dev-091` `[validation]`** (*Robertson v. Cartinhour*, D.D.C.):
   - Record Low: `Civil Action 09-01642 (ESH)`
   - Record High: `Civil Action No. 2009-1642`
   - Current Bug: `docket_numbers_conflict = 1` falsely triggered `unrelated`. Gold Label: `probable_version` (`min_hamming = 1`, `text_length_ratio = 0.98`).
3. **`dev-117` `[training]`** (*Jimenez v. State*, Florida 3rd DCA):
   - Record Low: `3D14-3142` (Florida 3rd DCA prefix format)
   - Record High: `14-3142` (Core year-sequence)
   - Current Bug: `docket_numbers_conflict = 1` triggered `unrelated`. Gold Label: `probable_version` (`min_hamming = 0`, `text_length_ratio = 0.99`).
4. **`dev-055` `[validation]`** (*People v. Dorris*, Ill. App. Ct.):
   - Record Low: `4-99-0979 Rel`
   - Record High: `No. 4 — 99—0979` (em-dashes and `No.` prefix vs `Rel` suffix)
   - Current Bug: `docket_numbers_conflict = 1` triggered `unrelated`. Gold Label: `probable_version` (`min_hamming = 0`, `text_length_ratio = 0.96`).
5. **`dev-075` `[training]`** (*State v. Larabee*, Ohio Ct. App.):
   - Record Low: `No. 93-200`
   - Record High: `1993-0200` (2-digit year + unpadded number vs 4-digit year + zero-padded number)
   - Current Bug: `docket_numbers_conflict = 1` triggered `unrelated`. Gold Label: `probable_version` (`min_hamming = 5`, `text_length_ratio = 0.97`).

### Problem B: Confusion Between `probable_version` vs `parallel_split_cluster` vs `probable_duplicate`
When court, docket, and dates match:
- **`dev-016` `[training]`** (*Weikel v. TCW Realty*, Cal. Ct. App.): `min_hamming = 14`, `text_length_ratio = 0.62`. Our system predicted `parallel_split_cluster`, but gold is `probable_version`.
- **`dev-024` `[training]`** (*Price v. Akaka*, 9th Cir.): `min_hamming = 7`, `text_length_ratio = 0.96`, `shared_citations = 2`. Our system predicted `parallel_split_cluster`, but gold is `probable_version`.
- **`dev-025` `[training]`** (*McMurray v. Dream Catcher USA*, Ariz. Ct. App.): `min_hamming = 4`, `text_length_ratio = 0.94`. Our system predicted `probable_duplicate`, but gold is `probable_version`.

---

## 3. Your Action Plan

1. **Implement Generalizable Docket Normalization in `triage/features.py`**:
   - Write a general function `normalize_docket_number(docket_str: str) -> str` that:
     - Strips procedural prefixes (`Civil Action No.`, `Civil Action`, `Case No.`, `No.`, `Appeal No.`) and trailing judge initials or status tokens (`(JDB)`, `(ESH)`, `Rel`).
     - Normalizes Unicode dashes (`—`, `–`) to standard hyphens (`-`).
     - Normalizes state district prefixes (`3D14-3142` $\rightarrow$ `14-3142`) and 2-digit vs 4-digit year-sequence pairs (`08-1963` / `2008-1963`, `09-01642` / `2009-1642`, `93-200` / `1993-0200`).
   - Update `extract_enriched_features()` so when normalized core docket tokens match, `doc_conflict` is overridden to `False` and `same_doc_num` is set to `True`.
2. **Refine Decision Boundaries in `triage/rules.py`**:
   - **`probable_duplicate`**: Reserve for near-zero divergence (`shared_sha1 == 1` or `shared_text_sha256 == 1`, or `min_hamming <= 1` and `text_length_ratio >= 0.98`).
   - **`probable_version`**: Alternate revisions, slip vs. published (`min_hamming` between 1 and 18, `text_length_ratio >= 0.60`, matching court and canonical docket, without disjoint parallel reporters).
   - **`parallel_split_cluster`**: Only assign when `disjoint_reporters == 1` (true disjoint reporter families with `shared_citations == 0`) or separate majority/concurrence split clusters.

---

## 4. How to Run Your Test Bench & Verify Generalization

1. **Run Role 1 Unit Bench + Anti-Leakage Guardrails**:
   ```bash
   .venv/bin/pytest tests/test_bench_role1_normalizer.py tests/test_anti_leakage.py -v
   ```
2. **Run Role 1 Split-Safe Scorecard (`training` vs. `validation`)**:
   ```bash
   .venv/bin/python3 -m triage.cli bench --role 1
   ```
   Check that **Noisy-Docket Merge Recovery** improves on *both* `Train` (currently `4/11`) and `Val` (currently `1/6`), and that `probable_version` Validation F1 improves without dropping `FLP Version-Pairs Validation Recall` (`99.0%`).
3. **Run Whole-Project Generalization Check**:
   ```bash
   .venv/bin/python3 -m triage.cli bench --role all
   .venv/bin/pytest -v
   ```
   Verify that `Generalization Gap (Train - Val)` stays `< 25.0%` and `git status` shows zero raw data files staged.
