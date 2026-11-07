# Agent Instructions: Ingestion Normalizer & Version Classifier (Role 1)

**Role:** Ingestion Normalizer & Version Disambiguator  
**Repository:** `flp-duplicate-detection`  
**Target Audience:** Free Law Project (FLP) / CourtListener Data Maintainers  
**Safety Mandate:** Wrongly merging distinct judicial decisions is catastrophic for citation networks and legal research. False merges must be prevented. However, superficial scraper formatting noise must not cause true duplicates or versions to be erroneously labeled as `unrelated`.

---

## 1. Context & Architecture

CourtListener ingests opinions from dozens of sources (scrapers, CAP/Harvard, RECAP, state feeds).
Our system operates a **3-Tier Ingestion Cascade**:
- **Tier 1:** Deterministic domain guardrails and calibrated feature scoring.
- **Tier 2:** Falsification Devil's Advocate agent.
- **Tier 3:** Priority review queue (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) with dry-run `ClusterRedirection` payloads.

### Your Module Ownership
You own the **docket normalization and version classification engine**:
- Primary files to inspect and modify:
  - `triage/features.py` (Feature extraction, docket normalization, text similarity)
  - `triage/rules.py` (Rules 6, 8, 9: `probable_duplicate`, `probable_version`, `parallel_split_cluster`)
  - `tests/test_triage.py` (Add regression tests for your edge cases)

---

## 2. Key Edge Cases & Failing Examples You Must Fix

Currently, our pipeline suffers from two major error categories in `dev_silver.jsonl`:

### Problem A: Scraper Formatting Noise Triggering False `docket_numbers_conflict`
CourtListener scrapers extract docket strings with varying noise, punctuation, and court prefixes. The baseline feature `docket_numbers_conflict = 1` fires on strings that describe the exact same case, causing Rule 2 to prematurely classify them as `unrelated`!

**Concrete Failing Test Pairs:**
1. **`dev-019`** (*Saldana v. Federal Bureau of Prisons*, D.D.C.):
   - Record Low: `Civil Action 08-1963 (JDB)` (Date: 2010-06-04)
   - Record High: `Civil Action No. 2008-1963` (Date: 2010-06-04)
   - Reality: Exact same case, `min_hamming = 2`, `text_length_ratio = 0.98`.
   - Current Bug: `docket_numbers_conflict = 1` falsely triggered `unrelated`.
   - Gold Silver Label: `probable_version`.
2. **`dev-117`** (*Jimenez v. State*, Florida 3rd DCA):
   - Record Low: `3D14-3142` (Florida 3rd DCA format)
   - Record High: `14-3142` (Raw sequence)
   - Current Bug: `docket_numbers_conflict = 1` triggered `unrelated`.
   - Gold Silver Label: `probable_version` (`min_hamming = 0`, `text_length_ratio = 0.99`).
3. **`dev-023`** (*In re H.N.S.*, Missouri Court of Appeals):
   - Record Low: `SD 30826`
   - Record High: `No. SD 80826` (or spacing/typo variant `SD 30826`)
   - Gold Silver Label: `probable_version` (`min_hamming = 3`, `text_length_ratio = 0.98`).

### Problem B: Confusion Between `probable_version` vs `parallel_split_cluster` vs `probable_duplicate`
When court, docket, and dates match:
- **`dev-015`** (*US v. Melick*, D.N.H.): `min_hamming = 8`, `text_length_ratio = 0.89`. Our system predicted `parallel_split_cluster`, but gold is `probable_version`.
- **`dev-016`** (*Weikel v. TCW Realty*, Cal. Ct. App.): `min_hamming = 14`, `text_length_ratio = 0.62`. Our system predicted `parallel_split_cluster`, but gold is `probable_version`.
- **`dev-025`** (*McMurray v. Dream Catcher USA*, Ariz. Ct. App.): `min_hamming = 4`, `text_length_ratio = 0.94`. Our system predicted `probable_duplicate`, but gold is `probable_version`.

---

## 3. Your Action Plan

1. **Implement Robust Docket Normalization in `triage/features.py`**:
   - Write a function `normalize_docket_number(docket_str: str) -> str` that:
     - Strips procedural prefixes: `Civil Action No.`, `Civil No.`, `No.`, `Case No.`, `Appeal No.`.
     - Normalizes Florida appellate prefixes (`3D14-3142` $\rightarrow$ `14-3142`), Texas criminal prefixes (`PD-1708-08` $\rightarrow$ `1708-08`), and year representations (`08-` $\rightarrow$ `2008-`).
     - Extracts the core sequence of digits and letter codes.
   - Update `extract_enriched_features()` to compute `canonical_docket_match`:
     - If raw docket strings conflict but normalized core numbers match, set `doc_conflict = False` and `same_doc_num = True`.
2. **Refine Decision Boundaries in `triage/rules.py`**:
   - **`probable_duplicate`**: Reserve for near-zero text divergence:
     - `min_hamming <= 2` AND `text_length_ratio >= 0.95` AND `same_court == True` AND `canonical_docket_match == True`.
   - **`probable_version`**: Alternate revisions, slip vs. published:
     - `min_hamming` between 3 and 18, `text_length_ratio >= 0.60`, with consistent court and canonical docket.
   - **`parallel_split_cluster`**: Only assign when citations show **true disjoint reporter families** (e.g. West regional reporter vs. official state reporter) or separate majority/dissent opinion documents.
3. **Verify with Regression Tests**:
   - Add unit tests in `tests/test_triage.py` checking `dev-019`, `dev-117`, and `dev-015`.
   - Run tests:
     ```bash
     pytest -v
     ```
   - Run validation evaluation:
     ```bash
     python -m triage.cli classify nyu-law-hackathon-2026-duplicates/student-packages/duplicates/data/dev_silver.jsonl --out outputs/silver_preds.jsonl
     python -m triage.cli evaluate outputs/silver_preds.jsonl --gold nyu-law-hackathon-2026-duplicates/student-packages/duplicates/data/dev_silver.jsonl --split validation
     ```

---

## 4. Ground Rules & Data Terms

- **Strict FLP Data Agreement:** NEVER commit any files matching `flp_*` or raw data from `nyu-law-hackathon-2026-duplicates/` to git.
- Check before committing:
  ```bash
  git status
  git ls-files | grep -i flp
  ```
