# Agent Instructions: Citation Topology & Reporter Page Disambiguator (Role 2)

**Role:** Citation Topology & Reporter Page Disambiguator  
**Repository:** `flp-duplicate-detection`  
**Target Audience:** Free Law Project (FLP) / CourtListener Data Maintainers  
**Safety Mandate:** Wrongly merging distinct decisions printed on the same reporter page is a frequent source of database corruption in CourtListener. You must ensure multi-case reporter pages are flagged as `shared_page_ambiguity` rather than collapsed or mislabeled as citation errors.

---

## 1. Context & Architecture

CourtListener bulk datasets contain hundreds of thousands of reporter volumes where multiple short decisions (certiorari denials, memorandum affirmances, attorney discipline, motions) appear on a single printed page.
When scrapers ingest these decisions, they frequently borrow or assign the identical volume/page citation to multiple records.

### Your Module Ownership
You own the **citation topology, reporter page span analysis, and data quality defect engine**:
- Primary files to inspect and modify:
  - `triage/rules.py` (Rule 1, Rule 3, Rule 4: `shared_page_ambiguity` vs. `citation_data_error` vs. `unrelated`)
  - `triage/features.py` (Page span extraction, citation string parsing, reporter detection)
  - `reports/data_quality_findings.md` (Corpus defect documentation: Sickels volumes, SCOTUS/CAFC defect)
  - `tests/test_triage.py` (Add regression tests for citation edge cases)

---

## 2. Key Edge Cases & Failing Examples You Must Fix

Our current evaluation reveals two major failure modes:

### Problem A: Shared Reporter Pages Falsely Falling Through to `unrelated`
In `dev_silver.jsonl`, a large cluster of pairs sharing a reporter page (`span >= 2` and `shared_citations >= 1`) are currently predicted as `unrelated` instead of `shared_page_ambiguity`.
- *Why this happens:* In our current decision list in `triage/rules.py`, Rule 2 (`doc_conflict`) and Rule 4 (`different courts`) trigger *before* the shared page logic. But in reality, when multiple distinct cases are printed on the same reporter page, they naturally have DIFFERENT dockets and often DIFFERENT courts! Having different dockets is *evidence of* `shared_page_ambiguity`, not a reason to skip it!

**Concrete Failing Test Pairs:**
1. **`dev-006`** (*McNeely v. Safe Haven* vs. *Combetta v. Ordoyne*, Louisiana Supreme Court):
   - Citations: Shared page with `min_page_cluster_span = 5`, `max_page_cluster_span = 5`.
   - Docket: `2006-OC-1352` vs. `2006-C-1353`.
   - Current Bug: Predicted `unrelated`. Gold Silver Label: `shared_page_ambiguity`.
2. **`dev-012`** (*State v. Brown* vs. *State v. Archie*, Louisiana Supreme Court):
   - Shared citation page with `span = (4, 4)`.
   - Current Bug: Predicted `unrelated`. Gold Silver Label: `shared_page_ambiguity`.
3. **`dev-014`** (*Smith v. Ferriday* vs. *State ex rel. Jeffers*, Louisiana Supreme Court):
   - Shared page with `span = (5, 5)`.
   - Current Bug: Predicted `unrelated`. Gold Silver Label: `shared_page_ambiguity`.
4. **`dev-108`** (*Driver v. State* vs. *Ex parte Hawthorne*, Tex. Crim. App.):
   - Shared citation page with `span = (2, 2)`.
   - Current Bug: Predicted `unrelated`. Gold Silver Label: `shared_page_ambiguity`.
5. **`dev-118`** (*United States v. Lowden* vs. *United States v. Rijos*, 1st Cir.):
   - Shared orders page with `span = (4, 4)`.
   - Current Bug: Predicted `unrelated`. Gold Silver Label: `shared_page_ambiguity`.

### Problem B: Disambiguating `shared_page_ambiguity` from `citation_data_error`
- In `citation_data_error`, two records represent completely distinct cases, but a citation attached to one was mistakenly assigned or scraped due to an OCR typo.
- In `shared_page_ambiguity`, the citation is physically correct (both cases literally appear on page 416 of 149 A.D.3d), but they are distinct decisions.
- **Rule of Thumb:**
  - If `max_page_span >= 2` (or cluster count on page $\ge 2$): Physical reporter page holds multiple cases $\rightarrow$ `shared_page_ambiguity`.
  - If `max_page_span <= 1` (a single isolated case occupies the page), but completely unrelated courts/titles carry the citation $\rightarrow$ `citation_data_error`.

---

## 3. Your Action Plan

1. **Restructure Shared Page Rule Precedence in `triage/rules.py`**:
   - Ensure that when `max_page_span >= 2` and `shared_citations >= 1`:
     - If text is near-identical (`min_hamming <= 15`, `length_ratio >= 0.65`, same court, matching title) $\rightarrow$ `probable_version` (same case in alternate versions sharing that citation).
     - Otherwise (conflicting dockets, different case captions, high Hamming distance) $\rightarrow$ definitively classify as **`shared_page_ambiguity`** with confidence $\ge 0.92$.
     - DO NOT let these cases fall through to `unrelated`!
2. **Handle Court Discrepancies on Shared Pages (e.g. SCOTUS / CAFC Bug)**:
   - Verify Rule 1 correctly identifies U.S. Reports Volume 543 cert denials misassigned to `cafc` (`dev-004`, `dev-021`, `dev-058`).
3. **Enhance Track 4 Data-Quality Scrubbing Scripts in `reports/data_quality_findings.md`**:
   - Provide SQL queries and Python audit scripts that CourtListener maintainers can run against their PostgreSQL database (`search_opinioncluster` and `search_citation`) to scrub `14 Sickels` nominative collisions and `cafc` cert orders.
4. **Verify with Regression Tests**:
   - Add unit tests in `tests/test_triage.py` for `dev-006`, `dev-014`, and `dev-108`.
   - Run tests and evaluation:
     ```bash
     pytest -v
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
