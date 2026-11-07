# Agent Instructions: Citation Topology & Reporter Page Disambiguator (Role 2)

**Role:** Citation Topology & Reporter Page Disambiguator  
**Repository:** `flp-duplicate-detection`  
**Target Audience:** Free Law Project (FLP) / CourtListener Data Maintainers  
**Safety Mandate:** Wrongly merging distinct decisions printed on the same reporter page is a frequent source of database corruption in CourtListener. You must ensure multi-case reporter pages are flagged as `shared_page_ambiguity` rather than collapsed or mislabeled as `unrelated`.

---

## 0. Mandatory Anti-Cheating & Anti-Overfitting Guardrails

Before writing any code, read [`AGENTS.md`](./AGENTS.md). You are strictly bound by the following rules (enforced automatically by `tests/test_anti_leakage.py`):
1. **THE `test` SPLIT IS LOCKED AND OFF-LIMITS**: Never inspect, filter for, load, or evaluate on rows with `split == "test"`. In Python, always use `triage.io.load_development_splits(path, splits=("training", "validation"))`. Never pass `allow_test_split=True` or `--unlock-test-set`.
2. **NO HARDCODED IDs OR CASE LOOKUPS**: Never hardcode `pair_id`, `pair_uid` (`dev-*`, `cand-*`, `hn-*`, `adj-*`), `cluster_id`, or case party names in `triage/*.py`. All citation and page-span rules must rely on general structural features.
3. **TRAIN VS. VALIDATION GENERALIZATION**: Fit models only on `split == "training"`. Verify every rule or feature change on **both** `training` and `validation` splits using `python -m triage.cli bench --role 2`. Reject any change that creates false merges on `hard_negatives` (`hn_cit_val_false_merges` must remain `0`).

---

## 1. Context & Architecture

CourtListener bulk datasets contain hundreds of thousands of reporter volumes where multiple short decisions (certiorari denials, memorandum affirmances, attorney discipline, motions) appear on a single printed page.
When scrapers ingest these decisions, they assign the identical volume/page citation to every decision printed on that page.

### Your Module Ownership
You own the **citation topology, reporter page span analysis, and data quality defect engine**:
- Primary files to inspect and modify:
  - `triage/rules.py` (Rule 1, Rule 2, Rule 3, Rule 4: `shared_page_ambiguity` vs. `citation_data_error` vs. `unrelated`)
  - `triage/features.py` (Page span extraction, citation string parsing, reporter detection)
  - `reports/data_quality_findings.md` (Corpus defect documentation: Sickels volumes, SCOTUS/CAFC defect)
  - `tests/test_bench_role2_citations.py` (Role 2 dedicated test bench)

---

## 2. Key Edge Cases & Failing Examples (`training` & `validation` Splits Only)

Our current evaluation on the development splits reveals two major failure modes:

### Problem A: Shared Reporter Pages Falsely Preempted by `docket_numbers_conflict` (`unrelated`)
In `dev_silver.jsonl`, a large group of pairs sharing a reporter page (`max_page_cluster_span >= 2` and `shared_citations >= 1`) are currently predicted as `unrelated` instead of `shared_page_ambiguity`.
- *Why this happens:* In `triage/rules.py`, Rule 2 (`if f["doc_conflict"] and not f["shared_sha1"]: return LABEL_UNRELATED`) currently runs **before** Rule 3 (`if f["max_page_span"] >= 2 and f["shared_cits"] > 0: return LABEL_SHARED_PAGE`).
- *Legal/Structural Reality:* When two distinct summary dispositions are printed on the top and bottom of the same physical reporter page, they are different cases—so they **always** have different docket numbers (`doc_conflict == True`)! Having conflicting docket numbers while sharing a multi-case reporter page (`max_page_span >= 2` and `shared_cits >= 1`) is the defining signature of `shared_page_ambiguity`, NOT `unrelated`!

**Concrete Failing Development Pairs (`training` & `validation` only):**
1. **`dev-006` `[training]`** (*McNeely v. Safe Haven* vs. *Combetta v. Ordoyne*, Louisiana Supreme Court):
   - Shared citation page with `min_page_cluster_span = 5`, `max_page_cluster_span = 5`, `shared_citations = 1`.
   - Docket: `2006-OC-1352` vs. `2006-C-1353` (`docket_numbers_conflict = 1`).
   - Current Bug: Rule 2 preempted Rule 3 and predicted `unrelated`. Gold Label: `shared_page_ambiguity`.
2. **`dev-012` `[validation]`** (*State v. Brown* vs. *State v. Archie*, Louisiana Supreme Court):
   - Shared citation page with `span = (4, 4)`, `shared_citations = 1`, `docket_numbers_conflict = 1`.
   - Current Bug: Predicted `unrelated`. Gold Label: `shared_page_ambiguity`.
3. **`dev-014` `[training]`** (*Smith v. Ferriday* vs. *State ex rel. Jeffers*, Louisiana Supreme Court):
   - Shared citation page with `span = (5, 5)`, `shared_citations = 1`, `docket_numbers_conflict = 1`.
   - Current Bug: Predicted `unrelated`. Gold Label: `shared_page_ambiguity`.
4. **`dev-108` `[validation]`** (*Driver v. State* vs. *Ex parte Hawthorne*, Tex. Crim. App.):
   - Shared citation page with `span = (2, 2)`, `shared_citations = 1`, `docket_numbers_conflict = 1`.
   - Current Bug: Predicted `unrelated`. Gold Label: `shared_page_ambiguity`.
5. **`dev-118` `[validation]`** (*United States v. Lowden* vs. *United States v. Rijos*, 1st Cir.):
   - Shared orders page with `span = (4, 4)`, `shared_citations = 2`, `docket_numbers_conflict = 1`.
   - Current Bug: Predicted `unrelated`. Gold Label: `shared_page_ambiguity`.

### Problem B: Disambiguating `shared_page_ambiguity` from `citation_data_error`
- **`shared_page_ambiguity`**: `shared_citations >= 1` and `max_page_span >= 2` (multiple distinct cases physically printed on the same reporter page, e.g., **543 U.S.** cert orders in **`dev-058` `[training]`**).
- **`citation_data_error`**: `shared_citations >= 1` and `max_page_span <= 1` (a single isolated case occupies the page), yet two completely unrelated cases with conflicting dockets/courts/titles carry the identical citation due to scraper/OCR misattachment (e.g., **`dev-073` `[validation]`**).

---

## 3. Your Action Plan

1. **Restructure Shared Page Rule Precedence in `triage/rules.py`**:
   - Evaluate shared-citation topology (`shared_cits > 0`) **before** the generic `doc_conflict -> unrelated` fallback:
     - When `max_page_span >= 2` and `shared_cits >= 1`:
       - If text and case identity match (`min_hamming <= 12`, `len_ratio >= 0.65`, `same_court`, and `title_exact` or `token_jaccard >= 0.5`) $\rightarrow$ `probable_version`.
       - Otherwise (different dockets, different captions, or different courts) $\rightarrow$ classify as **`shared_page_ambiguity`** (confidence $\ge 0.90$).
     - When `max_page_span <= 1` and `shared_cits >= 1` but `doc_conflict` or different courts/unrelated captions (`min_hamming > 20`) $\rightarrow$ classify as **`citation_data_error`** (confidence $\ge 0.85$).
2. **Preserve Zero False Merges on Citation Hard Negatives**:
   - Ensure no `shared_page_ambiguity` or `citation_data_error` pair is ever assigned a merge label.

---

## 4. How to Run Your Test Bench & Verify Generalization

1. **Run Role 2 Unit Bench + Anti-Leakage Guardrails**:
   ```bash
   .venv/bin/pytest tests/test_bench_role2_citations.py tests/test_anti_leakage.py -v
   ```
2. **Run Role 2 Split-Safe Scorecard (`training` vs. `validation`)**:
   ```bash
   .venv/bin/python3 -m triage.cli bench --role 2
   ```
   Check that `shared_page_ambiguity` Recall and F1 jump on **both** `Train` (currently `35.7%` recall) and `Val` (currently `55.6%` recall), `citation_data_error` Recall improves on `Train` and `Val`, and `Hard-Negative Citation/Page False Merges (Val)` stays `0/12`.
3. **Run Whole-Project Generalization Check**:
   ```bash
   .venv/bin/python3 -m triage.cli bench --role all
   .venv/bin/pytest -v
   ```
