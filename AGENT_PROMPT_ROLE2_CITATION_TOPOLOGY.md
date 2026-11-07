# Agent Instructions: Citation Topology & Reporter Page Disambiguator (Role 2)

**Role:** Citation Topology & Reporter Page Disambiguator  
**Repository:** `flp-duplicate-detection`  
**Target Audience:** Free Law Project (FLP) / CourtListener Data Maintainers  
**Safety Mandate:** Wrongly merging distinct decisions printed on the same reporter page is a frequent source of database corruption in CourtListener. Use `shared_page_ambiguity` when page-level evidence confirms multiple decisions; when that evidence is missing, use `citation_topology_unverified` rather than guessing or merging.

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

### Problem A: Shared-Page Candidates Falsely Preempted by `docket_numbers_conflict` (`unrelated`)
In `dev_silver.jsonl`, a group of pairs with shared citations and page-span lengths greater than one were predicted as `unrelated`. The page-span field alone does not establish that several cases occupy the same physical page. Treat these as `citation_topology_unverified` until page-level occupancy is supplied; do not call them unrelated or merge them based on the docket conflict alone.

When an authoritative page source confirms multiple decisions on the cited page, conflicting dockets support `shared_page_ambiguity`. The listed development pairs remain useful candidates for page-level verification, but their existing `shared_page_ambiguity` silver labels do not replace that missing evidence.

**Concrete Development Candidates (`training` & `validation` only):**
1. **`dev-006` `[training]`** (*McNeely v. Safe Haven* vs. *Combetta v. Ordoyne*, Louisiana Supreme Court):
   - Shared citation page with `min_page_cluster_span = 5`, `max_page_cluster_span = 5`, `shared_citations = 1`.
   - Docket: `2006-OC-1352` vs. `2006-C-1353` (`docket_numbers_conflict = 1`).
   - Current Bug: Docket conflict preempted citation review and predicted `unrelated`. Existing Gold Label: `shared_page_ambiguity`; page-occupancy evidence is still required to verify it.
2. **`dev-012` `[validation]`** (*State v. Brown* vs. *State v. Archie*, Louisiana Supreme Court):
   - Shared citation page with `span = (4, 4)`, `shared_citations = 1`, `docket_numbers_conflict = 1`.
   - Current Bug: Predicted `unrelated`. Existing Gold Label: `shared_page_ambiguity`; page-occupancy evidence is still required to verify it.
3. **`dev-014` `[training]`** (*Smith v. Ferriday* vs. *State ex rel. Jeffers*, Louisiana Supreme Court):
   - Shared citation page with `span = (5, 5)`, `shared_citations = 1`, `docket_numbers_conflict = 1`.
   - Current Bug: Predicted `unrelated`. Existing Gold Label: `shared_page_ambiguity`; page-occupancy evidence is still required to verify it.
4. **`dev-108` `[validation]`** (*Driver v. State* vs. *Ex parte Hawthorne*, Tex. Crim. App.):
   - Shared citation page with `span = (2, 2)`, `shared_citations = 1`, `docket_numbers_conflict = 1`.
   - Current Bug: Predicted `unrelated`. Existing Gold Label: `shared_page_ambiguity`; page-occupancy evidence is still required to verify it.
5. **`dev-118` `[validation]`** (*United States v. Lowden* vs. *United States v. Rijos*, 1st Cir.):
   - Shared orders page with `span = (4, 4)`, `shared_citations = 2`, `docket_numbers_conflict = 1`.
   - Current Bug: Predicted `unrelated`. Existing Gold Label: `shared_page_ambiguity`; page-occupancy evidence is still required to verify it.

### Problem B: Disambiguating `shared_page_ambiguity` from `citation_data_error`

The data dictionary defines `min_page_cluster_span` / `max_page_cluster_span` as the number of reporter pages each cluster spans. This does **not** count how many distinct decisions appear on one physical page. Do not use page-span length as a proxy for page occupancy.

- **`shared_page_ambiguity`**: shared citation plus page-level evidence in `distinct_decisions_on_shared_page >= 2` that multiple decisions occupy the cited page.
- **`citation_data_error`**: shared citation plus page-level evidence in `distinct_decisions_on_shared_page == 1`, together with conflicting case metadata showing the citation was misattached.
- **`citation_topology_unverified`**: records share a citation and have conflicting metadata, but no reliable page-level evidence distinguishes a multi-decision page from a citation attachment error. Route this to human/source review; never merge it.

---

## 3. Your Action Plan

1. **Restructure Shared Page Rule Precedence in `triage/rules.py`**:
   - Evaluate shared-citation topology (`shared_cits > 0`) **before** the generic `doc_conflict -> unrelated` fallback:
     - When `shared_cits >= 1`:
       - If text and case identity match (`min_hamming <= 12`, `len_ratio >= 0.65`, `same_court`, and `title_exact` or `token_jaccard >= 0.5`) $\rightarrow$ `probable_version`.
       - If `distinct_decisions_on_shared_page >= 2`, classify as **`shared_page_ambiguity`** (confidence $\ge 0.90$).
       - If `distinct_decisions_on_shared_page == 1` and case metadata conflicts, classify as **`citation_data_error`** (confidence $\ge 0.85$).
       - If page-level count is unavailable, classify as **`citation_topology_unverified`** (confidence $\ge 0.90$) and route for review.
2. **Preserve Zero False Merges on Citation Hard Negatives**:
   - Ensure no `shared_page_ambiguity`, `citation_data_error`, or `citation_topology_unverified` pair is ever assigned a merge label.

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
   Check that no ambiguous citation pair is assigned a merge label and `Hard-Negative Citation/Page False Merges (Val)` stays `0/12`. Report `shared_page_ambiguity`, `citation_data_error`, and `citation_topology_unverified` separately; do not tune thresholds against contradictory annotations.
3. **Run Whole-Project Generalization Check**:
   ```bash
   .venv/bin/python3 -m triage.cli bench --role all
   .venv/bin/pytest -v
   ```
