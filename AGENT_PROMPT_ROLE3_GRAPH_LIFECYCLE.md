# Agent Instructions: Litigation Lifecycle, Companion Decisions & Graph Consistency (Role 3)

**Role:** Litigation Lifecycle, Companion Decisions & Graph Consistency Engineer  
**Repository:** `flp-duplicate-detection`  
**Target Audience:** Free Law Project (FLP) / CourtListener Data Maintainers  
**Safety Mandate:** Real-world legal decisions do not exist in isolation—they form complex appeal chains, companion rulings, and multi-record components. You must ensure pairwise predictions never create transitive contradictions ($A = B \land B = C \implies A \neq C$) and that maintainers have a production-grade review experience.

---

## 1. Context & Architecture

Across the 2,100 candidates in CourtListener, there are **262 multi-pair connected components** (some with up to 16 clusters). Merging pairs without graph-wide consistency risks cascading errors where distinct appeals are accidentally joined.
Furthermore, courts regularly issue companion instruments (e.g. separate formal judgments and memorandum opinions) on the exact same date.

### Your Module Ownership
You own the **graph consistency engine, companion instrument logic, review workbench UX, and adjudication deliverables**:
- Primary files to inspect and modify:
  - `triage/graph.py` (Connected components, cycle detection, transitive contradiction reconciliation)
  - `triage/route.py` (Priority queue assignment: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`)
  - `app/streamlit_app.py` (Maintainer Review Workbench: Blind review, Co-Pilot, family graph views, keyboard shortcuts)
  - `reports/adjudications.md` (Worked adjudications for the 12 disagreement pairs in `adjudicate_me.jsonl` & Annotation Guide v2)
  - `tests/test_triage.py` (Add regression tests for graph contradictions)

---

## 2. Key Edge Cases & Failing Examples You Must Fix

### Problem A: Companion Instruments (Separate Judgment vs. Opinion) — The Texas Pattern
In states like Texas, the appellate court enters a 1-page formal decree ("Judgment") and a 15-page "Memorandum Opinion" on the exact same docket and date:
- **`adj-002`** (*Shakarzahi v. Malooly*, Tex. App. 3rd Dist., `03-15-00229-CV`): Claude called `parallel_split_cluster`, GPT called `related_decision`.
- **`adj-003`** (*Brockman v. State*, Tex. App. 6th Dist., `06-23-00067-CR`): Same pattern.
- **`adj-005`** (*Brockman v. State*, Tex. App. 6th Dist., `06-23-00066-CR`): Companion criminal appeal.
- **The Core Issue:** In Annotation Guide v1, there is no label for separate judgment and opinion! Annotators oscillate between `parallel_split_cluster` (inaccurate) and `related_decision` (misleading).
- **Your Solution:** Formalize the new canonical class **`companion_instrument`** in `reports/adjudications.md` and add specialized feature detection in `triage/features.py` and `app/streamlit_app.py`.

### Problem B: Multi-Stage Litigation & Companion Cases
- **`dev-112`** (*Delaware, Lackawanna & Western Railroad's Tax Assessment*, Pa. 1909): Appeal No. 5 vs. Appeal No. 7 decided on the same day. Related companion appeals! Currently misclassified as `unrelated`.
- **`dev-127`** (*Leshon Nelson v. Evi Cws Quill*, Ga. Ct. App.): Discretionary application (`A25D0223`) vs. direct appeal (`A26A1437`) 393 days apart. Gold is `related_decision`.

### Problem C: Transitive Graph Contradictions & Review Queue Priority
When candidate pairs form connected components:
- If Pair(A, B) is proposed as a `probable_duplicate`, and Pair(B, C) is proposed as `probable_duplicate`, but Pair(A, C) exists and is labeled `related_decision` or `unrelated`:
  - That is a **transitive contradiction** ($A = B \land B = C \implies A = C$).
  - Your engine must intercept these cycles, elevate priority to `CRITICAL`, and automatically downgrade the merge to `insufficient_evidence` until a maintainer adjudicates the family.

---

## 3. Your Action Plan

1. **Enhance Graph Consistency in `triage/graph.py`**:
   - Improve `detect_graph_contradictions()` to handle arbitrary cycles and star-topologies.
   - Calculate cluster degrees: flag any "hub" cluster proposed for merge with 4+ disparate clusters.
   - Ensure `reconcile_graph_predictions()` tags the output record with `component_id`, `family_size`, and contradiction warnings.
2. **Supercharge the Streamlit Review Workbench in `app/streamlit_app.py`**:
   - Add keyboard navigation shortcuts (`M` = Confirm Merge, `V` = Mark Version, `K` = Keep Separate, `D` = Defer).
   - In the Connected Family View, render an interactive network visualization or tabular comparison showing all clusters in the component.
   - Ensure maintainers can download both the `outputs/maintainer_decision_log.csv` and the dry-run `outputs/courtlistener_dry_run_payloads.json` directly from the UI.
3. **Refine Deliverables in `reports/adjudications.md`**:
   - Verify all 12 pairs in `adjudicate_me.jsonl` have complete, rigorous legal adjudications citing court, docket, and evidence.
   - Flesh out the **Annotation Guide v2** proposal with worked examples for `companion_instrument` and secondary error tags (`+citation_error`, `+rehearing_revision`).
4. **Verify with Regression Tests**:
   - Add unit tests in `tests/test_triage.py` for 3-node and 4-node transitive contradictions and companion instrument detection.
   - Run tests and launch app:
     ```bash
     pytest -v
     streamlit run app/streamlit_app.py
     ```

---

## 4. Ground Rules & Data Terms

- **Strict FLP Data Agreement:** NEVER commit any files matching `flp_*` or raw data from `nyu-law-hackathon-2026-duplicates/` to git.
- Check before committing:
  ```bash
  git status
  git ls-files | grep -i flp
  ```
