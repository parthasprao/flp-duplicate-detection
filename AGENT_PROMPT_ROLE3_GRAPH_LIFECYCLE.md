# Agent Instructions: Litigation Lifecycle, Companion Decisions & Graph Consistency (Role 3)

**Role:** Litigation Lifecycle, Companion Decisions & Graph Consistency Engineer  
**Repository:** `flp-duplicate-detection`  
**Target Audience:** Free Law Project (FLP) / CourtListener Data Maintainers  
**Safety Mandate:** Real-world legal decisions do not exist in isolation—they form complex appeal chains, companion rulings, and multi-record components. You must ensure pairwise predictions never create transitive contradictions ($A = B \land B = C \implies A \neq C$) and that recurring generic captions or companion decisions are never falsely merged.

---

## 0. Mandatory Anti-Cheating & Anti-Overfitting Guardrails

Before writing any code, read [`AGENTS.md`](./AGENTS.md). You are strictly bound by the following rules (enforced automatically by `tests/test_anti_leakage.py`):
1. **THE `test` SPLIT IS LOCKED AND OFF-LIMITS**: Never inspect, filter for, load, or evaluate on rows with `split == "test"`. In Python, always use `triage.io.load_development_splits(path, splits=("training", "validation"))`. Note that in `adjudicate_me.jsonl`, `adj-001` through `adj-009` and `adj-012` are `training`/`validation`, whereas held-out `test` rows must never be referenced in code or tests.
2. **NO HARDCODED IDs OR CASE LOOKUPS**: Never hardcode `pair_id`, `pair_uid` (`dev-*`, `cand-*`, `hn-*`, `adj-*`), `cluster_id`, or case party names in `triage/*.py`.
3. **TRAIN VS. VALIDATION GENERALIZATION**: Fit models only on `split == "training"`. Verify every rule, graph check, or feature change on **both** `training` and `validation` splits using `python -m triage.cli bench --role 3`.

---

## 1. Context & Architecture

Across the 2,100 candidates in CourtListener, there are **262 multi-pair connected components** (some with up to 16 clusters). Merging pairs without graph-wide consistency risks cascading errors where distinct appeals are accidentally joined.
Furthermore, courts regularly issue companion instruments (e.g. separate formal judgments and memorandum opinions) on the exact same date.

### Your Module Ownership
You own the **graph consistency engine, companion instrument logic, review workbench UX, and adjudication deliverables**:
- Primary files to inspect and modify:
  - `triage/graph.py` (Connected components, cycle detection, transitive contradiction reconciliation)
  - `triage/route.py` (Priority queue assignment: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`)
  - `triage/rules.py` & `triage/features.py` (Rules 5, 7, 10: `related_decision` vs. `unrelated` vs. companion instruments)
  - `app/streamlit_app.py` (Maintainer Review Workbench: Blind review, Co-Pilot, family graph views)
  - `reports/adjudications.md` (Adjudications & Annotation Guide v2)
  - `tests/test_bench_role3_graph_lifecycle.py` (Role 3 dedicated test bench)

---

## 2. Key Edge Cases & Failing Examples (`training` & `validation` Splits Only)

### Problem A: Companion Instruments (Separate Judgment vs. Opinion) — The Texas Pattern
In states like Texas, the appellate court enters a 1-page formal decree ("Judgment") and a multi-page "Memorandum Opinion" on the exact same docket and date:
- **`adj-002` `[training]`** (*Shakarzahi v. Malooly*, Tex. App. 3rd Dist., `03-15-00229-CV`): Claude called `parallel_split_cluster`, GPT called `related_decision`.
- **`adj-003` `[training]`** (*Brockman v. State*, Tex. App. 6th Dist., `06-23-00067-CR`): Same pattern.
- **`adj-005` `[training]`** (*Brockman v. State*, Tex. App. 6th Dist., `06-23-00066-CR`): Companion criminal appeal.
- **Your Solution:** Ensure `detect_judgment_vs_opinion()` in `triage/features.py` and Rule 5 in `triage/rules.py` reliably distinguish companion instruments and sequential companion docket numbers (e.g., `06-23-00066-CR` vs `06-23-00067-CR` decided on the same date by the same court with identical captions) as `related_decision` rather than merging them or dropping them into `unrelated`.

### Problem B: Multi-Stage Litigation & Companion Cases vs. `unrelated`
- **`dev-112` `[training]`** (*Delaware, Lackawanna & Western Railroad's Tax Assessment*, Pa. 1909): Companion tax assessment appeals (`Appeal No. 5` vs. `Appeal No. 7`) decided on the same day (`filing_date_delta_days = 0`, `same_court = 1`, `title_exact = 1`). Currently misclassified as `unrelated`; Gold Label is `related_decision`.
- **`dev-127` `[training]`** (*Leshon Nelson v. Evi Cws Quill*, Ga. Ct. App.): Discretionary application (`A25D0223`) vs. direct appeal (`A26A1437`) 393 days apart (`same_court = 1`, `title_exact = 1`). Currently misclassified as `unrelated`; Gold Label is `related_decision`.
- **Hard-Negative Trap to Guard Against (`hn_life_val_false_merges`)**:
  - Currently, `4/16` lifecycle/caption hard negatives in the `validation` split of `hard_negatives.jsonl` are falsely merged! Inspect why those validation hard negatives (e.g., `materially_different_similar_metadata` or `recurring_case_name`) are receiving merge predictions and tighten the text/length/citation guardrails so `hn_life_val_false_merges` drops to `0/16`.

### Problem C: Transitive Graph Contradictions & Hub Cluster Warnings
When candidate pairs form connected components:
- If Pair(A, B) is proposed as a merge, and Pair(B, C) is proposed as a merge, but Pair(A, C) is labeled `related_decision` or `unrelated`, `triage/graph.py` must flag the contradiction, downgrade low/medium-confidence merges to `insufficient_evidence`, and escalate priority to `CRITICAL`.
- Also flag "hub" clusters that have $\ge 4$ proposed merge edges in a single component so maintainers review the entire family before executing redirects.

---

## 3. How to Run Your Test Bench & Verify Generalization

1. **Run Role 3 Unit Bench + Anti-Leakage Guardrails**:
   ```bash
   .venv/bin/pytest tests/test_bench_role3_graph_lifecycle.py tests/test_anti_leakage.py -v
   ```
2. **Run Role 3 Split-Safe Scorecard (`training` vs. `validation`)**:
   ```bash
   .venv/bin/python3 -m triage.cli bench --role 3
   ```
   Check that `related_decision` and `unrelated` F1 improve on both `Train` and `Val`, and that `Hard-Negative Lifecycle/Caption False Merges (Val)` drops from `4/16` toward `0/16`.
3. **Run Whole-Project Generalization Check**:
   ```bash
   .venv/bin/python3 -m triage.cli bench --role all
   .venv/bin/pytest -v
   ```
