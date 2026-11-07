# UI walkthrough — Carla Bravo branch

## Scope

Browser-tested the synthetic demo on `carla-bravo/ui-workbench` and fixed output
presentation issues. This is UI QA, not human validation of legal classifications.
No `triage/` module, input dataset, model setting or classification rule changed.

Base for the follow-up: commit `c31a1d470ebe7a1e471ace29c61f9d12077bdf9c`.

## Fixes from the walkthrough

1. **Relationship names were truncated at laptop widths.** At 1024px the metric
   displayed “Shared r…” instead of “Shared reporter page.” Use normal wrapping
   text for the relationship name.
2. **Structural signal labels and values were truncated.** Six metric columns
   compressed labels and values such as docket status. Present the six signals
   in two rows of three using wrapping text.
3. **A source inconsistency was not explicitly explained.** The third synthetic
   demo pair supplies `same_court = 0`, while both displayed `court_id` values are
   `nyappdiv`. Show a warning when that supplied signal conflicts with the
   displayed IDs. Do not change the supplied value, fixture or prediction.
4. **Saving notes was implicit.** Add a short instruction explaining that a
   decision button saves the review notes and evidence fields.

## Executed browser checks

Used local headless Chromium driven by Playwright against the running Streamlit
server, not just AppTest. Inspected screenshots at desktop, laptop and mobile
viewports. Playwright is a local QA tool, not a new application dependency.

| Check | Result |
| --- | --- |
| Demo opens without private data, with empty downloads disabled and no live links for synthetic IDs | PASS |
| Side-by-side metadata, recommendation/evidence and text excerpts render | PASS |
| Reviewer, notes, evidence, chosen survivor, saved decision and progress update | PASS |
| Actual browser CSV download parses and matches the single recorded review, including quoted text and Unicode | PASS |
| Actual browser JSON download is valid, marked dry-run, and matches the selected source/target IDs | PASS |
| Downloads do not lose decisions; previous/next navigation restores saved notes and survivor | PASS |
| Offline falsification output shows tier/model and persists across unrelated rerenders without appearing on another pair | PASS |
| Reviewed/unreviewed filters and Defer operate on the displayed pair | PASS |
| Blind mode ignores prediction-based filters and hides machine/agent output until explicit reveal | PASS |
| Explicit reveal displays the recommendation | PASS |
| Desktop 1440px and laptop 1024px fit the viewport; long labels are legible after the fix | PASS |
| Mobile 390px review actions work, stack vertically and do not create horizontal page overflow | PASS |
| Browser page errors and error-level console messages in the completed desktop run | None observed |

The CSV contained only the recorded duplicate review, not the other unreviewed
predictions. Its matching JSON contained one proposal, `dry_run = true`, the
chosen surviving cluster, and an explicit synthetic-demo flag.

## Automated regression checks

- `python -m pytest -q`: **56 passed, 5 skipped**.
- Includes 12 UI/export tests and a new regression for wrapping relationship
  text and surfacing inconsistent supplied court signals without changing them.
- `git diff --check`: passed.
- Python 3.13.11; Streamlit 1.67.0.

## Remaining limits and team follow-up

- The full `python -m triage.cli bench --role all` command was attempted again
  and stopped because `dev_silver.jsonl` is unavailable locally. Five
  dataset-dependent tests remain skipped.
- A teammate with permitted local development data should run that benchmark and
  the full suite before marking the PR ready. Keep the held-out test split locked.
- Connected-family display is covered by synthetic AppTest fixtures, but this
  browser run did not validate a real multi-record family.
- Online providers, real CourtListener data, legal correctness, durability,
  accessibility compliance and all-browser compatibility are not established.
- Session-only decisions and the explicit-reveal nature of blind mode remain
  intentional limitations.

## Reproduce the main workflow

1. Run `streamlit run app/streamlit_app.py` and choose the synthetic demo.
2. Open the first pair; enter reviewer, notes and cited evidence; choose a
   surviving cluster and click Confirm Duplicate.
3. Download CSV and JSON. Confirm one recorded decision and one dry-run proposal
   with the chosen survivor. Confirm the filenames identify synthetic data.
4. Navigate away and back; verify saved fields are restored.
5. Run the offline check, edit an unrelated field, then navigate between pairs.
   Confirm the finding persists only for its own pair.
6. Use Unreviewed, then All pairs; defer a different pair.
7. Select a priority with no matches, switch to Blind Review and verify pairs
   remain available without machine output. Reveal explicitly to show it.
8. Open demo pair `1005-1006` at 1024px. Verify the full relationship label and
   signal text are legible and the source-data warning is visible.
9. At 390px, record a decision and verify the controls remain usable.
