"""Single-page CourtListener review UI. Analysis remains in the triage package."""

from __future__ import annotations

import difflib
import json
import sys
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from triage.agent import analyze_with_agent, run_offline_falsification
from triage.constants import (
    LABEL_INSUFFICIENT,
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    LABEL_RELATED_DECISION,
    MERGE_LABELS,
    VALID_LABELS,
)
from triage.export import export_courtlistener_dry_run_payloads, export_decision_log
from triage.graph import build_connected_components, reconcile_graph_predictions
from triage.io import find_default_data_dir, load_dataset, load_development_splits
from triage.model import CalibratedTriageClassifier
from triage.route import assign_queue_priority

PRIORITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
LABEL_NAMES = {
    "probable_duplicate": "Probable duplicate",
    "probable_version": "Alternate version",
    "parallel_split_cluster": "Parallel / split cluster",
    "related_decision": "Related but distinct decision",
    "shared_page_ambiguity": "Shared reporter page",
    "citation_data_error": "Citation data error",
    "unrelated": "Unrelated decisions",
    "insufficient_evidence": "Insufficient evidence",
}
DEMO_PATH = REPO_ROOT / "demo_data" / "sample_candidates.jsonl"


def display(value: Any) -> str:
    """Do not confuse missing information with a negative signal."""
    return "Not available" if value is None or value == "" else str(value)


def yes_no(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    if value in (True, 1, "1", "True", "true"):
        return "Yes"
    if value in (False, 0, "0", "False", "false"):
        return "No"
    return "Not available"


def courtlistener_url(record: dict, explicit: str | None = None) -> str | None:
    """Only offer CourtListener links; demo IDs never call this helper."""
    if explicit:
        parsed = urlparse(explicit)
        if parsed.scheme == "https" and parsed.netloc in {
            "courtlistener.com", "www.courtlistener.com",
        }:
            return explicit
    cid = str(record.get("cluster_id") or "")
    return f"https://www.courtlistener.com/opinion/{cid}/" if cid.isdigit() else None


@st.cache_data(show_spinner=False)
def load_all_data(
    source: str, source_mtime: int, training_path: str, training_mtime: int
) -> tuple[list[dict], dict]:
    """Cache by source identity; never use held-out data for UI development."""
    path = Path(source)
    if path.name in {"dev_silver.jsonl", "hard_negatives.jsonl", "adjudicate_me.jsonl"}:
        records = load_development_splits(path)
    else:
        records = [r for r in load_dataset(path) if r.get("split") != "test"]
    classifier = CalibratedTriageClassifier()
    if training_path:
        classifier.fit(load_development_splits(training_path, splits=("training",)))
    predictions = []
    for row in records:
        label, confidence, supporting, contradicting = classifier.predict_pair(row)
        predictions.append({
            "pair_id": row["pair_id"], "label": label, "confidence": confidence,
            "evidence": {"supporting": supporting, "contradicting": contradicting},
        })
    components = build_connected_components(records)
    reconciled = reconcile_graph_predictions(predictions, components)
    predictions_by_id = {
        p["pair_id"]: assign_queue_priority(p) for p in reconciled
    }
    for row in records:
        row.update(predictions_by_id[row["pair_id"]])
    records.sort(key=lambda r: (
        PRIORITIES.index(r.get("priority", "LOW")), str(r["pair_id"])
    ))
    return records, components


def export_bytes(decisions: list[dict]) -> tuple[bytes, bytes]:
    """Use the existing exporters without shared server-side output files."""
    if not decisions:
        return b"", b"[]"
    with TemporaryDirectory(prefix="courtlistener-review-") as directory:
        csv_path = Path(directory) / "decisions.csv"
        json_path = Path(directory) / "dry_run.json"
        export_decision_log(decisions, csv_path)
        export_courtlistener_dry_run_payloads(decisions, json_path)
        return csv_path.read_bytes(), json_path.read_bytes()


def record_decision(pair: dict, label: str, mode: str, demo: bool) -> None:
    """Button callback: save before rerendering so confirmations are never stale."""
    pid = pair["pair_id"]
    low = (pair.get("cluster_low") or {}).get("cluster_id")
    high = (pair.get("cluster_high") or {}).get("cluster_id")
    evidence = pair.get("evidence") or {}
    blind = mode == "Blind Review"
    revealed = pid in st.session_state.revealed_pairs
    decision = {
        "pair_id": pid, "decision": label,
        "reviewer": st.session_state.get("reviewer", "").strip() or "maintainer",
        "review_mode": mode,
        "reason": st.session_state.get(f"notes_{pid}", "").strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evidence_cited": st.session_state.get(f"evidence_{pid}", "").strip(),
        "evidence": {
            "reviewer_notes": st.session_state.get(f"notes_{pid}", "").strip(),
            "reviewer_citations": st.session_state.get(f"evidence_{pid}", "").strip(),
            "machine_evidence": evidence if not blind or revealed else {},
            "recommendation_visible": not blind or revealed,
            "synthetic_demo": demo,
        },
    }
    if label in MERGE_LABELS:
        keep = st.session_state.get(f"keep_{pid}", low)
        decision.update(keep_cluster_id=keep, delete_cluster_id=high if keep == low else low)
    st.session_state.decisions[pid] = decision
    # Freeze the first judgment separately from any later changes.
    st.session_state.initial_decisions.setdefault(pid, dict(decision))


def render_record(record: dict, side: str, explicit_url: str | None, demo: bool) -> None:
    st.subheader(f"Record {side}")
    st.markdown(
        f'<div class="record-name">{escape(display(record.get("case_name")))}</div>',
        unsafe_allow_html=True,
    )
    docket = record.get("docket") or {}
    court = record.get("court") or {}
    citations = record.get("citations") or []
    values = {
        "Cluster ID": record.get("cluster_id"),
        "Court": court.get("short_name") or record.get("court_id"),
        "Filed": record.get("date_filed"),
        "Docket": docket.get("docket_number") or record.get("docket_number"),
        "Citations": ", ".join(
            str(c.get("citation_key") or "") for c in citations if isinstance(c, dict)
        ) or None,
    }
    fields = "".join(
        '<div class="record-field">'
        f'<span class="record-field-label">{escape(title)}</span>'
        f'<span class="record-field-value">{escape(display(value))}</span></div>'
        for title, value in values.items()
    )
    st.markdown(f'<div class="record-fields">{fields}</div>', unsafe_allow_html=True)
    if record.get("case_name_full"):
        st.caption(record["case_name_full"])
    link = courtlistener_url(record, explicit_url) if not demo else None
    if link:
        st.link_button("View on CourtListener", link)


def render_evidence(pair: dict) -> None:
    with st.expander("Machine recommendation and evidence", expanded=True):
        left, right = st.columns([1, 2])
        with left:
            st.caption("Proposed relationship")
            # Metric values truncate long relationship names on laptop screens.
            # Normal text wraps without custom CSS or a responsive-layout framework.
            st.markdown(f"**{LABEL_NAMES.get(pair['label'], pair['label'])}**")
            st.metric("Model confidence", f"{pair.get('confidence', 0):.1%}")
            st.caption("Machine recommendation, not an approved action.")
        with right:
            evidence = pair.get("evidence") or {}
            for heading, key in [("Supporting evidence", "supporting"),
                                 ("Contradicting evidence", "contradicting")]:
                st.markdown(f"**{heading}**")
                for item in evidence.get(key) or ["None supplied"]:
                    st.write(f"• {item}")


def render_comparison(pair: dict, demo: bool) -> None:
    st.header("Case comparison")
    st.caption("Two records, side by side. Colors identify A and B only—not a match or a verdict.")
    low, high = pair.get("cluster_low") or {}, pair.get("cluster_high") or {}
    for column, record, side in zip(st.columns(2, gap="medium"), [low, high], ["A", "B"]):
        with column.container(border=True, key=f"record_{side}"):
            render_record(record, side, pair.get(
                "courtlistener_url_low" if side == "A" else "courtlistener_url_high"
            ), demo)
    features = pair.get("features") or {}
    conflict = yes_no(features.get("docket_numbers_conflict"))
    docket = "Conflict" if conflict == "Yes" else (
        "Same" if yes_no(features.get("same_docket_number")) == "Yes"
        else "Different" if conflict == "No" and yes_no(
            features.get("same_docket_number")) == "No" else "Not available"
    )
    signals = {
        "Same court": yes_no(features.get("same_court")),
        "Docket status": docket,
        "Date gap (days)": display(features.get("filing_date_delta_days")),
        "Shared citations": display(features.get("shared_citations")),
        "Hamming distance": display(features.get("min_hamming")),
        "Text length ratio": display(features.get("text_length_ratio")),
    }
    items = list(signals.items())
    st.caption("02 / COMPARISON SIGNALS")
    with st.container(border=True, key="comparison_signals"):
        for offset in range(0, len(items), 3):
            for column, (title, value) in zip(st.columns(3), items[offset:offset + 3]):
                column.markdown(f"**{title}**")
                column.write(value)
    st.caption("Signals are supplied by the data/pipeline; missing values are not mismatches.")
    court_a, court_b = low.get("court_id"), high.get("court_id")
    supplied_same_court = yes_no(features.get("same_court"))
    if (
        court_a and court_b and supplied_same_court != "Not available"
        and (supplied_same_court == "Yes") != (court_a == court_b)
    ):
        st.warning(
            "Source-data inconsistency: the supplied 'Same court' signal conflicts "
            "with the displayed court IDs. The UI has not changed the signal or "
            "prediction; check the source before relying on this evidence."
        )
    excerpts = [pair.get("excerpt_low") or "", pair.get("excerpt_high") or ""]
    with st.expander("Text excerpts and differences", expanded=False):
        for column, side, text in zip(st.columns(2), ["A", "B"], excerpts):
            with column:
                st.text_area(f"Record {side} excerpt", text or "No text excerpt available",
                             height=200, disabled=True, key=f"text_{pair['pair_id']}_{side}")
        if all(excerpts):
            difference = "\n".join(difflib.unified_diff(
                excerpts[0].splitlines(), excerpts[1].splitlines(),
                fromfile="Record A", tofile="Record B", lineterm="",
            ))
            if difference:
                st.code(difference, language="diff")
            else:
                st.caption("The available excerpts are identical; full texts may still differ.")


def render_decisions(pair: dict, mode: str, demo: bool) -> None:
    pid = pair["pair_id"]
    st.header("Human review")
    existing = st.session_state.decisions.get(pid)
    if existing:
        st.success(f"Recorded: {LABEL_NAMES[existing['decision']]} — {existing['timestamp']}")
        if existing.get("reason"):
            st.write(existing["reason"])
    for key, value in {
        f"notes_{pid}": (existing or {}).get("reason", ""),
        f"evidence_{pid}": (existing or {}).get("evidence_cited", ""),
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value
    with st.container(border=True, key="review_form"):
        notes_col, evidence_col = st.columns(2, gap="medium")
        with notes_col:
            st.text_area("Review notes", key=f"notes_{pid}", height=80)
        with evidence_col:
            st.text_input("Evidence cited (source link or passage)", key=f"evidence_{pid}")
        st.caption("Use a decision button below to save these fields with your review.")
        low = (pair.get("cluster_low") or {}).get("cluster_id")
        high = (pair.get("cluster_high") or {}).get("cluster_id")
        valid_ids = low is not None and high is not None and low != high
        keep_col, separate_col = st.columns(2, gap="medium")
        if valid_ids:
            if f"keep_{pid}" not in st.session_state:
                saved_keep = (existing or {}).get("keep_cluster_id", low)
                st.session_state[f"keep_{pid}"] = saved_keep if saved_keep in (low, high) else low
            keep_col.radio("Proposed surviving cluster (for duplicate/version proposals only)",
                           [low, high], format_func=lambda cid: f"Cluster {cid}",
                           horizontal=True, key=f"keep_{pid}")
        separate_labels = sorted(VALID_LABELS - MERGE_LABELS - {LABEL_INSUFFICIENT})
        if f"separate_{pid}" not in st.session_state:
            saved_label = (existing or {}).get("decision")
            st.session_state[f"separate_{pid}"] = (
                saved_label if saved_label in separate_labels else LABEL_RELATED_DECISION
            )
        separate = separate_col.selectbox("Keep-separate category", separate_labels,
                                         format_func=LABEL_NAMES.get, key=f"separate_{pid}")
        actions = [
            ("Confirm Duplicate", LABEL_PROBABLE_DUPLICATE),
            ("Mark as Version", LABEL_PROBABLE_VERSION),
            ("Keep Separate", separate),
            ("Defer", LABEL_INSUFFICIENT),
        ]
        for column, (title, label) in zip(st.columns(4), actions):
            column.button(title, key=f"decision_{pid}_{label}", use_container_width=True,
                          type="primary" if label == LABEL_PROBABLE_DUPLICATE else "secondary",
                          disabled=label in MERGE_LABELS and not valid_ids,
                          on_click=record_decision, args=(pair, label, mode, demo))
    st.caption("Records a review decision only. No CourtListener database changes are executed.")


def main() -> None:
    st.set_page_config(page_title="CourtListener Review Workbench", page_icon="⚖️", layout="wide")
    st.markdown(
        f"<style>{Path(__file__).with_name('workbench.css').read_text()}</style>",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        dark_mode = st.toggle("Dark mode", value=True, key="dark_mode",
                              help="Switch between the neon dark palette and the warm light palette. "
                                   "Your current review decisions stay in this session.")
    if dark_mode:
        st.markdown(
            f"<style>{Path(__file__).with_name('workbench-dark.css').read_text()}</style>",
            unsafe_allow_html=True,
        )
    st.markdown('<div class="review-brand"><span>■</span> CourtListener / Review workbench</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="review-eyebrow">Duplicate detection · Human validation</div>',
                unsafe_allow_html=True)
    st.title("CourtListener Case-Identity Review")
    st.caption("Compare records, inspect evidence, and prepare non-destructive review exports.")
    data_dir = find_default_data_dir().resolve()
    available = [p for p in [data_dir / "candidates.jsonl", data_dir / "candidates.csv",
                             data_dir / "dev_silver.jsonl"] if p.exists()]
    if DEMO_PATH.exists():
        available.append(DEMO_PATH)
    if not available:
        st.error("No candidate files found. Set TRIAGE_DATA_DIR or add the synthetic demo file.")
        return
    with st.sidebar:
        st.header("Review queue")
        st.caption("A considered review, one case at a time.")
        with st.expander("Your review guide"):
            st.markdown(
                "1. **Compare the records** and read the available excerpts.\n"
                "2. **Record your judgment** with notes and supporting sources.\n"
                "3. **Download your decisions** before leaving this session.\n\n"
                "Blind Review hides machine recommendations until you choose to reveal them. "
                "No action here changes CourtListener records."
            )
        source = st.selectbox("Dataset", available, key="dataset", format_func=lambda p: (
            "Synthetic demo (practice only)" if p == DEMO_PATH else p.name
        ))
    demo = source == DEMO_PATH
    if st.session_state.get("source") != str(source):
        for key in list(st.session_state):
            if key not in {"reviewer", "dataset", "dark_mode"}:
                del st.session_state[key]
        st.session_state.source = str(source)
    for key, default in [
        ("decisions", {}), ("initial_decisions", {}), ("revealed_pairs", set()), ("agent_results", {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
    training = data_dir / "dev_silver.jsonl"
    training_path = str(training) if training.exists() and not demo else ""
    try:
        with st.spinner("Loading the review queue…"):
            records, components = load_all_data(
                str(source), source.stat().st_mtime_ns, training_path,
                training.stat().st_mtime_ns if training_path else 0,
            )
    except (OSError, ValueError, KeyError, TypeError):
        st.error("Could not load this dataset. Check its JSONL/CSV format and required pair metadata.")
        return
    if demo:
        st.info("Synthetic demo data — practice only. IDs are not live CourtListener records.")
    with st.sidebar:
        st.divider()
        mode = st.radio("Review mode", ["Agent-Guided Co-Pilot", "Blind Review"])
        blind = mode == "Blind Review"
        priority = st.selectbox("Priority", ["ALL"] + PRIORITIES, disabled=blind)
        label = st.selectbox("Predicted relationship", ["ALL"] + sorted(VALID_LABELS),
                             format_func=lambda v: LABEL_NAMES.get(v, v), disabled=blind)
        if blind:
            st.caption("Prediction-based filters and priority ordering are not used in blind mode.")
        st.divider()
        status = st.radio("Review status", ["All pairs", "Unreviewed", "Reviewed"])
        st.text_input("Reviewer name", key="reviewer")
        reviewed = len(st.session_state.decisions)
        st.progress(reviewed / len(records) if records else 0)
        st.caption(f"{reviewed} / {len(records)} reviewed")
        st.warning("Decisions are session-only. Download them before leaving or restarting.")
        st.divider()
        st.subheader("Exports")
        st.caption("Only your recorded decisions are included — never unreviewed predictions.")
        decisions = list(st.session_state.decisions.values())
        csv_bytes, json_bytes = export_bytes(decisions)
        proposals = json.loads(json_bytes)
        prefix = "synthetic_demo_" if demo else ""
        st.download_button("Download decision log (CSV)", csv_bytes,
                           file_name=f"{prefix}decision_log.csv", mime="text/csv",
                           disabled=not decisions, use_container_width=True)
        st.download_button("Download dry-run proposals (JSON)", json_bytes,
                           file_name=f"{prefix}dry_run.json", mime="application/json",
                           disabled=not proposals, use_container_width=True)
        st.caption(f"{len(decisions)} decisions · {len(proposals)} dry-run proposals")
        if proposals:
            with st.expander("Preview dry-run proposals"):
                st.json(proposals)
    filtered = [
        r for r in records
        if (blind or priority == "ALL" or r["priority"] == priority)
        and (blind or label == "ALL" or r["label"] == label)
        and (status == "All pairs"
             or (status == "Reviewed") == (r["pair_id"] in st.session_state.decisions))
    ]
    if blind:
        filtered.sort(key=lambda r: str(r["pair_id"]))
    if not filtered:
        st.info("No pairs match these filters.")
        return
    ids = [r["pair_id"] for r in filtered]
    by_id = {r["pair_id"]: r for r in filtered}
    if st.session_state.get("selected_pair") not in ids:
        st.session_state.selected_pair = ids[0]

    def navigate(offset: int) -> None:
        index = ids.index(st.session_state.selected_pair)
        st.session_state.selected_pair = ids[max(0, min(index + offset, len(ids) - 1))]

    position = ids.index(st.session_state.selected_pair)
    previous, selector, following = st.columns([1, 4, 1], vertical_alignment="bottom")
    previous.button("Previous", use_container_width=True, disabled=position == 0, on_click=navigate, args=(-1,))
    with selector:
        pid = st.selectbox("Candidate pair", ids, key="selected_pair")
    following.button("Next", use_container_width=True, disabled=position == len(ids) - 1, on_click=navigate, args=(1,))
    pair = by_id[pid]
    st.caption(f"Pair {ids.index(pid) + 1} of {len(ids)} · ID: {pid}")
    revealed = pid in st.session_state.revealed_pairs
    # Show source records first, as in the approved comparison-focused design.
    render_comparison(pair, demo)
    if not blind or revealed:
        st.write(f"**Priority: {pair['priority']}** — {pair.get('routing_reason', '')}")
        render_evidence(pair)
    else:
        st.info("Blind review: the machine recommendation is hidden.")
        if st.button("Reveal machine recommendation"):
            st.session_state.revealed_pairs.add(pid)
            st.rerun()
    if not blind or revealed:
        st.divider()
        st.header("Falsification check")
        st.caption("Demo uses offline analysis. Other datasets use the configured provider, "
                   "which may send metadata/excerpts to an external model.")
        if st.button("Run Falsification Agent Check"):
            with st.spinner("Checking for contradictory evidence…"):
                try:
                    st.session_state.agent_results[pid] = (
                        run_offline_falsification(pair) if demo else analyze_with_agent(pair)
                    )
                except Exception as exc:
                    st.warning("The analysis could not complete. Your review decisions are unchanged.")
                    st.caption(f"Reason: {type(exc).__name__} — {exc}")
        result = st.session_state.agent_results.get(pid)
        if result:
            st.caption(f"Inference tier: {result.get('tier', 'offline')} · "
                       f"Model: {result.get('model_used', 'offline_analyzer')}")
            st.info(display(result.get("falsification_finding")))
            st.write(display(result.get("maintainer_summary")))
        component = components.get(pair.get("component_id"), {})
        if len(component.get("pairs", [])) > 1:
            with st.expander("Connected case family"):
                st.caption(f"{len(component['cluster_ids'])} clusters · "
                           f"{len(component['pairs'])} candidate pairs")
                st.dataframe(pd.DataFrame([{
                    "Pair": r["pair_id"],
                    "Record A": (r.get("cluster_low") or {}).get("case_name"),
                    "Record B": (r.get("cluster_high") or {}).get("case_name"),
                    "Prediction": LABEL_NAMES.get(r.get("label"), r.get("label")),
                    "Priority": r.get("priority"),
                } for r in component["pairs"]]), use_container_width=True, hide_index=True)
    st.divider()
    render_decisions(pair, mode, demo)


if __name__ == "__main__":
    main()
