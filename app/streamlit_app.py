"""CourtListener Case-Identity Triage: Maintainer Review Queue Workbench.

A dual-mode review application for Free Law Project / CourtListener data maintainers:
- Blind Review Mode: Unanchored human assessment with post-decision reveal.
- Co-Pilot Mode: Agent-guided falsification analysis, side-by-side metadata diffs, and connected-family consistency.
- Maintainer Export: Decision logs and dry-run ClusterRedirection ORM payloads.
"""

from __future__ import annotations

import difflib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path for robust execution across all launch methods
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from triage.agent import analyze_with_agent, run_offline_falsification
from triage.constants import (
    CLUSTER_REDIRECTION_MAP,
    LABEL_CITATION_TOPOLOGY_UNVERIFIED,
    LABEL_INSUFFICIENT,
    LABEL_PARALLEL_SPLIT,
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    LABEL_RELATED_DECISION,
    LABEL_SHARED_PAGE,
    LABEL_UNRELATED,
    MERGE_LABELS,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    VALID_LABELS,
)
from triage.export import (
    export_courtlistener_dry_run_payloads,
    export_decision_log,
)
from triage.graph import build_connected_components, reconcile_graph_predictions
from triage.io import find_default_data_dir, load_dataset
from triage.model import CalibratedTriageClassifier

# Page configuration
st.set_page_config(
    page_title="CourtListener Case-Identity Review Queue",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_all_data() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    data_dir = find_default_data_dir()
    candidates_path = data_dir / "candidates.jsonl"
    if not candidates_path.exists():
        candidates_path = data_dir / "candidates.csv"

    # Default to candidates if present, else dev_silver
    if candidates_path.exists():
        records = load_dataset(candidates_path)
    else:
        records = load_dataset(data_dir / "dev_silver.jsonl")

    # Load dev_silver to fit classifier
    silver_path = data_dir / "dev_silver.jsonl"
    training_rows = []
    if silver_path.exists():
        silver_all = load_dataset(silver_path)
        training_rows.extend([r for r in silver_all if r.get("split") == "training"])

    classifier = CalibratedTriageClassifier()
    if training_rows:
        classifier.fit(training_rows)

    initial_preds = []
    for r in records:
        lbl, conf, sup, con = classifier.predict_pair(r)
        r["pred_label"] = lbl
        r["pred_conf"] = conf
        r["pred_sup"] = sup
        r["pred_con"] = con
        initial_preds.append({
            "pair_id": r["pair_id"],
            "label": lbl,
            "confidence": conf,
            "evidence": {"supporting": sup, "contradicting": con},
        })

    components = build_connected_components(records)
    reconciled = reconcile_graph_predictions(initial_preds, components)

    rec_map = {p["pair_id"]: p for p in reconciled}
    for r in records:
        p = rec_map.get(r["pair_id"], {})
        r["priority"] = p.get("priority", PRIORITY_LOW)
        r["routing_reason"] = p.get("routing_reason", "")
        r["component_id"] = p.get("component_id", "")
        r["label"] = p.get("label", r["pred_label"])
        r["confidence"] = p.get("confidence", r["pred_conf"])
        r["evidence"] = p.get("evidence", {"supporting": r["pred_sup"], "contradicting": r["pred_con"]})

    return records, components


def main() -> None:
    st.title("⚖️ CourtListener Case-Identity Triage Workbench")
    st.caption("Evidence-backed review queue for Free Law Project / CourtListener data maintainers")

    records, components = load_all_data()

    # Session state for decisions
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "revealed_pairs" not in st.session_state:
        st.session_state.revealed_pairs = set()

    # Sidebar: Filters and Queue Metrics
    with st.sidebar:
        st.header("📋 Triage Queue Filters")
        prio_filter = st.selectbox(
            "Priority Tier",
            options=["ALL", PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW],
            index=0,
        )
        label_filter = st.selectbox(
            "Target Label",
            options=["ALL"] + sorted(list(VALID_LABELS)),
            index=0,
        )
        status_filter = st.radio(
            "Review Status",
            options=["All Pairs", "Unreviewed", "Reviewed"],
            index=0,
        )

        st.markdown("---")
        st.subheader("⚙️ Review Mode")
        review_mode = st.radio(
            "Mode",
            options=["Agent-Guided Co-Pilot", "Blind Review (Unanchored)"],
            index=0,
            help="Blind mode hides model recommendations until the reviewer renders an initial judgment.",
        )

        st.markdown("---")
        # Review Queue Progress
        total_pairs = len(records)
        reviewed_count = len(st.session_state.decisions)
        progress = reviewed_count / total_pairs if total_pairs > 0 else 0.0
        st.subheader("📊 Session Progress")
        st.progress(progress)
        st.write(f"Reviewed: **{reviewed_count}** / {total_pairs} ({progress:.1%})")

        # Export Actions
        st.markdown("---")
        st.subheader("💾 Maintainer Exports")
        if st.button("Export Decision Log (CSV)", use_container_width=True):
            log_path = Path("outputs/maintainer_decision_log.csv")
            dec_list = list(st.session_state.decisions.values())
            export_decision_log(dec_list, log_path)
            st.success(f"Saved {len(dec_list)} decisions to {log_path.name}")

        if st.button("Export Dry-Run Redirection Payloads", use_container_width=True):
            payload_path = Path("outputs/courtlistener_dry_run_payloads.json")
            dec_list = list(st.session_state.decisions.values())
            # Include both reviewed decisions and high-priority predictions
            items_to_export = dec_list if dec_list else [r for r in records if r["label"] in MERGE_LABELS]
            export_courtlistener_dry_run_payloads(items_to_export, payload_path)
            st.success(f"Generated payloads at {payload_path.name}")

    # Filter records
    filtered = []
    for idx, r in enumerate(records):
        pid = r["pair_id"]
        is_rev = pid in st.session_state.decisions
        if status_filter == "Unreviewed" and is_rev:
            continue
        if status_filter == "Reviewed" and not is_rev:
            continue
        if prio_filter != "ALL" and r.get("priority") != prio_filter:
            continue
        if label_filter != "ALL" and r.get("label") != label_filter:
            continue
        filtered.append((idx, r))

    if not filtered:
        st.warning("No candidate pairs match the current filter criteria.")
        return

    # Pagination / Pair Selector
    col_nav1, col_nav2, col_nav3 = st.columns([2, 5, 2])
    with col_nav1:
        if st.button("⬅️ Previous") and st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
    with col_nav3:
        if st.button("Next ➡️") and st.session_state.current_idx < len(filtered) - 1:
            st.session_state.current_idx += 1

    # Ensure index within bounds
    st.session_state.current_idx = min(st.session_state.current_idx, len(filtered) - 1)
    orig_idx, current_pair = filtered[st.session_state.current_idx]

    with col_nav2:
        st.markdown(
            f"<div style='text-align: center; font-weight: bold;'>Pair {st.session_state.current_idx + 1} of {len(filtered)} "
            f"(ID: <code>{current_pair['pair_id']}</code>)</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Priority & Routing Banner
    prio = current_pair.get("priority", PRIORITY_LOW)
    prio_color = {
        PRIORITY_CRITICAL: "red",
        PRIORITY_HIGH: "orange",
        PRIORITY_MEDIUM: "blue",
        PRIORITY_LOW: "gray",
    }.get(prio, "gray")

    st.markdown(
        f"**Queue Priority:** <span style='color:white; background-color:{prio_color}; padding:2px 8px; border-radius:4px; font-weight:bold;'>{prio}</span> "
        f"— *{current_pair.get('routing_reason')}*",
        unsafe_allow_html=True,
    )
    if current_pair.get("label") == LABEL_CITATION_TOPOLOGY_UNVERIFIED:
        st.info(
            "This pair shares a citation, but the data does not show how many distinct decisions appear on that reporter page. Verify the original page before choosing between shared-page ambiguity and citation-data error."
        )

    # Co-pilot vs Blind display
    is_blind = "Blind" in review_mode
    is_revealed = current_pair["pair_id"] in st.session_state.revealed_pairs

    if not is_blind or is_revealed:
        # Machine Recommendation Box
        with st.expander("🤖 Machine Ingestion Classification & Evidence", expanded=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Proposed Relationship", current_pair.get("label", "Unknown"))
                st.metric("Calibrated Confidence", f"{current_pair.get('confidence', 0.5):.1%}")
            with c2:
                sup = current_pair.get("evidence", {}).get("supporting", [])
                con = current_pair.get("evidence", {}).get("contradicting", [])
                st.markdown("**Supporting Evidence:**")
                for s in sup:
                    st.markdown(f"- ✅ {s}")
                if con:
                    st.markdown("**Contradicting Evidence:**")
                    for c in con:
                        st.markdown(f"- ⚠️ {c}")
    else:
        st.info("🔒 **Blind Review Mode Active:** Machine recommendations are hidden to prevent anchoring bias. Review the records below and choose your decision, or click 'Reveal AI Recommendation'.")
        if st.button("👁️ Reveal AI Recommendation"):
            st.session_state.revealed_pairs.add(current_pair["pair_id"])
            st.rerun()

    # Side-by-side Record Inspection
    low = current_pair.get("cluster_low", {})
    high = current_pair.get("cluster_high", {})
    feat = current_pair.get("features", {})

    st.subheader("🔍 Case Metadata Comparison")
    col_low, col_high = st.columns(2)

    with col_low:
        st.markdown(f"### Record Low (Cluster `{low.get('cluster_id')}`)")
        st.markdown(f"**Case Name:** `{low.get('case_name')}`")
        if low.get("case_name_full"):
            st.caption(f"Full: {low.get('case_name_full')}")
        st.markdown(f"**Court:** `{low.get('court', {}).get('short_name') or low.get('court_id')}`")
        st.markdown(f"**Date Filed:** `{low.get('date_filed')}`")
        st.markdown(f"**Docket:** `{low.get('docket', {}).get('docket_number')}`")
        cits_low = [c.get("citation_key", "") for c in low.get("citations", [])]
        st.markdown(f"**Citations:** `{', '.join(cits_low) if cits_low else 'None'}`")
        cl_url_low = current_pair.get("courtlistener_url_low") or f"https://www.courtlistener.com/opinion/{low.get('cluster_id')}/"
        st.markdown(f"[🌐 View on CourtListener]({cl_url_low})")

    with col_high:
        st.markdown(f"### Record High (Cluster `{high.get('cluster_id')}`)")
        st.markdown(f"**Case Name:** `{high.get('case_name')}`")
        if high.get("case_name_full"):
            st.caption(f"Full: {high.get('case_name_full')}")
        st.markdown(f"**Court:** `{high.get('court', {}).get('short_name') or high.get('court_id')}`")
        st.markdown(f"**Date Filed:** `{high.get('date_filed')}`")
        st.markdown(f"**Docket:** `{high.get('docket', {}).get('docket_number')}`")
        cits_high = [c.get("citation_key", "") for c in high.get("citations", [])]
        st.markdown(f"**Citations:** `{', '.join(cits_high) if cits_high else 'None'}`")
        cl_url_high = current_pair.get("courtlistener_url_high") or f"https://www.courtlistener.com/opinion/{high.get('cluster_id')}/"
        st.markdown(f"[🌐 View on CourtListener]({cl_url_high})")

    # Metadata Feature Badges
    st.markdown("---")
    st.markdown("#### ⚖️ Structural Feature Signals")
    f_cols = st.columns(7)
    with f_cols[0]:
        st.metric("Same Court", "Yes" if feat.get("same_court") else "No")
    with f_cols[1]:
        doc_status = "Conflict ❌" if feat.get("docket_numbers_conflict") else ("Same ✅" if feat.get("same_docket_number") else "Different")
        st.metric("Docket Status", doc_status)
    with f_cols[2]:
        st.metric("Date Gap", f"{feat.get('filing_date_delta_days')} days" if feat.get("filing_date_delta_days") is not None else "N/A")
    with f_cols[3]:
        st.metric("Shared Citations", f"{feat.get('shared_citations', 0)}")
    with f_cols[4]:
        st.metric("Hamming Distance", f"{feat.get('min_hamming')}" if feat.get("min_hamming") is not None else "N/A")
    with f_cols[5]:
        st.metric("Length Ratio", f"{feat.get('text_length_ratio'):.2f}" if feat.get("text_length_ratio") is not None else "N/A")
    with f_cols[6]:
        page_decisions = feat.get("distinct_decisions_on_shared_page")
        st.metric("Decisions on Page", page_decisions if page_decisions is not None else "Unverified")

    # Excerpt Comparison
    exc_low = current_pair.get("excerpt_low", "")
    exc_high = current_pair.get("excerpt_high", "")
    if exc_low or exc_high:
        st.markdown("---")
        st.subheader("📜 Text Excerpt Inspection")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.text_area("Low Cluster Excerpt", value=exc_low if exc_low else "[No text excerpt]", height=200, disabled=True)
        with t_col2:
            st.text_area("High Cluster Excerpt", value=exc_high if exc_high else "[No text excerpt]", height=200, disabled=True)

    # Tier 2 Falsification Agent Trigger
    st.markdown("---")
    st.subheader("🕵️ Tier 2: Falsification Devil's Advocate Analysis")
    if st.button("Run Falsification Agent Check"):
        with st.spinner("Analyzing merge hypothesis and testing for contradictions..."):
            agent_output = analyze_with_agent(current_pair)
            tier = agent_output.get("tier", "offline")
            model = agent_output.get("model_used", "offline_analyzer")
            tier_badge = {
                "fast": "⚡ GLM 5.3 Flash (Fast Inference)",
                "complex": "🧠 Nemotron Ultra (Deep Reasoning)",
                "offline": "🛡️ Offline Rule Analyzer",
            }.get(tier, tier)

            st.markdown(f"**Inference Tier:** `{tier_badge}` (Model: `{model}`)")
            st.info(f"**Agent Finding:** {agent_output.get('falsification_finding')}")
            st.write(f"**Maintainer Summary:** {agent_output.get('maintainer_summary')}")

    # Connected Family View (if component > 1)
    comp_id = current_pair.get("component_id")
    if comp_id and comp_id in components and len(components[comp_id]["pairs"]) > 1:
        comp_info = components[comp_id]
        with st.expander(f"🌐 Connected Case Family ({comp_id}: {len(comp_info['cluster_ids'])} clusters, {len(comp_info['pairs'])} pairs)"):
            st.write(f"Clusters in family: `{list(comp_info['cluster_ids'])}`")
            fam_rows = []
            for p in comp_info["pairs"]:
                fam_rows.append({
                    "pair_id": p["pair_id"],
                    "low": p.get("cluster_low", {}).get("case_name"),
                    "high": p.get("cluster_high", {}).get("case_name"),
                    "prediction": p.get("label"),
                    "priority": p.get("priority"),
                })
            st.dataframe(pd.DataFrame(fam_rows), use_container_width=True)

    # Maintainer Decision Action Section
    st.markdown("---")
    st.subheader("✍️ Maintainer Adjudication & Action")

    pid = current_pair["pair_id"]
    existing_dec = st.session_state.decisions.get(pid, {})

    dec_cols = st.columns(4)
    with dec_cols[0]:
        if st.button("✅ Confirm Duplicate (Merge)", use_container_width=True):
            st.session_state.decisions[pid] = {
                "pair_id": pid,
                "decision": LABEL_PROBABLE_DUPLICATE,
                "keep_cluster_id": low.get("cluster_id"),
                "delete_cluster_id": high.get("cluster_id"),
                "reviewer": "maintainer",
                "review_mode": review_mode,
                "reason": "Confirmed duplicate decision via review queue",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            st.rerun()

    with dec_cols[1]:
        if st.button("📄 Mark as Version", use_container_width=True):
            st.session_state.decisions[pid] = {
                "pair_id": pid,
                "decision": LABEL_PROBABLE_VERSION,
                "keep_cluster_id": low.get("cluster_id"),
                "delete_cluster_id": high.get("cluster_id"),
                "reviewer": "maintainer",
                "review_mode": review_mode,
                "reason": "Confirmed alternate version (slip/published)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            st.rerun()

    with dec_cols[2]:
        if st.button("❌ Keep Separate (Distinct)", use_container_width=True):
            st.session_state.decisions[pid] = {
                "pair_id": pid,
                "decision": LABEL_RELATED_DECISION if feat.get("same_court") else LABEL_UNRELATED,
                "reviewer": "maintainer",
                "review_mode": review_mode,
                "reason": "Records represent distinct legal proceedings",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            st.rerun()

    with dec_cols[3]:
        if st.button("⏳ Defer (Insufficient Data)", use_container_width=True):
            st.session_state.decisions[pid] = {
                "pair_id": pid,
                "decision": LABEL_INSUFFICIENT,
                "reviewer": "maintainer",
                "review_mode": review_mode,
                "reason": "Deferred: requires full text or docket inspection",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            st.rerun()

    if existing_dec:
        st.success(f"Recorded Decision: **{existing_dec.get('decision')}** ({existing_dec.get('reason')}) at {existing_dec.get('timestamp')}")


if __name__ == "__main__":
    main()
