"""Split-safe subcomponent and whole-project benchmark engine.

Enforces strict separation between `training` and `validation` splits and blocks access to `test` split rows.
Provides targeted diagnostic scorecards for:
- Role 1: Ingestion Normalizer & Version Classifier
- Role 2: Citation Topology & Reporter Page Disambiguator
- Role 3: Litigation Lifecycle, Companion Decisions & Graph Consistency
- All: Whole-project side-by-side Training vs. Validation generalization & safety audit
"""

from __future__ import annotations

import collections
import functools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from triage.constants import (
    KEEP_SEPARATE_LABELS,
    LABEL_CITATION_ERROR,
    LABEL_INSUFFICIENT,
    LABEL_PARALLEL_SPLIT,
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    LABEL_RELATED_DECISION,
    LABEL_SHARED_PAGE,
    LABEL_UNRELATED,
    MERGE_LABELS,
    PRIORITY_CRITICAL,
)
from triage.graph import (
    build_connected_components,
    detect_graph_contradictions,
    reconcile_graph_predictions,
)
from triage.io import find_default_data_dir, load_dataset, load_development_splits
from triage.model import CLASS_ORDER, CalibratedTriageClassifier
from triage.route import assign_queue_priority


def _compute_split_metrics(
    records: List[Dict[str, Any]],
    pred_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute comprehensive classification, safety, and per-class metrics on a split."""
    total = 0
    correct = 0
    baseline_correct = 0
    abstentions = 0
    false_merges = 0
    true_merges = 0
    pred_merges = 0
    gold_merges = 0
    high_conf_merge_tp = 0
    high_conf_merge_total = 0

    per_class_gold: Dict[str, int] = collections.defaultdict(int)
    per_class_pred: Dict[str, int] = collections.defaultdict(int)
    per_class_tp: Dict[str, int] = collections.defaultdict(int)
    confusion: Dict[Tuple[str, str], int] = collections.Counter()

    for r in records:
        pid = r["pair_id"]
        gold = r.get("silver_label") or r.get("gold_relation") or r.get("family")
        if not gold or pid not in pred_map:
            continue

        pred_item = pred_map[pid]
        pred = pred_item["label"]
        conf = float(pred_item.get("confidence", 0.5))
        base = r.get("rule_class") or r.get("baseline_prediction")

        total += 1
        per_class_gold[gold] += 1
        per_class_pred[pred] += 1
        confusion[(gold, pred)] += 1

        if pred == LABEL_INSUFFICIENT:
            abstentions += 1
        if pred == gold:
            correct += 1
            per_class_tp[gold] += 1
        if base == gold:
            baseline_correct += 1

        is_gold_merge = gold in MERGE_LABELS or gold == "same_decision"
        is_pred_merge = pred in MERGE_LABELS

        if is_gold_merge:
            gold_merges += 1
        if is_pred_merge:
            pred_merges += 1
            if is_gold_merge:
                true_merges += 1
            else:
                false_merges += 1

        if is_pred_merge and conf >= 0.80:
            high_conf_merge_total += 1
            if is_gold_merge:
                high_conf_merge_tp += 1

    per_class: Dict[str, Dict[str, float]] = {}
    f1_supported: List[float] = []
    for cls in CLASS_ORDER:
        g_cnt = per_class_gold[cls]
        p_cnt = per_class_pred[cls]
        tp = per_class_tp[cls]
        prec = tp / p_cnt if p_cnt > 0 else 0.0
        rec = tp / g_cnt if g_cnt > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        if g_cnt > 0:
            f1_supported.append(f1)
        per_class[cls] = {
            "gold": g_cnt,
            "pred": p_cnt,
            "tp": tp,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / total if total > 0 else 0.0,
        "macro_f1": sum(f1_supported) / len(f1_supported) if f1_supported else 0.0,
        "abstentions": abstentions,
        "abstention_rate": abstentions / total if total > 0 else 0.0,
        "coverage": (total - abstentions) / total if total > 0 else 0.0,
        "gold_merges": gold_merges,
        "pred_merges": pred_merges,
        "true_merges": true_merges,
        "false_merges": false_merges,
        "merge_precision": true_merges / pred_merges if pred_merges > 0 else 1.0,
        "merge_recall": true_merges / gold_merges if gold_merges > 0 else 0.0,
        "high_conf_merge_tp": high_conf_merge_tp,
        "high_conf_merge_total": high_conf_merge_total,
        "high_conf_merge_precision": (
            high_conf_merge_tp / high_conf_merge_total if high_conf_merge_total > 0 else 1.0
        ),
        "per_class": per_class,
        "confusion": dict(confusion),
    }


@functools.lru_cache(maxsize=16)
def _cached_dev_split(path_str: str, splits: Tuple[str, ...]) -> List[Dict[str, Any]]:
    return load_development_splits(Path(path_str), splits=splits)


def _predict_records(
    records: List[Dict[str, Any]],
    training_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Fit classifier strictly on training_records and return predictions map for records."""
    clf = CalibratedTriageClassifier()
    if training_records:
        clf.fit(training_records)

    initial_preds: List[Dict[str, Any]] = []
    for r in records:
        label, conf, sup, con = clf.predict_pair(r)
        initial_preds.append({
            "pair_id": r["pair_id"],
            "label": label,
            "confidence": conf,
            "evidence": {"supporting": sup, "contradicting": con},
        })

    comps = build_connected_components(records)
    reconciled = reconcile_graph_predictions(initial_preds, comps)
    routed = [assign_queue_priority(p) for p in reconciled]
    return {p["pair_id"]: p for p in routed}


def run_role1_bench(data_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, Any]:
    """Role 1 Bench: Ingestion Normalizer & Version Disambiguator (training & validation only)."""
    d_dir = data_dir or find_default_data_dir()
    silver_train = _cached_dev_split(str(d_dir / "dev_silver.jsonl"), ("training",))
    silver_val = _cached_dev_split(str(d_dir / "dev_silver.jsonl"), ("validation",))
    hn_train = _cached_dev_split(str(d_dir / "hard_negatives.jsonl"), ("training",))

    train_pool = silver_train + hn_train
    preds_train = _predict_records(silver_train, train_pool)
    preds_val = _predict_records(silver_val, train_pool)

    m_train = _compute_split_metrics(silver_train, preds_train)
    m_val = _compute_split_metrics(silver_val, preds_val)

    # Slice 1: Same-decision pairs with raw docket conflict or missing docket match
    noisy_docket_train = [
        r for r in silver_train
        if r.get("silver_label") in MERGE_LABELS
        and (r["features"].get("docket_numbers_conflict") or not r["features"].get("same_docket_number"))
    ]
    noisy_docket_val = [
        r for r in silver_val
        if r.get("silver_label") in MERGE_LABELS
        and (r["features"].get("docket_numbers_conflict") or not r["features"].get("same_docket_number"))
    ]
    noisy_train_recovered = sum(
        1 for r in noisy_docket_train if preds_train[r["pair_id"]]["label"] == r["silver_label"]
    )
    noisy_val_recovered = sum(
        1 for r in noisy_docket_val if preds_val[r["pair_id"]]["label"] == r["silver_label"]
    )

    # Slice 2: FLP version pairs recall on validation split (if available locally)
    flp_val_recall: Optional[float] = None
    flp_path = d_dir / "flp_version_pairs.jsonl"
    if flp_path.exists():
        flp_val = _cached_dev_split(str(flp_path), ("validation",))[:200]
        flp_preds = _predict_records(flp_val, train_pool)
        flp_hits = sum(1 for r in flp_val if flp_preds[r["pair_id"]]["label"] in MERGE_LABELS)
        flp_val_recall = flp_hits / len(flp_val) if flp_val else 0.0

    if verbose:
        print("\n" + "=" * 78)
        print("ROLE 1 BENCHMARK: Ingestion Normalizer & Version Classifier (TEST SPLIT LOCKED)")
        print("=" * 78)
        print(f"Merge Binary Precision (Train / Val): {m_train['merge_precision']:.1%} / {m_val['merge_precision']:.1%}")
        print(f"Merge Binary Recall    (Train / Val): {m_train['merge_recall']:.1%} / {m_val['merge_recall']:.1%}")
        for cls in (LABEL_PROBABLE_DUPLICATE, LABEL_PROBABLE_VERSION, LABEL_PARALLEL_SPLIT):
            ct = m_train["per_class"][cls]
            cv = m_val["per_class"][cls]
            print(
                f"  {cls:24s} | Train F1: {ct['f1']:5.1%} (P={ct['precision']:5.1%}, R={ct['recall']:5.1%}, n={ct['gold']}) "
                f"| Val F1: {cv['f1']:5.1%} (P={cv['precision']:5.1%}, R={cv['recall']:5.1%}, n={cv['gold']})"
            )
        print(
            f"Noisy-Docket Merge Recovery: Train {noisy_train_recovered}/{len(noisy_docket_train)} "
            f"| Val {noisy_val_recovered}/{len(noisy_docket_val)}"
        )
        if flp_val_recall is not None:
            print(f"FLP Version-Pairs Validation Recall (n=200): {flp_val_recall:.1%}")

    return {
        "train": m_train,
        "validation": m_val,
        "noisy_docket_train": (noisy_train_recovered, len(noisy_docket_train)),
        "noisy_docket_val": (noisy_val_recovered, len(noisy_docket_val)),
        "flp_val_recall": flp_val_recall,
    }


def run_role2_bench(data_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, Any]:
    """Role 2 Bench: Citation Topology & Reporter Page Disambiguator (training & validation only)."""
    d_dir = data_dir or find_default_data_dir()
    silver_train = _cached_dev_split(str(d_dir / "dev_silver.jsonl"), ("training",))
    silver_val = _cached_dev_split(str(d_dir / "dev_silver.jsonl"), ("validation",))
    hn_train = _cached_dev_split(str(d_dir / "hard_negatives.jsonl"), ("training",))
    hn_val = _cached_dev_split(str(d_dir / "hard_negatives.jsonl"), ("validation",))

    train_pool = silver_train + hn_train
    preds_train = _predict_records(silver_train, train_pool)
    preds_val = _predict_records(silver_val, train_pool)
    preds_hn_val = _predict_records(hn_val, train_pool)

    m_train = _compute_split_metrics(silver_train, preds_train)
    m_val = _compute_split_metrics(silver_val, preds_val)

    # Hard negative citation / shared-page families on validation
    cit_families = {"shared_page_ambiguity", "shared_reporter_page", "citation_data_error"}
    hn_cit_val = [r for r in hn_val if r.get("family") in cit_families]
    hn_cit_false_merges = sum(
        1 for r in hn_cit_val if preds_hn_val[r["pair_id"]]["label"] in MERGE_LABELS
    )

    if verbose:
        print("\n" + "=" * 78)
        print("ROLE 2 BENCHMARK: Citation Topology & Page Disambiguator (TEST SPLIT LOCKED)")
        print("=" * 78)
        for cls in (LABEL_SHARED_PAGE, LABEL_CITATION_ERROR):
            ct = m_train["per_class"][cls]
            cv = m_val["per_class"][cls]
            print(
                f"  {cls:24s} | Train F1: {ct['f1']:5.1%} (P={ct['precision']:5.1%}, R={ct['recall']:5.1%}, n={ct['gold']}) "
                f"| Val F1: {cv['f1']:5.1%} (P={cv['precision']:5.1%}, R={cv['recall']:5.1%}, n={cv['gold']})"
            )
        print(
            f"Hard-Negative Citation/Page False Merges (Val): {hn_cit_false_merges}/{len(hn_cit_val)} "
            "(Target: 0)"
        )

    return {
        "train": m_train,
        "validation": m_val,
        "hn_cit_val_false_merges": hn_cit_false_merges,
        "hn_cit_val_total": len(hn_cit_val),
    }


def run_role3_bench(data_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, Any]:
    """Role 3 Bench: Litigation Lifecycle, Companion Decisions & Graph Consistency (training & validation only)."""
    d_dir = data_dir or find_default_data_dir()
    silver_train = _cached_dev_split(str(d_dir / "dev_silver.jsonl"), ("training",))
    silver_val = _cached_dev_split(str(d_dir / "dev_silver.jsonl"), ("validation",))
    hn_train = _cached_dev_split(str(d_dir / "hard_negatives.jsonl"), ("training",))
    hn_val = _cached_dev_split(str(d_dir / "hard_negatives.jsonl"), ("validation",))
    adj_dev = _cached_dev_split(str(d_dir / "adjudicate_me.jsonl"), ("training", "validation"))

    train_pool = silver_train + hn_train
    preds_train = _predict_records(silver_train, train_pool)
    preds_val = _predict_records(silver_val, train_pool)
    preds_hn_val = _predict_records(hn_val, train_pool)
    preds_adj = _predict_records(adj_dev, train_pool)

    m_train = _compute_split_metrics(silver_train, preds_train)
    m_val = _compute_split_metrics(silver_val, preds_val)

    lifecycle_families = {
        "related_decision",
        "unrelated",
        "appellate_history",
        "recurring_case_name",
        "materially_different_similar_metadata",
    }
    hn_life_val = [r for r in hn_val if r.get("family") in lifecycle_families]
    hn_life_false_merges = sum(
        1 for r in hn_life_val if preds_hn_val[r["pair_id"]]["label"] in MERGE_LABELS
    )

    if verbose:
        print("\n" + "=" * 78)
        print("ROLE 3 BENCHMARK: Litigation Lifecycle & Graph Consistency (TEST SPLIT LOCKED)")
        print("=" * 78)
        for cls in (LABEL_RELATED_DECISION, LABEL_UNRELATED):
            ct = m_train["per_class"][cls]
            cv = m_val["per_class"][cls]
            print(
                f"  {cls:24s} | Train F1: {ct['f1']:5.1%} (P={ct['precision']:5.1%}, R={ct['recall']:5.1%}, n={ct['gold']}) "
                f"| Val F1: {cv['f1']:5.1%} (P={cv['precision']:5.1%}, R={cv['recall']:5.1%}, n={cv['gold']})"
            )
        print(
            f"Hard-Negative Lifecycle/Caption False Merges (Val): {hn_life_false_merges}/{len(hn_life_val)} "
            "(Target: 0)"
        )
        print(f"Adjudication Dev Set Routed (Train+Val, n={len(adj_dev)}): {len(preds_adj)} evaluated")

    return {
        "train": m_train,
        "validation": m_val,
        "hn_life_val_false_merges": hn_life_false_merges,
        "hn_life_val_total": len(hn_life_val),
    }


def run_all_benches(data_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, Any]:
    """Run all 3 subcomponent benches plus the Whole-Project Generalization & Safety Audit."""
    d_dir = data_dir or find_default_data_dir()
    r1 = run_role1_bench(d_dir, verbose=verbose)
    r2 = run_role2_bench(d_dir, verbose=verbose)
    r3 = run_role3_bench(d_dir, verbose=verbose)

    m_train = r1["train"]
    m_val = r1["validation"]
    gen_gap = m_train["accuracy"] - m_val["accuracy"]

    hn_train = _cached_dev_split(str(d_dir / "hard_negatives.jsonl"), ("training",))
    hn_val = _cached_dev_split(str(d_dir / "hard_negatives.jsonl"), ("validation",))
    silver_train = _cached_dev_split(str(d_dir / "dev_silver.jsonl"), ("training",))
    train_pool = silver_train + hn_train

    preds_hn_train = _predict_records(hn_train, train_pool)
    preds_hn_val = _predict_records(hn_val, train_pool)

    hn_train_fm = sum(1 for r in hn_train if preds_hn_train[r["pair_id"]]["label"] in MERGE_LABELS)
    hn_val_fm = sum(1 for r in hn_val if preds_hn_val[r["pair_id"]]["label"] in MERGE_LABELS)

    if verbose:
        print("\n" + "=" * 78)
        print("WHOLE-PROJECT GENERALIZATION & ANTI-OVERFITTING AUDIT (TEST SPLIT LOCKED)")
        print("=" * 78)
        print(
            f"Silver Accuracy     : Train = {m_train['correct']}/{m_train['total']} ({m_train['accuracy']:.1%}) "
            f"| Val = {m_val['correct']}/{m_val['total']} ({m_val['accuracy']:.1%}) "
            f"[Baseline Val = {m_val['baseline_accuracy']:.1%}]"
        )
        print(
            f"Macro-F1 (Supported): Train = {m_train['macro_f1']:.1%} | Val = {m_val['macro_f1']:.1%}"
        )
        print(
            f"Generalization Gap  : {gen_gap:+.1%} (Train - Val; warning threshold: +25.0%)"
        )
        if gen_gap > 0.25:
            print("  [WARNING] High generalization gap detected! Check rules/model for training-set memorization.")
        else:
            print("  [PASS] Generalization gap is within healthy bounds.")
        print(
            f"Hard-Negative Safety: Train False Merges = {hn_train_fm}/{len(hn_train)} ({hn_train_fm/len(hn_train):.1%}) "
            f"| Val False Merges = {hn_val_fm}/{len(hn_val)} ({hn_val_fm/len(hn_val):.1%})"
        )
        print("=" * 78 + "\n")

    return {
        "role1": r1,
        "role2": r2,
        "role3": r3,
        "generalization_gap": gen_gap,
        "hn_train_false_merges": hn_train_fm,
        "hn_val_false_merges": hn_val_fm,
    }
