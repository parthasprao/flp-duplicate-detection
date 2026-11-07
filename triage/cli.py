"""Command-line interface for the Case-Identity Triage system."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from triage.agent import analyze_with_agent
from triage.bench import (
    run_all_benches,
    run_role1_bench,
    run_role2_bench,
    run_role3_bench,
)
from triage.constants import (
    LABEL_INSUFFICIENT,
    MERGE_LABELS,
    REVIEW_REQUIRED_LABELS,
    VALID_LABELS,
)
from triage.export import (
    export_courtlistener_dry_run_payloads,
    export_submission,
)
from triage.graph import build_connected_components, reconcile_graph_predictions
from triage.io import (
    TestSetLockError,
    find_default_data_dir,
    load_dataset,
    load_development_splits,
)
from triage.model import METRIC_CLASS_ORDER, CalibratedTriageClassifier
from triage.route import assign_queue_priority
from triage.rules import classify_pair_rule


def run_classification_pipeline(
    records: List[Dict[str, Any]],
    training_records: Optional[List[Dict[str, Any]]] = None,
    use_agent_tier: bool = False,
) -> List[Dict[str, Any]]:
    """Execute the full 3-tier cascade on a dataset."""
    classifier = CalibratedTriageClassifier()
    if training_records:
        classifier.fit(training_records)

    # 1. Tier 1: Rules + Calibrated Model
    initial_preds: List[Dict[str, Any]] = []
    for r in records:
        label, conf, sup, con = classifier.predict_pair(r)
        initial_preds.append({
            "pair_id": r["pair_id"],
            "label": label,
            "confidence": conf,
            "evidence": {"supporting": sup, "contradicting": con},
        })

    # 2. Graph consistency & connected components
    components = build_connected_components(records)
    reconciled = reconcile_graph_predictions(initial_preds, components)

    # 3. Tier 2: Agent falsification for ambiguous or conflict pairs
    final_preds: List[Dict[str, Any]] = []
    pair_map = {r["pair_id"]: r for r in records}

    for p in reconciled:
        pid = p["pair_id"]
        # If use_agent_tier is requested and pair is uncertain or flagged
        if use_agent_tier and (p.get("label") in MERGE_LABELS or p.get("label") == LABEL_INSUFFICIENT):
            agent_res = analyze_with_agent(pair_map[pid])
            p["agent_finding"] = agent_res.get("falsification_finding")
            p["agent_tier"] = agent_res.get("tier")
            p["agent_model"] = agent_res.get("model_used")
            if agent_res.get("label") in VALID_LABELS:
                # If agent found a falsifying fact, downgrade merge
                if p["label"] in MERGE_LABELS and agent_res["label"] not in MERGE_LABELS:
                    p["label"] = agent_res["label"]
                    p["confidence"] = agent_res.get("confidence", 0.85)
                    p["evidence"]["contradicting"].extend(agent_res.get("contradicting_evidence", []))
                    p["evidence"]["supporting"].append(agent_res.get("falsification_finding", ""))
                elif p["label"] == LABEL_INSUFFICIENT and agent_res["label"] != LABEL_INSUFFICIENT:
                    p["label"] = agent_res["label"]
                    p["confidence"] = float(agent_res.get("confidence", 0.85))
                    p["evidence"]["supporting"].extend(agent_res.get("supporting_evidence", []))
                    p["evidence"]["contradicting"].extend(agent_res.get("contradicting_evidence", []))

        # 4. Tier 3: Priority routing
        routed = assign_queue_priority(p)
        final_preds.append(routed)

    return final_preds


def classify_command(args: argparse.Namespace) -> None:
    """CLI action: Classify an input candidate file."""
    in_path = Path(args.input)
    out_path = Path(args.out)

    print(f"Loading input file: {in_path}...")
    records = load_dataset(in_path)
    print(f"Loaded {len(records)} candidate pairs.")

    if args.agent != "none":
        os.environ["TRIAGE_LLM"] = args.agent

    # Check for training data to fit calibrated model
    data_dir = find_default_data_dir()
    train_silver_path = data_dir / "dev_silver.jsonl"
    train_hn_path = data_dir / "hard_negatives.jsonl"

    training_records: List[Dict[str, Any]] = []
    if train_silver_path.exists():
        training_records.extend(load_development_splits(train_silver_path, splits=("training",)))
    if train_hn_path.exists():
        training_records.extend(load_development_splits(train_hn_path, splits=("training",)))

    print(f"Running 3-tier triage pipeline (training set size: {len(training_records)})...")
    preds = run_classification_pipeline(
        records,
        training_records=training_records if training_records else None,
        use_agent_tier=args.agent != "none",
    )

    fmt = "csv" if out_path.suffix == ".csv" else "jsonl"
    export_submission(preds, out_path, format=fmt)
    print(f"Predictions saved to: {out_path}")

    # Also export dry-run payloads
    if args.dry_run_payloads:
        payload_path = Path(args.dry_run_payloads)
        export_courtlistener_dry_run_payloads(preds, payload_path)
        print(f"CourtListener dry-run payloads saved to: {payload_path}")


def evaluate_command(args: argparse.Namespace) -> None:
    """CLI action: Evaluate predictions against silver/gold labels."""
    preds_path = Path(args.predictions)
    gold_path = Path(args.gold)

    # Load predictions
    preds_list = []
    if preds_path.suffix == ".jsonl":
        with open(preds_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    preds_list.append(json.loads(line))
    elif preds_path.suffix == ".csv":
        import csv
        with open(preds_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            preds_list = list(reader)
    else:
        raise ValueError(f"Unsupported prediction format: {preds_path.suffix}. Expected .jsonl or .csv")
    pred_map = {p["pair_id"]: p for p in preds_list}

    # Load gold with split-lock enforcement (defaults to validation)
    target_split = args.split or "validation"
    unlock_test = getattr(args, "unlock_test_set", False)
    gold_records = load_dataset(
        gold_path,
        split=target_split,
        allow_test_split=unlock_test,
    )

    print(f"\n=======================================================")
    print(f"EVALUATION REPORT: {preds_path.name} vs {gold_path.name}")
    print(f"Split Filter: {target_split} (Count: {len(gold_records)})")
    print(f"=======================================================\n")

    correct = 0
    total = 0
    baseline_correct = 0

    per_class_gold: Dict[str, int] = collections.defaultdict(int)
    per_class_pred: Dict[str, int] = collections.defaultdict(int)
    per_class_tp: Dict[str, int] = collections.defaultdict(int)

    high_conf_merge_tp = 0
    high_conf_merge_total = 0
    abstentions = 0

    for r in gold_records:
        pid = r["pair_id"]
        gold_label = r.get("silver_label") or r.get("gold_relation") or r.get("family")
        if not gold_label:
            continue

        base_label = r.get("rule_class") or r.get("baseline_prediction")
        pred_item = pred_map.get(pid)
        if not pred_item:
            continue

        pred_label = pred_item["label"]
        conf = float(pred_item.get("confidence", 0.5))

        total += 1
        per_class_gold[gold_label] += 1
        per_class_pred[pred_label] += 1

        if pred_label in REVIEW_REQUIRED_LABELS:
            abstentions += 1

        if pred_label == gold_label:
            correct += 1
            per_class_tp[gold_label] += 1

        if base_label == gold_label:
            baseline_correct += 1

        if pred_label in MERGE_LABELS and conf >= 0.80:
            high_conf_merge_total += 1
            if gold_label in MERGE_LABELS:
                high_conf_merge_tp += 1

    overall_acc = correct / total if total > 0 else 0.0
    baseline_acc = baseline_correct / total if total > 0 else 0.0

    print(f"Overall Accuracy:        {correct}/{total} ({overall_acc:.2%})")
    print(f"Baseline Agreement:      {baseline_correct}/{total} ({baseline_acc:.2%})")
    print(f"Relative Improvement:    +{(overall_acc - baseline_acc):.2%}\n")

    if high_conf_merge_total > 0:
        hcm_prec = high_conf_merge_tp / high_conf_merge_total
        print(f"High-Conf Merge Prec:    {high_conf_merge_tp}/{high_conf_merge_total} ({hcm_prec:.2%}) [SAFETY METRIC]")
    print(f"Abstention Rate:         {abstentions}/{total} ({abstentions/total:.2%})")
    print(f"Coverage Rate:           {(total - abstentions)}/{total} ({(total - abstentions)/total:.2%})\n")

    print(f"{'Class':<25} | {'Gold':<6} | {'Pred':<6} | {'TP':<4} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}")
    print("-" * 75)

    f1_list = []
    for cls in METRIC_CLASS_ORDER:
        g_cnt = per_class_gold[cls]
        p_cnt = per_class_pred[cls]
        tp = per_class_tp[cls]
        prec = tp / p_cnt if p_cnt > 0 else 0.0
        rec = tp / g_cnt if g_cnt > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        if g_cnt > 0:
            f1_list.append(f1)
        print(f"{cls:<25} | {g_cnt:<6} | {p_cnt:<6} | {tp:<4} | {prec:<7.1%} | {rec:<7.1%} | {f1:<7.1%}")

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    print("-" * 75)
    print(f"Macro-F1 (supported):     {macro_f1:.2%}\n")


def bench_command(args: argparse.Namespace) -> None:
    """CLI action: Run split-safe subcomponent or whole-project benchmarks."""
    data_dir = Path(args.data_dir) if args.data_dir else None
    role = str(args.role).lower()
    if role == "1":
        run_role1_bench(data_dir=data_dir, verbose=True)
    elif role == "2":
        run_role2_bench(data_dir=data_dir, verbose=True)
    elif role == "3":
        run_role3_bench(data_dir=data_dir, verbose=True)
    else:
        run_all_benches(data_dir=data_dir, verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CourtListener Duplicate Detection & Case-Identity Triage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Classify
    p_classify = subparsers.add_parser("classify", help="Classify candidate pairs file")
    p_classify.add_argument("input", help="Path to input candidate file (.jsonl or .csv)")
    p_classify.add_argument("--out", "-o", required=True, help="Path for output predictions (.jsonl or .csv)")
    p_classify.add_argument("--agent", choices=["offline", "openrouter", "gemini", "none"], default="offline", help="Agent tier mode")
    p_classify.add_argument("--dry-run-payloads", help="Path to export CourtListener ClusterRedirection payloads")
    p_classify.set_defaults(func=classify_command)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate predictions against gold/silver labels")
    p_eval.add_argument("predictions", help="Path to predictions file (.jsonl)")
    p_eval.add_argument("--gold", "-g", required=True, help="Path to gold dataset (.jsonl or .csv)")
    p_eval.add_argument(
        "--split",
        choices=["training", "validation", "train_val", "test"],
        default="validation",
        help="Split to evaluate (default: validation; 'test' is locked unless --unlock-test-set is set)",
    )
    p_eval.add_argument(
        "--unlock-test-set",
        action="store_true",
        help="Explicitly unlock the held-out test split (logged to outputs/.test_access_audit.log; forbidden for agents)",
    )
    p_eval.set_defaults(func=evaluate_command)

    # Bench
    p_bench = subparsers.add_parser(
        "bench",
        help="Run split-safe subcomponent (Role 1/2/3) or whole-project benchmark on training & validation splits",
    )
    p_bench.add_argument(
        "--role",
        "-r",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which benchmark suite to run (1=Normalizer/Versions, 2=Citations/SharedPage, 3=Lifecycle/Graph, all=Full)",
    )
    p_bench.add_argument("--data-dir", help="Optional override for hackathon data directory")
    p_bench.set_defaults(func=bench_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
