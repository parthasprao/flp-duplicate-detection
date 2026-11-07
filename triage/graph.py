"""Graph consistency and connected component analysis for CourtListener clusters.

Legal case-identity reasoning operates over families of records, not isolated pairs.
This module:
1. Builds connected components across candidate pairs.
2. Identifies transitive contradictions:
   e.g. A = B (merge) and B = C (merge), but A != C (keep separate).
3. Detects star-topology conflicts (e.g. one cluster proposed for merge with multiple conflicting dockets).
4. Flags high-risk components and adjusts confidence to protect against unsafe cascading merges.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Set, Tuple

from triage.constants import (
    LABEL_INSUFFICIENT,
    MERGE_LABELS,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
)


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


def build_connected_components(pairs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Cluster candidate pairs into connected components of cluster IDs.

    Returns:
        Mapping of component_id -> {
            "cluster_ids": set of cluster ids,
            "pair_ids": list of pair_ids,
            "pairs": list of pair dicts,
        }
    """
    uf = UnionFind()
    pair_nodes: List[Tuple[str, int, int]] = []

    for r in pairs:
        pair_id = r["pair_id"]
        feat = r.get("features", {})
        low = r.get("cluster_low", {})
        high = r.get("cluster_high", {})

        cid_low = feat.get("cluster_id_low") or low.get("cluster_id")
        cid_high = feat.get("cluster_id_high") or high.get("cluster_id")

        if cid_low is not None and cid_high is not None:
            c1, c2 = int(cid_low), int(cid_high)
            uf.union(c1, c2)
            pair_nodes.append((pair_id, c1, c2))

    # Group pairs by root
    components: Dict[int, Dict[str, Any]] = collections.defaultdict(
        lambda: {"cluster_ids": set(), "pair_ids": [], "pairs": []}
    )

    pair_map = {r["pair_id"]: r for r in pairs}

    for pid, c1, c2 in pair_nodes:
        root = uf.find(c1)
        components[root]["cluster_ids"].add(c1)
        components[root]["cluster_ids"].add(c2)
        components[root]["pair_ids"].append(pid)
        components[root]["pairs"].append(pair_map[pid])

    # Stringify keys
    result: Dict[str, Dict[str, Any]] = {}
    for idx, (root, comp) in enumerate(components.items(), start=1):
        comp_id = f"comp-{idx:04d}"
        comp["component_id"] = comp_id
        result[comp_id] = comp

    return result


def detect_graph_contradictions(
    predictions: List[Dict[str, Any]],
    components: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Detect transitive merge contradictions within components.

    If Pair(A, B) is a MERGE and Pair(B, C) is a MERGE, but Pair(A, C) exists and is labeled KEEP_SEPARATE,
    this constitutes a direct transitive contradiction (A=B and B=C implies A=C).

    Returns:
        Mapping of pair_id -> list of contradiction warnings.
    """
    pred_map = {p["pair_id"]: p for p in predictions}
    contradictions: Dict[str, List[str]] = collections.defaultdict(list)

    for comp_id, comp in components.items():
        comp_pairs = comp["pairs"]
        if len(comp_pairs) < 3:
            continue

        # Build adjacency for merge edges and separation edges
        merge_edges: Set[Tuple[int, int]] = set()
        separate_edges: Set[Tuple[int, int]] = set()

        for r in comp_pairs:
            pid = r["pair_id"]
            pred = pred_map.get(pid, {})
            label = pred.get("label", r.get("rule_class"))

            feat = r.get("features", {})
            c1 = int(feat.get("cluster_id_low") or r.get("cluster_low", {}).get("cluster_id"))
            c2 = int(feat.get("cluster_id_high") or r.get("cluster_high", {}).get("cluster_id"))
            u, v = min(c1, c2), max(c1, c2)

            if label in MERGE_LABELS:
                merge_edges.add((u, v))
            else:
                separate_edges.add((u, v))

        # Check for arbitrary-length transitive merge contradictions (3-cycles, 4-cycles, n-cycles)
        merge_uf = UnionFind()
        for u, v in merge_edges:
            merge_uf.union(u, v)

        for sep_u, sep_v in separate_edges:
            if merge_uf.find(sep_u) == merge_uf.find(sep_v):
                msg = (
                    f"Transitive contradiction in {comp_id}: clusters {sep_u} and {sep_v} are connected "
                    f"by a chain of proposed merges, but pair {sep_u}-{sep_v} is marked keep-separate"
                )
                contradictions[f"{sep_u}-{sep_v}"].append(msg)
                root = merge_uf.find(sep_u)
                for mu, mv in merge_edges:
                    if merge_uf.find(mu) == root:
                        contradictions[f"{mu}-{mv}"].append(msg)

    return dict(contradictions)


HUB_MERGE_DEGREE = 4


def _pair_clusters(r: Dict[str, Any]) -> Tuple[int, int]:
    feat = r.get("features", {})
    c1 = int(feat.get("cluster_id_low") or r.get("cluster_low", {}).get("cluster_id"))
    c2 = int(feat.get("cluster_id_high") or r.get("cluster_high", {}).get("cluster_id"))
    return min(c1, c2), max(c1, c2)


def detect_hub_clusters(
    predictions: List[Dict[str, Any]],
    components: Dict[str, Dict[str, Any]],
    min_degree: int = HUB_MERGE_DEGREE,
) -> Dict[str, List[str]]:
    """Flag clusters proposed for merging with `min_degree` or more other clusters.

    A hub means one redirect would fold a whole family together, so every merge
    edge touching the hub is routed for family-level review.

    Returns:
        Mapping of pair_id -> list of hub warnings.
    """
    pred_map = {p["pair_id"]: p for p in predictions}
    warnings: Dict[str, List[str]] = collections.defaultdict(list)

    for comp_id, comp in components.items():
        merge_pairs: List[Tuple[str, int, int]] = []
        degree: Dict[int, int] = collections.Counter()
        for r in comp["pairs"]:
            pid = r["pair_id"]
            label = pred_map.get(pid, {}).get("label", r.get("rule_class"))
            if label not in MERGE_LABELS:
                continue
            u, v = _pair_clusters(r)
            merge_pairs.append((pid, u, v))
            degree[u] += 1
            degree[v] += 1

        hubs = {c for c, d in degree.items() if d >= min_degree}
        for hub in sorted(hubs):
            msg = (
                f"Hub cluster {hub} in {comp_id} has {degree[hub]} proposed merges; "
                f"review the whole family of {len(comp['cluster_ids'])} clusters before any redirect"
            )
            for pid, u, v in merge_pairs:
                if hub in (u, v):
                    warnings[pid].append(msg)

    return dict(warnings)


def reconcile_graph_predictions(
    predictions: List[Dict[str, Any]],
    components: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Enforce graph consistency and safety over predictions.

    When transitive contradictions are detected:
    - Escalate priority to CRITICAL.
    - Add contradiction warning to evidence.
    - If confidence is not near-certain (> 0.95), downgrade merge to insufficient_evidence to prevent unsafe merge.
    """
    contradictions = detect_graph_contradictions(predictions, components)
    hubs = detect_hub_clusters(predictions, components)

    # Component map by pair_id
    pair_to_comp: Dict[str, str] = {}
    for comp_id, comp in components.items():
        for pid in comp["pair_ids"]:
            pair_to_comp[pid] = comp_id

    reconciled: List[Dict[str, Any]] = []
    for p in predictions:
        pid = p["pair_id"]
        rec = dict(p)
        # Copy the evidence lists so warnings never leak back into the caller's predictions.
        ev = p.get("evidence") or {}
        rec["evidence"] = {k: list(v) if isinstance(v, list) else v for k, v in ev.items()}
        rec["component_id"] = pair_to_comp.get(pid, "unknown")

        if pid in contradictions:
            warnings = contradictions[pid]
            rec["priority"] = PRIORITY_CRITICAL
            rec["routing_reason"] = "Transitive contradiction across case family"
            con_ev = rec.setdefault("evidence", {}).setdefault("contradicting", [])
            con_ev.extend(warnings)

            # Safety fallback: if there is a contradiction in the graph, do not auto-merge
            if rec.get("label") in MERGE_LABELS and rec.get("confidence", 0.0) < 0.95:
                rec["label"] = LABEL_INSUFFICIENT
                rec["confidence"] = 0.50
                rec.setdefault("evidence", {}).setdefault("supporting", []).append(
                    "Transitive contradiction between candidate pairs requires human adjudication"
                )

        if pid in hubs:
            # A hub is a review-order signal, not evidence against the merge: escalate only.
            rec["priority"] = PRIORITY_CRITICAL
            if pid not in contradictions:
                rec["routing_reason"] = "Hub cluster: review the whole case family before merging"
            rec.setdefault("evidence", {}).setdefault("contradicting", []).extend(hubs[pid])

        reconciled.append(rec)

    return reconciled
