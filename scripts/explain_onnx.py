"""
scripts/explain_onnx.py

Proof-of-Concept ONNX-Native GNNExplainer for software architectural smell detection.
Performs model-agnostic marginal attribution on ONNX models, extracts explanatory subgraphs,
computes Fidelity+/Fidelity-/Sparsity metrics, and produces visual reports.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from scripts.common import (
    CLASS_METRIC_NAMES,
    COMPONENT_METRIC_NAMES,
    EDGE_TYPES,
    NodeFeatureScaler,
    ONNXInferenceRunner,
    SyntheticGraphBuilder,
    custom_json_serializer,
)

logger = logging.getLogger("ArchDiver.Explainer")


# ============================================================================
# Explanation Data Structures
# ============================================================================


@dataclass
class EdgeAttribution:
    edge_type: Tuple[str, str, str]
    src_id: int
    dst_id: int
    src_name: str
    dst_name: str
    importance: float


@dataclass
class ExplanationResult:
    target_node_id: int
    target_node_name: str
    target_class: int
    predicted_class: int
    base_probability: float
    fidelity_plus: float
    fidelity_minus: float
    sparsity: float
    top_edges: List[EdgeAttribution] = field(default_factory=list)
    class_feature_importances: Dict[str, float] = field(default_factory=dict)
    component_feature_importances: Dict[str, float] = field(default_factory=dict)
    subgraph_nodes: Dict[str, List[int]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 65,
            f"GNN Explanation for Component Node #{self.target_node_id}: {self.target_node_name}",
            "=" * 65,
            f"Target Class: {self.target_class} | Predicted Class: {self.predicted_class} | P(Smell) = {self.base_probability:.4f}",
            f"Metrics: Fidelity+ = {self.fidelity_plus:.4f} | Fidelity- = {self.fidelity_minus:.4f} | Sparsity = {self.sparsity:.4f}",
            "\n--- Top Explanatory Edges (Structural Influences) ---",
        ]
        for idx, e in enumerate(self.top_edges, 1):
            src_t, rel, dst_t = e.edge_type
            lines.append(
                f"  {idx:2d}. [{src_t}] {e.src_name} --({rel})--> [{dst_t}] {e.dst_name} (score: {e.importance:.4f})"
            )

        lines.append("\n--- Top Explanatory Class Metrics (Internal Anomaly) ---")
        for metric, score in sorted(
            self.class_feature_importances.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  - {metric:<24}: {score:.4f}")

        lines.append("\n--- Component Metrics ---")
        for metric, score in self.component_feature_importances.items():
            lines.append(f"  - {metric:<24}: {score:.4f}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["top_edges"] = [
            {**asdict(e), "edge_type": list(e.edge_type)} for e in self.top_edges
        ]
        return d


# ============================================================================
# ONNX GNNExplainer Algorithm
# ============================================================================


class ONNXGNNExplainer:
    """
    Model-agnostic Explainer operating directly on ONNX models via marginal edge and feature perturbation.
    """

    def __init__(
        self,
        onnx_model_path: str,
        num_hops: int = 2,
        top_k_edges: int = 10,
    ) -> None:
        self.runner = ONNXInferenceRunner(onnx_model_path)
        self.num_hops = num_hops
        self.top_k_edges = top_k_edges

    def extract_subgraph(
        self,
        graph: HeteroData,
        target_node_id: int,
    ) -> Tuple[Dict[str, Set[int]], Dict[Tuple[str, str, str], List[int]]]:
        """
        Extracts the k-hop computational ego-network around the target Component node.
        """
        comp_nodes: Set[int] = {target_node_id}
        class_nodes: Set[int] = set()
        edge_subsets: Dict[Tuple[str, str, str], List[int]] = {
            et: [] for et in EDGE_TYPES
        }

        e_cc = graph[EDGE_TYPES[0]].edge_index.cpu().numpy()
        e_cl = graph[EDGE_TYPES[1]].edge_index.cpu().numpy()
        e_cbc = graph[EDGE_TYPES[2]].edge_index.cpu().numpy()
        e_ll = graph[EDGE_TYPES[3]].edge_index.cpu().numpy()

        for _ in range(self.num_hops):
            # 1. Component contains Component
            if e_cc.shape[1] > 0:
                for idx in range(e_cc.shape[1]):
                    u, v = int(e_cc[0, idx]), int(e_cc[1, idx])
                    if u in comp_nodes or v in comp_nodes:
                        comp_nodes.add(u)
                        comp_nodes.add(v)
                        if idx not in edge_subsets[EDGE_TYPES[0]]:
                            edge_subsets[EDGE_TYPES[0]].append(idx)

            # 2. Component contains Class
            if e_cl.shape[1] > 0:
                for idx in range(e_cl.shape[1]):
                    u, v = int(e_cl[0, idx]), int(e_cl[1, idx])
                    if u in comp_nodes:
                        class_nodes.add(v)
                        if idx not in edge_subsets[EDGE_TYPES[1]]:
                            edge_subsets[EDGE_TYPES[1]].append(idx)

            # 3. Class contained by Component
            if e_cbc.shape[1] > 0:
                for idx in range(e_cbc.shape[1]):
                    u, v = int(e_cbc[0, idx]), int(e_cbc[1, idx])
                    if v in comp_nodes or u in class_nodes:
                        class_nodes.add(u)
                        comp_nodes.add(v)
                        if idx not in edge_subsets[EDGE_TYPES[2]]:
                            edge_subsets[EDGE_TYPES[2]].append(idx)

            # 4. Class imports Class
            if e_ll.shape[1] > 0:
                for idx in range(e_ll.shape[1]):
                    u, v = int(e_ll[0, idx]), int(e_ll[1, idx])
                    if u in class_nodes or v in class_nodes:
                        class_nodes.add(u)
                        class_nodes.add(v)
                        if idx not in edge_subsets[EDGE_TYPES[3]]:
                            edge_subsets[EDGE_TYPES[3]].append(idx)

        return {"Component": comp_nodes, "Class": class_nodes}, edge_subsets

    def explain(
        self,
        graph: HeteroData,
        target_node_id: int,
        target_class: int = 1,
        class_mapping: Optional[Dict[int, str]] = None,
        component_mapping: Optional[Dict[int, str]] = None,
    ) -> ExplanationResult:
        c_map = class_mapping or {}
        comp_map = component_mapping or {}

        # 1. Base inference
        _, base_probs = self.runner.predict_graph(graph)
        p0 = float(base_probs[target_node_id, target_class])
        predicted_cls = int(np.argmax(base_probs[target_node_id]))

        # 2. Extract computation neighborhood
        sub_nodes, sub_edges = self.extract_subgraph(graph, target_node_id)

        # 3. Edge Attribution
        x_c = graph["Class"].x.cpu().numpy()
        x_comp = graph["Component"].x.cpu().numpy()
        edges_base = {et: graph[et].edge_index.cpu().numpy() for et in EDGE_TYPES}

        ranked_edges: List[EdgeAttribution] = []
        for et in EDGE_TYPES:
            n_edges = edges_base[et].shape[1]
            active_indices = sub_edges[et]

            for idx in active_indices:
                mask = np.ones(n_edges, dtype=bool)
                mask[idx] = False
                e_ablated = dict(edges_base)
                e_ablated[et] = edges_base[et][:, mask]

                _, probs_ablated = self.runner.predict_raw(
                    x_c,
                    x_comp,
                    e_ablated[EDGE_TYPES[0]],
                    e_ablated[EDGE_TYPES[1]],
                    e_ablated[EDGE_TYPES[2]],
                    e_ablated[EDGE_TYPES[3]],
                )
                p_ablated = float(probs_ablated[target_node_id, target_class])
                drop = max(0.0, p0 - p_ablated)

                src_id = int(edges_base[et][0, idx])
                dst_id = int(edges_base[et][1, idx])
                src_t, rel, dst_t = et

                src_name = (
                    c_map.get(src_id, f"Class_{src_id}")
                    if src_t == "Class"
                    else comp_map.get(src_id, f"Comp_{src_id}")
                )
                dst_name = (
                    c_map.get(dst_id, f"Class_{dst_id}")
                    if dst_t == "Class"
                    else comp_map.get(dst_id, f"Comp_{dst_id}")
                )

                ranked_edges.append(
                    EdgeAttribution(
                        edge_type=et,
                        src_id=src_id,
                        dst_id=dst_id,
                        src_name=src_name,
                        dst_name=dst_name,
                        importance=drop,
                    )
                )

        ranked_edges.sort(key=lambda x: x.importance, reverse=True)
        top_edges = ranked_edges[: self.top_k_edges]

        # 4. Class Feature Attribution
        class_feat_importances: Dict[str, float] = {}
        rel_class_nodes = list(sub_nodes["Class"])
        if x_c.shape[1] > 0 and len(rel_class_nodes) > 0:
            for f_idx in range(x_c.shape[1]):
                f_name = (
                    CLASS_METRIC_NAMES[f_idx]
                    if f_idx < len(CLASS_METRIC_NAMES)
                    else f"class_feat_{f_idx}"
                )
                x_c_ablated = x_c.copy()
                x_c_ablated[rel_class_nodes, f_idx] = 0.0

                _, p_ablated = self.runner.predict_raw(
                    x_c_ablated,
                    x_comp,
                    edges_base[EDGE_TYPES[0]],
                    edges_base[EDGE_TYPES[1]],
                    edges_base[EDGE_TYPES[2]],
                    edges_base[EDGE_TYPES[3]],
                )
                class_feat_importances[f_name] = max(
                    0.0, p0 - float(p_ablated[target_node_id, target_class])
                )

        # 5. Component Feature Attribution
        comp_feat_importances: Dict[str, float] = {}
        rel_comp_nodes = list(sub_nodes["Component"])
        if x_comp.shape[1] > 0:
            for f_idx in range(x_comp.shape[1]):
                f_name = (
                    COMPONENT_METRIC_NAMES[f_idx]
                    if f_idx < len(COMPONENT_METRIC_NAMES)
                    else f"comp_feat_{f_idx}"
                )
                x_comp_ablated = x_comp.copy()
                x_comp_ablated[rel_comp_nodes, f_idx] = 0.0

                _, p_ablated = self.runner.predict_raw(
                    x_c,
                    x_comp_ablated,
                    edges_base[EDGE_TYPES[0]],
                    edges_base[EDGE_TYPES[1]],
                    edges_base[EDGE_TYPES[2]],
                    edges_base[EDGE_TYPES[3]],
                )
                comp_feat_importances[f_name] = max(
                    0.0, p0 - float(p_ablated[target_node_id, target_class])
                )

        # 6. Fidelity+ (Comprehensiveness) and Fidelity- (Sufficiency)
        top_edge_keys = set((e.edge_type, e.src_id, e.dst_id) for e in top_edges)
        e_without: Dict[Tuple[str, str, str], np.ndarray] = {}
        e_only: Dict[Tuple[str, str, str], np.ndarray] = {}

        for et in EDGE_TYPES:
            ed = edges_base[et]
            keep_mask = np.zeros(ed.shape[1], dtype=bool)
            drop_mask = np.ones(ed.shape[1], dtype=bool)

            for i in range(ed.shape[1]):
                u, v = int(ed[0, i]), int(ed[1, i])
                if (et, u, v) in top_edge_keys:
                    keep_mask[i] = True
                    drop_mask[i] = False

            e_only[et] = ed[:, keep_mask]
            e_without[et] = ed[:, drop_mask]

        _, p_without = self.runner.predict_raw(
            x_c,
            x_comp,
            e_without[EDGE_TYPES[0]],
            e_without[EDGE_TYPES[1]],
            e_without[EDGE_TYPES[2]],
            e_without[EDGE_TYPES[3]],
        )
        _, p_only = self.runner.predict_raw(
            x_c,
            x_comp,
            e_only[EDGE_TYPES[0]],
            e_only[EDGE_TYPES[1]],
            e_only[EDGE_TYPES[2]],
            e_only[EDGE_TYPES[3]],
        )

        fidelity_plus = max(0.0, p0 - float(p_without[target_node_id, target_class]))
        fidelity_minus = float(p0 - float(p_only[target_node_id, target_class]))

        total_sub_edges = sum(len(s) for s in sub_edges.values())
        sparsity = 1.0 - (len(top_edges) / max(1, total_sub_edges))

        target_name = comp_map.get(target_node_id, f"Component_{target_node_id}")

        return ExplanationResult(
            target_node_id=target_node_id,
            target_node_name=target_name,
            target_class=target_class,
            predicted_class=predicted_cls,
            base_probability=p0,
            fidelity_plus=fidelity_plus,
            fidelity_minus=fidelity_minus,
            sparsity=max(0.0, sparsity),
            top_edges=top_edges,
            class_feature_importances=class_feat_importances,
            component_feature_importances=comp_feat_importances,
            subgraph_nodes={k: sorted(list(v)) for k, v in sub_nodes.items()},
        )


# ============================================================================
# Visualizer
# ============================================================================


class ExplanationVisualizer:
    """
    Renders high-quality two-panel explanation figures:
    Left: NetworkX computational subgraph with highlighted explanatory edges.
    Right: Horizontal bar chart of metric attributions.
    """

    @staticmethod
    def render(
        graph: HeteroData,
        result: ExplanationResult,
        class_mapping: Dict[int, str],
        component_mapping: Dict[int, str],
        save_path: str,
        show: bool = False,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig, (ax_graph, ax_bar) = plt.subplots(
            1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.3, 0.7]}
        )

        G = nx.DiGraph()
        comp_nodes = set(result.subgraph_nodes.get("Component", []))
        class_nodes = set(result.subgraph_nodes.get("Class", []))

        for c_id in comp_nodes:
            lbl = os.path.basename(component_mapping.get(c_id, f"Comp_{c_id}"))
            G.add_node(f"Component_{c_id}", label=lbl, type="Component")

        for cl_id in class_nodes:
            lbl = class_mapping.get(cl_id, f"Class_{cl_id}").split(".")[-1]
            G.add_node(f"Class_{cl_id}", label=lbl, type="Class")

        top_edge_scores = {}
        for e in result.top_edges:
            sp = "Class" if e.edge_type[0] == "Class" else "Component"
            dp = "Class" if e.edge_type[2] == "Class" else "Component"
            top_edge_scores[(f"{sp}_{e.src_id}", f"{dp}_{e.dst_id}")] = e.importance

        normal_edges, important_edges, weights = [], [], []

        for et in EDGE_TYPES:
            src_t, rel, dst_t = et
            e_idx = graph[et].edge_index.cpu().numpy()
            for i in range(e_idx.shape[1]):
                u, v = int(e_idx[0, i]), int(e_idx[1, i])
                sn = f"{src_t}_{u}"
                dn = f"{dst_t}_{v}"
                if sn in G and dn in G:
                    G.add_edge(sn, dn)
                    if (sn, dn) in top_edge_scores and top_edge_scores[(sn, dn)] > 1e-4:
                        important_edges.append((sn, dn))
                        weights.append(top_edge_scores[(sn, dn)])
                    else:
                        normal_edges.append((sn, dn))

        pos = nx.spring_layout(G, seed=42, k=1.4)
        target_name = f"Component_{result.target_node_id}"
        other_comps = [
            n
            for n in G.nodes()
            if G.nodes[n].get("type") == "Component" and n != target_name
        ]
        classes = [n for n in G.nodes() if G.nodes[n].get("type") == "Class"]

        if other_comps:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=other_comps,
                node_color="#ffcc66",
                node_size=1100,
                ax=ax_graph,
                label="Component",
            )
        if target_name in G:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[target_name],
                node_color="#ff3333",
                node_size=1600,
                ax=ax_graph,
                label="Target (Smelly) Component",
            )
        if classes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=classes,
                node_color="#66b3ff",
                node_size=700,
                ax=ax_graph,
                label="Class",
            )

        if normal_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=normal_edges,
                edge_color="#cccccc",
                width=1.0,
                alpha=0.6,
                ax=ax_graph,
            )
        if important_edges:
            w_norm = [max(1.8, min(6.0, w * 15.0 + 2.0)) for w in weights]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=important_edges,
                edge_color="#cc0000",
                width=w_norm,
                alpha=0.9,
                ax=ax_graph,
            )

        labels = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(
            G, pos, labels=labels, font_size=8, font_weight="bold", ax=ax_graph
        )

        ax_graph.set_title(
            f"Explanatory Subgraph (ONNX) - {os.path.basename(result.target_node_name)}\n"
            f"P(Smell) = {result.base_probability:.3f} | Fid+ = {result.fidelity_plus:.3f}, Sparsity = {result.sparsity:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax_graph.legend(loc="upper right", fontsize=9)
        ax_graph.axis("off")

        # Metric Importances Plot
        feats = dict(result.class_feature_importances)
        for k, v in result.component_feature_importances.items():
            feats[f"comp_{k}"] = v

        sorted_feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_feats]
        vals = [x[1] for x in sorted_feats]
        y_pos = np.arange(len(names))

        colors = ["#ff4d4d" if v > 0.005 else "#4da6ff" for v in vals]
        bars = ax_bar.barh(y_pos, vals, color=colors, edgecolor="black", alpha=0.85)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(names, fontsize=10)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Attribution Score (Drop in P(Smell))", fontsize=11)
        ax_bar.set_title("Node Metric Importance", fontsize=12, fontweight="bold")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            w = bar.get_width()
            ax_bar.text(
                w + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        logger.info(f"Saved explanation visualization: {save_path}")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="ONNX GNNExplainer PoC for Architectural Smells"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="output/models/arch_smell_model.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/models/arch_smell_model.pt",
        help="Path to checkpoint with scaler",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="raw_data/labeled graphs/graph_1.pt",
        help="Path to graph .pt file",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic test graph"
    )
    parser.add_argument(
        "--node", type=int, default=None, help="Target Component node ID to explain"
    )
    parser.add_argument(
        "--hops", type=int, default=2, help="Number of hops for computation subgraph"
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Top explanatory edges to extract"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/explanations", help="Output directory"
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")

    args = parser.parse_args()

    # 1. Initialize ONNX Explainer
    explainer = ONNXGNNExplainer(
        onnx_model_path=args.model,
        num_hops=args.hops,
        top_k_edges=args.top_k,
    )

    # 2. Load Graph
    if args.synthetic:
        logger.info("Generating synthetic graph with injected smell...")
        graph, c_map, comp_map = SyntheticGraphBuilder.build_graph(
            inject_smell=True, seed=42
        )
        base_name = "synthetic_demo"
    else:
        logger.info(f"Loading target graph from: {args.graph}...")
        loaded = torch.load(args.graph, weights_only=False)
        graph = loaded["graph_data"]
        c_map = loaded.get("class_mapping", {})
        comp_map = loaded.get("component_mapping", {})
        base_name = os.path.basename(args.graph).replace(".pt", "")

    # 3. Apply Scaler if available
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, weights_only=False)
        if "scaler_state" in ckpt:
            scaler = NodeFeatureScaler()
            scaler.load_state_dict(ckpt["scaler_state"])
            graph = scaler.transform_single(graph)

    # 4. Select Target Node
    if args.node is not None:
        target_nodes = [args.node]
    else:
        if hasattr(graph["Component"], "y") and graph["Component"].y is not None:
            smelly_nodes = np.where(graph["Component"].y.cpu().numpy() == 1)[0].tolist()
            if smelly_nodes:
                target_nodes = smelly_nodes
            else:
                _, probs = explainer.runner.predict_graph(graph)
                target_nodes = [int(np.argmax(probs[:, 1]))]
        else:
            _, probs = explainer.runner.predict_graph(graph)
            target_nodes = [int(np.argmax(probs[:, 1]))]

    os.makedirs(args.output_dir, exist_ok=True)

    for node_id in target_nodes:
        logger.info(f"Running ONNX GNNExplainer on Component #{node_id}...")
        res = explainer.explain(
            graph=graph,
            target_node_id=node_id,
            target_class=1,
            class_mapping=c_map,
            component_mapping=comp_map,
        )

        print("\n" + res.summary() + "\n")

        # Save JSON
        json_file = os.path.join(
            args.output_dir, f"{base_name}_comp_{node_id}_explanation.json"
        )
        with open(json_file, "w") as f:
            json.dump(res.to_dict(), f, indent=2, default=custom_json_serializer)
        logger.info(f"Saved explanation JSON: {json_file}")

        # Save Plot
        png_file = os.path.join(
            args.output_dir, f"{base_name}_comp_{node_id}_explanation.png"
        )
        ExplanationVisualizer.render(
            graph, res, c_map, comp_map, png_file, show=args.show
        )


if __name__ == "__main__":
    main()
