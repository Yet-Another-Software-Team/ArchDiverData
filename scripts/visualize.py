"""
scripts/visualize.py

Comprehensive visualization tool for ArchDiver software architecture graphs and GNN explanations.

Capabilities:
1. Full Graph View: Renders complete heterogeneous architecture graphs with color-coded nodes and edge relationships.
2. Ego-Network View: Renders focused k-hop neighborhood subgraphs around specified Component nodes.
3. GNN Explanation View: Renders dual-panel explanation reports (highlighted explanatory edges + metric attribution charts).
4. Metric Profiling: Renders metric distribution bar charts for Classes belonging to a target Component.
"""

from __future__ import annotations
import os
import sys

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.data import HeteroData

from scripts.common import (
    EDGE_TYPES,
    CLASS_METRIC_NAMES,
    COMPONENT_METRIC_NAMES,
    SyntheticGraphBuilder,
)
from scripts.explain_onnx import ONNXGNNExplainer, ExplanationResult, ExplanationVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ArchDiver.Visualizer")

# Color palette for distinct visualization
COLOR_CLEAN_COMP = "#ffbf47"      # Orange/Amber
COLOR_SMELLY_COMP = "#ff3333"     # Red
COLOR_TARGET_COMP = "#b30000"     # Deep Red
COLOR_CLASS = "#4da6ff"           # Blue
EDGE_COLORS = {
    ("Component", "contains", "Component"): "#e68a00",  # Dark Amber
    ("Component", "contains", "Class"): "#808080",      # Gray
    ("Class", "contained_by", "Component"): "#999999",  # Light Gray
    ("Class", "imports", "Class"): "#2b6cb0",           # Steel Blue
}


class ArchitectureVisualizer:
    """
    Visualizer class providing various graph rendering methods.
    """

    @staticmethod
    def build_networkx_graph(
        graph: HeteroData,
        class_mapping: Optional[Dict[int, str]] = None,
        component_mapping: Optional[Dict[int, str]] = None,
        target_component_id: Optional[int] = None,
        k_hops: Optional[int] = None,
    ) -> Tuple[nx.DiGraph, Dict[str, str], Dict[str, str]]:
        """
        Converts HeteroData into a NetworkX DiGraph, optionally extracting a k-hop subgraph.
        """
        c_map = class_mapping or {}
        comp_map = component_mapping or {}
        G = nx.DiGraph()

        has_labels = hasattr(graph["Component"], "y") and graph["Component"].y is not None
        y_labels = graph["Component"].y.cpu().numpy() if has_labels else None

        num_comps = graph["Component"].x.shape[0] if hasattr(graph["Component"], "x") else len(comp_map)
        num_classes = graph["Class"].x.shape[0] if hasattr(graph["Class"], "x") else len(c_map)

        # 1. Filter nodes if k_hops is requested
        active_comps: Set[int] = set(range(num_comps))
        active_classes: Set[int] = set(range(num_classes))

        if target_component_id is not None and k_hops is not None:
            active_comps = {target_component_id}
            active_classes = set()
            for _ in range(k_hops):
                for et in EDGE_TYPES:
                    src_t, _, dst_t = et
                    if et in graph.edge_types:
                        e_idx = graph[et].edge_index.cpu().numpy()
                        for i in range(e_idx.shape[1]):
                            u, v = int(e_idx[0, i]), int(e_idx[1, i])
                            if src_t == "Component" and u in active_comps:
                                if dst_t == "Component":
                                    active_comps.add(v)
                                else:
                                    active_classes.add(v)
                            if dst_t == "Component" and v in active_comps:
                                if src_t == "Component":
                                    active_comps.add(u)
                                else:
                                    active_classes.add(u)
                            if src_t == "Class" and dst_t == "Class" and (u in active_classes or v in active_classes):
                                active_classes.add(u)
                                active_classes.add(v)

        # 2. Add Component Nodes
        for c_id in active_comps:
            node_id = f"Component_{c_id}"
            raw_name = comp_map.get(c_id, f"Component_{c_id}")
            label = os.path.basename(raw_name) or raw_name
            is_smelly = bool(y_labels[c_id] == 1) if y_labels is not None and c_id < len(y_labels) else False
            G.add_node(
                node_id,
                label=label,
                node_type="Component",
                is_smelly=is_smelly,
                raw_id=c_id,
                full_name=raw_name,
            )

        # 3. Add Class Nodes
        for cl_id in active_classes:
            node_id = f"Class_{cl_id}"
            raw_name = c_map.get(cl_id, f"Class_{cl_id}")
            label = raw_name.split(".")[-1]
            G.add_node(
                node_id,
                label=label,
                node_type="Class",
                is_smelly=False,
                raw_id=cl_id,
                full_name=raw_name,
            )

        # 4. Add Edges
        for et in EDGE_TYPES:
            src_t, rel, dst_t = et
            if et in graph.edge_types:
                e_idx = graph[et].edge_index.cpu().numpy()
                for i in range(e_idx.shape[1]):
                    u, v = int(e_idx[0, i]), int(e_idx[1, i])
                    sn = f"{src_t}_{u}"
                    dn = f"{dst_t}_{v}"
                    if sn in G and dn in G:
                        G.add_edge(sn, dn, edge_type=et, rel=rel)

        return G, c_map, comp_map

    @classmethod
    def plot_graph(
        cls,
        graph: HeteroData,
        class_mapping: Optional[Dict[int, str]] = None,
        component_mapping: Optional[Dict[int, str]] = None,
        target_component_id: Optional[int] = None,
        k_hops: Optional[int] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Renders a full or ego-subgraph layout of the software architecture.
        """
        G, c_map, comp_map = cls.build_networkx_graph(
            graph,
            class_mapping=class_mapping,
            component_mapping=component_mapping,
            target_component_id=target_component_id,
            k_hops=k_hops,
        )

        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, seed=42, k=1.6 / max(1.0, np.sqrt(G.number_of_nodes()) * 0.3))

        # Node categories
        clean_comps = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Component" and not d.get("is_smelly")]
        smelly_comps = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Component" and d.get("is_smelly")]
        classes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Class"]

        # Draw Nodes
        if clean_comps:
            nx.draw_networkx_nodes(G, pos, nodelist=clean_comps, node_color=COLOR_CLEAN_COMP, node_size=1200, edgecolors="#666666", label="Clean Component")
        if smelly_comps:
            nx.draw_networkx_nodes(G, pos, nodelist=smelly_comps, node_color=COLOR_SMELLY_COMP, node_size=1500, edgecolors="#800000", linewidths=2.5, label="Smelly Component")
        if classes:
            nx.draw_networkx_nodes(G, pos, nodelist=classes, node_color=COLOR_CLASS, node_size=650, edgecolors="#333333", label="Class")

        # Highlight target if requested
        if target_component_id is not None:
            tn = f"Component_{target_component_id}"
            if tn in G:
                nx.draw_networkx_nodes(G, pos, nodelist=[tn], node_color=COLOR_TARGET_COMP, node_size=1800, edgecolors="yellow", linewidths=3, label="Target Component")

        # Draw Edges by Type
        for et, color in EDGE_COLORS.items():
            edgelist = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == et]
            if edgelist:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edgelist,
                    edge_color=color,
                    width=1.8 if "imports" in et[1] else 1.2,
                    alpha=0.75,
                    arrowstyle="-|>",
                    arrowsize=18,
                    connectionstyle="arc3,rad=0.08",
                )

        # Draw Labels
        labels = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")

        plot_title = title or f"Software Architecture Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)"
        if target_component_id is not None:
            plot_title += f" [Ego Subgraph around Component #{target_component_id}]"

        plt.title(plot_title, fontsize=14, fontweight="bold", pad=20)
        plt.axis("off")

        # Legend patches
        legend_patches = [
            mpatches.Patch(color=COLOR_CLEAN_COMP, label="Clean Component"),
            mpatches.Patch(color=COLOR_SMELLY_COMP, label="Smelly Component"),
            mpatches.Patch(color=COLOR_CLASS, label="Class"),
            mpatches.Patch(color=EDGE_COLORS[EDGE_TYPES[3]], label="Class Imports Class"),
            mpatches.Patch(color=EDGE_COLORS[EDGE_TYPES[0]], label="Component Contains Component"),
        ]
        plt.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.9)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Graph visualization saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    @classmethod
    def plot_explanation_from_file(
        cls,
        json_explanation_path: str,
        graph_path: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Visualizes a saved explanation JSON report alongside its graph.
        """
        if not os.path.exists(json_explanation_path):
            raise FileNotFoundError(f"Explanation file not found: {json_explanation_path}")

        with open(json_explanation_path, "r") as f:
            data = json.load(f)

        target_node_id = data["target_node_id"]

        # Load or generate graph
        if graph_path and os.path.exists(graph_path):
            loaded = torch.load(graph_path, weights_only=False)
            graph = loaded["graph_data"]
            c_map = loaded.get("class_mapping", {})
            comp_map = loaded.get("component_mapping", {})
        else:
            logger.info("No valid graph .pt provided; constructing graph view from explanation payload...")
            graph = HeteroData()
            c_map = {}
            comp_map = {target_node_id: data["target_node_name"]}

        out_img = save_path or json_explanation_path.replace(".json", ".png")

        # Convert back to ExplanationResult
        from scripts.explain_onnx import EdgeAttribution
        top_edges = [
            EdgeAttribution(
                edge_type=tuple(e["edge_type"]) if isinstance(e["edge_type"], list) else tuple(eval(e["edge_type"])),
                src_id=int(e["src_id"]),
                dst_id=int(e["dst_id"]),
                src_name=str(e["src_name"]),
                dst_name=str(e["dst_name"]),
                importance=float(e["importance"]),
            )
            for e in data.get("top_edges", [])
        ]

        result = ExplanationResult(
            target_node_id=int(data["target_node_id"]),
            target_node_name=str(data["target_node_name"]),
            target_class=int(data.get("target_class", 1)),
            predicted_class=int(data.get("predicted_class", 1)),
            base_probability=float(data.get("base_probability", 1.0)),
            fidelity_plus=float(data.get("fidelity_plus", 0.0)),
            fidelity_minus=float(data.get("fidelity_minus", 0.0)),
            sparsity=float(data.get("sparsity", 0.0)),
            top_edges=top_edges,
            class_feature_importances=data.get("class_feature_importances", {}),
            component_feature_importances=data.get("component_feature_importances", {}),
            subgraph_nodes=data.get("subgraph_nodes", {}),
        )

        ExplanationVisualizer.render(graph, result, c_map, comp_map, out_img, show=show)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ArchDiver Graph and Explanation Visualization Tool")
    parser.add_argument("-g", "--graph", type=str, default=None, help="Path to input graph .pt file")
    parser.add_argument("-e", "--explanation", type=str, default=None, help="Path to saved explanation .json file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic graph for demonstration")
    parser.add_argument("-n", "--node", type=int, default=None, help="Focus on specific Component node ID")
    parser.add_argument("--hops", type=int, default=None, help="Ego-network hop radius (e.g., 1 or 2)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output image file path (.png)")
    parser.add_argument("--show", action="store_true", help="Display interactive window")

    args = parser.parse_args()

    # Mode 1: Render from Explanation JSON
    if args.explanation:
        logger.info(f"Visualizing explanation from: {args.explanation}")
        ArchitectureVisualizer.plot_explanation_from_file(
            json_explanation_path=args.explanation,
            graph_path=args.graph,
            save_path=args.output,
            show=args.show,
        )
        return

    # Mode 2: Render Graph (.pt or synthetic)
    if args.synthetic:
        logger.info("Generating synthetic graph...")
        graph, c_map, comp_map = SyntheticGraphBuilder.build_graph(num_components=6, classes_per_component=3, inject_smell=True, seed=42)
        base_name = "synthetic_graph"
    elif args.graph:
        logger.info(f"Loading graph: {args.graph}")
        loaded = torch.load(args.graph, weights_only=False)
        graph = loaded["graph_data"]
        c_map = loaded.get("class_mapping", {})
        comp_map = loaded.get("component_mapping", {})
        base_name = os.path.basename(args.graph).replace(".pt", "")
    else:
        logger.error("Please provide --graph, --explanation, or --synthetic.")
        parser.print_help()
        return

    out_file = args.output or f"output/explanations/{base_name}_view.png"
    ArchitectureVisualizer.plot_graph(
        graph=graph,
        class_mapping=c_map,
        component_mapping=comp_map,
        target_component_id=args.node,
        k_hops=args.hops,
        save_path=out_file,
        show=args.show,
    )


if __name__ == "__main__":
    main()
