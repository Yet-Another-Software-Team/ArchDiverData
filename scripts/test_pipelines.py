"""
scripts/test_pipelines.py

Automated unit and integration test suite validating:
- Feature scaling and dataset loading
- Synthetic graph generation with ground-truth smells
- GNN training forward pass & Focal loss
- ONNX model export with dynamic axes
- ONNX Runtime inference & numerical parity
- ONNX GNNExplainer execution, metrics, and visualization
- Full graph and ego-subgraph visualizer
"""

from __future__ import annotations
import os
import sys

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import unittest
import numpy as np
import torch
from torch_geometric.data import HeteroData

from scripts.common import (
    ArchSmellClassifier,
    FocalLoss,
    NodeFeatureScaler,
    SyntheticGraphBuilder,
    export_model_to_onnx,
    ONNXInferenceRunner,
    EDGE_TYPES,
)
from scripts.train_pipeline import TrainingConfig, run_training
from scripts.explain_onnx import ONNXGNNExplainer, ExplanationVisualizer
from scripts.visualize import ArchitectureVisualizer


class TestArchDiverPipelines(unittest.TestCase):

    def setUp(self):
        self.test_dir = "/tmp/archdiver_test_output"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_synthetic_graph_builder(self):
        graph, c_map, comp_map = SyntheticGraphBuilder.build_graph(
            num_components=5,
            classes_per_component=3,
            inject_smell=True,
            seed=42,
        )
        self.assertEqual(graph["Class"].x.shape[0], 15)
        self.assertEqual(graph["Class"].x.shape[1], 6)
        self.assertEqual(graph["Component"].x.shape[0], 5)
        self.assertEqual(graph["Component"].x.shape[1], 1)
        self.assertEqual(graph["Component"].y[1].item(), 1)  # Smelly node

        for et in EDGE_TYPES:
            self.assertIn(et, graph.edge_types)

    def test_feature_scaler(self):
        graphs = [
            SyntheticGraphBuilder.build_graph(num_components=4, classes_per_component=2, seed=i)[0]
            for i in range(3)
        ]
        scaler = NodeFeatureScaler()
        scaler.fit(graphs)

        state = scaler.state_dict()
        self.assertIn("Class", state)
        self.assertIn("Component", state)

        scaled = scaler.transform_all(graphs)
        self.assertEqual(len(scaled), 3)
        self.assertEqual(scaled[0]["Class"].x.shape, graphs[0]["Class"].x.shape)

    def test_model_forward_and_loss(self):
        graph, _, _ = SyntheticGraphBuilder.build_graph(num_components=4, classes_per_component=2, seed=42)
        model = ArchSmellClassifier(graph.metadata(), hidden_channels=16, num_classes=2, dropout_rate=0.0)
        model.eval()

        with torch.no_grad():
            out = model(graph.x_dict, graph.edge_index_dict)
        self.assertEqual(out.shape, (4, 2))

        criterion = FocalLoss(gamma=2.0)
        loss = criterion(out, graph["Component"].y)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_onnx_export_and_inference_parity(self):
        graph, _, _ = SyntheticGraphBuilder.build_graph(num_components=4, classes_per_component=2, seed=42)
        model = ArchSmellClassifier(graph.metadata(), hidden_channels=16, num_classes=2, dropout_rate=0.0)
        model.eval()

        with torch.no_grad():
            pt_out = model(graph.x_dict, graph.edge_index_dict)

        onnx_file = os.path.join(self.test_dir, "test_model.onnx")
        export_model_to_onnx(model, graph, onnx_file)
        self.assertTrue(os.path.exists(onnx_file))

        runner = ONNXInferenceRunner(onnx_file)
        logits, probs = runner.predict_graph(graph)
        self.assertEqual(logits.shape, (4, 2))
        np.testing.assert_allclose(pt_out.detach().cpu().numpy(), logits, atol=1e-4)

    def test_onnx_gnn_explainer_end_to_end(self):
        graph, c_map, comp_map = SyntheticGraphBuilder.build_graph(num_components=5, classes_per_component=3, inject_smell=True, seed=99)
        model = ArchSmellClassifier(graph.metadata(), hidden_channels=16, num_classes=2, dropout_rate=0.0)
        model.eval()

        onnx_file = os.path.join(self.test_dir, "explainer_model.onnx")
        export_model_to_onnx(model, graph, onnx_file)

        explainer = ONNXGNNExplainer(onnx_model_path=onnx_file, num_hops=2, top_k_edges=5)
        res = explainer.explain(
            graph=graph,
            target_node_id=1,
            target_class=1,
            class_mapping=c_map,
            component_mapping=comp_map,
        )

        self.assertEqual(res.target_node_id, 1)
        self.assertGreaterEqual(res.base_probability, 0.0)
        self.assertLessEqual(res.base_probability, 1.0)
        self.assertGreaterEqual(res.sparsity, 0.0)
        self.assertIn("class_lcom", res.class_feature_importances)

        # Visualizer check
        png_file = os.path.join(self.test_dir, "test_explanation.png")
        ExplanationVisualizer.render(graph, res, c_map, comp_map, png_file, show=False)
        self.assertTrue(os.path.exists(png_file))
        self.assertGreater(os.path.getsize(png_file), 0)

    def test_architecture_visualizer(self):
        graph, c_map, comp_map = SyntheticGraphBuilder.build_graph(num_components=4, classes_per_component=2, seed=42)
        out_full = os.path.join(self.test_dir, "test_vis_full.png")
        out_sub = os.path.join(self.test_dir, "test_vis_sub.png")

        # Full graph rendering
        ArchitectureVisualizer.plot_graph(graph, c_map, comp_map, save_path=out_full, show=False)
        self.assertTrue(os.path.exists(out_full))
        self.assertGreater(os.path.getsize(out_full), 0)

        # Ego subgraph rendering
        ArchitectureVisualizer.plot_graph(graph, c_map, comp_map, target_component_id=1, k_hops=1, save_path=out_sub, show=False)
        self.assertTrue(os.path.exists(out_sub))
        self.assertGreater(os.path.getsize(out_sub), 0)


if __name__ == "__main__":
    unittest.main()
