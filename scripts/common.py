"""
scripts/common.py

Shared utilities, model definitions, loss functions, ONNX wrappers,
and dataset processors for the ArchDiver GNN training and explanation pipelines.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ArchDiver")

# Heterogeneous canonical edge types
EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("Component", "contains", "Component"),
    ("Component", "contains", "Class"),
    ("Class", "contained_by", "Component"),
    ("Class", "imports", "Class"),
]

# Canonical feature names
CLASS_METRIC_NAMES: List[str] = [
    "class_lcom",
    "num_attributes",
    "num_methods_declared",
    "num_methods_actual",
    "avg_method_lcom",
    "avg_params",
]

COMPONENT_METRIC_NAMES: List[str] = [
    "num_files_and_folders",
]


# ============================================================================
# Model & Loss Definitions
# ============================================================================


class ArchSmellClassifier(nn.Module):
    """
    Heterogeneous Graph Neural Network for classifying architectural smells
    in software Component nodes using heterogeneous GraphSAGE.
    """

    def __init__(
        self,
        metadata: Tuple[Sequence[str], Sequence[Tuple[str, str, str]]],
        hidden_channels: int = 64,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        base_sage = GraphSAGE(
            in_channels=-1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout_rate,
        )
        self.sage = to_hetero(base_sage, metadata, aggr="sum")
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass producing logits for 'Component' nodes.
        """
        h_dict = self.sage(x_dict, edge_index_dict)
        h = h_dict["Component"]
        h = F.relu(h)
        h = self.dropout(h)
        return self.classifier(h)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance between clean and smelly components.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class OnnxGraphWrapper(nn.Module):
    """
    Positional adapter wrapper required to trace PyG HeteroData inputs for ONNX export.
    """

    def __init__(
        self, model: nn.Module, edge_types: List[Tuple[str, str, str]]
    ) -> None:
        super().__init__()
        self.model = model
        self.edge_types = edge_types

    def forward(
        self,
        x_class: torch.Tensor,
        x_comp: torch.Tensor,
        e_cc: torch.Tensor,
        e_cl: torch.Tensor,
        e_cbc: torch.Tensor,
        e_ll: torch.Tensor,
    ) -> torch.Tensor:
        x_dict = {"Class": x_class, "Component": x_comp}
        edge_index_dict = {
            self.edge_types[0]: e_cc,
            self.edge_types[1]: e_cl,
            self.edge_types[2]: e_cbc,
            self.edge_types[3]: e_ll,
        }
        return self.model(x_dict, edge_index_dict)


# ============================================================================
# Feature Scaling & Dataset Processing
# ============================================================================


class NodeFeatureScaler:
    """
    Manages standardization of Class and Component continuous metrics across graphs.
    """

    def __init__(self) -> None:
        self.scalers: Dict[str, StandardScaler] = {}

    def fit(
        self,
        graphs: Sequence[HeteroData],
        node_types: Sequence[str] = ("Class", "Component"),
    ) -> "NodeFeatureScaler":
        for n_type in node_types:
            feats = [
                g[n_type].x
                for g in graphs
                if hasattr(g[n_type], "x") and g[n_type].x.numel() > 0
            ]
            if feats:
                concat_feats = torch.cat(feats, dim=0).cpu().numpy()
                scaler = StandardScaler()
                scaler.fit(concat_feats)
                self.scalers[n_type] = scaler
        return self

    def transform_single(self, graph: HeteroData) -> HeteroData:
        g_copy = copy.deepcopy(graph)
        for n_type, scaler in self.scalers.items():
            if hasattr(g_copy[n_type], "x") and g_copy[n_type].x.numel() > 0:
                feat_np = g_copy[n_type].x.cpu().numpy()
                norm_feat = scaler.transform(feat_np)
                g_copy[n_type].x = torch.from_numpy(norm_feat).float()
        return g_copy

    def transform_all(self, graphs: Sequence[HeteroData]) -> List[HeteroData]:
        return [self.transform_single(g) for g in graphs]

    def state_dict(self) -> Dict[str, Any]:
        return {
            k: {
                "mean": s.mean_.tolist(),
                "scale": s.scale_.tolist(),
                "var": s.var_.tolist(),
            }
            for k, s in self.scalers.items()
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "NodeFeatureScaler":
        self.scalers = {}
        for k, v in state.items():
            s = StandardScaler()
            s.mean_ = np.array(v["mean"], dtype=np.float64)
            s.scale_ = np.array(v["scale"], dtype=np.float64)
            s.var_ = np.array(v["var"], dtype=np.float64)
            self.scalers[k] = s
        return self


def load_labeled_graphs(
    data_dir: str,
) -> List[Tuple[HeteroData, Dict[int, str], Dict[int, str], str]]:
    """
    Loads labeled code graphs from the specified directory.
    Returns: List of tuples (HeteroData, class_mapping, component_mapping, filename).
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Target data directory does not exist: {data_dir}")

    results = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            loaded = torch.load(path, weights_only=False)
            if isinstance(loaded, dict) and "graph_data" in loaded:
                g = loaded["graph_data"]
                c_map = loaded.get("class_mapping", {})
                comp_map = loaded.get("component_mapping", {})
            elif isinstance(loaded, HeteroData):
                g, c_map, comp_map = loaded, {}, {}
            else:
                continue

            if hasattr(g["Component"], "y") and g["Component"].y is not None:
                results.append((g, c_map, comp_map, fname))
        except Exception as ex:
            logger.warning(f"Skipping corrupted or incompatible file '{fname}': {ex}")

    logger.info(f"Loaded {len(results)} valid labeled graphs from '{data_dir}'")
    return results


# ============================================================================
# Synthetic Dataset Generator for Testing & Simulation
# ============================================================================


class SyntheticGraphBuilder:
    """
    Builds synthetic software architecture graphs with controlled smell patterns (e.g. cyclic dependencies, god components).
    """

    @staticmethod
    def build_graph(
        num_components: int = 5,
        classes_per_component: int = 3,
        inject_smell: bool = True,
        seed: int = 42,
    ) -> Tuple[HeteroData, Dict[int, str], Dict[int, str]]:
        np.random.seed(seed)
        torch.manual_seed(seed)

        data = HeteroData()
        c_map: Dict[int, str] = {}
        comp_map: Dict[int, str] = {}

        total_classes = num_components * classes_per_component

        # 1. Component node features
        comp_features = []
        for c_id in range(num_components):
            comp_map[c_id] = f"src/com/synthetic/app/module_{c_id}"
            comp_features.append([float(classes_per_component + 1)])
        data["Component"].x = torch.tensor(comp_features, dtype=torch.float)

        # 2. Containment and Class Features
        class_features = []
        edge_cc: Tuple[List[int], List[int]] = ([], [])
        edge_cl: Tuple[List[int], List[int]] = ([], [])
        edge_cbc: Tuple[List[int], List[int]] = ([], [])
        edge_ll: Tuple[List[int], List[int]] = ([], [])

        for c_id in range(1, num_components):
            edge_cc[0].append((c_id - 1) // 2)
            edge_cc[1].append(c_id)

        class_id = 0
        y_labels = torch.zeros(num_components, dtype=torch.long)

        for c_id in range(num_components):
            for k in range(classes_per_component):
                c_map[class_id] = f"Module{c_id}Service{k}"
                # Injected smell node has elevated cohesion & complexity metrics
                if inject_smell and c_id == 1:
                    feat = [0.85, 18.0, 30.0, 30.0, 0.80, 4.5]
                else:
                    feat = [0.10, 3.0, 5.0, 5.0, 0.12, 1.2]
                class_features.append(feat)

                edge_cl[0].append(c_id)
                edge_cl[1].append(class_id)
                edge_cbc[0].append(class_id)
                edge_cbc[1].append(c_id)
                class_id += 1

        data["Class"].x = torch.tensor(class_features, dtype=torch.float)

        # Injected cyclic dependencies across components
        if inject_smell:
            y_labels[1] = 1
            for sc in [3, 4, 5]:
                for ec in [0, 1, 6]:
                    if sc < total_classes and ec < total_classes:
                        edge_ll[0].extend([sc, ec])
                        edge_ll[1].extend([ec, sc])

        data["Component"].y = y_labels

        def _to_edge_tensor(tup: Tuple[List[int], List[int]]) -> torch.Tensor:
            if tup[0]:
                return torch.tensor(tup, dtype=torch.long)
            return torch.empty((2, 0), dtype=torch.long)

        data[EDGE_TYPES[0]].edge_index = _to_edge_tensor(edge_cc)
        data[EDGE_TYPES[1]].edge_index = _to_edge_tensor(edge_cl)
        data[EDGE_TYPES[2]].edge_index = _to_edge_tensor(edge_cbc)
        data[EDGE_TYPES[3]].edge_index = _to_edge_tensor(edge_ll)

        return data, c_map, comp_map

    @classmethod
    def build_dataset(
        cls,
        num_graphs: int = 10,
        seed: int = 42,
    ) -> List[Tuple[HeteroData, Dict[int, str], Dict[int, str], str]]:
        dataset = []
        for i in range(num_graphs):
            g, c_map, comp_map = cls.build_graph(
                num_components=int(np.random.randint(4, 8)),
                classes_per_component=int(np.random.randint(2, 5)),
                inject_smell=(i % 2 == 1),
                seed=seed + i,
            )
            dataset.append((g, c_map, comp_map, f"synthetic_graph_{i}.pt"))
        return dataset


# ============================================================================
# ONNX Export & Execution Runner
# ============================================================================


def export_model_to_onnx(
    model: ArchSmellClassifier,
    sample_data: HeteroData,
    onnx_path: str,
    device: torch.device = torch.device("cpu"),
    opset_version: int = 18,
) -> str:
    """
    Exports a trained ArchSmellClassifier to an ONNX model file with dynamic axes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    model = model.to(device)

    # Initialize lazy parameters if uninitialized
    with torch.no_grad():
        x_dev = {k: v.to(device) for k, v in sample_data.x_dict.items()}
        e_dev = {k: v.to(device) for k, v in sample_data.edge_index_dict.items()}
        _ = model(x_dev, e_dev)

    model.eval()
    wrapper = OnnxGraphWrapper(model, EDGE_TYPES).to(device)
    wrapper.eval()

    dummy_inputs = (
        sample_data["Class"].x.to(device),
        sample_data["Component"].x.to(device),
        sample_data[EDGE_TYPES[0]].edge_index.to(device),
        sample_data[EDGE_TYPES[1]].edge_index.to(device),
        sample_data[EDGE_TYPES[2]].edge_index.to(device),
        sample_data[EDGE_TYPES[3]].edge_index.to(device),
    )

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        onnx_path,
        opset_version=opset_version,
        input_names=["x_class", "x_comp", "e_cc", "e_cl", "e_cbc", "e_ll"],
        output_names=["logits"],
        dynamic_axes={
            "x_class": {0: "num_class_nodes"},
            "x_comp": {0: "num_comp_nodes"},
            "e_cc": {1: "num_cc_edges"},
            "e_cl": {1: "num_cl_edges"},
            "e_cbc": {1: "num_cbc_edges"},
            "e_ll": {1: "num_ll_edges"},
            "logits": {0: "num_comp_nodes"},
        },
    )
    logger.info(f"ONNX model successfully saved: {onnx_path}")
    return onnx_path


class ONNXInferenceRunner:
    """
    Executes ONNX Runtime inference on heterogeneous graph tensors.
    """

    def __init__(self, onnx_path_or_session: Union[str, ort.InferenceSession]) -> None:
        if isinstance(onnx_path_or_session, ort.InferenceSession):
            self.session = onnx_path_or_session
        else:
            if not os.path.exists(onnx_path_or_session):
                raise FileNotFoundError(f"ONNX model not found: {onnx_path_or_session}")
            self.session = ort.InferenceSession(
                onnx_path_or_session, providers=["CPUExecutionProvider"]
            )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def predict_raw(
        self,
        x_class: np.ndarray,
        x_comp: np.ndarray,
        e_cc: np.ndarray,
        e_cl: np.ndarray,
        e_cbc: np.ndarray,
        e_ll: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference and returns (logits, softmax_probabilities).
        """
        inputs = {
            "x_class": x_class.astype(np.float32),
            "x_comp": x_comp.astype(np.float32),
            "e_cc": e_cc.astype(np.int64),
            "e_cl": e_cl.astype(np.int64),
            "e_cbc": e_cbc.astype(np.int64),
            "e_ll": e_ll.astype(np.int64),
        }
        logits = self.session.run(self.output_names, inputs)[0]
        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return logits, probs

    def predict_graph(self, graph: HeteroData) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict_raw(
            graph["Class"].x.cpu().numpy(),
            graph["Component"].x.cpu().numpy(),
            graph[EDGE_TYPES[0]].edge_index.cpu().numpy(),
            graph[EDGE_TYPES[1]].edge_index.cpu().numpy(),
            graph[EDGE_TYPES[2]].edge_index.cpu().numpy(),
            graph[EDGE_TYPES[3]].edge_index.cpu().numpy(),
        )


def custom_json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer handling numpy integers, floats, and arrays.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (tuple, list, set)):
        return [custom_json_serializer(x) for x in obj]
    return str(obj)
