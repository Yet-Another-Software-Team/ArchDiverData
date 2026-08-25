# ArchDiver Scripts & GNN Pipeline Guide

This directory contains the complete pipeline for data processing, GNN training, dynamic ONNX model export, model-agnostic ONNX-native GNNExplainer, visualization tools, and automated testing for detecting and interpreting software architectural smells.

---

## Table of Contents

1. [Overview & Directory Structure](#1-overview--directory-structure)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [Quick Start Cheat Sheet](#3-quick-start-cheat-sheet)
4. [Heterogeneous Graph Schema & Data Model](#4-heterogeneous-graph-schema--data-model)
5. [Training Pipeline (`train_pipeline.py`)](#5-training-pipeline-train_pipelinepy)
6. [ONNX GNNExplainer PoC (`explain_onnx.py`)](#6-onnx-gnnexplainer-poc-explain_onnxpy)
7. [Architecture & Explanation Visualizer (`visualize.py`)](#7-architecture--explanation-visualizer-visualizepy)
8. [Automated Test Suite (`test_pipelines.py`)](#8-automated-test-suite-test_pipelinespy)
9. [Data Ingestion & Labeling Tools](#9-data-ingestion--labeling-tools)
10. [Algorithm Deep-Dive: How ONNX GNNExplainer Works](#10-algorithm-deep-dive-how-onnx-gnnexplainer-works)

---

## 1. Overview & Directory Structure

```
scripts/
├── common.py             # Shared GNN model (ArchSmellClassifier), FocalLoss, NodeFeatureScaler,
│                         # SyntheticGraphBuilder, ONNXInferenceRunner, and serialization utils
├── train_pipeline.py     # Main GNN training pipeline with Focal Loss, dynamic ONNX export, and --explain
├── explain_onnx.py       # Pure ONNX Runtime GNNExplainer (edge ablation, metric attribution, fidelity)
├── visualize.py          # Unified visualizer for full graphs, ego-neighborhoods, and on-the-fly --explain
├── test_pipelines.py     # Automated unit & integration test suite (6 passing test cases)
├── requirements.txt      # Python runtime dependencies
├── graph_creation.py     # Parses Java AST and metric CSVs to build PyG HeteroData graphs
├── graph_labeler.py      # Queries Postgres smell tables to attach ground-truth labels
├── graph_viewer.py       # Legacy viewer reference
├── repo_cloner.py        # Clones Java repositories and invokes DesigniteJava analyzer
├── link_checker.py       # Checks validity of harvested GitHub repository URLs
└── README.md             # This comprehensive guide
```

---

## 2. Prerequisites & Installation

Ensure you have Python 3.10+ and a virtual environment set up:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r scripts/requirements.txt
```

---

## 3. Quick Start Cheat Sheet

| Task                               | Command                                                                                       |
| :--------------------------------- | :-------------------------------------------------------------------------------------------- |
| **Run Unit Tests**                 | `python scripts/test_pipelines.py`                                                            |
| **Train & Export to ONNX**         | `python scripts/train_pipeline.py --epochs 40 --batch_size 8`                                 |
| **Train & Auto-Explain**           | `python scripts/train_pipeline.py --epochs 30 --explain`                                      |
| **Explain a Real Graph**           | `python scripts/explain_onnx.py --graph "raw_data/labeled graphs/graph_1.pt"`                 |
| **Explain a Specific Node**        | `python scripts/explain_onnx.py --graph "raw_data/labeled graphs/graph_1.pt" --node 14`       |
| **Synthetic Explain Demo**         | `python scripts/explain_onnx.py --synthetic`                                                  |
| **Visualize Full Graph**           | `python scripts/visualize.py --graph "raw_data/labeled graphs/graph_1.pt"`                    |
| **Visualize Ego Subgraph**         | `python scripts/visualize.py --graph "raw_data/labeled graphs/graph_1.pt" --node 14 --hops 2` |
| **Visualize & Explain (One-Step)** | `python scripts/visualize.py --graph "raw_data/labeled graphs/graph_1.pt" --explain`          |

---

## 4. Heterogeneous Graph Schema & Data Model

Each software architecture graph is stored as a PyTorch Geometric `HeteroData` object containing two node types and four canonical directed edge types:

### Node Types & Metric Features

1. **`Class` Nodes** (`x` shape: `[N_classes, 6]`):
   - `class_lcom`: Lack of Cohesion in Methods of the class.
   - `num_attributes`: Total attributes/fields declared.
   - `num_methods_declared`: Explicitly declared methods.
   - `num_methods_actual`: Actual callable methods (including inherited).
   - `avg_method_lcom`: Average LCOM score across methods.
   - `avg_params`: Average parameter count per method.
2. **`Component` Nodes** (`x` shape: `[N_components, 1]`):
   - `num_files_and_folders`: Count of contained files and folders.
   - Target Label `Component.y`: `0` = Clean, `1` = Architectural Smell (e.g., Cyclic Dependency, God Component).

### Canonical Edge Types

- `('Component', 'contains', 'Component')`: Package hierarchy nesting.
- `('Component', 'contains', 'Class')`: Component to member class containment.
- `('Class', 'contained_by', 'Component')`: Inverse containment link.
- `('Class', 'imports', 'Class')`: Static code dependency / import relation between classes.

---

## 5. Training Pipeline (`train_pipeline.py`)

Trains a heterogeneous GraphSAGE network with Focal Loss and exports the trained model directly to dynamic ONNX format.

### Key Capabilities:

- **Feature Scaling**: Automatically fits a `NodeFeatureScaler` across training graphs and stores normalization parameters in the checkpoint.
- **Focal Loss**: Balances severe class imbalance (e.g. 1995 clean vs 226 smelly components) using $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$.
- **Dynamic ONNX Export**: Automatically initializes lazy GNN layers and traces dynamic dimensions for both nodes and edges.
- **Automated Post-Training Explanation (`--explain`)**: Automatically triggers explanation generation and visualization on validation graphs right after training completes.

### CLI Options:

```text
--data_dir DIR            Path to labeled graphs directory (default: "raw_data/labeled graphs")
--synthetic               Train on synthetic graph dataset
--num_synthetic N         Number of synthetic graphs if --synthetic (default: 20)
--epochs N                Number of training epochs (default: 40)
--batch_size N            Batch size (default: 8)
--lr LR                   Learning rate (default: 0.002)
--hidden_channels N       GNN hidden layer dimensions (default: 64)
--num_layers N            Number of GraphSAGE layers (default: 2)
--dropout D               Dropout rate (default: 0.3)
--focal_gamma G           Focal loss gamma parameter (default: 2.0)
--model_name NAME         Saved model basename (default: "arch_smell_model")
--output_dir DIR          Model output directory (default: "output/models")
--explain_output_dir DIR  Explanation output directory (default: "output/explanations")
--explain                 Run automated ONNX explanation after training completes
--cpu                     Force CPU training execution
--seed N                  Random seed (default: 42)
```

### Examples:

```bash
# Standard training on labeled dataset
python scripts/train_pipeline.py --epochs 40 --batch_size 8

# Train and immediately generate explanation reports
python scripts/train_pipeline.py --epochs 40 --explain

# Quick synthetic test run
python scripts/train_pipeline.py --synthetic --num_synthetic 15 --epochs 20 --explain
```

---

## 6. ONNX GNNExplainer PoC (`explain_onnx.py`)

Explains why a Component was classified as smelly using **pure ONNX Runtime** (`onnxruntime.InferenceSession`), requiring no PyTorch autograd engine at inference time.

### Quantitative Explanation Metrics:

- **Fidelity+ (Comprehensiveness)**:
  $$\text{Fidelity}^+ = P(Y \mid G) - P(Y \mid G \setminus G_{\text{exp}})$$
  _Higher score means the identified explanatory elements were essential for the smell prediction._
- **Fidelity- (Sufficiency)**:
  $$\text{Fidelity}^- = P(Y \mid G) - P(Y \mid G_{\text{exp}})$$
  _Lower score means the explanatory subgraph alone is sufficient to preserve the smell prediction._
- **Sparsity**:
  $$\text{Sparsity} = 1 - \frac{|E_{\text{exp}}|}{|E_{\text{subgraph}}|}$$
  _Measures conciseness of the explanation._

### CLI Options:

```text
--model PATH              Path to ONNX model (default: "output/models/arch_smell_model.onnx")
--checkpoint PATH         Path to checkpoint containing feature scaler (default: "output/models/arch_smell_model.pt")
--graph PATH              Path to graph .pt file (default: "raw_data/labeled graphs/graph_1.pt")
--synthetic               Generate a synthetic test graph with an injected smell
--node ID                 Specific Component node ID to explain (default: auto-detect smelly node)
--hops N                  Computational neighborhood radius (default: 2)
--top_k N                 Top explanatory edges to rank (default: 10)
--output_dir DIR          Output directory for JSON and PNG artifacts (default: "output/explanations")
--show                    Display plot interactively in GUI
```

### Example Usage:

```bash
# Explain ground-truth or top smelly node
python scripts/explain_onnx.py --graph "raw_data/labeled graphs/graph_1.pt"

# Explain a specific component (e.g. Component #14) with 2-hop radius
python scripts/explain_onnx.py --graph "raw_data/labeled graphs/graph_1.pt" --node 14 --hops 2 --top_k 8
```

---

## 7. Architecture & Explanation Visualizer (`visualize.py`)

Unified visualizer supporting full architecture layouts, $k$-hop ego-network subgraphs, and dual-panel explanation reports.

### Color & Styling Palette:

- **Clean Component**: Amber / Yellow (`#ffbf47`)
- **Smelly Component**: Red (`#ff3333`)
- **Class Node**: Blue (`#4da6ff`)
- **Explanatory Edges**: Thick Red (`#cc0000`)
- **Import Dependency Edges**: Steel Blue (`#2b6cb0`)

### CLI Options:

```text
-g, --graph PATH          Path to input graph .pt file
-e, --explain [JSON_PATH] Run on-the-fly explanation or load saved explanation JSON file
-m, --model PATH          Path to ONNX model for on-the-fly explanation
-c, --checkpoint PATH     Path to checkpoint (.pt) containing feature scaler
--synthetic               Use synthetic graph for demonstration
-n, --node ID             Focus on specific Component node ID
--hops N                  Ego-network hop radius (default: 2)
-o, --output PATH         Output image file path (.png)
--show                    Display interactive window
```

### Example Modes:

```bash
# Mode 1: Full Graph Visualization
python scripts/visualize.py --graph "raw_data/labeled graphs/graph_1.pt" --output "output/explanations/graph_1_full.png"

# Mode 2: Focused 2-Hop Ego Subgraph
python scripts/visualize.py --graph "raw_data/labeled graphs/graph_1.pt" --node 14 --hops 2 --output "output/explanations/graph_1_node_14_subgraph.png"

# Mode 3: On-the-Fly GNN Explanation & Visualization (--explain)
python scripts/visualize.py --graph "raw_data/labeled graphs/graph_1.pt" --node 14 --explain

# Mode 4: Replot from a Saved Explanation JSON File
python scripts/visualize.py --explain "output/explanations/graph_1_comp_14_explanation.json" --graph "raw_data/labeled graphs/graph_1.pt"
```

---

## 8. Automated Test Suite (`test_pipelines.py`)

Comprehensive unit and integration test suite with zero external mock dependencies.

```bash
# Run all tests
python scripts/test_pipelines.py

# Or via unittest with verbose output
python -m unittest -v scripts/test_pipelines.py
```

### Verified Test Cases:

1. `test_synthetic_graph_builder`: Ensures synthetic graph dimensions, node features, edge connectivity, and injected smell anomalies match specifications.
2. `test_feature_scaler`: Validates feature standardization, state dictionary serialization, and deserialization.
3. `test_model_forward_and_loss`: Checks heterogeneous forward pass and `FocalLoss` computation.
4. `test_onnx_export_and_inference_parity`: Asserts numerical parity ($10^{-4}$ tolerance) between PyTorch forward pass and ONNX Runtime inference.
5. `test_onnx_gnn_explainer_end_to_end`: Validates ONNX-native subgraph extraction, marginal drop ranking, fidelity calculations, and report generation.
6. `test_architecture_visualizer`: Tests full graph and ego-subgraph image rendering.

---

## 9. Data Ingestion & Labeling Tools

For end-to-end data harvesting and graph construction from raw Java repositories and Postgres database dumps:

1. **Clone Repositories & Run DesigniteJava**:
   ```bash
   python scripts/repo_cloner.py
   ```
2. **Build Heterogeneous Graph (`HeteroData`)**:
   ```bash
   python scripts/graph_creation.py -f <path_to_cloned_repo_output>
   ```
3. **Attach Postgres Ground-Truth Smell Labels**:
   ```bash
   python scripts/graph_labeler.py -f <path_to_created_graph.pt>
   ```
4. **Validate Repository Links**:
   ```bash
   python scripts/link_checker.py
   ```

---

## 10. Algorithm Deep-Dive: How ONNX GNNExplainer Works

### The Challenge

Original GNNExplainer (_Ying et al., NeurIPS 2019_) learns continuous edge/feature masks $M, F \in [0, 1]$ via gradient descent:
$$\min_{M, F} -\log P_\Phi(Y = c \mid G \odot \sigma(M), X \odot \sigma(F)) + \lambda_1 \|M\|_1 + \lambda_2 \mathcal{H}(M)$$
This requires an active autograd computation graph with backward differentiation, which is **not supported by inference-only ONNX runtimes**.

### The Solution: Forward-Only Marginal Attribution

Our implementation uses **Model-Agnostic Structural Perturbation (Forward Marginal Attribution)**:

```
 ┌────────────────────────────────────────────────────────┐
 │ 1. Local Receptive Field Extraction (2-hop Ego Graph) │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. Base ONNX Inference: P₀ = P(Smell | G, X)          │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Forward Perturbation Loops via ONNX Runtime:        │
 │    • For each local edge e_i:                          │
 │        P_ablated = ONNX(G \ {e_i}, X)                  │
 │        Score(e_i) = max(0, P₀ - P_ablated)            │
 │                                                        │
 │    • For each metric feature f_j:                     │
 │        P_ablated = ONNX(G, X with f_j masked)          │
 │        Score(f_j) = max(0, P₀ - P_ablated)            │
 └──────────────────────────┬─────────────────────────────┘
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 4. Rank Top Influences + Compute Fidelity & Sparsity   │
 └────────────────────────────────────────────────────────┘
```

1. **Locality Filtering**: Because a 2-layer GraphSAGE has a 2-hop receptive field, any edge beyond 2 hops has a gradient of exactly 0. We isolate the local ego-network ($\sim 10\text{--}50$ edges).
2. **Edge Ablation**: We remove one edge $e_i$ at a time and measure the probability drop $\Delta P(e_i) = \max(0, P_0 - P_{\text{ablated}})$.
3. **Metric Attribution**: We set individual metric columns to baseline and measure the impact on prediction confidence.
4. **Fidelity Metrics**: We evaluate the prediction drop when removing the top edges ($\text{Fidelity}^+$) vs. preserving only the top edges ($\text{Fidelity}^-$).

This achieves explanation in milliseconds with zero GPU or PyTorch autograd dependencies.
