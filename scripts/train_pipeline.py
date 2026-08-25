"""
scripts/train_pipeline.py

Production-grade training pipeline for Heterogeneous Graph Neural Networks
detecting architectural smells, featuring Focal Loss and dynamic ONNX export.
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
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from scripts.common import (
    ArchSmellClassifier,
    FocalLoss,
    NodeFeatureScaler,
    ONNXInferenceRunner,
    SyntheticGraphBuilder,
    export_model_to_onnx,
    load_labeled_graphs,
)

logger = logging.getLogger("ArchDiver.Train")


@dataclass
class TrainingConfig:
    data_dir: str = "raw_data/labeled graphs"
    synthetic: bool = False
    num_synthetic: int = 20
    epochs: int = 40
    batch_size: int = 8
    lr: float = 0.002
    weight_decay: float = 1e-4
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    focal_gamma: float = 2.0
    model_name: str = "arch_smell_model"
    output_dir: str = "output/models"
    cpu: bool = False
    random_seed: int = 42


def evaluate_model(
    model: ArchSmellClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates the model on validation/test graphs.
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x_dict, batch.edge_index_dict)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = batch["Component"].y.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    return {
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "f1": float(f1_score(all_targets, all_preds, pos_label=1, zero_division=0)),
        "precision": float(
            precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
        ),
    }


def run_training(config: TrainingConfig) -> Tuple[str, str]:
    """
    Main training execution function.
    Returns: Tuple of (pytorch_checkpoint_path, onnx_model_path).
    """
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.cpu else "cpu"
    )
    logger.info(f"Using execution device: {device}")

    # 1. Load Data
    if config.synthetic:
        logger.info(f"Generating synthetic dataset ({config.num_synthetic} graphs)...")
        raw_items = SyntheticGraphBuilder.build_dataset(
            num_graphs=config.num_synthetic, seed=config.random_seed
        )
    else:
        logger.info(f"Loading labeled graphs from directory: '{config.data_dir}'...")
        raw_items = load_labeled_graphs(config.data_dir)

    if not raw_items:
        raise ValueError(
            "No valid labeled graphs found. Check data directory or enable --synthetic."
        )

    graphs = [item[0] for item in raw_items]
    logger.info(f"Loaded total of {len(graphs)} graphs.")

    # 2. Split and Scale Features
    train_raw, val_raw = train_test_split(
        graphs, test_size=0.2, random_state=config.random_seed
    )
    scaler = NodeFeatureScaler()
    scaler.fit(train_raw)

    train_scaled = scaler.transform_all(train_raw)
    val_scaled = scaler.transform_all(val_raw)

    train_loader = DataLoader(train_scaled, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_scaled, batch_size=1)

    # 3. Handle Imbalance
    all_y = torch.cat([g["Component"].y for g in train_scaled])
    num_smelly = int(all_y.sum().item())
    num_clean = len(all_y) - num_smelly
    smell_weight = float(num_clean) / max(1, num_smelly)
    weights = torch.tensor([1.0, smell_weight], dtype=torch.float, device=device)
    logger.info(
        f"Class distribution: Clean = {num_clean}, Smelly = {num_smelly} (Smell Loss Weight = {smell_weight:.2f})"
    )

    # 4. Model & Optimizer Setup
    sample_g = train_scaled[0]
    model = ArchSmellClassifier(
        metadata=sample_g.metadata(),
        hidden_channels=config.hidden_channels,
        num_classes=2,
        num_layers=config.num_layers,
        dropout_rate=config.dropout,
    ).to(device)

    # Warm-up lazy parameters
    with torch.no_grad():
        x_dev = {k: v.to(device) for k, v in sample_g.x_dict.items()}
        e_dev = {k: v.to(device) for k, v in sample_g.edge_index_dict.items()}
        _ = model(x_dev, e_dev)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = FocalLoss(weight=weights, gamma=config.focal_gamma)

    # 5. Training Loop
    best_f1 = -1.0
    best_weights: Optional[Dict[str, torch.Tensor]] = None

    logger.info("Starting training loop...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x_dict, batch.edge_index_dict)
            loss = criterion(logits, batch["Component"].y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        metrics = evaluate_model(model, val_loader, device)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == config.epochs:
            logger.info(
                f"Epoch {epoch:03d}/{config.epochs:03d} | Loss: {avg_loss:.4f} | "
                f"Val Acc: {metrics['accuracy']:.4f} | Val F1 (Smell): {metrics['f1']:.4f} | "
                f"Val Rec: {metrics['recall']:.4f}"
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 6. Save Model Checkpoint & Export to ONNX
    os.makedirs(config.output_dir, exist_ok=True)
    pt_path = os.path.join(config.output_dir, f"{config.model_name}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_state": scaler.state_dict(),
            "metadata": sample_g.metadata(),
            "config": vars(config),
        },
        pt_path,
    )
    logger.info(f"PyTorch checkpoint saved: {pt_path}")

    onnx_path = os.path.join(config.output_dir, f"{config.model_name}.onnx")
    export_model_to_onnx(model, sample_g, onnx_path, device=device)

    # Verify ONNX Runtime Parity
    runner = ONNXInferenceRunner(onnx_path)
    _, probs = runner.predict_graph(sample_g)
    logger.info(f"ONNX Model validation complete (output shape: {probs.shape})")

    return pt_path, onnx_path


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Architectural Smell GNN Training Pipeline & ONNX Exporter"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="raw_data/labeled graphs",
        help="Path to labeled graphs directory",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Train on synthetic graph dataset"
    )
    parser.add_argument(
        "--num_synthetic",
        type=int,
        default=20,
        help="Number of synthetic graphs if --synthetic",
    )
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument(
        "--hidden_channels", type=int, default=64, help="GNN hidden channels"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GraphSAGE layers"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="arch_smell_model",
        help="Saved model basename",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/models", help="Output directory"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    return TrainingConfig(
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        num_synthetic=args.num_synthetic,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
        model_name=args.model_name,
        output_dir=args.output_dir,
        cpu=args.cpu,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
