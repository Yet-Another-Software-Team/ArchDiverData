import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, to_hetero
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

# --- Model Definitions ---

class ArchSmellClassifier(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_classes, num_layers=2, dropout_rate=0.5):
        super().__init__()
        base_sage = GraphSAGE(
            in_channels=-1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout_rate
        )
        self.sage = to_hetero(base_sage, metadata, aggr='sum')
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_dict, edge_index_dict):
        h_dict = self.sage(x_dict, edge_index_dict)
        h = h_dict['Component']
        h = F.relu(h)
        h = self.dropout(h)
        logits = self.classifier(h)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# --- Helper Functions ---

def load_dataset(data_dir):
    dataset = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found.")
    for file in os.listdir(data_dir):
        if file.endswith(".pt"):
            path = os.path.join(data_dir, file)
            data_obj = torch.load(path, weights_only=False)['graph_data']
            if hasattr(data_obj['Component'], 'y'):
                dataset.append(data_obj)
    return dataset

def fit_node_scalers(train_graphs, node_types=['Class', 'Component']):
    scalers = {}
    for n_type in node_types:
        feats = [g[n_type].x for g in train_graphs if hasattr(g[n_type], 'x') and g[n_type].x.numel() > 0]
        if feats:
            all_features = torch.cat(feats, dim=0).numpy()
            scalers[n_type] = StandardScaler().fit(all_features)
    return scalers

def apply_node_scalers(graphs, scalers):
    # Process a copy to prevent modifying the source list in unexpected ways
    processed_graphs = []
    for g in graphs:
        g_copy = copy.deepcopy(g)
        for n_type, scaler in scalers.items():
            if hasattr(g_copy[n_type], 'x'):
                features_np = g_copy[n_type].x.numpy()
                normalized = scaler.transform(features_np)
                g_copy[n_type].x = torch.from_numpy(normalized).float()
        processed_graphs.append(g_copy)
    return processed_graphs

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x_dict, batch.edge_index_dict)
        loss = criterion(logits, batch['Component'].y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x_dict, batch.edge_index_dict)
            preds = logits.argmax(dim=1)
            preds_list.extend(preds.cpu().numpy())
            targets_list.extend(batch['Component'].y.cpu().numpy())
    
    f1 = f1_score(targets_list, preds_list, pos_label=1, zero_division=0)
    acc = accuracy_score(targets_list, preds_list)
    return f1, acc, targets_list, preds_list

def save_onnx(model, sample_data, device, path="arch_smell_model.onnx"):
    model.eval()
    edge_types = [('Component', 'contains', 'Component'), ('Component', 'contains', 'Class'),
                  ('Class', 'contained_by', 'Component'), ('Class', 'imports', 'Class')]
    
    if not os.path.exists("output/models"):
        os.makedirs("output/models")

    path = os.path.join("output/models", path)
    
    class OnnxWrapper(nn.Module):
        def __init__(self, m, et): 
            super().__init__()
            self.m = m
            self.et = et
        def forward(self, xc, xcom, ecc, ecl, ecbc, ell):
            xd = {'Class': xc, 'Component': xcom}
            ed = {self.et[0]: ecc, self.et[1]: ecl, self.et[2]: ecbc, self.et[3]: ell}
            return self.m(xd, ed)

    wrapper = OnnxWrapper(model, edge_types).to(device)
    dummy_inputs = (
        sample_data['Class'].x.to(device), sample_data['Component'].x.to(device),
        sample_data[edge_types[0]].edge_index.to(device), sample_data[edge_types[1]].edge_index.to(device),
        sample_data[edge_types[2]].edge_index.to(device), sample_data[edge_types[3]].edge_index.to(device)
    )
    
    torch.onnx.export(wrapper, dummy_inputs, path, opset_version=15, 
                      input_names=["x_class", "x_comp", "e_cc", "e_cl", "e_cbc", "e_ll"], 
                      output_names=["logits"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hc', type=int, default=64, help='Hidden channels')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--f', type=float, default=1.5, help='Focal Loss gamma')
    parser.add_argument('--ex', type=str, default="ArchSmell_Detection", help='MLflow experiment name')
    parser.add_argument('--n', type=str, default="ArchSmell_Detection_Baseline", help='Model artifact name')
    args = parser.parse_args()

    mlflow.set_experiment(args.ex)
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        
        # 1. Load and Split Raw Data
        raw_graphs = load_dataset(args.data_dir)
        train_val_raw, test_raw = train_test_split(raw_graphs, test_size=0.2, random_state=42)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        best_overall_model = None
        best_overall_f1 = -1.0
        fold_f1_scores = []

        # 2. K-Fold Cross-Validation
        for fold, (t_idx, v_idx) in enumerate(kf.split(train_val_raw)):
            print(f"\n--- Starting Fold {fold + 1} ---")
            
            # Create fold-specific data slices
            train_subset = [train_val_raw[i] for i in t_idx]
            val_subset = [train_val_raw[i] for i in v_idx]

            # Fit scalers
            scalers = fit_node_scalers(train_subset)
            train_fold = apply_node_scalers(train_subset, scalers)
            val_fold = apply_node_scalers(val_subset, scalers)

            train_loader = DataLoader(train_fold, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_fold, batch_size=args.batch_size)

            # Compute weights for this fold's training distribution
            all_labels = torch.cat([d['Component'].y for d in train_fold])
            num_smelly = all_labels.sum().item()
            num_clean = len(all_labels) - num_smelly
            weights = torch.tensor([1.0, num_clean / max(num_smelly, 1)]).to(device)

            # Initialize Model
            model = ArchSmellClassifier(raw_graphs[0].metadata(), args.hc, 2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            criterion = FocalLoss(weight=weights, gamma=args.f)

            best_fold_f1 = -1.0
            best_epoch_per_fold = []
            f1_scores_per_epoch = []
            for epoch in range(1, args.epochs + 1):
                loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                f1, acc, _, _ = evaluate(model, val_loader, device)

                f1_scores_per_epoch.append(f1)

                if f1 > best_fold_f1:
                    best_fold_f1 = f1
                    best_fold_epoch = epoch
                    if f1 > best_overall_f1:
                        best_overall_f1 = f1
            
            print(f"Fold {fold} Best F1: {best_fold_f1:.4f}")
            mlflow.log_metric(f"fold_{fold}_f1", best_fold_f1)
            mlflow.log_metric(f"fold_{fold}_best_epoch", best_fold_epoch)
            fold_f1_scores.append(np.mean(f1_scores_per_epoch))
            best_epoch_per_fold.append(best_fold_epoch)

        mlflow.log_metric("avg_cv_f1", np.mean(fold_f1_scores))

        final_scalers    = fit_node_scalers(train_val_raw)
        train_val_scaled = apply_node_scalers(train_val_raw, final_scalers)
        test_processed   = apply_node_scalers(test_raw, final_scalers)

        full_loader = DataLoader(train_val_scaled, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_processed, batch_size=1)

        # compute weights on full train_val
        all_labels    = torch.cat([d['Component'].y for d in train_val_scaled])
        num_smelly    = all_labels.sum().item()
        num_clean     = len(all_labels) - num_smelly
        final_weights = torch.tensor([1.0, num_clean / max(num_smelly, 1)]).to(device)

        final_model     = ArchSmellClassifier(raw_graphs[0].metadata(), args.hc, 2).to(device)
        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=5e-4)
        final_criterion = FocalLoss(weight=final_weights, gamma=args.f)

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(final_model, full_loader, final_optimizer, final_criterion, device)
            if epoch % 10 == 0:
                print(f"  Retrain epoch {epoch}/{args.epochs} — loss: {loss:.4f}")

        test_f1, test_acc, y_true, y_pred = evaluate(final_model, test_loader, device)

        # Log average cross-validation performance
        mlflow.log_metric("avg_cv_f1", np.mean(fold_f1_scores))

        # 3. Final Test Evaluation
        # Scale test set using the scalers associated with the best training session
        test_processed = apply_node_scalers(test_raw, final_scalers)
        test_loader = DataLoader(test_processed, batch_size=1)
        
        test_f1, test_acc, y_true, y_pred = evaluate(final_model, test_loader, device)
        
        print(f"\nFinal Test Results -> F1: {test_f1:.4f}, Acc: {test_acc:.4f}")
        mlflow.log_metrics({"test_f1": test_f1, "test_acc": test_acc})
        
        # 4. Save Model and Artifacts
        mlflow.pytorch.log_model(final_model, args.ex + "_" + args.n)
        save_onnx(final_model, test_processed[0], device, args.n + ".onnx")
        mlflow.log_artifact("output/models/" + args.n + ".onnx")
        
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
        plt.title(f"Confusion Matrix (F1: {test_f1:.2f})")

        if not os.path.exists("output/graphs"):
            os.makedirs("output/graphs")

        plt.savefig("output/graphs/" + args.n + "_" + "cm.png")
        mlflow.log_artifact("output/graphs/" + args.n + "_" + "cm.png")