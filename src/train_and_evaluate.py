"""
src/train_and_evaluate.py
CardioPulse: Comprehensive Model Training & Evaluation Pipeline
Trains and evaluates:
  1. NumPy Neural Network (2-Layer MLP: 187 -> 96 ReLU -> 5 Softmax)
  2. 1D-CNN (3 Conv1D blocks with BatchNorm, MaxPool, Dropout, Dense)
  3. Random Forest (100 Trees with balanced class weights)
Computes:
  - Accuracy, Macro & Weighted F1
  - Per-class Precision, Recall, F1
  - Macro One-vs-Rest ROC-AUC
  - Confusion Matrices (Raw counts & Row-normalized)
  - Inference Latency per beat (ms)
  - 5-fold Stratified Cross-Validation on Main Model
  - Saves all plots to reports/figures/ and metrics to reports/metrics_results.json
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, auc, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular / PVC (V)",
    "Fusion (F)",
    "Unknown / Paced (Q)"
]
CLASS_SHORT = ["N", "S", "V", "F", "Q"]
CLASS_COLORS = ["#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#6B7280"]

def load_data(train_path="data/mitbih_train.csv", test_path="data/mitbih_test.csv"):
    print(f"Loading {train_path} and {test_path}...")
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)

    X_train = df_train.iloc[:, :-1].values.astype(np.float32)
    y_train = df_train.iloc[:, -1].values.astype(np.int64)

    X_test = df_test.iloc[:, :-1].values.astype(np.float32)
    y_test = df_test.iloc[:, -1].values.astype(np.int64)

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    return X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------
# 1. NumPy Neural Network (2-Layer MLP)
# --------------------------------------------------------------------------
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class NumPyNeuralClassifier:
    def __init__(self, input_dim=187, hidden_dim=96, num_classes=5, lr=0.05):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        # He normal initialization
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, num_classes).astype(np.float32) * np.sqrt(2.0 / hidden_dim).astype(np.float32)
        self.b2 = np.zeros((1, num_classes), dtype=np.float32)
        self.loss_history = []

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0.0, self.Z1)  # ReLU
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return softmax(self.Z2)

    def fit(self, X, y, epochs=30, batch_size=128, class_weights=None):
        num_samples = X.shape[0]
        if class_weights is None:
            weights_per_sample = np.ones(num_samples, dtype=np.float32)
        else:
            weights_per_sample = np.array([class_weights[c] for c in y], dtype=np.float32)
            weights_per_sample /= np.mean(weights_per_sample)

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            w_shuffled = weights_per_sample[indices]

            for i in range(0, num_samples, batch_size):
                xb = X_shuffled[i:i+batch_size]
                yb = y_shuffled[i:i+batch_size]
                wb = w_shuffled[i:i+batch_size]
                m = xb.shape[0]

                probs = self.forward(xb)
                
                # One-hot target
                Y_one_hot = np.zeros((m, self.num_classes), dtype=np.float32)
                Y_one_hot[np.arange(m), yb] = 1.0

                # Weighted gradient
                dZ2 = (probs - Y_one_hot) * wb[:, None] / m
                dW2 = np.dot(self.A1.T, dZ2)
                db2 = np.sum(dZ2, axis=0, keepdims=True)

                dA1 = np.dot(dZ2, self.W2.T)
                dZ1 = dA1 * (self.Z1 > 0)
                dW1 = np.dot(xb.T, dZ1)
                db1 = np.sum(dZ1, axis=0, keepdims=True)

                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            # Monitor loss on subset to save time
            sub_idx = indices[:5000]
            sub_probs = self.forward(X_shuffled[:5000])
            eps = 1e-9
            loss = -np.mean(np.log(sub_probs[np.arange(5000), y_shuffled[:5000]] + eps))
            self.loss_history.append(float(loss))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [NumPy NN] Epoch {epoch+1:02d}/{epochs} - Loss: {loss:.4f}")

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Saved NumPy NN weights to {filepath}")


# --------------------------------------------------------------------------
# 2. PyTorch 1D-CNN Architecture
# --------------------------------------------------------------------------
class Conv1DHeartbeatClassifier(nn.Module):
    """
    3-Block 1D Convolutional Neural Network matching the report specification.
    Architecture:
      Conv1D(1 -> 32, k=5, p=2) -> BatchNorm -> ReLU -> MaxPool(2)
      Conv1D(32 -> 64, k=5, p=2) -> BatchNorm -> ReLU -> MaxPool(2)
      Conv1D(64 -> 128, k=3, p=1) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout(0.3)
      Flatten -> Dense(128 * 23 -> 64) -> ReLU -> Dropout(0.3) -> Dense(64 -> 5)
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 187 -> 93
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 93 -> 46
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 46 -> 23
            nn.Dropout(p=0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 23, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 1, 187)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.classifier(x)
        return logits


def train_1d_cnn(X_train, y_train, class_weights_tensor, epochs=12, batch_size=128, lr=0.001):
    model = Conv1DHeartbeatClassifier(num_classes=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    dataset = TensorDataset(torch.from_numpy(X_train).unsqueeze(1), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    print("\nTraining 1D-CNN Model...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)

        epoch_loss = total_loss / len(dataset)
        loss_history.append(epoch_loss)
        print(f"  [1D-CNN] Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.4f}")

    return model, loss_history


def predict_1d_cnn(model, X):
    model.eval()
    with torch.no_grad():
        bx = torch.from_numpy(X).unsqueeze(1)
        logits = model(bx)
        probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    return preds, probs


# --------------------------------------------------------------------------
# Evaluation & Metrics Aggregator
# --------------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, y_probs, latency_ms):
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # Class-wise metrics
    p_class, r_class, f1_class, supp_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # Macro One-vs-Rest ROC-AUC
    # One-hot encode y_true for AUC
    y_true_oh = np.zeros((len(y_true), 5))
    y_true_oh[np.arange(len(y_true)), y_true] = 1.0
    roc_auc_macro = roc_auc_score(y_true_oh, y_probs, average='macro', multi_class='ovr')

    cm_counts = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    cm_norm = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]

    metrics = {
        "model_name": name,
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro": float(roc_auc_macro),
        "latency_per_beat_ms": float(latency_ms),
        "per_class": {
            CLASS_NAMES[i]: {
                "class_id": i,
                "precision": float(p_class[i]),
                "recall": float(r_class[i]),
                "f1_score": float(f1_class[i]),
                "support": int(supp_class[i])
            }
            for i in range(5)
        },
        "confusion_matrix_counts": cm_counts.tolist(),
        "confusion_matrix_norm": cm_norm.tolist()
    }
    return metrics


def measure_latency(predict_fn, sample_beat, num_repeats=500):
    # Warmup
    for _ in range(50):
        predict_fn(sample_beat)
    t0 = time.perf_counter()
    for _ in range(num_repeats):
        predict_fn(sample_beat)
    t1 = time.perf_counter()
    return ((t1 - t0) / num_repeats) * 1000.0  # in ms


# --------------------------------------------------------------------------
# Main Execution Pipeline
# --------------------------------------------------------------------------
def main():
    figures_dir = "reports/figures"
    models_dir = "models"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()

    # Compute balanced class weights
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {c: float(w) for c, w in zip(classes, cw)}
    print("\nBalanced Class Weights for Training:")
    for c in classes:
        print(f"  Class {c} ({CLASS_SHORT[c]}): {class_weights_dict[c]:.4f}")

    cw_tensor = torch.tensor(cw, dtype=torch.float32)

    # -------------------------------------------------------------
    # 1. Train & Eval: NumPy Neural Network
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [1/3] Training NumPy Neural Network (2-Layer MLP)")
    print("=" * 60)
    numpy_model = NumPyNeuralClassifier(input_dim=187, hidden_dim=96, num_classes=5, lr=0.08)
    numpy_model.fit(X_train, y_train, epochs=30, batch_size=128, class_weights=class_weights_dict)

    # Save weights for app and HTML dashboard
    np_weights_path = os.path.join(models_dir, "cardio_neural_weights.npz")
    numpy_model.save(np_weights_path)

    # Predict & measure latency
    sample_beat = X_test[0:1]
    latency_np = measure_latency(lambda b: numpy_model.predict(b), sample_beat)
    preds_np = numpy_model.predict(X_test)
    probs_np = numpy_model.predict_proba(X_test)
    metrics_np = evaluate_model("NumPy Neural Network", y_test, preds_np, probs_np, latency_np)

    # -------------------------------------------------------------
    # 2. Train & Eval: 1D-CNN (PyTorch)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [2/3] Training 1D-CNN (3 Conv1D Blocks)")
    print("=" * 60)
    cnn_model, cnn_loss_history = train_1d_cnn(X_train, y_train, cw_tensor, epochs=12, batch_size=128, lr=0.001)

    # Save 1D-CNN weights
    cnn_save_path = os.path.join(models_dir, "cardio_1d_cnn.pt")
    torch.save(cnn_model.state_dict(), cnn_save_path)
    print(f"Saved 1D-CNN weights to {cnn_save_path}")

    # Predict & measure latency
    latency_cnn = measure_latency(lambda b: predict_1d_cnn(cnn_model, b), sample_beat)
    preds_cnn, probs_cnn = predict_1d_cnn(cnn_model, X_test)
    metrics_cnn = evaluate_model("1D Convolutional Neural Network (1D-CNN)", y_test, preds_cnn, probs_cnn, latency_cnn)

    # -------------------------------------------------------------
    # 3. Train & Eval: Random Forest Baseline
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [3/3] Training Random Forest Baseline (100 Trees)")
    print("=" * 60)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    t_rf_start = time.time()
    rf_model.fit(X_train, y_train)
    print(f"Random Forest fit completed in {time.time() - t_rf_start:.2f}s")

    # Predict & measure latency
    latency_rf = measure_latency(lambda b: rf_model.predict(b), sample_beat)
    preds_rf = rf_model.predict(X_test)
    probs_rf = rf_model.predict_proba(X_test)
    metrics_rf = evaluate_model("Random Forest Baseline", y_test, preds_rf, probs_rf, latency_rf)

    # -------------------------------------------------------------
    # 4. 5-Fold Stratified Cross-Validation on Main Model (1D-CNN)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [4] Running 5-Fold Stratified Cross-Validation on 1D-CNN")
    print("=" * 60)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    cv_macro_f1s = []

    fold = 1
    for train_idx, val_idx in skf.split(X_train, y_train):
        print(f"  Fold {fold}/5...")
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        m_fold, _ = train_1d_cnn(X_tr, y_tr, cw_tensor, epochs=6, batch_size=128, lr=0.001)
        p_val, _ = predict_1d_cnn(m_fold, X_va)
        
        f_acc = accuracy_score(y_va, p_val)
        _, _, f_f1, _ = precision_recall_fscore_support(y_va, p_val, average='macro', zero_division=0)
        cv_accuracies.append(f_acc)
        cv_macro_f1s.append(f_f1)
        print(f"    Fold {fold} - Accuracy: {f_acc*100:.2f}%, Macro F1: {f_f1:.4f}")
        fold += 1

    cv_results = {
        "num_folds": 5,
        "fold_accuracies": [float(a) for a in cv_accuracies],
        "fold_macro_f1s": [float(f) for f in cv_macro_f1s],
        "mean_accuracy": float(np.mean(cv_accuracies)),
        "std_accuracy": float(np.std(cv_accuracies)),
        "mean_macro_f1": float(np.mean(cv_macro_f1s)),
        "std_macro_f1": float(np.std(cv_macro_f1s))
    }
    print(f"\n5-Fold CV Results: Accuracy = {cv_results['mean_accuracy']*100:.2f}% +/- {cv_results['std_accuracy']*100:.2f}%, "
          f"Macro F1 = {cv_results['mean_macro_f1']:.4f} +/- {cv_results['std_macro_f1']:.4f}")

    # -------------------------------------------------------------
    # Consolidate and Save Metrics JSON
    # -------------------------------------------------------------
    all_results = {
        "imbalance_handling": "Inverse class frequency weighting (class_weight='balanced') applied across all models",
        "dataset_split": {
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "class_distribution_train": {int(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()},
            "class_distribution_test": {int(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()}
        },
        "models": {
            "numpy_neural_net": metrics_np,
            "1d_cnn": metrics_cnn,
            "random_forest": metrics_rf
        },
        "cross_validation_1d_cnn": cv_results
    }

    metrics_json_path = "reports/metrics_results.json"
    with open(metrics_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved consolidated metrics to {metrics_json_path}")

    # -------------------------------------------------------------
    # Plotting Figures
    # -------------------------------------------------------------
    print("\nGenerating evaluation figures...")

    # Figure 1: Confusion Matrices Comparison (Counts and Row-Normalized)
    fig, axes = plt.subplots(3, 2, figsize=(13, 15), dpi=150)
    models_cm = [
        ("1D-CNN (Proposed)", metrics_cnn),
        ("NumPy Neural Network", metrics_np),
        ("Random Forest Baseline", metrics_rf)
    ]

    for idx, (title, met) in enumerate(models_cm):
        cm_c = np.array(met["confusion_matrix_counts"])
        cm_n = np.array(met["confusion_matrix_norm"])

        # Counts
        sns.heatmap(cm_c, annot=True, fmt='d', cmap='Blues', ax=axes[idx, 0],
                    xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, cbar=False)
        axes[idx, 0].set_title(f"{title} - Raw Counts", fontsize=11, fontweight='bold')
        axes[idx, 0].set_xlabel("Predicted Label")
        axes[idx, 0].set_ylabel("True Label")

        # Row Normalized
        sns.heatmap(cm_n, annot=True, fmt='.3f', cmap='Greens', ax=axes[idx, 1],
                    xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, cbar=False)
        axes[idx, 1].set_title(f"{title} - Row Normalized (Recall)", fontsize=11, fontweight='bold')
        axes[idx, 1].set_xlabel("Predicted Label")
        axes[idx, 1].set_ylabel("True Label")

    plt.tight_layout()
    cm_fig_path = os.path.join(figures_dir, "04_confusion_matrices_all.png")
    plt.savefig(cm_fig_path)
    plt.close()
    print(f"Saved {cm_fig_path}")

    # Figure 2: Per-Class ROC Curves for 1D-CNN
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    y_test_oh = np.zeros((len(y_test), 5))
    y_test_oh[np.arange(len(y_test)), y_test] = 1.0

    for i in range(5):
        fpr, tpr, _ = roc_curve(y_test_oh[:, i], probs_cnn[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2,
                label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance (AUC = 0.5000)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(f"1D-CNN Per-Class One-vs-Rest ROC Curves (Macro AUC = {metrics_cnn['roc_auc_macro']:.4f})",
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_fig_path = os.path.join(figures_dir, "06_roc_curves_1d_cnn.png")
    plt.savefig(roc_fig_path)
    plt.close()
    print(f"Saved {roc_fig_path}")

    # Figure 3: Model Comparison Bar Chart (Accuracy, Macro F1, Weighted F1, ROC-AUC)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    model_labels = ["NumPy Neural Net", "1D-CNN (Proposed)", "Random Forest"]
    accs = [metrics_np["accuracy"] * 100, metrics_cnn["accuracy"] * 100, metrics_rf["accuracy"] * 100]
    macro_f1s = [metrics_np["macro_f1"] * 100, metrics_cnn["macro_f1"] * 100, metrics_rf["macro_f1"] * 100]
    weighted_f1s = [metrics_np["weighted_f1"] * 100, metrics_cnn["weighted_f1"] * 100, metrics_rf["weighted_f1"] * 100]
    aucs = [metrics_np["roc_auc_macro"] * 100, metrics_cnn["roc_auc_macro"] * 100, metrics_rf["roc_auc_macro"] * 100]

    x = np.arange(len(model_labels))
    width = 0.2

    ax.bar(x - 1.5*width, accs, width, label='Accuracy (%)', color='#3B82F6')
    ax.bar(x - 0.5*width, macro_f1s, width, label='Macro F1 (%)', color='#10B981')
    ax.bar(x + 0.5*width, weighted_f1s, width, label='Weighted F1 (%)', color='#F59E0B')
    ax.bar(x + 1.5*width, aucs, width, label='Macro ROC-AUC (%)', color='#8B5CF6')

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Comprehensive Model Performance Comparison on MIT-BIH Test Set", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10, fontweight='bold')
    ax.set_ylim([70, 102])
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    comp_fig_path = os.path.join(figures_dir, "07_model_benchmark_comparison.png")
    plt.savefig(comp_fig_path)
    plt.close()
    print(f"Saved {comp_fig_path}")

    # Figure 4: Training curves for 1D-CNN and NumPy NN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    ax1.plot(numpy_model.loss_history, color='#EF4444', lw=2, marker='o', markersize=4)
    ax1.set_title("NumPy Neural Network Convergence", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch Iteration")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(cnn_loss_history, color='#3B82F6', lw=2, marker='s', markersize=4)
    ax2.set_title("1D-CNN Training Loss Curve", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Epoch Iteration")
    ax2.set_ylabel("Weighted Cross-Entropy Loss")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    train_curves_path = os.path.join(figures_dir, "05_training_curves.png")
    plt.savefig(train_curves_path)
    plt.close()
    print(f"Saved {train_curves_path}")

    # Also save the single primary confusion matrix as 04_confusion_matrix.png for backward compatibility
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    sns.heatmap(np.array(metrics_cnn["confusion_matrix_counts"]), annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, ax=ax)
    ax.set_title("CardioPulse 1D-CNN Confusion Matrix (Test Set)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_confusion_matrix.png"))
    plt.close()

    print("\n" + "=" * 60)
    print("      ALL MODELS TRAINED AND EVALUATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
