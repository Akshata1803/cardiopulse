"""
src/fast_train_evaluate.py
Evaluates all three models on the real MIT-BIH dataset:
  1. NumPy Neural Network (weights already trained or fit fast)
  2. 1D-CNN (weights loaded from models/cardio_1d_cnn.pt)
  3. Random Forest (100 trees, balanced class weights)
Runs 5-fold stratified CV for the main model (1D-CNN) efficiently.
Saves all figures to reports/figures/ and metrics to reports/metrics_results.json.
"""

import os
import sys
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

class Conv1DHeartbeatClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class NumPyNeuralClassifier:
    def __init__(self, input_dim=187, hidden_dim=96, num_classes=5, lr=0.08):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, num_classes).astype(np.float32) * np.sqrt(2.0 / hidden_dim).astype(np.float32)
        self.b2 = np.zeros((1, num_classes), dtype=np.float32)
        self.loss_history = []

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0.0, self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return softmax(self.Z2)

    def fit(self, X, y, epochs=15, batch_size=128, class_weights=None):
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
                Y_one_hot = np.zeros((m, self.num_classes), dtype=np.float32)
                Y_one_hot[np.arange(m), yb] = 1.0

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

            sub_probs = self.forward(X_shuffled[:5000])
            loss = -np.mean(np.log(sub_probs[np.arange(5000), y_shuffled[:5000]] + 1e-9))
            self.loss_history.append(float(loss))

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def load(self, filepath):
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)


def predict_1d_cnn(model, X):
    model.eval()
    with torch.no_grad():
        bx = torch.from_numpy(X).unsqueeze(1)
        logits = model(bx)
        probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    return preds, probs


def evaluate_model(name, y_true, y_pred, y_probs, latency_ms):
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    p_class, r_class, f1_class, supp_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    y_true_oh = np.zeros((len(y_true), 5))
    y_true_oh[np.arange(len(y_true)), y_true] = 1.0
    roc_auc_macro = roc_auc_score(y_true_oh, y_probs, average='macro', multi_class='ovr')

    cm_counts = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    cm_norm = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]

    return {
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


def measure_latency(predict_fn, sample_beat, num_repeats=300):
    for _ in range(30):
        predict_fn(sample_beat)
    t0 = time.perf_counter()
    for _ in range(num_repeats):
        predict_fn(sample_beat)
    t1 = time.perf_counter()
    return ((t1 - t0) / num_repeats) * 1000.0


def main():
    print("[1/5] Loading datasets...")
    df_train = pd.read_csv("data/mitbih_train.csv", header=None)
    df_test = pd.read_csv("data/mitbih_test.csv", header=None)

    X_train = df_train.iloc[:, :-1].values.astype(np.float32)
    y_train = df_train.iloc[:, -1].values.astype(np.int64)
    X_test = df_test.iloc[:, :-1].values.astype(np.float32)
    y_test = df_test.iloc[:, -1].values.astype(np.int64)

    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {c: float(w) for c, w in zip(classes, cw)}
    cw_tensor = torch.tensor(cw, dtype=torch.float32)
    print("Class weights (balanced):", class_weights_dict)

    sample_beat = X_test[0:1]

    # --- MODEL 1: NumPy Neural Net ---
    print("\n[2/5] Training/loading NumPy Neural Network...")
    np_model = NumPyNeuralClassifier(input_dim=187, hidden_dim=96, num_classes=5, lr=0.08)
    np_weights_path = "models/cardio_neural_weights.npz"
    if os.path.exists(np_weights_path):
        try:
            np_model.load(np_weights_path)
            print("Loaded existing NumPy NN weights.")
        except Exception:
            np_model.fit(X_train, y_train, epochs=15, batch_size=128, class_weights=class_weights_dict)
            np_model.save(np_weights_path)
    else:
        np_model.fit(X_train, y_train, epochs=15, batch_size=128, class_weights=class_weights_dict)
        np_model.save(np_weights_path)

    preds_np = np_model.predict(X_test)
    probs_np = np_model.predict_proba(X_test)
    latency_np = measure_latency(lambda b: np_model.predict(b), sample_beat)
    metrics_np = evaluate_model("NumPy Neural Network", y_test, preds_np, probs_np, latency_np)
    print(f"NumPy NN -> Acc: {metrics_np['accuracy']*100:.2f}%, Macro F1: {metrics_np['macro_f1']:.4f}, AUC: {metrics_np['roc_auc_macro']:.4f}, Latency: {latency_np:.3f} ms")

    # --- MODEL 2: 1D-CNN ---
    print("\n[3/5] Loading/evaluating 1D-CNN (PyTorch)...")
    cnn_model = Conv1DHeartbeatClassifier(num_classes=5)
    cnn_weights_path = "models/cardio_1d_cnn.pt"
    if os.path.exists(cnn_weights_path):
        cnn_model.load_state_dict(torch.load(cnn_weights_path, weights_only=True))
        print("Loaded existing 1D-CNN weights.")
    else:
        # Quick train 3 epochs if missing
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_train).unsqueeze(1), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
        for ep in range(4):
            cnn_model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(cnn_model(bx), by)
                loss.backward()
                optimizer.step()
        torch.save(cnn_model.state_dict(), cnn_weights_path)

    preds_cnn, probs_cnn = predict_1d_cnn(cnn_model, X_test)
    latency_cnn = measure_latency(lambda b: predict_1d_cnn(cnn_model, b), sample_beat)
    metrics_cnn = evaluate_model("1D Convolutional Neural Network (1D-CNN)", y_test, preds_cnn, probs_cnn, latency_cnn)
    print(f"1D-CNN   -> Acc: {metrics_cnn['accuracy']*100:.2f}%, Macro F1: {metrics_cnn['macro_f1']:.4f}, AUC: {metrics_cnn['roc_auc_macro']:.4f}, Latency: {latency_cnn:.3f} ms")

    # --- MODEL 3: Random Forest ---
    print("\n[4/5] Training/evaluating Random Forest Baseline...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=16, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    preds_rf = rf_model.predict(X_test)
    probs_rf = rf_model.predict_proba(X_test)
    latency_rf = measure_latency(lambda b: rf_model.predict(b), sample_beat)
    metrics_rf = evaluate_model("Random Forest Baseline", y_test, preds_rf, probs_rf, latency_rf)
    print(f"RF       -> Acc: {metrics_rf['accuracy']*100:.2f}%, Macro F1: {metrics_rf['macro_f1']:.4f}, AUC: {metrics_rf['roc_auc_macro']:.4f}, Latency: {latency_rf:.3f} ms")

    # --- 5-Fold Stratified Cross Validation on Main Model ---
    print("\n[5/5] Running 5-Fold Stratified Cross-Validation on 1D-CNN...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    cv_f1s = []

    # Run on stratified folds using 1D-CNN (1 epoch per fold for rapid verification)
    fold_i = 1
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        m_fold = Conv1DHeartbeatClassifier(num_classes=5)
        crit = nn.CrossEntropyLoss(weight=cw_tensor)
        opt = optim.Adam(m_fold.parameters(), lr=0.002)
        ld = DataLoader(TensorDataset(torch.from_numpy(X_tr).unsqueeze(1), torch.from_numpy(y_tr)), batch_size=256, shuffle=True)
        m_fold.train()
        for bx, by in ld:
            opt.zero_grad()
            l = crit(m_fold(bx), by)
            l.backward()
            opt.step()

        p_va, _ = predict_1d_cnn(m_fold, X_va)
        f_acc = accuracy_score(y_va, p_va)
        _, _, f_f1, _ = precision_recall_fscore_support(y_va, p_va, average='macro', zero_division=0)
        cv_accs.append(float(f_acc))
        cv_f1s.append(float(f_f1))
        print(f"  Fold {fold_i}: Accuracy = {f_acc*100:.2f}%, Macro F1 = {f_f1:.4f}")
        fold_i += 1

    cv_results = {
        "num_folds": 5,
        "fold_accuracies": cv_accs,
        "fold_macro_f1s": cv_f1s,
        "mean_accuracy": float(np.mean(cv_accs)),
        "std_accuracy": float(np.std(cv_accs)),
        "mean_macro_f1": float(np.mean(cv_f1s)),
        "std_macro_f1": float(np.std(cv_f1s))
    }
    print(f"5-Fold CV Mean Acc: {cv_results['mean_accuracy']*100:.2f}% +/- {cv_results['std_accuracy']*100:.2f}%, Mean Macro F1: {cv_results['mean_macro_f1']:.4f} +/- {cv_results['std_macro_f1']:.4f}")

    # --- Save Consolidated Metrics JSON ---
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

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved reports/metrics_results.json successfully!")

    # --- Generate Publication Quality Figures ---
    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Confusion Matrix All Models (Counts and Row Normalized)
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), dpi=150)
    models_cm = [
        ("1D-CNN (Proposed)", metrics_cnn),
        ("NumPy Neural Network", metrics_np),
        ("Random Forest Baseline", metrics_rf)
    ]
    for idx, (title, met) in enumerate(models_cm):
        cm_c = np.array(met["confusion_matrix_counts"])
        cm_n = np.array(met["confusion_matrix_norm"])

        sns.heatmap(cm_c, annot=True, fmt='d', cmap='Blues', ax=axes[idx, 0],
                    xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, cbar=False)
        axes[idx, 0].set_title(f"{title} - Raw Counts", fontsize=11, fontweight='bold')
        axes[idx, 0].set_xlabel("Predicted")
        axes[idx, 0].set_ylabel("True")

        sns.heatmap(cm_n, annot=True, fmt='.3f', cmap='Greens', ax=axes[idx, 1],
                    xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, cbar=False)
        axes[idx, 1].set_title(f"{title} - Row-Normalised (Recall)", fontsize=11, fontweight='bold')
        axes[idx, 1].set_xlabel("Predicted")
        axes[idx, 1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_confusion_matrices_all.png"))
    plt.close()

    # Also single primary 04_confusion_matrix.png
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(np.array(metrics_cnn["confusion_matrix_counts"]), annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, ax=ax)
    ax.set_title("CardioPulse 1D-CNN Confusion Matrix (Test Set)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Diagnostic Label")
    ax.set_ylabel("Ground Truth Diagnostic Label")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_confusion_matrix.png"))
    plt.close()

    # 2. Per-Class ROC Curves
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
    plt.savefig(os.path.join(figures_dir, "06_roc_curves_1d_cnn.png"))
    plt.close()

    # 3. Model Benchmark Comparison
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    m_names = ["NumPy Neural Net", "1D-CNN (Proposed)", "Random Forest"]
    accs = [metrics_np["accuracy"]*100, metrics_cnn["accuracy"]*100, metrics_rf["accuracy"]*100]
    mf1s = [metrics_np["macro_f1"]*100, metrics_cnn["macro_f1"]*100, metrics_rf["macro_f1"]*100]
    wf1s = [metrics_np["weighted_f1"]*100, metrics_cnn["weighted_f1"]*100, metrics_rf["weighted_f1"]*100]
    aucs = [metrics_np["roc_auc_macro"]*100, metrics_cnn["roc_auc_macro"]*100, metrics_rf["roc_auc_macro"]*100]
    x = np.arange(len(m_names))
    w = 0.18
    ax.bar(x - 1.5*w, accs, w, label='Accuracy (%)', color='#3B82F6')
    ax.bar(x - 0.5*w, mf1s, w, label='Macro F1 (%)', color='#10B981')
    ax.bar(x + 0.5*w, wf1s, w, label='Weighted F1 (%)', color='#F59E0B')
    ax.bar(x + 1.5*w, aucs, w, label='Macro ROC-AUC (%)', color='#8B5CF6')
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("CardioPulse Model Benchmark on MIT-BIH Test Set (21,892 Beats)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(m_names, fontweight='bold')
    ax.set_ylim([70, 102])
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "07_model_benchmark_comparison.png"))
    plt.close()

    print("All figures saved to reports/figures/.")

if __name__ == "__main__":
    main()
