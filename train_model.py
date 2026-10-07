"""
train_model.py - Deliverable 2: Model Training & Evaluation Pipeline
Constructs and trains a 2-Layer Deep Neural Classifier (with ReLU, Softmax & Cross-Entropy)
in pure NumPy. Generates the Confusion Matrix, Training Loss Curve, and saves model weights.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular / PVC (V)",
    "Fusion (F)",
    "Unknown / Paced (Q)"
]
CLASS_SHORT = ["N", "S", "V", "F", "Q"]

def load_ecg_csv(filepath):
    if not os.path.exists(filepath):
        print(f"[!] Dataset missing at {filepath}. Generating starter data...")
        import generate_samples
        generate_samples.create_datasets()

    rows = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append([float(val) for val in row])
    data = np.array(rows)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class NeuralECGClassifier:
    """
    Lightweight, high-speed 2-Layer Neural Classifier implemented in NumPy.
    Architecture: 187 Inputs -> 96 Hidden (ReLU) -> 5 Outputs (Softmax).
    """
    def __init__(self, input_dim=187, hidden_dim=96, num_classes=5, lr=0.01):
        np.random.seed(42)
        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, num_classes))
        self.lr = lr
        self.loss_history = []

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.probs = softmax(self.Z2)
        return self.probs

    def fit(self, X, y, epochs=50, batch_size=64):
        num_samples = X.shape[0]
        num_classes = 5

        # One-hot encode targets
        Y_one_hot = np.zeros((num_samples, num_classes))
        Y_one_hot[np.arange(num_samples), y] = 1.0

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y_one_hot[indices]

            for i in range(0, num_samples, batch_size):
                xb = X_shuffled[i:i+batch_size]
                yb = Y_shuffled[i:i+batch_size]
                m = xb.shape[0]

                # Forward
                probs = self.forward(xb)

                # Backward
                dZ2 = (probs - yb) / m
                dW2 = np.dot(self.A1.T, dZ2)
                db2 = np.sum(dZ2, axis=0, keepdims=True)

                dA1 = np.dot(dZ2, self.W2.T)
                dZ1 = dA1 * (self.Z1 > 0)
                dW1 = np.dot(xb.T, dZ1)
                db1 = np.sum(dZ1, axis=0, keepdims=True)

                # Gradient descent step
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            # Compute epoch loss
            full_probs = self.forward(X)
            loss = -np.mean(np.log(full_probs[np.arange(num_samples), y] + 1e-9))
            self.loss_history.append(loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                preds = np.argmax(full_probs, axis=1)
                acc = np.mean(preds == y) * 100
                print(f"    Epoch {epoch+1:2d}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.2f}%")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.forward(X)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"[OK] Model weights saved to: {filepath}")

    def load(self, filepath):
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

def compute_metrics(y_true, y_pred):
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracies = np.sum(y_true == y_pred) / len(y_true)
    precisions, recalls, f1s = [], [], []

    print("\n[3] Class-Wise Performance Metrics:")
    print(f"    {'Category':<26} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("    " + "-" * 56)

    for i in range(5):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        print(f"    {CLASS_NAMES[i]:<26} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")

    macro_f1 = np.mean(f1s)
    print("    " + "-" * 56)
    print(f"    {'Macro Average':<26} {np.mean(precisions):<10.4f} {np.mean(recalls):<10.4f} {macro_f1:<10.4f}")
    return cm, accuracies, macro_f1

def train_and_evaluate(models_dir="models", figures_dir="reports/figures"):
    print("=" * 60)
    print("      CARDIOPULSE: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    X_train, y_train = load_ecg_csv("data/mitbih_train.csv")
    X_test, y_test = load_ecg_csv("data/mitbih_test.csv")
    print(f"\n[1] Training samples: {X_train.shape[0]:,} | Test samples: {X_test.shape[0]:,}")

    print("\n[2] Training Deep Neural ECG Classifier...")
    model = NeuralECGClassifier(input_dim=187, hidden_dim=96, num_classes=5, lr=0.08)
    model.fit(X_train, y_train, epochs=50, batch_size=64)

    # Predictions & Evaluation
    y_preds = model.predict(X_test)
    cm, acc, macro_f1 = compute_metrics(y_test, y_preds)

    print("\n" + "=" * 60)
    print(f"    FINAL TEST ACCURACY: {acc * 100:.2f}% | MACRO F1-SCORE: {macro_f1:.4f}")
    print("=" * 60)

    # -------------------------------------------------------------
    # Plot 1: Confusion Matrix Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(5),
        yticks=np.arange(5),
        xticklabels=CLASS_SHORT,
        yticklabels=CLASS_SHORT,
        title="CardioPulse Arrhythmia Confusion Matrix",
        ylabel="Ground Truth Diagnostic Label",
        xlabel="Model Predicted Diagnostic Label"
    )

    thresh = cm.max() / 2.
    for i in range(5):
        for j in range(5):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")

    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "04_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"\n[OK] Saved Confusion Matrix to: {cm_path}")

    # -------------------------------------------------------------
    # Plot 2: Loss Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(model.loss_history, label="Training Cross-Entropy Loss", color="#EF4444", linewidth=2)
    ax.set_title("Deep Neural Model Convergence (Loss Curve)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch Iteration", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    loss_path = os.path.join(figures_dir, "05_training_curves.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"[OK] Saved Training Loss Curve to: {loss_path}")

    # -------------------------------------------------------------
    # Save the Trained Model
    # -------------------------------------------------------------
    model_save_path = os.path.join(models_dir, "cardio_neural_weights.npz")
    model.save(model_save_path)

    print("\n" + "=" * 60)
    print("      MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate()
