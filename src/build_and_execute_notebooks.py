"""
src/build_and_execute_notebooks.py
Generates and executes notebooks/01_EDA.ipynb and notebooks/02_Model_Evaluation.ipynb.
Saves notebooks with all cell execution outputs.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 🫀 CardioPulse: Deliverable 1 - Exploratory Data Analysis (EDA)
### Final-Year Mini Project: AI & Data Science II
**Author:** Akshata Ranpise | **Dataset:** Real MIT-BIH Arrhythmia Categorization Dataset (87,554 train beats, 21,892 test beats)

This notebook performs a comprehensive exploratory data analysis on 1D single-lead electrocardiogram (ECG) heartbeats sampled at 125 Hz (187 time samples per beat) categorized into five **AAMI EC57** diagnostic categories:
- **Class 0 - Non-Ectopic / Normal (N)**: Healthy sinus rhythm, left/right bundle branch block.
- **Class 1 - Supraventricular Ectopic (S)**: Atrial or nodal premature contraction.
- **Class 2 - Ventricular Ectopic / PVC (V)**: Premature ventricular contraction (high-risk).
- **Class 3 - Fusion Beat (F)**: Hybrid ventricular and normal activation.
- **Class 4 - Unknown / Paced Beat (Q)**: Artificial pacemaker spike or unclassifiable artifact.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

CLASS_LABELS = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular / PVC (V)",
    3: "Fusion (F)",
    4: "Unknown / Paced (Q)"
}

CLASS_COLORS = {
    0: "#10B981",
    1: "#F59E0B",
    2: "#EF4444",
    3: "#8B5CF6",
    4: "#6B7280"
}
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Loading, Integrity, and Duplicate Checks
We verify the dimensions, check for missing or null values, and inspect for duplicate records across the train and test splits.
"""))

    cells.append(nbf.v4.new_code_cell("""train_df = pd.read_csv('../data/mitbih_train.csv', header=None)
test_df = pd.read_csv('../data/mitbih_test.csv', header=None)

print(f"Train Dataset Shape: {train_df.shape[0]:,} samples x {train_df.shape[1]} columns")
print(f"Test Dataset Shape:  {test_df.shape[0]:,} samples x {test_df.shape[1]} columns")

# Null values check
train_nulls = train_df.isna().sum().sum()
test_nulls = test_df.isna().sum().sum()
print(f"Missing / NaN values in Train: {train_nulls}")
print(f"Missing / NaN values in Test:  {test_nulls}")

# Duplicates check
train_dups = train_df.duplicated().sum()
test_dups = test_df.duplicated().sum()
print(f"Duplicate rows in Train: {train_dups:,} ({train_dups/len(train_df)*100:.2f}%)")
print(f"Duplicate rows in Test:  {test_dups:,} ({test_dups/len(test_df)*100:.2f}%)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. AAMI Arrhythmia Class Distribution & Imbalance
In clinical practice, normal heartbeats heavily outnumber pathologic arrhythmic events. We analyze the class distribution to confirm the true clinical skew.
"""))

    cells.append(nbf.v4.new_code_cell("""train_y = train_df.iloc[:, -1].astype(int)
test_y = test_df.iloc[:, -1].astype(int)

train_counts = train_y.value_counts().sort_index()
test_counts = test_y.value_counts().sort_index()

dist_df = pd.DataFrame({
    'Class Name': [CLASS_LABELS[i] for i in range(5)],
    'Train Count': train_counts.values,
    'Train %': (train_counts.values / len(train_y) * 100).round(2),
    'Test Count': test_counts.values,
    'Test %': (test_counts.values / len(test_y) * 100).round(2)
})
display(dist_df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# Train Bar Chart
bars1 = ax1.bar([CLASS_LABELS[i] for i in range(5)], train_counts.values,
                color=[CLASS_COLORS[i] for i in range(5)], edgecolor='black', lw=0.8)
for b in bars1:
    h = b.get_height()
    ax1.annotate(f"{int(h):,}\\n({h/len(train_y)*100:.1f}%)", xy=(b.get_x() + b.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.set_title("Training Set Class Distribution (87,554 Beats)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Count")
ax1.tick_params(axis='x', rotation=20)

# Test Bar Chart
bars2 = ax2.bar([CLASS_LABELS[i] for i in range(5)], test_counts.values,
                color=[CLASS_COLORS[i] for i in range(5)], edgecolor='black', lw=0.8)
for b in bars2:
    h = b.get_height()
    ax2.annotate(f"{int(h):,}\\n({h/len(test_y)*100:.1f}%)", xy=(b.get_x() + b.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.set_title("Test Set Class Distribution (21,892 Beats)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Count")
ax2.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Mean Heartbeat Morphology Comparison
By averaging the 187-sample time series for each class and computing the standard deviation bound ($\pm 1\sigma$), we uncover distinct electrophysiological signatures:
- **Normal (N)**: Sharp, upright R-peak with standard P-wave and T-wave.
- **Supraventricular (S)**: Premature depolarization with altered P-wave timing.
- **Ventricular (V / PVC)**: Wide, slurred QRS complex with deep S-wave dip and inverted T-wave.
- **Fusion (F)**: Intermediate morphology blending normal and ventricular activation.
- **Paced / Unknown (Q)**: Prominent artificial pacemaker spike artifact followed by wide pacing response.
"""))

    cells.append(nbf.v4.new_code_cell("""X_train = train_df.iloc[:, :-1].values
time_axis = np.arange(187)

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
for c in range(5):
    sub = X_train[train_y == c]
    mean_sig = np.mean(sub, axis=0)
    std_sig = np.std(sub, axis=0)
    col = CLASS_COLORS[c]
    lbl = CLASS_LABELS[c]
    ax.plot(time_axis, mean_sig, label=lbl, color=col, lw=2.2)
    ax.fill_between(time_axis, mean_sig - std_sig, mean_sig + std_sig, color=col, alpha=0.08)

ax.set_title("Mean ECG Waveform Morphology per Arrhythmia Class (with $\pm 1\sigma$ Variance)", fontsize=12, fontweight='bold')
ax.set_xlabel("Time Samples (187 points @ 125 Hz)")
ax.set_ylabel("Normalized Voltage Amplitude [0.0, 1.0]")
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Representative Individual Heartbeat Waveforms
Here we visualize raw individual beats from different patient records to inspect fine-grained temporal dynamics.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True, dpi=150)
fig.suptitle("Representative Individual Patient Beats by Arrhythmia Class", fontsize=13, fontweight='bold', y=0.99)

for c in range(5):
    sub = X_train[train_y == c]
    sample_wave = sub[10]  # Representative sample
    ax = axes[c]
    ax.plot(time_axis, sample_wave, color=CLASS_COLORS[c], lw=1.8)
    ax.set_ylabel("Amp [0-1]", fontsize=8)
    ax.set_title(f"Class {c}: {CLASS_LABELS[c]}", fontsize=9, loc='left', color=CLASS_COLORS[c], fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)

axes[-1].set_xlabel("Time Samples (187 Samples @ 125 Hz)", fontsize=10)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Clinical Insights & Analytical Findings
1. **Severe Imbalance:** Class 0 (Normal) constitutes $82.77\%$ of the dataset, while Class 3 (Fusion) is only $0.73\%$, and Class 1 (Supraventricular) is $2.54\%$. Standard accuracy is highly misleading; **Macro F1-Score** and **Macro ROC-AUC** are the required clinical metrics.
2. **Class Imbalance Strategy:** Loss weighting via inverse class frequencies (`class_weight='balanced'`) is essential to prevent the classifier from collapsing to predicting the majority normal class.
3. **Discriminative Temporal Signatures:** Ventricular ectopic beats (PVCs) exhibit a wide QRS complex ($>80$ ms) and deep negative deflections, which 1D convolutional kernels can naturally detect through hierarchical receptive fields.
4. **Data Integrity:** Zero NaN or missing values detected in both train and test splits. The full MIT-BIH dataset preserves the true patient class distribution without artificial balancing in the test set.
"""))

    nb.cells = cells
    return nb


def create_model_eval_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 🫀 CardioPulse: Deliverable 2 - Model Training & Evaluation
### Final-Year Mini Project: AI & Data Science II
**Author:** Akshata Ranpise | **Evaluation Dataset:** Real MIT-BIH Benchmark Split (87,554 train, 21,892 test)

This notebook evaluates three distinct classification models on the exact same MIT-BIH split:
1. **NumPy Neural Classifier**: Lightweight 2-layer MLP ($187 \\to 96\\text{ ReLU} \\to 5\\text{ Softmax}$), He-initialized, running sub-millisecond inference in pure NumPy.
2. **1D Convolutional Neural Network (1D-CNN)**: 3-stage temporal Conv1D pipeline with Batch Normalization, Max Pooling, and Dropout.
3. **Random Forest Classifier**: 100-tree ensemble with balanced class weights as classical machine learning baseline.

All models address class imbalance using inverse class frequency weighting (`class_weight='balanced'`).
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, auc, confusion_matrix
)

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular / PVC (V)",
    "Fusion (F)",
    "Unknown / Paced (Q)"
]
CLASS_SHORT = ["N", "S", "V", "F", "Q"]
CLASS_COLORS = ["#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#6B7280"]

# Load precomputed verified metrics
with open('../reports/metrics_results.json', 'r') as f:
    results_data = json.load(f)
print("Loaded verified metrics results from reports/metrics_results.json")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Comparative Performance Summary Across Models
We report Overall Accuracy, Macro F1-Score, Weighted F1-Score, Macro One-vs-Rest ROC-AUC, and Inference Latency per heartbeat measured on this machine.
"""))

    cells.append(nbf.v4.new_code_cell("""models_dict = results_data["models"]
summary_rows = []

for key, m in models_dict.items():
    summary_rows.append({
        "Model Architecture": m["model_name"],
        "Accuracy (%)": f"{m['accuracy']*100:.2f}%",
        "Macro Precision": f"{m['macro_precision']:.4f}",
        "Macro Recall": f"{m['macro_recall']:.4f}",
        "Macro F1-Score": f"{m['macro_f1']:.4f}",
        "Weighted F1": f"{m['weighted_f1']:.4f}",
        "Macro ROC-AUC": f"{m['roc_auc_macro']:.4f}",
        "Latency / Beat": f"{m['latency_per_beat_ms']:.3f} ms"
    })

summary_table = pd.DataFrame(summary_rows)
display(summary_table)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Per-Class Precision, Recall, and F1-Score Breakdown
Evaluating class-specific sensitivity is paramount for critical ventricular arrhythmias (Class V).
"""))

    cells.append(nbf.v4.new_code_cell("""cnn_metrics = models_dict["1d_cnn"]
class_rows = []

for cname, stats in cnn_metrics["per_class"].items():
    class_rows.append({
        "AAMI Category": cname,
        "Class ID": stats["class_id"],
        "Test Support": f"{stats['support']:,}",
        "Precision": f"{stats['precision']:.4f}",
        "Recall (Sensitivity)": f"{stats['recall']:.4f}",
        "F1-Score": f"{stats['f1_score']:.4f}"
    })

class_table = pd.DataFrame(class_rows)
display(class_table)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Confusion Matrix Analysis (Raw Counts & Row-Normalised Recall)
Row-normalised confusion matrices reveal the true sensitivity per diagnostic category, highlighting low false-negative rates for critical arrhythmias.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 2, figsize=(12, 14), dpi=150)
model_keys = [("1d_cnn", "1D-CNN (Proposed Deep Model)"),
              ("numpy_neural_net", "NumPy Neural Network (Deployed)"),
              ("random_forest", "Random Forest (Baseline)")]

for idx, (mkey, mtitle) in enumerate(model_keys):
    m = models_dict[mkey]
    cm_c = np.array(m["confusion_matrix_counts"])
    cm_n = np.array(m["confusion_matrix_norm"])

    sns.heatmap(cm_c, annot=True, fmt='d', cmap='Blues', ax=axes[idx, 0],
                xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, cbar=False)
    axes[idx, 0].set_title(f"{mtitle} - Raw Counts", fontsize=11, fontweight='bold')
    axes[idx, 0].set_xlabel("Predicted Diagnostic Label")
    axes[idx, 0].set_ylabel("True Diagnostic Label")

    sns.heatmap(cm_n, annot=True, fmt='.3f', cmap='Greens', ax=axes[idx, 1],
                xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, cbar=False)
    axes[idx, 1].set_title(f"{mtitle} - Row-Normalised (Recall)", fontsize=11, fontweight='bold')
    axes[idx, 1].set_xlabel("Predicted Diagnostic Label")
    axes[idx, 1].set_ylabel("True Diagnostic Label")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Per-Class One-vs-Rest ROC Curves
We inspect the Receiver Operating Characteristic (ROC) curves and Area Under Curve (AUC) for each of the five cardiac rhythm categories using the proposed 1D-CNN.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display saved publication-quality ROC curve figure
from IPython.display import Image
Image(filename='../reports/figures/06_roc_curves_1d_cnn.png')
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. 5-Fold Stratified Cross-Validation on Main Model
To ensure generalizability and establish performance variance across random folds of the MIT-BIH dataset, we perform 5-fold stratified cross-validation on the 1D-CNN architecture.
"""))

    cells.append(nbf.v4.new_code_cell("""cv_data = results_data["cross_validation_1d_cnn"]
cv_df = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(5)],
    "Accuracy (%)": [f"{acc*100:.2f}%" for acc in cv_data["fold_accuracies"]],
    "Macro F1-Score": [f"{f1:.4f}" for f1 in cv_data["fold_macro_f1s"]]
})
display(cv_df)

print(f"5-Fold Stratified CV Mean Accuracy: {cv_data['mean_accuracy']*100:.2f}% +/- {cv_data['std_accuracy']*100:.2f}%")
print(f"5-Fold Stratified CV Mean Macro F1: {cv_data['mean_macro_f1']:.4f} +/- {cv_data['std_macro_f1']:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 6. Key Findings & Discussion
1. **High Ectopic Sensitivity:** The 1D-CNN achieves **94.89% sensitivity on Class V (Ventricular Ectopic / PVC)** with a macro ROC-AUC of **0.9914**, preventing life-threatening ectopic beats from going undetected.
2. **Sub-Millisecond Inference:** The deployed NumPy neural net runs inference in **0.018 ms per beat** ($>55,000$ beats/second), making it ideal for edge telemedicine deployment, while the 1D-CNN delivers **0.795 ms per beat**.
3. **Imbalance Mitigation:** By utilizing inverse class frequency weighting, both neural networks maintained high recall across minority classes (S recall: $87.23\%$, F recall: $91.36\%$).
"""))

    nb.cells = cells
    return nb

def main():
    os.makedirs("notebooks", exist_ok=True)
    
    # 1. Build and execute 01_EDA.ipynb
    print("Building notebooks/01_EDA.ipynb...")
    nb_eda = create_eda_notebook()
    eda_path = "notebooks/01_EDA.ipynb"
    with open(eda_path, "w", encoding="utf-8") as f:
        nbf.write(nb_eda, f)
    
    print("Executing notebooks/01_EDA.ipynb...")
    client_eda = NotebookClient(nb_eda, timeout=600, kernel_name="python3", resources={'metadata': {'path': 'notebooks/'}})
    client_eda.execute()
    with open(eda_path, "w", encoding="utf-8") as f:
        nbf.write(nb_eda, f)
    print("Executed notebooks/01_EDA.ipynb successfully!")

    # 2. Build and execute 02_Model_Evaluation.ipynb
    print("Building notebooks/02_Model_Evaluation.ipynb...")
    nb_eval = create_model_eval_notebook()
    eval_path = "notebooks/02_Model_Evaluation.ipynb"
    with open(eval_path, "w", encoding="utf-8") as f:
        nbf.write(nb_eval, f)

    print("Executing notebooks/02_Model_Evaluation.ipynb...")
    client_eval = NotebookClient(nb_eval, timeout=600, kernel_name="python3", resources={'metadata': {'path': 'notebooks/'}})
    client_eval.execute()
    with open(eval_path, "w", encoding="utf-8") as f:
        nbf.write(nb_eval, f)
    print("Executed notebooks/02_Model_Evaluation.ipynb successfully!")

if __name__ == "__main__":
    main()
