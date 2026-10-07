"""
01_eda.py - Deliverable 1: Exploratory Data Analysis (EDA)
Loads ECG data using numpy, checks for missing values, plots class distribution,
and visualizes the average heartbeat morphology across all 5 AAMI classes.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Set clean scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Class labels and metadata
CLASS_LABELS = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular / PVC (V)",
    3: "Fusion (F)",
    4: "Unknown / Paced (Q)"
}

CLASS_COLORS = {
    0: "#10B981",  # Green
    1: "#F59E0B",  # Orange
    2: "#EF4444",  # Red
    3: "#8B5CF6",  # Purple
    4: "#6B7280"   # Gray
}

def load_ecg_csv(data_path):
    rows = []
    with open(data_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append([float(val) for val in row])
    data = np.array(rows)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

def run_eda(data_path="data/mitbih_train.csv", output_dir="reports/figures"):
    print("=" * 60)
    print("      CARDIOPULSE: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"[!] File not found at {data_path}. Running starter data generator first...")
        import generate_samples
        generate_samples.create_datasets()

    print(f"\n[1] Loading dataset from: {data_path}")
    X, y = load_ecg_csv(data_path)
    print(f"    Dataset Dimensions: {X.shape[0]:,} rows (heartbeats), {X.shape[1] + 1} columns")

    # 1. Check Missing / Null values
    null_count = np.isnan(X).sum() + np.isnan(y).sum()
    print(f"    Missing / NaN values: {null_count}")

    # 2. Class Distribution Summary
    print("\n[2] Class Distribution Summary:")
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    total_samples = len(y)

    for class_id in range(5):
        count = class_counts.get(class_id, 0)
        pct = (count / total_samples) * 100
        print(f"    Class {class_id} - {CLASS_LABELS[class_id]:<25}: {count:6,} ({pct:.2f}%)")

    # -------------------------------------------------------------
    # Plot 1: Class Distribution Bar Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    classes = [c for c in range(5) if c in class_counts]
    bars = ax.bar(
        [CLASS_LABELS[c] for c in classes],
        [class_counts[c] for c in classes],
        color=[CLASS_COLORS[c] for c in classes],
        edgecolor="black",
        linewidth=0.8
    )
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title("AAMI Cardiac Arrhythmia Class Distribution", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Heartbeat Category", fontsize=10)
    ax.set_ylabel("Total Beat Count", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "01_class_distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()
    print(f"\n[OK] Saved plot: {dist_plot_path}")

    # -------------------------------------------------------------
    # Plot 2: Average Heartbeat Morphology Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    time_points = np.arange(X.shape[1])

    for class_id in range(5):
        if class_id in class_counts:
            class_beats = X[y == class_id]
            mean_beat = np.mean(class_beats, axis=0)
            std_beat = np.std(class_beats, axis=0)
            color = CLASS_COLORS[class_id]
            label = CLASS_LABELS[class_id]

            ax.plot(time_points, mean_beat, label=label, color=color, linewidth=2)
            ax.fill_between(time_points, mean_beat - std_beat, mean_beat + std_beat, color=color, alpha=0.1)

    ax.set_title("Mean ECG Morphology Across Arrhythmia Classes (with +/- 1 sigma variance)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Time Sample (at 125 Hz)", fontsize=10)
    ax.set_ylabel("Normalized Voltage Amplitude [0, 1]", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    mean_plot_path = os.path.join(output_dir, "02_mean_heartbeat_morphology.png")
    plt.savefig(mean_plot_path)
    plt.close()
    print(f"[OK] Saved plot: {mean_plot_path}")

    # -------------------------------------------------------------
    # Plot 3: Subplot of 5 Distinct Single Heartbeat Samples
    # -------------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True, dpi=150)
    fig.suptitle("Representative Individual Heartbeat Waveforms by Class", fontsize=13, fontweight='bold')

    for class_id in range(5):
        matching_indices = np.where(y == class_id)[0]
        if len(matching_indices) > 0:
            sample_beat = X[matching_indices[0]]
            ax = axes[class_id]
            ax.plot(time_points, sample_beat, color=CLASS_COLORS[class_id], linewidth=1.8)
            ax.set_ylabel("Amp [0-1]", fontsize=8)
            ax.set_title(f"Class {class_id}: {CLASS_LABELS[class_id]}", fontsize=9, loc='left', color=CLASS_COLORS[class_id], fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)

    axes[-1].set_xlabel("Time Points (187 Samples @ 125 Hz)", fontsize=10)
    plt.tight_layout()
    sample_plot_path = os.path.join(output_dir, "03_individual_beat_samples.png")
    plt.savefig(sample_plot_path)
    plt.close()
    print(f"[OK] Saved plot: {sample_plot_path}")

    print("\n" + "=" * 60)
    print("      EDA COMPLETED SUCCESSFULLY!")
    print(f"      All figures saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
