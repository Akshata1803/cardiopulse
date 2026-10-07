"""
generate_samples.py - Generates sample ECG data for all 5 AAMI classes.
Uses pure Python (math and csv) so it runs with zero dependency issues.
"""

import math
import random
import csv
import os

NUM_POINTS = 187  # Standard MIT-BIH sequence length

CLASS_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular Ectopic (S)",
    2: "Ventricular Ectopic / PVC (V)",
    3: "Fusion Beat (F)",
    4: "Unknown / Paced (Q)"
}

def generate_ecg_beat(class_id: int, noise_level: float = 0.02):
    """Generates a realistic 187-point synthetic ECG heartbeat for class_id (0 to 4)."""
    signal = [0.0] * NUM_POINTS

    for i in range(NUM_POINTS):
        t = i / (NUM_POINTS - 1)
        val = 0.0

        if class_id == 0:  # Normal Sinus Rhythm (N)
            val += 0.15 * math.exp(-((t - 0.20) ** 2) / (2 * (0.025 ** 2)))   # P-wave
            val -= 0.10 * math.exp(-((t - 0.32) ** 2) / (2 * (0.010 ** 2)))   # Q-wave
            val += 1.00 * math.exp(-((t - 0.35) ** 2) / (2 * (0.015 ** 2)))   # R-peak
            val -= 0.25 * math.exp(-((t - 0.38) ** 2) / (2 * (0.012 ** 2)))   # S-wave
            val += 0.30 * math.exp(-((t - 0.55) ** 2) / (2 * (0.050 ** 2)))   # T-wave

        elif class_id == 1:  # Supraventricular Ectopic (S)
            val += 0.10 * math.sin(2 * math.pi * 5 * t) * math.exp(-((t - 0.18) ** 2) / (2 * (0.03 ** 2)))
            val += 0.85 * math.exp(-((t - 0.30) ** 2) / (2 * (0.014 ** 2)))
            val -= 0.20 * math.exp(-((t - 0.33) ** 2) / (2 * (0.012 ** 2)))
            val += 0.25 * math.exp(-((t - 0.48) ** 2) / (2 * (0.045 ** 2)))

        elif class_id == 2:  # Ventricular Ectopic (PVC / V) - High Risk
            val += 1.15 * math.exp(-((t - 0.28) ** 2) / (2 * (0.045 ** 2)))   # Wide R peak
            val -= 0.70 * math.exp(-((t - 0.38) ** 2) / (2 * (0.035 ** 2)))   # Deep S dip
            val -= 0.35 * math.exp(-((t - 0.60) ** 2) / (2 * (0.070 ** 2)))   # Inverted T wave

        elif class_id == 3:  # Fusion Beat (F)
            val += 0.10 * math.exp(-((t - 0.15) ** 2) / (2 * (0.020 ** 2)))
            val += 0.75 * math.exp(-((t - 0.32) ** 2) / (2 * (0.030 ** 2)))
            val -= 0.40 * math.exp(-((t - 0.40) ** 2) / (2 * (0.025 ** 2)))
            val += 0.20 * math.exp(-((t - 0.55) ** 2) / (2 * (0.045 ** 2)))

        else:  # Unknown / Paced Beat (Q)
            val += 0.95 * math.exp(-((t - 0.25) ** 2) / (2 * (0.005 ** 2)))   # Pacemaker spike
            val += 0.50 * math.exp(-((t - 0.35) ** 2) / (2 * (0.050 ** 2)))
            val -= 0.20 * math.exp(-((t - 0.45) ** 2) / (2 * (0.040 ** 2)))

        # Add Gaussian noise
        noise = random.gauss(0, noise_level)
        signal[i] = val + noise

    # Min-Max normalize to [0.0, 1.0]
    s_min = min(signal)
    s_max = max(signal)
    diff = s_max - s_min
    if diff > 1e-6:
        signal = [round((x - s_min) / diff, 6) for x in signal]

    return signal

def write_csv(filepath, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"[OK] Successfully created: {filepath} ({len(rows)} samples)")

def create_datasets(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    # 1. Training set: 5,000 samples (1,000 per class)
    train_rows = []
    for class_id in range(5):
        for _ in range(1000):
            beat = generate_ecg_beat(class_id)
            beat.append(float(class_id))
            train_rows.append(beat)
    write_csv(os.path.join(data_dir, "mitbih_train.csv"), train_rows)

    # 2. Testing set: 1,000 samples (200 per class)
    test_rows = []
    for class_id in range(5):
        for _ in range(200):
            beat = generate_ecg_beat(class_id)
            beat.append(float(class_id))
            test_rows.append(beat)
    write_csv(os.path.join(data_dir, "mitbih_test.csv"), test_rows)

    # 3. Compact sample beats: 10 samples (2 per class for quick UI testing)
    sample_rows = []
    for class_id in range(5):
        for _ in range(2):
            beat = generate_ecg_beat(class_id)
            beat.append(float(class_id))
            sample_rows.append(beat)
    write_csv(os.path.join(data_dir, "sample_test_beats.csv"), sample_rows)

if __name__ == "__main__":
    create_datasets()
