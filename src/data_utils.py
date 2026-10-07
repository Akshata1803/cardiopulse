"""
data_utils.py - CardioPulse Data Utilities and Preprocessing
Provides dataset loading, synthetic demo data generation, and signal preprocessing.
"""

import numpy as np
import pandas as pd
import os

# Standard AAMI ECG Class Labels
CLASS_MAPPING = {
    0: {
        "code": "N",
        "name": "Normal Beat (Non-Ectopic)",
        "description": "Normal sinus rhythm originating from the SA node.",
        "status": "Normal",
        "color": "#10B981"  # Emerald Green
    },
    1: {
        "code": "S",
        "name": "Supraventricular Ectopic",
        "description": "Atrial or nodal premature beat originating above ventricles.",
        "status": "Warning",
        "color": "#F59E0B"  # Amber Orange
    },
    2: {
        "code": "V",
        "name": "Ventricular Ectopic (PVC)",
        "description": "Premature Ventricular Contraction. High-risk cardiac irregularity.",
        "status": "Critical",
        "color": "#EF4444"  # Red
    },
    3: {
        "code": "F",
        "name": "Fusion Beat",
        "description": "Combined electrical activation of ventricular and normal beats.",
        "status": "Warning",
        "color": "#8B5CF6"  # Purple
    },
    4: {
        "code": "Q",
        "name": "Unknown / Unclassified",
        "description": "Paced beat, unclassifiable or anomalous electrical artifact.",
        "status": "Inconclusive",
        "color": "#6B7280"  # Gray
    }
}

NUM_SAMPLES = 187  # Standard length of preprocessed MIT-BIH heartbeat


def generate_synthetic_heartbeat(class_id: int, noise_level: float = 0.02) -> np.ndarray:
    """
    Generates a realistic synthetic ECG heartbeat vector (187 points)
    mimicking the morphological waveform of each specific AAMI class.
    Used for instant testing and standalone UI demo.
    """
    t = np.linspace(0, 1, NUM_SAMPLES)
    signal = np.zeros(NUM_SAMPLES)

    if class_id == 0:  # Normal Sinus Rhythm (N)
        # P-wave: small early bump
        signal += 0.15 * np.exp(-((t - 0.20) ** 2) / (2 * (0.025 ** 2)))
        # Q-wave: slight negative deflection
        signal -= 0.10 * np.exp(-((t - 0.32) ** 2) / (2 * (0.010 ** 2)))
        # R-peak: sharp prominent spike
        signal += 1.00 * np.exp(-((t - 0.35) ** 2) / (2 * (0.015 ** 2)))
        # S-wave: negative dip
        signal -= 0.25 * np.exp(-((t - 0.38) ** 2) / (2 * (0.012 ** 2)))
        # T-wave: medium recovery wave
        signal += 0.30 * np.exp(-((t - 0.55) ** 2) / (2 * (0.050 ** 2)))

    elif class_id == 1:  # Supraventricular Ectopic (S)
        # Premature, inverted or biphasic P-wave, normal QRS
        signal += 0.10 * np.sin(2 * np.pi * 5 * t) * np.exp(-((t - 0.18) ** 2) / (2 * (0.03 ** 2)))
        signal += 0.85 * np.exp(-((t - 0.30) ** 2) / (2 * (0.014 ** 2)))
        signal -= 0.20 * np.exp(-((t - 0.33) ** 2) / (2 * (0.012 ** 2)))
        signal += 0.25 * np.exp(-((t - 0.48) ** 2) / (2 * (0.045 ** 2)))

    elif class_id == 2:  # Ventricular Ectopic / PVC (V)
        # Wide, bizarre QRS complex, no P-wave, inverted T-wave
        signal += 1.15 * np.exp(-((t - 0.28) ** 2) / (2 * (0.045 ** 2)))  # Wide R peak
        signal -= 0.70 * np.exp(-((t - 0.38) ** 2) / (2 * (0.035 ** 2)))  # Deep S dip
        signal -= 0.35 * np.exp(-((t - 0.60) ** 2) / (2 * (0.070 ** 2)))  # Inverted T wave

    elif class_id == 3:  # Fusion Beat (F)
        # Hybrid morph: intermediate duration QRS
        signal += 0.10 * np.exp(-((t - 0.15) ** 2) / (2 * (0.020 ** 2)))
        signal += 0.75 * np.exp(-((t - 0.32) ** 2) / (2 * (0.030 ** 2)))
        signal -= 0.40 * np.exp(-((t - 0.40) ** 2) / (2 * (0.025 ** 2)))
        signal += 0.20 * np.exp(-((t - 0.55) ** 2) / (2 * (0.045 ** 2)))

    else:  # Unknown / Paced Beat (Q)
        # Sharp pacemaker spike followed by broad artificial response
        signal += 0.95 * np.exp(-((t - 0.25) ** 2) / (2 * (0.005 ** 2)))  # Pacing spike
        signal += 0.50 * np.exp(-((t - 0.35) ** 2) / (2 * (0.050 ** 2)))
        signal -= 0.20 * np.exp(-((t - 0.45) ** 2) / (2 * (0.040 ** 2)))

    # Add subtle physiological baseline noise
    noise = np.random.normal(0, noise_level, NUM_SAMPLES)
    signal = signal + noise
    
    # Min-Max normalize to [0, 1] range as in MIT-BIH
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max > signal_min:
        signal = (signal - signal_min) / (signal_max - signal_min)

    return signal


def create_sample_dataset(output_dir: str = "data/samples"):
    """
    Creates ready-to-test sample CSV files for all 5 classes so the application
    runs immediately even before downloading the large dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples = []
    
    for class_id in range(5):
        # Generate 20 samples per class for quick testing
        for _ in range(20):
            beat = generate_synthetic_heartbeat(class_id)
            row = np.append(beat, float(class_id))
            samples.append(row)
            
    df = pd.DataFrame(samples)
    sample_file = os.path.join(output_dir, "sample_ecg_dataset.csv")
    df.to_csv(sample_file, index=False, header=False)
    print(f"[✓] Created sample dataset: {sample_file} ({df.shape[0]} records, {df.shape[1]} columns)")
    return sample_file


def load_dataset(csv_path: str):
    """
    Loads dataset from CSV file.
    Expects 188 columns (columns 0-186: features, column 187: label).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)
    return X, y
