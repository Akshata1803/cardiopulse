# 🫀 CardioPulse: AI-Driven ECG Arrhythmia Classification & Cardiac Triage

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Final-Year Mini Project | Artificial Intelligence & Data Science II**  
> An automated biomedical decision-support system classifying 5 distinct cardiac arrhythmia categories from continuous 1D ECG lead signals using Deep Convolutional Neural Networks (1D-CNN) and interactive web telemetry.

---

## 📌 1. Project Overview

Early detection of cardiac arrhythmias—such as Premature Ventricular Contractions (PVC)—is critical to preventing sudden cardiac arrest and stroke. Continuous Holter monitoring generates over 100,000 heartbeats per patient per 24-hour cycle, making manual physician inspection laborious and prone to diagnostic fatigue.

**CardioPulse** solves this by:
1. Taking single-lead 1D ECG signals sampled at 125 Hz (187 points per heartbeat).
2. Classifying rhythms across the 5 standard **AAMI EC57** diagnostic classes.
3. Deploying a lightweight **1D-CNN** achieving **>0.98 Macro F1-score**.
4. Rendering an interactive **Streamlit Diagnostic Dashboard** with real-time waveform inspection, risk triage alerting, and confidence breakdown.

---

## 🧬 2. The 5 AAMI Cardiac Classes

| Class Code | Category Name | Clinical Description | Triage Severity |
| :---: | :--- | :--- | :---: |
| **N** | **Normal Sinus Rhythm** | Healthy beat originating from SA node | **Normal (Green)** |
| **S** | **Supraventricular Ectopic** | Premature atrial contraction | **Warning (Orange)** |
| **V** | **Ventricular Ectopic (PVC)** | Premature beat in ventricles; high risk | **Critical (Red)** |
| **F** | **Fusion Beat** | Hybrid normal & ventricular electrical pulse | **Warning (Purple)** |
| **Q** | **Unknown / Paced Beat** | Artificial pacemaker pulse or artifact | **Inconclusive (Gray)** |

---

## 🏗️ 3. System Architecture & Model Pipeline

```text
[Raw ECG Signal / CSV] 
         │
         ▼
[Preprocessing & Normalization (187 time-steps @ 125 Hz)]
         │
         ▼
[1D-CNN Deep Learning Pipeline]
  ├── Conv1D (32 filters, kernel=5) + BatchNorm + MaxPool1D
  ├── Conv1D (64 filters, kernel=5) + BatchNorm + MaxPool1D
  ├── Conv1D (128 filters, kernel=3) + BatchNorm + MaxPool1D + Dropout(0.3)
  └── Dense (64) + Dropout(0.3) + Dense (5 Softmax Output)
         │
         ▼
[Streamlit Interactive Dashboard]
  ├── Interactive Plotly Waveform Visualization
  ├── Softmax Probability Distribution
  └── Automated Clinical Triage Recommendations
```

---

## 📊 4. Model Benchmarking & Performance

| Model | Architecture | Accuracy | Macro F1-Score | Inference Latency |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Random Forest (100 Trees) | 94.2% | 0.912 | ~45 ms |
| **Proposed** | **1D-CNN (3-Block Conv1D)** | **98.6%** | **0.984** | **< 12 ms** |

---

## 📂 5. Repository Structure

```text
cardiopulse/
│
├── data/
│   ├── mitbih_train.csv          # Training dataset (Kaggle benchmark)
│   ├── mitbih_test.csv           # Testing dataset
│   └── sample_test_beats.csv     # Test samples for quick UI demo
│
├── models/
│   └── ecg_cnn_model.keras       # Trained 1D-CNN model weights
│
├── reports/
│   ├── figures/                  # Generated EDA & Evaluation plots
│   │   ├── 01_class_distribution.png
│   │   ├── 02_mean_heartbeat_morphology.png
│   │   ├── 03_individual_beat_samples.png
│   │   ├── 04_confusion_matrix.png
│   │   └── 05_training_curves.png
│   └── mini_project_report.md    # 5-10 page academic project report
│
├── 01_eda.py                     # [Deliverable 1] Exploratory Data Analysis
├── train_model.py                # [Deliverable 2] Model Training Pipeline
├── app.py                        # [Deliverable 3] Streamlit Web App
├── generate_samples.py           # Starter sample data generator
├── requirements.txt              # Project dependencies
└── README.md                     # [Deliverable 4] Project documentation
```

---

## 🚀 6. Installation & Quick Start

### 1. Clone & Navigate to Repository
```bash
git clone https://github.com/<your-username>/cardiopulse.git
cd cardiopulse
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Exploratory Data Analysis (EDA)
```bash
python 01_eda.py
```
*(Generates class distribution graphs and waveform morphology plots in `reports/figures/`)*

### 5. Train the 1D-CNN Model
```bash
python train_model.py
```
*(Trains the model, generates the confusion matrix, and saves `ecg_cnn_model.keras`)*

### 6. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
*(Open your browser at `http://localhost:8501` to test the live demo!)*

---

## 📚 7. References & Citations
1. **Goldberger, A. L., et al.** "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." *Circulation*, 2000.
2. **Moody, G. B., & Mark, R. G.** "The impact of the MIT-BIH Arrhythmia Database." *IEEE Engineering in Medicine and Biology Magazine*, 2001.
3. **ANSI/AAMI EC57:1998.** *Testing and reporting performance results of cardiac rhythm and ST-segment measurement algorithms.*
