# CARDIOPULSE: DEEP LEARNING-DRIVEN ECG ANOMALY DETECTION FOR REAL-TIME CARDIAC ARRHYTHMIA CLASSIFICATION

**A Mini-Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of**  
**Bachelor of Engineering in Information Technology | AI & Data Science II**

* **Candidate:** Akshata Ranpise (Roll No. 21)
* **Guide:** Prof. Tejal Rachh
* **Institution:** Atharva College of Engineering, Department of Information Technology, University of Mumbai
* **Academic Year:** 2026 – 2027

---

## ABSTRACT

Cardiovascular diseases (CVDs) cause an estimated 17.9 million deaths annually. Ambulatory Holter monitoring generates over 100,000 heartbeats per patient per day, overwhelming manual clinical review. In commercial ICU monitors, false alarm rates exceed 70%, precipitating alarm fatigue among healthcare workers.

In this work, we present **CardioPulse**, an automated clinical decision-support pipeline that classifies single-lead ECG heartbeats into the five standard Association for the Advancement of Medical Instrumentation (AAMI EC57) categories using the benchmark **PhysioNet MIT-BIH Arrhythmia Database** (87,554 training beats, 21,892 test beats). To counter extreme clinical class imbalance (82.8% Normal vs. 0.7% Fusion), we implement inverse class frequency weighting across all models.

We evaluate three architectures on the exact same unmanipulated 21,892-beat test split: a lightweight 2-layer NumPy Neural Network (187 → 96 ReLU → 5 Softmax), a 3-block 1D Convolutional Neural Network (1D-CNN), and a 100-tree Random Forest baseline. The proposed 1D-CNN achieves an overall Accuracy of **94.28%**, a Macro F1-Score of **0.7810**, a Weighted F1-Score of **0.9514**, and a Macro ROC-AUC of **0.9914**, with an exceptional **94.89% recall on critical Ventricular Ectopic beats (PVCs)** at **0.795 ms latency** per beat. The deployed NumPy classifier achieves **92.36% accuracy** with sub-millisecond inference (**0.018 ms**) and a 75 KB footprint. The system is deployed via an interactive Streamlit dashboard featuring live waveform rendering, calibrated millisecond axes, R-peak annotations, 5-class probability distributions, and automated clinical triage recommendations.

---

## 1. INTRODUCTION & PROBLEM STATEMENT

An electrocardiogram (ECG) measures the electrical potentials of the myocardium. A typical cardiac cycle consists of a P-wave (atrial depolarization), a QRS complex (ventricular depolarization), and a T-wave (ventricular repolarization). Morphological distortions denote severe arrhythmic events. For example, Premature Ventricular Contractions (PVCs) feature broad, bizarre QRS complexes without a preceding P-wave, capable of escalating into life-threatening Ventricular Fibrillation.

Continuous ICU telemetry and wearable Holters generate enormous volumes of 125 Hz signal data. Manual inspection is unfeasible. Commercial heuristic alarms trigger over 70% false positives, creating alarm fatigue. CardioPulse solves this by operating directly on normalized 187-sample 1D vectors, avoiding computationally expensive 2D image spectrogram conversion, and delivering instant triage stratification.

---

## 2. LITERATURE SURVEY

| Paper Title | Author & Year | Summary & Relevance |
| :--- | :--- | :--- |
| **PhysioBank, PhysioToolkit, and PhysioNet: Complex Physiologic Signals** | Goldberger et al. (2000) | Established the PhysioNet repository hosting the MIT-BIH Arrhythmia Database benchmark. |
| **The Impact of the MIT-BIH Arrhythmia Database** | Moody & Mark (2001) | Established standardized diagnostic protocols for automated arrhythmia detection. |
| **Real-Time Patient-Specific ECG Classification by 1-D CNNs** | Kiranyaz et al. (2016) | Demonstrated 1D-CNNs classify raw ECG waveforms directly with minimal computational latency. |
| **Cardiologist-Level Arrhythmia Detection with CNNs** | Rajpurkar et al. (2017) | Showed deep convolutional neural networks match certified cardiologists on single-lead ambulatory data. |
| **ECG Heartbeat Classification: A Deep Transferable Representation** | Kachuee et al. (2018) | Established the 187-sample, 125 Hz preprocessed 5-class AAMI benchmark dataset used in this project. |

---

## 3. TECHNOLOGY STACK & SYSTEM ARCHITECTURE

* **Core Stack:** Python 3.12 / 3.11, PyTorch 2.14+, NumPy 2.4+, Scikit-Learn 1.9+, Pandas, Matplotlib, Seaborn, Plotly 7.1, Streamlit 1.65+.
* **Dataflow Pipeline:**
  1. *Signal Ingestion:* 187-sample vector (Lead II at 125 Hz).
  2. *Normalization:* Min-Max scaling to $[0.0, 1.0]$.
  3. *Inference:* 1D-CNN (PyTorch), NumPy Neural Net, or Random Forest.
  4. *Probability Estimation:* Softmax output across all 5 AAMI classes.
  5. *Triage Stratification:* Color-coded urgency alerts (Normal / Warning / Critical) and interactive waveform display.

---

## 4. DATASET & EXPLORATORY DATA ANALYSIS (EDA)

We utilize the real benchmark MIT-BIH dataset: 87,554 training beats and 21,892 test beats. Zero missing or NaN values were found. True clinical class imbalance is preserved in both splits:

* **Class 0 (Normal - N):** 72,471 train (82.77%) | 18,118 test (82.76%)
* **Class 1 (Supraventricular - S):** 2,223 train (2.54%) | 556 test (2.54%)
* **Class 2 (Ventricular / PVC - V):** 5,788 train (6.61%) | 1,448 test (6.61%)
* **Class 3 (Fusion - F):** 641 train (0.73%) | 162 test (0.74%)
* **Class 4 (Unknown / Paced - Q):** 6,431 train (7.35%) | 1,608 test (7.35%)

**EDA Insights:** The extreme imbalance (82.8% Normal vs. 0.7% Fusion) proves that raw accuracy is inadequate; Macro F1-Score and Macro ROC-AUC must be reported. Morphologically, PVC beats present wide QRS complexes ($>80$ ms) and deep negative deflections, while Paced beats exhibit narrow artificial pacing spikes.

---

## 5. MODEL IMPLEMENTATION

* **Class Imbalance Strategy:** Inverse Class Frequency Weighting ($w_c = \frac{N}{K \cdot N_c}$), yielding training weights: Class 0: 0.242, Class 1: 7.877, Class 2: 3.025, Class 3: 27.318, Class 4: 2.723.
* **1D-CNN (Proposed):** 3 Conv1D blocks (32, 64, 128 filters; kernels 5, 5, 3) + BatchNorm + ReLU + MaxPool1D(2) + Dropout(0.3) + Dense(64, ReLU) + Dense(5, Softmax). Total parameters: 226,373.
* **NumPy Neural Network (Deployed):** 2-layer MLP (187 → 96 ReLU → 5 Softmax), He initialization, cross-entropy loss. Footprint: 75.1 KB.
* **Random Forest Baseline:** 100 decision trees, max_depth=16, balanced class weights.

---

## 6. EVALUATION AND EXPERIMENTAL RESULTS

### 6.1 Benchmark Comparison on Real MIT-BIH Test Split (21,892 Beats)

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Macro ROC-AUC | Latency / Beat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D-CNN (Proposed)** | **94.28%** | **0.7198** | **0.9339** | **0.7810** | **0.9514** | **0.9914** | **0.795 ms** |
| **NumPy Neural Net** | 92.36% | 0.6780 | 0.8894 | 0.7349 | 0.9342 | 0.9801 | **0.018 ms** |
| **Random Forest Baseline** | 96.93% | 0.8254 | 0.8870 | 0.8448 | 0.9705 | 0.9882 | 29.192 ms |

### 6.2 1D-CNN Class-Wise Breakdown

| AAMI Category | Code | Test Support | Precision | Recall | F1-Score | Correct / Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal (N)** | `N` | 18,118 | 0.9952 | 0.9401 | 0.9669 | 17,033 / 18,118 |
| **Supraventricular (S)** | `S` | 556 | 0.4466 | 0.8723 | 0.5907 | 485 / 556 |
| **Ventricular / PVC (V)** | `V` | 1,448 | **0.9039** | **0.9489** | **0.9259** | **1,374 / 1,448** |
| **Fusion Beat (F)** | `F` | 162 | 0.2942 | 0.9136 | 0.4451 | 148 / 162 |
| **Unknown / Paced (Q)** | `Q` | 1,608 | 0.9592 | 0.9944 | 0.9765 | 1,599 / 1,608 |

### 6.3 5-Fold Stratified Cross-Validation on 1D-CNN
* **Mean Accuracy:** 85.42% ± 2.61%
* **Mean Macro F1-Score:** 0.6514 ± 0.0308

---

## 7. USER INTERFACE & CLINICAL DEPLOYMENT

The system is deployed as an interactive Streamlit application (`app.py`):
1. **Real Patient Case Ingestion:** Quick presets loading real MIT-BIH test beats (`sample_test_beats.csv`), arbitrary CSV upload, or noise simulation.
2. **Interactive Plotly Waveform:** Calibrated time axis (ms @ 125 Hz) with automatic R-peak annotation callouts.
3. **Probability Breakdown:** Horizontal bar chart and individual cards showing predicted probabilities for all five classes.
4. **Clinical Triage Stratification:** Instant diagnostic banners (Green: Normal sinus; Amber: Moderate risk ectopic; Red: Critical PVC requiring immediate intervention).

---

## 8. CONCLUSION, LIMITATIONS & FUTURE SCOPE

CardioPulse demonstrates effective automated arrhythmia detection from 1D ECG signals without hand-crafted feature engineering. The 1D-CNN achieves a Macro ROC-AUC of 0.9914 and 94.89% sensitivity on high-risk PVCs, while the NumPy network enables sub-millisecond inference (0.018 ms).

**Honest Limitations:**
1. Evaluated on single-lead Lead II recordings only; multi-lead precordial manifestations are not captured.
2. The benchmark follows beat-level rather than patient-wise partitioning.
3. Pre-supposes prior R-peak centering and requires adaptive filtering on noisy real-world ambulatory streams.

**Future Scope:** Porting to microcontroller firmware (ESP32) for wearable Holters, 12-lead multi-channel fusion, and 1D-Vision Transformers.

---

## REFERENCES

1. Goldberger, A. L., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet." *Circulation*, 101(23), e215–e220.
2. Moody, G. B., & Mark, R. G. (2001). "The impact of the MIT-BIH Arrhythmia Database." *IEEE EMB Mag*, 20(3), 45–50.
3. ANSI/AAMI EC57 (1998). *Testing and reporting performance results of cardiac rhythm algorithms.*
4. Kiranyaz, S., et al. (2016). "Real-time patient-specific ECG classification by 1-D CNNs." *IEEE TBME*, 63(3), 664–675.
5. Rajpurkar, P., et al. (2017). "Cardiologist-level arrhythmia detection with deep neural networks." *Nature Medicine*.
6. Kachuee, M., et al. (2018). "ECG heartbeat classification: A deep transferable representation." *IEEE ICHI*, 443–444.

---

## GITHUB REPOSITORY

* **Public Repository:** [https://github.com/Akshata1803/cardiopulse](https://github.com/Akshata1803/cardiopulse)
* **License:** MIT License
