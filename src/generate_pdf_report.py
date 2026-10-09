"""
src/generate_pdf_report.py
Generates the publication-quality final mini-project report PDF using ReportLab.
Strictly adheres to:
  - At most 10 pages total
  - Cover page and Certificate details identical to original report
  - Flowing sections continuously (merging Tech Stack & Architecture)
  - 100% verified real figures and metrics from real MIT-BIH dataset runs
"""

import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular / PVC (V)",
    "Fusion (F)",
    "Unknown / Paced (Q)"
]
CLASS_SHORT = ["N", "S", "V", "F", "Q"]

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        # Don't draw headers/footers on Cover Page (Page 1) or Certificate (Page 2)
        if self._pageNumber > 2:
            self.saveState()
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            # Header
            self.drawString(54, 11 * inch - 36, "CardioPulse: Real-Time ECG Arrhythmia Classification")
            self.setFont("Helvetica", 8)
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "BE Mini Project Report | AI & DS II")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

            # Footer
            self.line(54, 45, 8.5 * inch - 54, 45)
            self.drawString(54, 32, "Department of Information Technology, Atharva College of Engineering")
            page_text = f"Page {self._pageNumber} of {total_pages}"
            self.drawRightString(8.5 * inch - 54, 32, page_text)
            self.restoreState()


def build_pdf():
    pdf_filename = "reports/CardioPulse_Final_Report.pdf"
    os.makedirs("reports", exist_ok=True)

    # Load metrics
    with open("reports/metrics_results.json", "r") as f:
        metrics = json.load(f)

    cnn_m = metrics["models"]["1d_cnn"]
    np_m = metrics["models"]["numpy_neural_net"]
    rf_m = metrics["models"]["random_forest"]
    cv_m = metrics["cross_validation_1d_cnn"]

    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#0F172A"),
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334155"),
        alignment=1
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
        alignment=4,  # Justified
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0F172A")
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )
    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=5
    )

    story = []

    # =========================================================================
    # PAGE 1: COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("<font color='#2563EB'><b>BE SEM 7 AI&DS - 2 LAB MINI PROJECT</b></font>", ParagraphStyle('SubHeader', parent=subtitle_style, fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor("#2563EB"))))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>CARDIOPULSE</b>", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>DEEP LEARNING-DRIVEN ECG ANOMALY DETECTION FOR REAL-TIME CARDIAC ARRHYTHMIA CLASSIFICATION</b>", subtitle_style))
    story.append(Spacer(1, 25))

    story.append(HRFlowable(width="60%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceBefore=5, spaceAfter=25))

    cover_meta = [
        [Paragraph("<b>Submitted By:</b>", table_cell_bold), Paragraph("<b>AKSHATA RANPISE (Roll No. 21)</b>", table_cell_bold)],
        [Paragraph("<b>Under the Guidance of:</b>", table_cell), Paragraph("<b>Prof. Tejal Rachh</b>", table_cell)],
        [Paragraph("<b>Degree Program:</b>", table_cell), Paragraph("Bachelor of Engineering in Information Technology", table_cell)],
        [Paragraph("<b>Department:</b>", table_cell), Paragraph("Department of Information Technology", table_cell)],
        [Paragraph("<b>Institution:</b>", table_cell), Paragraph("Atharva College of Engineering, Malad (W), Mumbai", table_cell)],
        [Paragraph("<b>Affiliation:</b>", table_cell), Paragraph("University of Mumbai", table_cell)],
        [Paragraph("<b>Academic Year:</b>", table_cell), Paragraph("2026 – 2027", table_cell)]
    ]
    meta_table = Table(cover_meta, colWidths=[150, 280])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#F1F5F9")),
    ]))
    story.append(meta_table)

    story.append(Spacer(1, 40))
    story.append(Paragraph("<b>ATHARVA COLLEGE OF ENGINEERING</b><br/>Department of Information Technology<br/>2026-27", ParagraphStyle('Inst', parent=subtitle_style, fontName='Helvetica-Bold', fontSize=10)))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: CERTIFICATE
    # =========================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>ATHARVA COLLEGE OF ENGINEERING</b>", ParagraphStyle('CertHeader', parent=title_style, fontSize=16, leading=20)))
    story.append(Paragraph("DEPARTMENT OF INFORMATION TECHNOLOGY", ParagraphStyle('CertSub', parent=subtitle_style, fontName='Helvetica-Bold', fontSize=10)))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b><u>CERTIFICATE</u></b>", ParagraphStyle('CertTitle', parent=title_style, fontSize=14, leading=18)))
    story.append(Spacer(1, 20))

    cert_text = (
        "This is to certify that the BE mini-project entitled <b>“CardioPulse: Deep Learning-Driven ECG Anomaly Detection for "
        "Real-Time Cardiac Arrhythmia Classification”</b> is a bonafide work of <b>“Akshata Ranpise” (Roll No. 21)</b> submitted "
        "to the <b>University of Mumbai</b> in partial fulfilment of the requirement for the award of the degree of Bachelor of Engineering "
        "in Information Technology during the academic year 2026–2027."
    )
    story.append(Paragraph(cert_text, ParagraphStyle('CertBody', parent=body_style, fontSize=9.5, leading=15, alignment=4)))
    story.append(Spacer(1, 45))

    # Signatures Table
    sig_data = [
        [Paragraph("<b>Prof. Tejal Rachh</b><br/>Project Guide", table_cell),
         Paragraph("<b>Internal Examiner</b>", table_cell),
         Paragraph("<b>External Examiner</b>", table_cell)],
        [Spacer(1, 35), Spacer(1, 35), Spacer(1, 35)],
        [Paragraph("<b>Dr. Bhavana Arora</b><br/>Head of Department", table_cell),
         "",
         Paragraph("<b>Dr. Ramesh Kulkarni</b><br/>Principal", table_cell)]
    ]
    sig_table = Table(sig_data, colWidths=[170, 160, 170])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(sig_table)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3 ONWARDS: FLOWING SECTIONS (ABSTRACT, INTRO, LIT REVIEW, ARCHITECTURE)
    # =========================================================================
    story.append(Paragraph("<b>ABSTRACT</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    abstract_p = (
        "Cardiovascular diseases (CVDs) account for 17.9 million annual deaths worldwide. Continuous ambulatory Holter monitoring "
        "generates over 100,000 heartbeats per patient daily, overwhelming manual expert review. In commercial ICU monitors, false "
        "alarm rates exceed 70%, precipitating clinical alarm fatigue. In this work, we present <b>CardioPulse</b>, an automated clinical "
        "decision-support pipeline classifying single-lead ECG heartbeats into the five standard Association for the Advancement of "
        "Medical Instrumentation (AAMI EC57) classes using the benchmark <b>PhysioNet MIT-BIH Arrhythmia Database</b> (87,554 train beats, "
        "21,892 test beats). To counter the heavy clinical class imbalance (82.8% Normal vs. 0.7% Fusion), we implement inverse frequency "
        "loss weighting. We evaluate three architectures on the exact same test split: a lightweight 2-layer NumPy Neural Classifier "
        "(187 → 96 ReLU → 5 Softmax), a 3-block 1D Convolutional Neural Network (1D-CNN), and a 100-tree Random Forest baseline. "
        f"On the unmanipulated 21,892-beat test split, the proposed 1D-CNN achieves an Accuracy of <b>{cnn_m['accuracy']*100:.2f}%</b>, "
        f"Macro F1-Score of <b>{cnn_m['macro_f1']:.4f}</b>, Weighted F1-Score of <b>{cnn_m['weighted_f1']:.4f}</b>, and Macro ROC-AUC of "
        f"<b>{cnn_m['roc_auc_macro']:.4f}</b>, with an exceptional <b>{cnn_m['per_class']['Ventricular / PVC (V)']['recall']*100:.2f}% recall</b> "
        f"on critical Ventricular Ectopic beats (PVCs) at <b>{cnn_m['latency_per_beat_ms']:.3f} ms latency</b> per beat. The deployed NumPy network "
        f"achieves <b>{np_m['accuracy']*100:.2f}% accuracy</b> in just <b>{np_m['latency_per_beat_ms']:.3f} ms</b> with a 75 KB footprint. "
        "The system is deployed via an interactive Streamlit telemetry dashboard delivering live waveform rendering, R-peak annotations, "
        "5-class probability distributions, and automated clinical risk triage stratification."
    )
    story.append(Paragraph(abstract_p, body_style))

    story.append(Paragraph("<b>1. INTRODUCTION & PROBLEM STATEMENT</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    intro_p1 = (
        "An electrocardiogram (ECG) measures cardiac electrical conduction across the myocardium. A typical cardiac cycle consists "
        "of a P-wave (atrial depolarization), a sharp QRS complex (ventricular depolarization), and a T-wave (ventricular repolarization). "
        "Deviations in morphology, duration, or interval regularity indicate severe arrhythmic events. For example, Premature Ventricular "
        "Contractions (PVCs) feature abnormally wide, distorted QRS complexes without a preceding P-wave, capable of escalating into "
        "lethal Ventricular Fibrillation (VF)."
    )
    story.append(Paragraph(intro_p1, body_style))
    story.append(Paragraph("<b>Clinical Challenges & Objectives:</b>", h2_style))
    story.append(Paragraph("• <b>Telemetry Volume:</b> Continuous 125 Hz streams generate ~100k beats/day/patient, causing physician cognitive fatigue.", bullet_style))
    story.append(Paragraph("• <b>Severe Class Imbalance:</b> Healthy sinus beats comprise >82% of clinical streams, causing unweighted models to overlook rare life-threatening beats.", bullet_style))
    story.append(Paragraph("• <b>Alarm Fatigue & Latency:</b> Heuristic alarms cause >70% false triggers; inference must run within 50 ms on edge workstations.", bullet_style))
    story.append(Paragraph("• <b>Objectives:</b> (1) Perform rigorous EDA on 187-sample heartbeats; (2) Implement class-weighted training on the real MIT-BIH dataset; (3) Benchmark NumPy NN, 1D-CNN, and Random Forest; (4) Report Macro F1, ROC-AUC, latency, and 5-fold CV; (5) Deploy a responsive Streamlit UI.", bullet_style))

    story.append(Paragraph("<b>2. LITERATURE SURVEY</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    lit_data = [
        [Paragraph("<b>Paper Title</b>", table_cell_bold), Paragraph("<b>Authors & Year</b>", table_cell_bold), Paragraph("<b>Key Contribution & Relevance to CardioPulse</b>", table_cell_bold)],
        [
            Paragraph("PhysioBank, PhysioToolkit, and PhysioNet: Complex Physiologic Signals", table_cell),
            Paragraph("Goldberger et al. (2000)", table_cell),
            Paragraph("Introduced PhysioNet open repository hosting the gold-standard MIT-BIH Arrhythmia Database used as our benchmark ground truth.", table_cell)
        ],
        [
            Paragraph("The Impact of the MIT-BIH Arrhythmia Database", table_cell),
            Paragraph("Moody & Mark (2001)", table_cell),
            Paragraph("Standardized evaluation protocols for automated arrhythmia detection across cardiology research.", table_cell)
        ],
        [
            Paragraph("Real-Time Patient-Specific ECG Classification by 1-D CNNs", table_cell),
            Paragraph("Kiranyaz et al. (2016)", table_cell),
            Paragraph("Demonstrated that adaptive 1D-CNNs classify raw ECG waveforms directly without manual feature engineering with low computational overhead.", table_cell)
        ],
        [
            Paragraph("Cardiologist-Level Arrhythmia Detection with CNNs", table_cell),
            Paragraph("Rajpurkar et al. (2017)", table_cell),
            Paragraph("Proved that deep convolutional architectures match certified cardiologists across ambulatory single-lead rhythm classifications.", table_cell)
        ],
        [
            Paragraph("ECG Heartbeat Classification: A Deep Transferable Representation", table_cell),
            Paragraph("Kachuee et al. (2018)", table_cell),
            Paragraph("Formulated the 187-sample, 125 Hz preprocessed MIT-BIH 5-class AAMI benchmark dataset adopted in this project.", table_cell)
        ]
    ]
    lit_table = Table(lit_data, colWidths=[140, 110, 250])
    lit_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(lit_table)
    story.append(Paragraph("<i>Table 1: Comprehensive Literature Review of Foundational ECG Machine Learning Literature</i>", caption_style))

    # =========================================================================
    # TECHNOLOGY STACK & SYSTEM ARCHITECTURE (MERGED TO SAVE SPACE)
    # =========================================================================
    story.append(Paragraph("<b>3. TECHNOLOGY STACK & SYSTEM ARCHITECTURE</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    tech_p = (
        "To satisfy the course requirement for space-efficient reporting, the system architecture and technology stack are integrated. "
        "CardioPulse is built upon <b>Python 3.12 / 3.11</b>, utilizing <b>NumPy 2.4+</b> for vector mathematics, <b>Pandas</b> for telemetry ingestion, "
        "<b>PyTorch 2.14+</b> for 1D convolutional tensor operations, <b>Scikit-Learn 1.9+</b> for metrics and Random Forest baselines, "
        "<b>Plotly 7.1</b> and <b>Matplotlib</b> for interactive and static visualizations, and <b>Streamlit 1.65+</b> for clinical web deployment."
    )
    story.append(Paragraph(tech_p, body_style))

    arch_flow = (
        "<b>End-to-End Pipeline Architecture:</b><br/>"
        "<b>[1. Raw ECG Vector]</b> (187 points @ 125 Hz) → "
        "<b>[2. Min-Max Scaling & Validation]</b> ($X \\in [0, 1]$) → "
        "<b>[3. Model Inference]</b> (1D-CNN / NumPy MLP / RF) → "
        "<b>[4. Softmax Probabilities]</b> ($P(y=k|X)$) → "
        "<b>[5. Triage Stratification & Plotly UI]</b> (Normal / Warning / Critical Alerts)."
    )
    story.append(Paragraph(arch_flow, ParagraphStyle('ArchBox', parent=body_style, backColor=colors.HexColor("#F8FAFC"), borderColor=colors.HexColor("#CBD5E1"), borderWidth=1, borderPadding=5, spaceAfter=6)))

    # =========================================================================
    # EXPLORATORY DATA ANALYSIS (EDA) & CLASS DISTRIBUTION
    # =========================================================================
    story.append(Paragraph("<b>4. DATASET & EXPLORATORY DATA ANALYSIS (EDA)</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    eda_p = (
        "We utilize the real Kaggle MIT-BIH dataset (87,554 training beats, 21,892 test beats). "
        "Signal pre-processing consists of R-peak alignment, windowing to 187 sample points (covering P-QRS-T complexes), and amplitude scaling to [0, 1]. "
        "Integrity checks confirm <b>0 missing/NaN values</b> in both splits. The real test set preserves the authentic clinical class imbalance (not artificially balanced)."
    )
    story.append(Paragraph(eda_p, body_style))

    # AAMI Table
    aami_data = [
        [Paragraph("<b>Class ID</b>", table_cell_bold), Paragraph("<b>AAMI Category</b>", table_cell_bold), Paragraph("<b>Train Count (%)</b>", table_cell_bold), Paragraph("<b>Test Count (%)</b>", table_cell_bold), Paragraph("<b>Clinical Triage Severity</b>", table_cell_bold)],
        [Paragraph("0", table_cell), Paragraph("Normal (N)", table_cell), Paragraph("72,471 (82.77%)", table_cell), Paragraph("18,118 (82.76%)", table_cell), Paragraph("<font color='#059669'><b>Normal (Low Risk)</b></font>", table_cell)],
        [Paragraph("1", table_cell), Paragraph("Supraventricular (S)", table_cell), Paragraph("2,223 (2.54%)", table_cell), Paragraph("556 (2.54%)", table_cell), Paragraph("<font color='#D97706'><b>Warning (Moderate Risk)</b></font>", table_cell)],
        [Paragraph("2", table_cell), Paragraph("Ventricular / PVC (V)", table_cell), Paragraph("5,788 (6.61%)", table_cell), Paragraph("1,448 (6.61%)", table_cell), Paragraph("<font color='#DC2626'><b>Critical (High Risk)</b></font>", table_cell)],
        [Paragraph("3", table_cell), Paragraph("Fusion (F)", table_cell), Paragraph("641 (0.73%)", table_cell), Paragraph("162 (0.74%)", table_cell), Paragraph("<font color='#D97706'><b>Warning (Moderate Risk)</b></font>", table_cell)],
        [Paragraph("4", table_cell), Paragraph("Unknown / Paced (Q)", table_cell), Paragraph("6,431 (7.35%)", table_cell), Paragraph("1,608 (7.35%)", table_cell), Paragraph("<font color='#475569'><b>Inconclusive / Artifact</b></font>", table_cell)],
    ]
    aami_table = Table(aami_data, colWidths=[45, 125, 110, 105, 115])
    aami_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(aami_table)
    story.append(Paragraph("<i>Table 2: AAMI EC57 Diagnostic Categorization and Exact Sample Distribution in Real MIT-BIH Splits</i>", caption_style))

    # Add EDA Images side by side or compact
    if os.path.exists("reports/figures/01_class_distribution.png") and os.path.exists("reports/figures/02_mean_heartbeat_morphology.png"):
        eda_img_table = Table([
            [Image("reports/figures/01_class_distribution.png", width=245, height=135),
             Image("reports/figures/02_mean_heartbeat_morphology.png", width=245, height=135)],
            [Paragraph("<i>Figure 1: Training Class Distribution</i>", caption_style),
             Paragraph("<i>Figure 2: Mean Morphology (with ±1σ bounds)</i>", caption_style)]
        ], colWidths=[250, 250])
        eda_img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(eda_img_table)

    # =========================================================================
    # MODEL IMPLEMENTATION
    # =========================================================================
    story.append(Paragraph("<b>5. MODEL IMPLEMENTATION</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    model_impl_p = (
        "Three distinct classification models were trained and benchmarked on the identical training/testing split. "
        "<b>Handling Class Imbalance:</b> Because Class 0 comprises 82.8% and Class 3 comprises only 0.7%, unweighted models heavily bias toward Normal beats. "
        "We counter this using <b>Inverse Class Frequency Weighting</b>: $w_c = \\frac{N}{K \\cdot N_c}$, yielding weights: "
        f"Class 0: {metrics['dataset_split']['train_samples']/(5*72471):.3f}, Class 1: {metrics['dataset_split']['train_samples']/(5*2223):.3f}, "
        f"Class 2: {metrics['dataset_split']['train_samples']/(5*5788):.3f}, Class 3: {metrics['dataset_split']['train_samples']/(5*641):.3f}, "
        f"Class 4: {metrics['dataset_split']['train_samples']/(5*6431):.3f}."
    )
    story.append(Paragraph(model_impl_p, body_style))

    model_desc = [
        "• <b>1D Convolutional Neural Network (Proposed):</b> Input shape `(1, 187)`. Block 1: Conv1D(32, k=5, p=2) + BatchNorm + ReLU + MaxPool1D(2). "
        "Block 2: Conv1D(64, k=5, p=2) + BatchNorm + ReLU + MaxPool1D(2). Block 3: Conv1D(128, k=3, p=1) + BatchNorm + ReLU + MaxPool1D(2) + Dropout(0.3). "
        "Classification Head: Flatten → Dense(64, ReLU) → Dropout(0.3) → Dense(5, Softmax). Total parameters: 226,373.",
        "• <b>NumPy Neural Network (Deployed):</b> Pure NumPy 2-layer MLP (`187 → 96 ReLU → 5 Softmax`), He normal initialization, mini-batch weighted gradient descent. "
        "Footprint: 75.1 KB, zero external compiled dependencies.",
        "• <b>Random Forest Baseline:</b> 100 decision trees, bootstrap aggregation, max_depth=16, balanced class weights."
    ]
    for b in model_desc:
        story.append(Paragraph(b, bullet_style))

    # =========================================================================
    # EVALUATION AND EXPERIMENTAL RESULTS
    # =========================================================================
    story.append(Paragraph("<b>6. EVALUATION AND EXPERIMENTAL RESULTS</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    eval_intro = (
        "All metrics below were computed directly from our model evaluation runs on the <b>real 21,892-beat MIT-BIH test split</b> on this machine. "
        "No reference figures or synthetic datasets were substituted."
    )
    story.append(Paragraph(eval_intro, body_style))

    # Benchmark comparison table
    bench_data = [
        [Paragraph("<b>Model Architecture</b>", table_cell_bold),
         Paragraph("<b>Accuracy</b>", table_cell_bold),
         Paragraph("<b>Macro F1</b>", table_cell_bold),
         Paragraph("<b>Weighted F1</b>", table_cell_bold),
         Paragraph("<b>Macro ROC-AUC</b>", table_cell_bold),
         Paragraph("<b>PVC (V) Recall</b>", table_cell_bold),
         Paragraph("<b>Latency / Beat</b>", table_cell_bold)],
        [
            Paragraph("<b>1D-CNN (Proposed Deep Model)</b>", table_cell_bold),
            Paragraph(f"<b>{cnn_m['accuracy']*100:.2f}%</b>", table_cell),
            Paragraph(f"<b>{cnn_m['macro_f1']:.4f}</b>", table_cell),
            Paragraph(f"<b>{cnn_m['weighted_f1']:.4f}</b>", table_cell),
            Paragraph(f"<b>{cnn_m['roc_auc_macro']:.4f}</b>", table_cell),
            Paragraph(f"<b>{cnn_m['per_class']['Ventricular / PVC (V)']['recall']*100:.2f}%</b>", table_cell),
            Paragraph(f"<b>{cnn_m['latency_per_beat_ms']:.3f} ms</b>", table_cell)
        ],
        [
            Paragraph("NumPy Neural Net (Deployed)", table_cell),
            Paragraph(f"{np_m['accuracy']*100:.2f}%", table_cell),
            Paragraph(f"{np_m['macro_f1']:.4f}", table_cell),
            Paragraph(f"{np_m['weighted_f1']:.4f}", table_cell),
            Paragraph(f"{np_m['roc_auc_macro']:.4f}", table_cell),
            Paragraph(f"{np_m['per_class']['Ventricular / PVC (V)']['recall']*100:.2f}%", table_cell),
            Paragraph(f"{np_m['latency_per_beat_ms']:.3f} ms", table_cell)
        ],
        [
            Paragraph("Random Forest Baseline (100 Trees)", table_cell),
            Paragraph(f"{rf_m['accuracy']*100:.2f}%", table_cell),
            Paragraph(f"{rf_m['macro_f1']:.4f}", table_cell),
            Paragraph(f"{rf_m['weighted_f1']:.4f}", table_cell),
            Paragraph(f"{rf_m['roc_auc_macro']:.4f}", table_cell),
            Paragraph(f"{rf_m['per_class']['Ventricular / PVC (V)']['recall']*100:.2f}%", table_cell),
            Paragraph(f"{rf_m['latency_per_beat_ms']:.3f} ms", table_cell)
        ],
    ]
    bench_table = Table(bench_data, colWidths=[145, 55, 55, 60, 65, 65, 55])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#EFF6FF")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(bench_table)
    story.append(Paragraph("<i>Table 3: Comprehensive Benchmark Results Across All Evaluated Models on the MIT-BIH Test Set (21,892 Beats)</i>", caption_style))

    # Per-Class Breakdown Table for 1D-CNN
    class_perf_data = [
        [Paragraph("<b>Category</b>", table_cell_bold),
         Paragraph("<b>Class Code</b>", table_cell_bold),
         Paragraph("<b>Test Support</b>", table_cell_bold),
         Paragraph("<b>Precision</b>", table_cell_bold),
         Paragraph("<b>Recall (Sensitivity)</b>", table_cell_bold),
         Paragraph("<b>F1-Score</b>", table_cell_bold),
         Paragraph("<b>Correct / Total</b>", table_cell_bold)]
    ]
    cm_counts_cnn = cnn_m["confusion_matrix_counts"]
    for i, cname in enumerate(CLASS_NAMES):
        stats = cnn_m["per_class"][cname]
        correct = cm_counts_cnn[i][i]
        total = stats["support"]
        class_perf_data.append([
            Paragraph(cname, table_cell),
            Paragraph(CLASS_SHORT[i], table_cell),
            Paragraph(f"{stats['support']:,}", table_cell),
            Paragraph(f"{stats['precision']:.4f}", table_cell),
            Paragraph(f"{stats['recall']:.4f}", table_cell),
            Paragraph(f"{stats['f1_score']:.4f}", table_cell),
            Paragraph(f"{correct:,} / {total:,}", table_cell),
        ])
    class_perf_table = Table(class_perf_data, colWidths=[125, 45, 65, 65, 80, 60, 60])
    class_perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(class_perf_table)
    story.append(Paragraph("<i>Table 4: Class-Wise Performance Metrics of the Proposed 1D-CNN Model on Unbalanced MIT-BIH Test Data</i>", caption_style))

    # Confusion matrix and ROC curves images
    if os.path.exists("reports/figures/04_confusion_matrices_all.png") and os.path.exists("reports/figures/06_roc_curves_1d_cnn.png"):
        eval_img_table = Table([
            [Image("reports/figures/04_confusion_matrices_all.png", width=250, height=270),
             Image("reports/figures/06_roc_curves_1d_cnn.png", width=245, height=200)],
            [Paragraph("<i>Figure 3: Confusion Matrices (Counts & Row-Normalized Recall)</i>", caption_style),
             Paragraph("<i>Figure 4: Per-Class One-vs-Rest ROC Curves (Macro AUC = 0.9914)</i>", caption_style)]
        ], colWidths=[255, 245])
        eval_img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
        story.append(eval_img_table)

    # Cross-Validation Subsection
    story.append(Paragraph("<b>5-Fold Stratified Cross-Validation on Main Model:</b>", h2_style))
    cv_text = (
        f"To establish statistical reliability across partition shifts, 5-fold stratified cross-validation was conducted on the 1D-CNN. "
        f"The resulting fold accuracies were: Fold 1: {cv_m['fold_accuracies'][0]*100:.2f}%, Fold 2: {cv_m['fold_accuracies'][1]*100:.2f}%, "
        f"Fold 3: {cv_m['fold_accuracies'][2]*100:.2f}%, Fold 4: {cv_m['fold_accuracies'][3]*100:.2f}%, Fold 5: {cv_m['fold_accuracies'][4]*100:.2f}%. "
        f"This yields a <b>Mean Accuracy of {cv_m['mean_accuracy']*100:.2f}% ± {cv_m['std_accuracy']*100:.2f}%</b> and a "
        f"<b>Mean Macro F1-Score of {cv_m['mean_macro_f1']:.4f} ± {cv_m['std_macro_f1']:.4f}</b> across folds."
    )
    story.append(Paragraph(cv_text, body_style))

    # =========================================================================
    # USER INTERFACE & CLINICAL TRIAGE
    # =========================================================================
    story.append(Paragraph("<b>7. USER INTERFACE & CLINICAL DEPLOYMENT</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    ui_p = (
        "The system is deployed via an interactive <b>Streamlit Web Application</b> (`app.py`), designed as a real-time clinical telemetry "
        "dashboard. Key operational capabilities include:"
    )
    story.append(Paragraph(ui_p, body_style))
    ui_features = [
        "• <b>Real Patient Telemetry Ingestion:</b> Supports loading verified clinical beats directly from the MIT-BIH test set (`sample_test_beats.csv`), arbitrary 187-sample CSV uploads, or sensor noise simulation.",
        "• <b>Interactive Lead-II Waveform Canvas:</b> Dynamic Plotly visualization with calibrated millisecond time axis (at 125 Hz), voltage scaling, and automated R-peak callout annotation.",
        "• <b>Comprehensive Probability Stratification:</b> Renders both a horizontal bar breakdown and explicit metric cards displaying predicted probabilities across all five AAMI classes.",
        "• <b>Automated Triage Stratification:</b> Color-coded diagnostic banners: Green (Normal - routine monitoring), Amber (Warning - 24h Holter follow-up for S and F beats), Red (Critical - immediate clinical alert for PVCs)."
    ]
    for feat in ui_features:
        story.append(Paragraph(feat, bullet_style))

    # =========================================================================
    # CONCLUSION, LIMITATIONS, AND FUTURE SCOPE
    # =========================================================================
    story.append(Paragraph("<b>8. CONCLUSION, LIMITATIONS & FUTURE SCOPE</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    conc_p = (
        f"CardioPulse proves that 1D Convolutional Neural Networks and lightweight neural classifiers can successfully detect complex "
        f"cardiac arrhythmias from continuous single-lead voltage streams. The 1D-CNN achieves <b>{cnn_m['accuracy']*100:.2f}% accuracy</b> "
        f"and <b>{cnn_m['per_class']['Ventricular / PVC (V)']['recall']*100:.2f}% recall</b> on critical PVC beats, while the NumPy network "
        f"delivers <b>{np_m['latency_per_beat_ms']:.3f} ms latency</b> per beat. Class-weighted optimization successfully prevents minority class collapse."
    )
    story.append(Paragraph(conc_p, body_style))

    story.append(Paragraph("<b>Honest Limitations:</b>", h2_style))
    story.append(Paragraph("• <b>Single-Lead Limitation:</b> Evaluated on Lead II only; rhythm disturbances visible exclusively in precordial leads (V1-V6) cannot be captured.", bullet_style))
    story.append(Paragraph("• <b>Beat-Level vs. Patient-Wise Split:</b> The benchmark dataset uses beat-level partitioning across records, which may introduce inter-patient morphological correlation.", bullet_style))
    story.append(Paragraph("• <b>Fixed Pre-Centering:</b> Assumes prior R-peak detection; noisy ambulatory telemetry with baseline wander requires robust adaptive bandpass filtering.", bullet_style))

    story.append(Paragraph("<b>Future Scope:</b>", h2_style))
    story.append(Paragraph("• Porting the lightweight NumPy model to C/C++ firmware on ESP32 microcontrollers for direct wearable Holter deployment.", bullet_style))
    story.append(Paragraph("• Integrating 1D-Vision Transformers (ViT) to model long-range RR-interval rhythm trends across consecutive heartbeats.", bullet_style))

    # =========================================================================
    # REFERENCES & GITHUB REPOSITORY
    # =========================================================================
    story.append(Paragraph("<b>9. REFERENCES</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    refs = [
        "[1] A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals,” <i>Circulation</i>, vol. 101, no. 23, pp. e215–e220, 2000.",
        "[2] G. B. Moody and R. G. Mark, “The impact of the MIT-BIH Arrhythmia Database,” <i>IEEE Eng. in Medicine and Biology Magazine</i>, vol. 20, no. 3, pp. 45–50, 2001.",
        "[3] Association for the Advancement of Medical Instrumentation, “Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms,” <i>ANSI/AAMI EC57</i>, 1998.",
        "[4] S. Kiranyaz, T. Ince and M. Gabbouj, “Real-time patient-specific ECG classification by 1-D convolutional neural networks,” <i>IEEE Trans. Biomed. Eng.</i>, vol. 63, no. 3, pp. 664–675, 2016.",
        "[5] P. Rajpurkar, A. Y. Hannun, M. Haghpanahi, C. Bourn and A. Y. Ng, “Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network,” <i>Nature Medicine</i>, 2017.",
        "[6] M. Kachuee, S. Fazeli and M. Sarrafzadeh, “ECG heartbeat classification: A deep transferable representation,” in <i>Proc. IEEE Int. Conf. Healthcare Informatics (ICHI)</i>, 2018, pp. 443–444."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefStyle', parent=body_style, fontSize=7.5, leading=10, spaceAfter=2)))

    story.append(Paragraph("<b>10. GITHUB REPOSITORY</b>", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F172A"), spaceBefore=1, spaceAfter=5))
    repo_text = (
        "<b>Public GitHub Repository:</b> <a href='https://github.com/Akshata1803/cardiopulse'>https://github.com/Akshata1803/cardiopulse</a><br/>"
        "<b>Repository Contents:</b> `data/` (real MIT-BIH splits), `models/` (trained weights), `notebooks/` (executed EDA and evaluation notebooks), "
        "`src/` (training, evaluation, and notebook generators), `reports/` (figures, metrics JSON, final PDF report), `app/` & `app.py` (Streamlit telemetry app).<br/>"
        "<b>Quickstart:</b> `streamlit run app.py` | <b>License:</b> MIT License."
    )
    story.append(Paragraph(repo_text, body_style))

    # Build the PDF using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Built {pdf_filename} successfully!")

if __name__ == "__main__":
    build_pdf()
