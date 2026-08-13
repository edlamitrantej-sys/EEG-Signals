<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/MNE--Python-1.6+-0C479D?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🧠 NeuroScan — Epileptic Seizure Detection System</h1>

<p align="center">
  <b>A real-time, AI-powered EEG analysis platform for epileptic seizure detection using a hybrid CNN-Transformer architecture with Continuous Wavelet Transform (CWT) feature engineering. </b>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-high-level-architecture">Architecture</a> •
  <a href="#-pipeline-stages">Pipeline</a> •
  <a href="#-model-architecture">Model</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-dataset">Dataset</a>
</p>

---

## 📌 Overview

NeuroScan is an end-to-end clinical-grade EEG analysis system that classifies brain signals into **four neurological states** in real time:

| State | Description | Clinical Significance |
|:---:|---|---|
| 🟢 **Normal** | Healthy background cortical activity | No intervention required |
| 🟡 **Preictal** | Pre-seizure warning phase | Alert clinician, prepare rescue medication |
| 🔴 **Seizure (Ictal)** | Active epileptic seizure event | Immediate clinical response required |
| 🟠 **Postictal** | Post-seizure recovery state | Monitor for secondary events |

**Key Features:**
- 🔬 Real CNN-Transformer hybrid model trained on **CHB-MIT Scalp EEG Database**
- 🌊 Continuous Wavelet Transform (CWT) scalogram visualization
- 📋 Auto-generated clinical reports with confidence scoring
- 🔐 Session-based clinician authentication (8-hour tokens)
- 📁 Supports `.edf` (medical), `.csv`, and `.txt` EEG file formats
- 🎮 Built-in demo mode with synthetic signals for all 4 seizure states

---

## 🏗 High-Level Architecture

```mermaid
flowchart TD
    subgraph INPUT["📥  INPUT SOURCES"]
        A1["🗂️ EDF File\n(.edf medical format)"]
        A2["📄 CSV / TXT File"]
        A3["🔴 Live Demo Signal\n(Normal / Preictal / Seizure / Postictal)"]
        A4["📡 JSON API Call"]
    end

    subgraph S1["⚙️  STAGE 1 — Signal Preprocessing"]
        B1["Notch Filter\n60 Hz removal"]
        B2["Bandpass Filter\n0.5 – 50 Hz"]
        B3["Epoch Extraction\n2s windows @ 256 Hz"]
        B4["Artifact Rejection\n> 300 µV discarded"]
        B5["Z-Score Normalization\nper channel"]
    end

    subgraph S2["🌊  STAGE 2 — CWT Feature Engineering"]
        C1["Complex Morlet Wavelet\ncmor1.5-1.0"]
        C2["Scalogram Generation\n22 ch × 50 freq × 512 time"]
        C3["PNG Heatmaps\n(base64, for dashboard)"]
    end

    subgraph S3["🤖  STAGE 3 — CNN-Transformer Model"]
        D1["2D CNN Blocks\nSpatial Feature Extraction"]
        D2["Reshape Bridge\n→ Sequence of 128 time steps"]
        D3["Transformer Encoder\n2 layers · 8 heads · d=768"]
        D4["Classifier Head\nSoftmax → 4 classes"]
    end

    subgraph S4["📋  STAGE 4 — Clinical Output"]
        E1["Prediction Label\n+ Confidence Scores"]
        E2["Medical Report\nAuto-generated notes"]
        E3["Clinical Dashboard\nLive waveform · Scalograms"]
    end

    A1 & A2 & A3 & A4 --> B1
    B1 --> B2 --> B3 --> B4 --> B5
    B5 --> C1 --> C2 --> C3
    C2 --> D1 --> D2 --> D3 --> D4
    D4 --> E1
    C3 --> E3
    E1 --> E2 --> E3
```

---

## 🔄 Data Flow

```mermaid
sequenceDiagram
    actor C as 👨‍⚕️ Clinician
    participant D as 🖥️ Dashboard
    participant A as 🐍 Flask API
    participant M as 🤖 CNN-Transformer

    C->>D: Login (Practice ID + Key)
    D->>A: POST /api/auth
    A-->>D: Session Token (8 hr)

    C->>D: Click "Seizure" demo button
    D->>A: GET /api/demo-signal/seizure
    A-->>D: Synthetic 22ch × 512 EEG signal

    D->>A: POST /api/predict
    Note over A: Stage 1 — Notch · Bandpass · Epoch · Z-Score
    Note over A: Stage 2 — CWT → (22, 50, 512) scalogram
    A->>M: Forward pass
    Note over M: CNN → Transformer → Softmax
    M-->>A: [0.01, 0.01, 0.97, 0.01]
    Note over A: Stage 4 — Generate clinical report

    A-->>D: prediction + confidences + scalograms + notes
    D-->>C: Dashboard updates in real time
```

---

## 🔬 Pipeline Stages

### Stage 1 — Data Ingestion & Signal Preprocessing

```mermaid
flowchart LR
    RAW["Raw EEG\n22ch × N samples\n(µV)"]
    N["Notch Filter\n60 Hz IIR\nRemoves power-line noise"]
    BP["Bandpass Filter\n0.5–50 Hz Butterworth 4th order\nRetains brain frequencies"]
    EP["Epoch Extraction\n2s sliding window · 512 samples\n50% overlap"]
    AR["Artifact Rejection\nRejects epochs > 300 µV\n(muscle/electrode pop)"]
    ZS["Z-Score Normalization\nMean=0 · Std=1\nper channel"]
    OUT["Clean Epoch\n22 × 512"]

    RAW --> N --> BP --> EP --> AR --> ZS --> OUT
```

**Supported Input Formats:**

| Format | Parser | Use Case |
|---|---|---|
| `.edf` | MNE-Python | Medical EEG files (European Data Format) |
| `.csv` / `.txt` | Built-in CSV reader | Exported recordings, research data |
| JSON body | Direct API input | Real-time streaming, integrations |
| Demo button | Synthetic generator | Testing & demonstration |

---

### Stage 2 — CWT Scalogram Generation

```mermaid
flowchart LR
    SIG["1D Signal\nper channel\n512 samples"]
    CWT["Complex Morlet CWT\ncmor1.5-1.0\n1–50 Hz · 50 bins"]
    SCALO["2D Scalogram\n50 freq × 512 time\nper channel"]
    TENSOR["Final Tensor\n22 × 50 × 512\nall channels"]
    PNG["PNG Heatmaps\nbase64 encoded\nfor dashboard"]

    SIG --> CWT --> SCALO
    SCALO -->|"×22 channels"| TENSOR
    TENSOR --> PNG
```

> **Why CWT?** Raw EEG is a noisy 1D voltage trace. The CWT decomposes it into a 2D frequency-time heatmap where seizure-specific patterns — rhythmic 3–8 Hz bursts and spike-wave complexes — become **visually distinct** and far easier for the CNN to detect.

---

### Stage 3 — CNN-Transformer Hybrid Model

```mermaid
flowchart TD
    IN["Input Tensor\nBatch × 22 × 50 × 512\n22ch · 50 freq bins · 512 time"]

    subgraph CNN["🖼️  2D CNN — Spatial Feature Extraction"]
        C1["Conv2D 22→32 · BatchNorm · ReLU\nMaxPool 2×2\n→ Batch × 32 × 25 × 256"]
        C2["Conv2D 32→64 · BatchNorm · ReLU\nMaxPool 2×2\n→ Batch × 64 × 12 × 128"]
    end

    BRIDGE["🔀 Reshape Bridge\nPermute + Flatten\n→ Batch × 128 timesteps × 768 features"]

    subgraph TF["⚡  Transformer Encoder — Temporal Attention"]
        T1["Self-Attention Layer 1\n8 heads · d_model=768"]
        T2["Self-Attention Layer 2\n8 heads · d_ff=2048"]
    end

    POOL["📉 Global Average Pooling\nMean over 128 timesteps\n→ Batch × 768"]

    subgraph CLS["🎯  Classifier Head"]
        L1["Linear 768→256 · ReLU · Dropout 50%"]
        L2["Linear 256→4 · Softmax"]
    end

    OUT["Output\n4 Class Probabilities\nNormal · Preictal · Seizure · Postictal"]

    IN --> C1 --> C2 --> BRIDGE --> T1 --> T2 --> POOL --> L1 --> L2 --> OUT
```

**Training Details:**

| Parameter | Value |
|---|---|
| Dataset | CHB-MIT Scalp EEG Database (24 pediatric patients) |
| Optimizer | AdamW · lr=1e-4 · weight_decay=1e-2 |
| Scheduler | ReduceLROnPlateau · factor=0.5 · patience=3 |
| Loss | Cross-Entropy with class weights `[0.1, 0.4, 0.9, 0.4]` |
| Regularization | Dropout 50% · BatchNorm · weight decay |
| Precision | Mixed FP16 (AMP on GPU) |

> **Why class weighting?** In real EEG data, seizures are <1% of recording time. Without weighting the model just predicts "Normal" always and still gets 99% accuracy — being clinically useless. Heavier weight on seizure class forces the model to focus on rare events.

---

### Stage 4 — Clinical Dashboard & Output

| Output | Description |
|---|---|
| **Classification** | Top predicted class with confidence % |
| **Confidence Scores** | Full probability distribution across all 4 classes |
| **CWT Scalograms** | Heatmap images from channels FP1-F7 and C3-P3 |
| **Signal Preview** | 512-point raw waveform for visual inspection |
| **Medical Report** | Auto-generated clinical notes with recommended actions |
| **Inference Time** | End-to-end pipeline latency in milliseconds |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/ScriptOrbit-132/EEG-Signals.git
cd EEG-Signals
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
python code/app.py
```

You should see:

```
========================================================
  NeuroScan EEG Analysis API - v1.0
  Model mode : real
  Classes    : ['Normal', 'Preictal', 'Seizure (Ictal)', 'Postictal']
  -------------------------------------------------
  Demo credentials:
    DEMO_CLINIC          -> key: NS2026
    chb_research         -> key: CHB_MIT_001
    neuroscan_dev        -> key: DEV_9999
  -------------------------------------------------
  Open: http://localhost:5000
========================================================
```

### 4. Open the Dashboard

Navigate to **http://localhost:5000** and login:
- **Practice ID:** `DEMO_CLINIC`
- **Access Key:** `NS2026`

---

## 🎮 Demo

| Button | Simulates | Expected Confidence |
|---|---|---|
| 🟢 Normal | Healthy resting-state EEG | ~97% Normal |
| 🟡 Preictal | Pre-seizure warning signals | ~89% Preictal |
| 🔴 Seizure | Active ictal event with spike-wave bursts | ~97% Seizure |
| 🟠 Postictal | Post-seizure delta slowing | ~93% Postictal |

**File Upload:** Drag-and-drop a `.edf`, `.csv`, or `.txt` file for real inference against the trained model.

---

## 📁 Project Structure

```
EEG-Signals/
│
├── 📄 neuroscan_dashboard.html       # Frontend — clinical dashboard UI
├── 🧠 backend_model_completed.pt     # Pre-trained model weights (~45 MB)
├── 📊 processed_metadata.csv         # Training data index mapping
├── 📦 requirements.txt               # Python dependencies
├── 📖 README.md                      # This file
├── 📋 SETUP.md                       # Quick setup guide
│
└── code/
    ├── 🐍 app.py                     # Flask API — all 4 pipeline stages
    ├── 🤖 model.py                   # EEG_2D_Hybrid_Model definition
    ├── 🌊 preprocess_features.py     # CWT feature extraction (train + inference)
    ├── 🔧 preprocess_seizure_only.py # Targeted seizure data preprocessor
    ├── 🔍 find_hardest_seizure.py    # Edge case seizure locator
    ├── 🧪 test_manual_seizure.py     # Model accuracy verification
    ├── 📈 train_full.py              # Full training pipeline
    ├── ⚡ train_optimized.py          # Memory-optimized training
    ├── 🔄 run_pipeline.py            # End-to-end pipeline runner
    ├── 📊 resource_monitor.py        # Hardware safety guard
    ├── ⚙️ training_config.yaml       # Resource threshold config
    └── 🌐 render.yaml                # Render.com deployment config
```

---

## 🔌 API Reference

### Authentication
```http
POST /api/auth
Content-Type: application/json

{ "practice_id": "DEMO_CLINIC", "key": "NS2026" }
```

### Predict — JSON
```http
POST /api/predict
Content-Type: application/json
X-Auth-Token: <token>

{ "signal_data": [[...22 channels...]], "patient_id": "patient_001" }
```

### Predict — File Upload
```http
POST /api/predict
Content-Type: multipart/form-data
X-Auth-Token: <token>

file: eeg_recording.edf
```

### Demo Signal
```http
GET /api/demo-signal/{normal|preictal|seizure|postictal}
```

### Health Check
```http
GET /api/health
```

---

## 🧬 Dataset

**CHB-MIT Scalp EEG Database** (PhysioNet):

- 24 pediatric patients with intractable epilepsy
- 22-channel EEG (international 10-20 system)
- 256 Hz sampling rate · ~983 hours · 198 annotated seizures

```
FP1-F7  F7-T7  T7-P7  P7-O1  FP1-F3  F3-C3  C3-P3  P3-O1
FP2-F4  F4-C4  C4-P4  P4-O2  FP2-F8  F8-T8  T8-P8  P8-O2
FZ-CZ   CZ-PZ  P7-T7  T7-FT9 FT9-FT10 FT10-T8
```

> Shoeb, A. H. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD Thesis, MIT.

---

## ⚙️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Backend | Flask 3.0+ | REST API server |
| Frontend | HTML / CSS / JS | Clinical dashboard UI |
| Deep Learning | PyTorch 2.2+ | CNN-Transformer model |
| Signal Processing | SciPy | IIR / Butterworth filters |
| Wavelet Transform | PyWavelets | CWT scalogram generation |
| EEG Parsing | MNE-Python | Medical EDF file reader |
| Visualization | Matplotlib | Scalogram PNG rendering |
| Deployment | Gunicorn + Render | Production WSGI server |

---

## 📚 References

1. Abiyev, R. et al. (2020). *Identification of Epileptic EEG Signals Using CNN.* Applied Sciences, 10(12), 4089.
2. Shoeb, A. H. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD Thesis, MIT.
3. Goldberger, A. et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 101(23), e215–e220.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built with 🧠 by ScriptOrbit</b>
</p>
