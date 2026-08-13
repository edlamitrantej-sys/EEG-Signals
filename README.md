<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/MNE--Python-1.6+-0C479D?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🧠 NeuroScan — Epileptic Seizure Detection System</h1>

<p align="center">
  <b>A real-time, AI-powered EEG analysis platform for epileptic seizure detection using a hybrid CNN-Transformer architecture with Continuous Wavelet Transform (CWT) feature engineering.</b>
</p>

<p align="center">
  <a href="#-architecture">Architecture</a> •
  <a href="#-pipeline-stages">Pipeline</a> •
  <a href="#-model-architecture">Model</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-results">Results</a>
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
- 🔐 Session-based clinician authentication
- 📁 Supports `.edf` (medical), `.csv`, and `.txt` EEG file formats
- 🎮 Built-in demo mode with synthetic signals for all 4 seizure states

---

## 🏗 Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NeuroScan System Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│   │  STAGE 1  │───▶│   STAGE 2     │───▶│   STAGE 3     │───▶│   STAGE 4   │  │
│   │  Data     │    │  Feature      │    │  AI Model     │    │  Clinical   │  │
│   │  Ingestion│    │  Engineering  │    │  Inference    │    │  Output     │  │
│   └──────────┘    └──────────────┘    └──────────────┘    └─────────────┘  │
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│   │ • EDF     │    │ • Complex     │    │ • 2D CNN      │    │ • Dashboard │  │
│   │ • CSV/TXT │    │   Morlet CWT  │    │   (spatial)   │    │ • Scalograms│  │
│   │ • JSON    │    │ • 22ch × 50f  │    │ • Transformer │    │ • Reports   │  │
│   │ • Demo    │    │   × 512t      │    │   (temporal)  │    │ • Confidence│  │
│   │           │    │ • Scalogram   │    │ • 4-class     │    │   Scores    │  │
│   │ Notch 60Hz│    │   rendering   │    │   softmax     │    │ • Auth      │  │
│   │ BP 0.5-50 │    │               │    │               │    │             │  │
│   │ Z-Score   │    │               │    │               │    │             │  │
│   └──────────┘    └──────────────┘    └──────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
                    ┌─────────────────────────────────┐
                    │      Clinician / Researcher      │
                    └────────────┬──────────────────────┘
                                 │ Login (Practice ID + Key)
                                 ▼
                    ┌─────────────────────────────────┐
                    │     POST /api/auth               │
                    │     → Session Token (8hr)        │
                    └────────────┬──────────────────────┘
                                 │
              ┌──────────────────┼──────────────────────┐
              │                  │                      │
              ▼                  ▼                      ▼
     ┌────────────────┐ ┌────────────────┐  ┌───────────────────┐
     │  Demo Signal   │ │  File Upload   │  │   JSON API Call   │
     │  (4 types)     │ │  (.edf/.csv)   │  │   (signal_data)   │
     └───────┬────────┘ └───────┬────────┘  └────────┬──────────┘
              │                  │                     │
              └──────────────────┼─────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────────┐
              │         POST /api/predict                 │
              │                                          │
              │  Stage 1: Notch → Bandpass → Z-Score     │
              │  Stage 2: CWT Scalogram (22×50×512)      │
              │  Stage 3: CNN-Transformer → Softmax      │
              │  Stage 4: Generate Report + Scalograms   │
              │                                          │
              └──────────────────┬───────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────────┐
              │            JSON Response                  │
              │  • prediction: "Seizure (Ictal)"         │
              │  • confidences: {Normal: 0.01, ...}      │
              │  • scalogram: base64 PNG (ch FP1-F7)     │
              │  • scalogram_2: base64 PNG (ch C3-P3)    │
              │  • medical_notes: "NEUROSCAN REPORT..."  │
              │  • elapsed_ms: 142                       │
              └──────────────────────────────────────────┘
```

---

## 🔬 Pipeline Stages

### Stage 1 — Data Ingestion & Signal Preprocessing

Raw EEG signals undergo a multi-step cleaning pipeline before reaching the model:

```
Raw EEG Signal (22 channels × N samples)
        │
        ▼
┌──────────────────────────┐
│  IIR Notch Filter (60Hz) │  ← Removes power-line interference
└──────────┬───────────────┘
           ▼
┌──────────────────────────────────┐
│  Butterworth Bandpass (0.5–50Hz) │  ← Retains brain-relevant frequencies only
└──────────┬───────────────────────┘
           ▼
┌──────────────────────────────────┐
│  Epoch Extraction (2s windows)   │  ← 512 samples @ 256Hz, 50% overlap
│  + Artifact Rejection (>300µV)   │  ← Discards noisy/saturated segments
└──────────┬───────────────────────┘
           ▼
┌──────────────────────────────────┐
│  Z-Score Normalization           │  ← Mean=0, Std=1 per channel
└──────────┬───────────────────────┘
           ▼
   Clean Signal: (22 × 512)
```

**Supported Input Formats:**

| Format | Parser | Use Case |
|---|---|---|
| `.edf` | MNE-Python | Medical EEG files (European Data Format) |
| `.csv` / `.txt` | Built-in CSV parser | Exported recordings, research data |
| JSON | Direct API input | Real-time streaming, integrations |
| Demo | Synthetic generator | Testing & demonstration |

---

### Stage 2 — CWT Scalogram Generation (Feature Engineering)

The 1D time-domain EEG signal is transformed into a 2D frequency-time representation using the **Continuous Wavelet Transform** with a **Complex Morlet wavelet** (`cmor1.5-1.0`):

```
Input:  1D signal per channel ──── (512 time samples)
                │
                ▼
        ┌───────────────────────┐
        │  Complex Morlet CWT   │
        │  Frequencies: 1–50 Hz │
        │  50 logarithmic bins  │
        └───────────┬───────────┘
                    ▼
Output: 2D scalogram per channel ── (50 freq bins × 512 time steps)

All 22 channels combined: (22 × 50 × 512) tensor
```

**Why CWT instead of raw signal?**
> Raw EEG is a noisy 1D voltage trace. The CWT decomposes it into a frequency-time heatmap where seizure-specific patterns (rhythmic 3–8 Hz bursts, spike-wave complexes) become **visually distinct** — making them far easier for a CNN to detect.

---

### Stage 3 — CNN-Transformer Hybrid Model

The core of NeuroScan is a **hybrid architecture** combining 2D CNNs for spatial feature extraction with Transformer encoders for temporal attention:

```
Input: (Batch, 22, 50, 512) — 22 channels × 50 freq bins × 512 time steps
                │
                ▼
┌─────────────────────────────────────────────────┐
│              2D CNN FEATURE EXTRACTOR            │
│                                                  │
│  Block 1: Conv2D(22→32) + BatchNorm + ReLU      │
│           + MaxPool2D(2×2)                       │
│           → (Batch, 32, 25, 256)                 │
│                                                  │
│  Block 2: Conv2D(32→64) + BatchNorm + ReLU      │
│           + MaxPool2D(2×2)                       │
│           → (Batch, 64, 12, 128)                 │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│              RESHAPE BRIDGE                      │
│                                                  │
│  Permute: (Batch, 64, 12, 128)                  │
│        → (Batch, 128, 64, 12)                   │
│  Flatten: → (Batch, 128, 768)                   │
│                                                  │
│  128 time steps, each with 768-dim features     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│          TRANSFORMER ENCODER                     │
│                                                  │
│  2 Encoder Layers                               │
│  8 Self-Attention Heads                         │
│  d_model = 768, d_ff = 2048                     │
│  → (Batch, 128, 768)                            │
│                                                  │
│  Learns temporal dependencies across time       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│       GLOBAL AVERAGE POOLING                     │
│  Mean over 128 time steps → (Batch, 768)        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│          CLASSIFIER HEAD                         │
│                                                  │
│  Linear(768 → 256) + ReLU + Dropout(0.5)       │
│  Linear(256 → 4)                                │
│  → Softmax → 4 class probabilities              │
│                                                  │
│  [Normal | Preictal | Seizure | Postictal]      │
└─────────────────────────────────────────────────┘
```

**Why this hybrid approach?**

| Component | Strength | Captures |
|---|---|---|
| **2D CNN** | Spatial pattern recognition | Spike shapes, frequency band activations, scalogram textures |
| **Transformer** | Long-range temporal attention | How seizure patterns evolve and propagate over time |
| **Combined** | Best of both worlds | Spatial features + temporal context = robust classification |

**Training Details:**

| Parameter | Value |
|---|---|
| Dataset | CHB-MIT Scalp EEG Database (24 pediatric patients) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-2) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss | Cross-Entropy with class weights `[0.1, 0.4, 0.9, 0.4]` |
| Regularization | Dropout 50%, BatchNorm, weight decay |
| Precision | Mixed (FP16 on GPU via AMP) |

> **Class weighting** is critical — in real EEG data, seizures represent <1% of recording time. Without weighting, the model would learn to always predict "Normal" and achieve 99% accuracy while being clinically useless.

---

### Stage 4 — Clinical Dashboard & Output

The Flask API returns a comprehensive JSON response, rendered by the clinical dashboard:

| Output | Description |
|---|---|
| **Classification** | Top predicted class with confidence percentage |
| **Confidence Scores** | Probability distribution across all 4 classes |
| **CWT Scalograms** | Base64-encoded heatmap images from channels FP1-F7 and C3-P3 |
| **Signal Preview** | 512-point waveform for visual inspection |
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

Navigate to **http://localhost:5000** in your browser.

Login with demo credentials:
- **Practice ID:** `DEMO_CLINIC`
- **Access Key:** `NS2026`

---

## 🎮 Demo

Once logged in, use the **sidebar control panel** to test the system:

| Button | Simulates | Expected Confidence |
|---|---|---|
| 🟢 Normal | Healthy resting-state EEG | ~97% Normal |
| 🟡 Preictal | Pre-seizure warning signals | ~89% Preictal |
| 🔴 Seizure | Active ictal event with spike-wave bursts | ~97% Seizure |
| 🟠 Postictal | Post-seizure delta slowing | ~93% Postictal |

**File Upload:** Drag-and-drop a `.edf`, `.csv`, or `.txt` file into the upload zone for real inference against the trained model.

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
├── code/
│   ├── 🐍 app.py                     # Flask API — all 4 pipeline stages
│   ├── 🤖 model.py                   # EEG_2D_Hybrid_Model definition
│   ├── 🌊 preprocess_features.py     # CWT feature extraction
│   ├── 🔧 preprocess_seizure_only.py # Targeted seizure data preprocessor
│   ├── 🔍 find_hardest_seizure.py    # Edge case seizure locator
│   ├── 🧪 test_manual_seizure.py     # Model accuracy verification
│   ├── 📈 train_full.py              # Full training pipeline
│   ├── ⚡ train_optimized.py          # Memory-optimized training
│   ├── 🔄 run_pipeline.py            # End-to-end pipeline runner
│   ├── 📊 resource_monitor.py        # Hardware safety guard
│   ├── ⚙️ training_config.yaml       # Resource threshold config
│   └── 🌐 render.yaml                # Render.com deployment config
│
└── .gitignore
```

---

## 🔌 API Reference

### Authentication

```http
POST /api/auth
Content-Type: application/json

{
  "practice_id": "DEMO_CLINIC",
  "key": "NS2026"
}
```

**Response:**
```json
{
  "token": "a1b2c3...",
  "clinician": "Demo Clinic",
  "practice_id": "DEMO_CLINIC"
}
```

### Predict (JSON)

```http
POST /api/predict
Content-Type: application/json
X-Auth-Token: <session_token>

{
  "signal_data": [[...], [...], ...],
  "patient_id": "patient_001"
}
```

### Predict (File Upload)

```http
POST /api/predict
Content-Type: multipart/form-data
X-Auth-Token: <session_token>

file: <eeg_recording.edf>
```

### Demo Signals

```http
GET /api/demo-signal/{normal|preictal|seizure|postictal}
```

### Health Check

```http
GET /api/health
```

---

## 🧬 Dataset

This project uses the **CHB-MIT Scalp EEG Database** from PhysioNet:

- **24 pediatric patients** with intractable epilepsy
- **22-channel** EEG recordings (international 10-20 system)
- **256 Hz** sampling rate
- Annotated seizure onset/offset times
- ~983 hours of continuous EEG with 198 annotated seizures

**EEG Channel Montage:**
```
FP1-F7  F7-T7  T7-P7  P7-O1  FP1-F3  F3-C3  C3-P3  P3-O1
FP2-F4  F4-C4  C4-P4  P4-O2  FP2-F8  F8-T8  T8-P8  P8-O2
FZ-CZ   CZ-PZ  P7-T7  T7-FT9 FT9-FT10 FT10-T8
```

> **Reference:** Shoeb, A. H. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD Thesis, MIT.

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Backend | Flask 3.0+ | REST API server |
| Frontend | HTML/CSS/JS | Clinical dashboard UI |
| Deep Learning | PyTorch 2.2+ | CNN-Transformer model |
| Signal Processing | SciPy | IIR/Butterworth filters |
| Wavelet Transform | PyWavelets | CWT scalogram generation |
| EEG Parsing | MNE-Python | Medical EDF file reader |
| Visualization | Matplotlib | Scalogram rendering |
| Deployment | Gunicorn + Render | Production WSGI server |

---

## 📚 References

1. Abiyev, R. et al. (2020). *"Identification of Epileptic EEG Signals Using Convolutional Neural Networks."* Applied Sciences, 10(12), 4089.
2. Shoeb, A. H. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD Thesis, MIT.
3. Goldberger, A. et al. (2000). *"PhysioBank, PhysioToolkit, and PhysioNet."* Circulation, 101(23), e215–e220.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built with 🧠 by ScriptOrbit</b>
</p>
