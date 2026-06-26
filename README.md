# TinyML Predictive Maintenance System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![ML](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost%20%7C%20LightGBM%20%7C%20SVM%20%7C%20NN-green)
![Edge](https://img.shields.io/badge/Edge-TFLite%20Micro%20%7C%20emlearn-red)
![Embedded](https://img.shields.io/badge/Target-RPi%20%7C%20STM32%20%7C%20Arduino-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Predict bearing and motor faults from raw vibration data using signal processing and machine learning — with edge deployment on Raspberry Pi, STM32, or Arduino-class microcontrollers via **two** interchangeable paths: a decision-tree model exported with emlearn, and a quantized neural network running on **TensorFlow Lite for Microcontrollers (TFLM)**.

## System Architecture

```
Accelerometer (IMU / MPU-6050)
         │
         ▼
  Signal Acquisition  ←── simulator.py (synthetic data)
         │
         ▼
   Preprocessing       ←── normalize → window (Hann)
         │
         ▼
  Feature Extraction   ←── 11 time domain + 11 frequency domain = 22 features
         │
         ▼
  Trained Model        ←── RandomForest / XGBoost / LightGBM / SVM   (host)
         │                 └── int8 Neural Network                   (edge, TFLM)
         ▼
  Fault Prediction
  Healthy | Inner Race | Outer Race | Ball Fault
         │
         ├─→ Dashboard / Terminal           (host)
         ├─→ emlearn C model                (STM32 / Arduino, decision tree)
         └─→ TFLite Micro int8 model        (STM32 / Arduino, neural network)
```

## Features Extracted

**Time Domain (11):** RMS, Mean, Variance, Std Dev, Kurtosis, Skewness, Peak, Peak-to-Peak, Crest Factor, Shape Factor, Impulse Factor

**Frequency Domain (11):** Dominant Frequency, Spectral Centroid, Spectral Spread, Band Energy (0–1kHz, 1–3kHz, 3–6kHz), FFT Peaks (top 3), Spectral Entropy, Spectral Flatness

## Quickstart

```bash
git clone https://github.com/dushyanth1409/tinyml-predictive-maintenance.git
cd tinyml-predictive-maintenance
pip install -r requirements.txt

# Train all models + evaluate
python run.py

# Run inference simulation
python run.py --simulate

# Live terminal dashboard
python src/dashboard/monitor.py
```

### Neural network + TensorFlow Lite Micro

```bash
pip install -r requirements-nn.txt        # TensorFlow + sklearn/pandas/scipy

# Train the NN, quantize to int8, and emit the firmware C arrays in one go
python run.py --nn --no-plot

# …or run the two stages directly
python src/models/train_nn.py             # Keras → int8 .tflite (+ nn_metadata.json)
python src/models/export_tflm.py          # → firmware/tflm/model_data.{cc,h}, model_settings.h
```

## Fault Classes

| Class | Description | Key Indicator |
|---|---|---|
| Healthy | Normal operation | Kurtosis ~3, low crest factor |
| Inner Race | Fault on inner ring | BPFI harmonics (~162 Hz), AM modulation |
| Outer Race | Fault on outer ring | BPFO harmonics (~107 Hz), fixed amplitude |
| Ball | Rolling element fault | BSF harmonics (~70 Hz), FTF modulation |

Bearing fault frequencies based on SKF 6205 geometry (CWRU dataset parameters) at 1797 RPM.

## Model Comparison

Evaluated on 800 samples (200/class), 5-fold stratified CV (tree models) / held-out test split (NN):

| Model | Accuracy | F1 | Notes |
|---|---|---|---|
| Random Forest | ~99% | ~0.99 | |
| XGBoost | ~99% | ~0.99 | |
| LightGBM | ~99% | ~0.99 | |
| SVM (RBF) | ~97% | ~0.97 | |
| Neural Network (int8, TFLM) | ~100% | ~1.00 | 5.1 KB model, no accuracy loss vs float32 |

> **How to read these numbers.** The synthetic generator places each fault class
> around distinct, well-separated bearing frequencies (BPFI / BPFO / BSF), so the
> four classes are nearly linearly separable — which is exactly why every model
> lands in the 97–100% range. These results validate that the **pipeline**
> (acquisition → features → model → int8 quantization → C export) is correct and
> that quantization is lossless here; they are **not** a real-world fault-detection
> rate. On measured data (e.g. the CWRU bearing dataset or live MPU-6050 captures)
> accuracy would be lower and the float-vs-int8 gap would become visible. Validating
> against real recordings is the natural next step.

## Repository Structure

```
tinyml-predictive-maintenance/
├── src/
│   ├── acquisition/
│   │   └── simulator.py        ← Synthetic bearing fault signal generator
│   ├── preprocessing/
│   │   └── pipeline.py         ← Normalize + Hann windowing
│   ├── features/
│   │   ├── time_domain.py      ← RMS, kurtosis, crest factor...
│   │   ├── frequency_domain.py ← FFT peaks, spectral centroid...
│   │   └── extractor.py        ← Combined feature pipeline
│   ├── models/
│   │   ├── train.py            ← RF, XGBoost, LightGBM, SVM + CV
│   │   ├── evaluate.py         ← Confusion matrix, per-class metrics
│   │   ├── train_nn.py         ← Keras NN training + int8 TFLite conversion
│   │   └── export_tflm.py      ← int8 .tflite → C arrays for TFLM firmware
│   ├── inference/
│   │   └── inference.py        ← Edge inference for RPi
│   └── dashboard/
│       └── monitor.py          ← Real-time terminal dashboard
├── firmware/
│   ├── inference_stub.c        ← C feature extraction for STM32/Arduino
│   └── tflm/                   ← TensorFlow Lite Micro neural-network firmware
│       ├── bearing_inference.cc/.h   ← Portable TFLM setup + classify()
│       ├── bearing_nn.ino            ← Arduino demo sketch
│       ├── model_data.cc/.h          ← (generated) model bytes as a C array
│       ├── model_settings.h          ← (generated) labels + StandardScaler params
│       └── README.md                 ← Build/deploy notes (Arduino + STM32)
├── models/saved/               ← Trained models: *.pkl, nn_int8.tflite, nn_metadata.json
├── run.py                      ← One-command pipeline (add --nn for the NN path)
├── requirements.txt
└── requirements-nn.txt         ← Extra deps for the NN/TFLM path
```

## Embedded Deployment

Both edge paths consume the same 22-feature vector produced by the feature
extractor (`firmware/inference_stub.c` implements that extraction in bare-metal
C, no stdlib dependencies). Pick whichever model suits your target.

### Option A — Decision tree (emlearn)

Export the trained RandomForest to inline C:

```bash
pip install emlearn
python -c "
import joblib, emlearn
model = joblib.load('models/saved/RandomForest.pkl')
clf = model.named_steps['clf']
cmodel = emlearn.convert(clf, method='inline')
cmodel.save(file='firmware/model_generated.h')
"
```

### Option B — Neural network (TensorFlow Lite Micro)

A small dense network (22 → 32 → 16 → 4) trained on the same features, fully
int8-quantized to ~5 KB. Feature scaling is applied **on-device** using the
StandardScaler parameters baked into `model_settings.h`, which keeps the int8
input tensor well-conditioned despite the very different feature ranges
(RMS vs Hz vs entropy).

```bash
python run.py --nn --no-plot
# generates firmware/tflm/model_data.{cc,h} and model_settings.h
```

Then build the firmware:

- **Arduino** (Nano 33 BLE Sense, Portenta, ESP32, …): drop the `firmware/tflm/`
  files into a sketch folder, install a TFLM library (`Arduino_TensorFlowLite`
  or `Chirale_TensorFlowLite`), and upload.
- **STM32 / bare-metal:** add the TFLM sources plus CMSIS-NN kernels to your
  build (X-CUBE-AI or a manual `tflite-micro` integration), then call
  `bearing_nn_init()` once and `bearing_nn_classify()` per window.

See [`firmware/tflm/README.md`](firmware/tflm/README.md) for the full build and
tuning notes (arena sizing, op resolver, CMSIS-NN).

## License
MIT — [Dushyanth](https://github.com/dushyanth1409), M.Sc. Electromobility (AI & Autonomous Driving), FAU Erlangen-Nürnberg
