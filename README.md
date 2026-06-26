# TinyML Predictive Maintenance System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![ML](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost%20%7C%20LightGBM%20%7C%20SVM-green)
![Embedded](https://img.shields.io/badge/Target-RPi%20%7C%20STM32%20%7C%20Arduino-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Predict bearing and motor faults from raw vibration data using signal processing and machine learning — with edge deployment on Raspberry Pi or STM32.

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
  Trained ML Model     ←── Random Forest / XGBoost / LightGBM / SVM
         │
         ▼
  Fault Prediction
  Healthy | Inner Race | Outer Race | Ball Fault
         │
         ▼
  Dashboard / Terminal
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

## Fault Classes

| Class | Description | Key Indicator |
|---|---|---|
| Healthy | Normal operation | Kurtosis ~3, low crest factor |
| Inner Race | Fault on inner ring | BPFI harmonics (~162 Hz), AM modulation |
| Outer Race | Fault on outer ring | BPFO harmonics (~107 Hz), fixed amplitude |
| Ball | Rolling element fault | BSF harmonics (~70 Hz), FTF modulation |

Bearing fault frequencies based on SKF 6205 geometry (CWRU dataset parameters) at 1797 RPM.

## Model Comparison

Evaluated on 800 samples (200/class), 5-fold stratified CV:

| Model | Accuracy | F1 |
|---|---|---|
| Random Forest | ~99% | ~0.99 |
| XGBoost | ~99% | ~0.99 |
| LightGBM | ~99% | ~0.99 |
| SVM (RBF) | ~97% | ~0.97 |

## Repository Structure

```
tinyml-predictive-maintenance/
├── src/
│   ├── acquisition/
│   │   └── simulator.py       ← Synthetic bearing fault signal generator
│   ├── preprocessing/
│   │   └── pipeline.py        ← Normalize + Hann windowing
│   ├── features/
│   │   ├── time_domain.py     ← RMS, kurtosis, crest factor...
│   │   ├── frequency_domain.py← FFT peaks, spectral centroid...
│   │   └── extractor.py       ← Combined feature pipeline
│   ├── models/
│   │   ├── train.py           ← RF, XGBoost, LightGBM, SVM + CV
│   │   └── evaluate.py        ← Confusion matrix, per-class metrics
│   ├── inference/
│   │   └── inference.py       ← Edge inference for RPi
│   └── dashboard/
│       └── monitor.py         ← Real-time terminal dashboard
├── firmware/
│   └── inference_stub.c       ← C feature extraction for STM32/Arduino
├── models/saved/              ← Trained model .pkl files
├── run.py                     ← One-command pipeline
└── requirements.txt
```

## Embedded Deployment

The `firmware/inference_stub.c` implements time-domain feature extraction in bare-metal C (no stdlib dependencies). Export the trained RandomForest to C using [emlearn](https://emlearn.org):

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

## License
MIT — [Dushyanth](https://github.com/dushyanth1409), M.Sc. Electromobility (AI & Autonomous Driving), FAU Erlangen-Nürnberg
