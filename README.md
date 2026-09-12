# SSD
# Automatic Sleep/Wake Event Detection from Wrist-Worn Accelerometer Data

A production-ready machine learning and deep learning framework for detecting **sleep onset** and **wake-up events** from continuous raw wrist-worn wearable accelerometer signals (`anglez` and `enmo`).

---

## Target Architecture & Methodology

The repository is built as a **progressive experiment pipeline**, establishing benchmarks at each level of complexity:

```
Raw Wearable Signals (anglez, enmo)
        ↓
Data Understanding & Series-Level Split (GroupKFold by series_id to prevent leakage)
        ↓
Signal Preprocessing & Robustness Perturbation Testing
        ↓
Baseline 0: Rule-Based Activity Threshold
        ↓
Classical ML (Logistic Regression, Random Forest, LightGBM) on Rolling Window Features
        ↓
Temporal Deep Learning (1D-CNN & Conv-GRU) on Continuous 2-Channel Sequences
        ↓
Event Post-Processing (Probability Smoothing + Hysteresis Persistence + Refractory Period Filtering)
        ↓
Two-Level Evaluation (State Classification & Multi-Tolerance Event Bipartite Matching: ±30s, ±1m, ±2m, ±5m, ±15m)
```

---


## Quick Start

### 1. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Complete Benchmark End-to-End
Runs Baseline 0, Logistic Regression, Random Forest, LightGBM, 1D CNN, Ablations, and Sensor Robustness experiments with one command:
```bash
python3 scripts/run_all.py
```

### 3. Run Individual Components
- **Baseline 0 (Activity Threshold):**
  ```bash
  python3 scripts/run_baseline.py
  ```
- **Classical Models (Logistic Regression, Random Forest, LightGBM):**
  ```bash
  python3 scripts/train_classical.py
  ```
- **1D Convolutional Neural Network:**
  ```bash
  python3 scripts/train_cnn.py
  ```
- **Ablation Studies:**
  ```bash
  python3 scripts/run_ablations.py
  ```
- **Sensor Perturbation & Robustness:**
  ```bash
  python3 scripts/run_robustness.py
  ```

---

## Key Methodological Highlights

1. **Strict Series-Level Validation:**
   - Splitting is performed exclusively on `series_id` (recordings/subjects), preventing window leakage where adjacent 30-second windows contaminate the validation fold.
2. **Feature Engineering Families:**
   - **Signal Statistics:** Mean, standard deviation, min, max, median, 25%/75% quantiles, IQR, range.
   - **Dynamic Features:** First-order differences, absolute mean rate of change, rolling variance.
   - **Spectral / Frequency:** Spectral energy, spectral entropy, dominant frequency (via FFT).
   - **Temporal Context:** 24-hour cyclical sine/cosine encodings.
3. **Event Post-Processing (Hysteresis):**
   - Raw probabilities often oscillate near decision boundaries. A two-threshold hysteresis state machine with persistence requirements ($\ge 1\text{ min}$ for onset, $\ge 30\text{s}$ for wake) and minimum refractory separation eliminates high-frequency false-positive transitions.
4. **Tolerance-Based Evaluation:**
   - Measures event detection precision, recall, and F1 across multiple realistic tolerances ($\pm 30\text{s}$, $\pm 1\text{m}$, $\pm 2\text{m}$, $\pm 5\text{m}$, $\pm 15\text{m}$) using bipartite matching.
5. **Real-World Sensor Robustness:**
   - Evaluates performance under Gaussian sensor chatter, 10% packet drops / off-wrist chunks, mechanical impact spikes, and static sensor angle drift.
