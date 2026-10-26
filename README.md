```markdown
# Jet Engine Remaining Useful Life (RUL) Prediction

> An end-to-end deep learning pipeline for predictive maintenance, estimating the remaining operational cycles of jet engines using the NASA C-MAPSS dataset (FD001) and a hybrid CNN-LSTM network in PyTorch.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement and RUL Definition](#problem-statement-and-rul-definition)
3. [Dataset Description](#dataset-description)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [Model Architecture](#model-architecture)
6. [Training Strategy](#training-strategy)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Repository Structure](#repository-structure)
9. [Installation and Requirements](#installation-and-requirements)
10. [Usage Guide](#usage-guide)
11. [Model Persistence and Serialization](#model-persistence-and-serialization)
12. [Results and Benchmarks](#results-and-benchmarks)
13. [Hyperparameter Tuning](#hyperparameter-tuning)
14. [Troubleshooting](#troubleshooting)
15. [References](#references)
16. [License](#license)

---

## Project Overview

This repository provides an end-to-end data-driven Prognostics and Health Management (PHM) pipeline to forecast the **Remaining Useful Life (RUL)** of turbofan jet engines using time-series sensor degradation data.

The project demonstrates a production-grade machine learning workflow using **PyTorch**, covering data acquisition, sensor filtering, rolling feature engineering, temporal sequence windowing, network architecture design, and asymmetric performance evaluation.

---

## Problem Statement and RUL Definition

Predictive maintenance aims to determine when an industrial component will fail so maintenance can be scheduled proactively—preventing catastrophic failure while avoiding premature replacement.

### Remaining Useful Life (RUL)

At any given operational cycle $t$ for a specific engine:

$$\text{RUL}(t) = t_{\text{failure}} - t$$

- **Input:** A temporal sequence of multi-variate sensor readings over a historical window of $N$ cycles ($SEQ\_LEN = 30$).
- **Output:** A continuous scalar prediction representing the expected number of remaining safe operating cycles.

---

## Dataset Description

The project uses the **FD001** benchmark subset from the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset.

### Subset Characteristics (FD001)

- **Training Set:** 100 engine units run continuously until complete failure.
- **Test Set:** 100 engine units with truncated operating histories (the goal is to predict RUL at the final recorded cycle).
- **Operating Conditions:** Single operating condition (Sea Level).
- **Fault Mode:** Single failure mode (High-Pressure Compressor degradation).

### Raw Features per Cycle

Each row contains 26 attributes:
- `unit_nr`: Unique engine identification number (1 to 100).
- `time_cycle`: Current operational cycle counter.
- `setting_1`, `setting_2`, `setting_3`: Operational condition settings.
- `s_1` through `s_21`: 21 continuous sensor measurements (temperatures, pressures, shaft speeds, flow ratios).

---

## Data Preprocessing Pipeline

To maximize feature quality and prevent data leakage, the pipeline applies structured data transformations:

### 1. Target Label Engineering (Piecewise Linear Capping)
Early in an engine's life cycle, component wear is negligible and sensors show no degradation signal. To prevent the model from attempting to fit linear decay during healthy operation, RUL targets are capped at **125 cycles**:

$$\text{Target RUL}(t) = \min(\text{True RUL}(t), 125)$$

### 2. Feature Removal (Zero-Variance Channels)
Channels with zero variance across cycles carry no signal and are dropped:
- **Operational Settings:** `setting_1`, `setting_2`, `setting_3` (constant in FD001).
- **Flat Sensors:** `s_1`, `s_5`, `s_6`, `s_10`, `s_16`, `s_18`, `s_19`.

This filters the original 21 sensors down to **14 informative sensor channels**.

### 3. Feature Scaling
Standardization ($\mu = 0, \sigma = 1$) is applied using `StandardScaler`. To prevent data leakage, the scaler is **fitted strictly on the training set** and applied to the test set.

### 4. Rolling Feature Engineering
A 10-cycle rolling mean (`ROLL_WIN = 10`) is computed per engine unit for each active sensor channel to smooth high-frequency sensor noise and highlight underlying degradation trajectories. This expands the input space to **28 features per time step**.

### 5. Temporal Sequence Windowing
The tabular matrix is transformed into overlapping sliding windows:
- **Training Samples:** Input tensor shape `(Batch, 30, 28)` paired with the target RUL at cycle $t$.
- **Test Samples:** The final 30 cycles of each test engine are extracted to evaluate RUL at the last recorded operational state.

---

## Model Architecture

The model uses a hybrid **1D CNN + LSTM (RUL_CNNLSTM)** architecture designed for time-series regression.

```text
Input Matrix: (Batch Size, 30 Time Steps, 28 Features)
       |
       v
1D Convolution Layer (In: 28, Out: 32, Kernel: 3, Padding: 1)
       |
       v
Batch Normalization (1D) + ReLU Activation
       |
       v
2-Layer LSTM Network (Hidden Size: 64, Batch First: True)
       |
       v
Extract Final Sequence State: (Batch Size, 64)
       |
       v
Layer Normalization (64)
       |
       v
Linear Layer 1 (64 -> 32) + ReLU Activation
       |
       v
Linear Layer 2 (32 -> 1)
       |
       v
Output Scalar: Predicted Remaining Useful Life

```

---

## Training Strategy

* **Loss Function:** `HuberLoss(delta=1.0)` is used instead of Mean Squared Error to increase robustness against extreme RUL outliers.
* **Optimizer:** `AdamW` (Learning Rate = $10^{-3}$, Weight Decay = $10^{-4}$).
* **Learning Rate Scheduler:** `ReduceLROnPlateau` halves the learning rate when validation loss plateaus for 3 consecutive epochs.
* **Gradient Clipping:** Gradients are clipped to `max_norm=1.0` to stabilize recurrent backpropagation.
* **Early Stopping:** Training terminates if validation loss does not improve for 8 consecutive epochs (`PATIENCE = 8`). The model automatically reloads the state checkpoint with the lowest validation loss.

---

## Evaluation Metrics

Model performance is evaluated on the test set (`RUL_FD001.txt`) using two standards:

### 1. Root Mean Squared Error (RMSE)

Measures average magnitude error in cycles:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})^2}$$

### 2. NASA Asymmetric Scoring Function

In aerospace maintenance, **overestimating** RUL (predicting an engine is healthy when it is near failure) is far more dangerous than **underestimating** RUL (replacing an engine early). The NASA score penalizes optimistic errors exponentially higher:

$$S = \sum_{i=1}^{N} s_i$$

where $d_i = y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)}$:

$$s_i = \begin{cases} e^{-d_i / 13} - 1, & d_i < 0 \quad (\text{Late / Pessimistic Prediction}) \\ e^{d_i / 10} - 1, & d_i \ge 0 \quad (\text{Early / Optimistic Prediction}) \end{cases}$$

Lower scores indicate safer and more accurate system performance.

---

## Repository Structure

```text
.
├── README.md                  # Project documentation
├── rul_lstm.py                # Single end-to-end execution script
├── best_lstm.pt               # Saved PyTorch state checkpoint
├── turbofan_rul_model.pkl     # Pickled model state dictionary
└── cmapss_data/               # Extracted dataset directory (auto-created)
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt

```

---

## Installation and Requirements

### Environment Requirements

* Python 3.8 or higher
* PyTorch 2.0 or higher
* CUDA-compatible GPU (Optional, CPU execution supported)

### Install Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib

```

---

## Usage Guide

To run the complete data acquisition, preprocessing, training, evaluation, and plotting pipeline in a single command:

```bash
python rul_lstm.py

```

The script will automatically handle dataset downloading if the `cmapss_data` directory is missing.

---

## Model Persistence and Serialization

### Saving Model Weights to `.pkl` Format

To serialize the trained state dictionary using Python's `pickle` module:

```python
import pickle
import torch

def save_model_pkl(model, filepath="turbofan_rul_model.pkl"):
    """Serializes model state dictionary to a binary pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(model.state_dict(), f)
    print(f"Model successfully saved to {filepath}")

# Usage after training:
save_model_pkl(model, "turbofan_rul_model.pkl")

```

### Loading Model Weights from `.pkl` Format

To load saved weights into a newly instantiated model object:

```python
import pickle
import torch

def load_model_pkl(model, filepath="turbofan_rul_model.pkl", device='cpu'):
    """Loads weights from a pickle file into the model instance."""
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model successfully loaded from {filepath}")
    return model

# Usage for inference:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model = RUL_CNNLSTM(input_dim=28).to(device)
loaded_model = load_model_pkl(loaded_model, "turbofan_rul_model.pkl", device=device)

```

---

## Results and Benchmarks

Typical baseline metrics obtained on the **FD001 test set**:

| Metric | Target / Benchmark Range |
| --- | --- |
| **Test RMSE (vs. Capped RUL)** | **~12.5 – 15.0 cycles** |
| **Test RMSE (vs. True RUL)** | **~16.0 – 19.0 cycles** |
| **NASA Asymmetric Score** | **~250 – 450** |

---

## Hyperparameter Tuning

Key configuration constants defined in the pipeline:

```python
RUL_CAP   = 125  # Piecewise linear target cap
SEQ_LEN   = 30   # Input sliding sequence length (cycles)
ROLL_WIN  = 10   # Rolling mean smoothing window size
EPOCHS    = 80   # Maximum training iterations
PATIENCE  = 8    # Early stopping patience

```

---

## Troubleshooting

* **Dataset Download Failures:** If the automated script cannot fetch `CMAPSSData.zip` due to network restrictions, download the ZIP archive manually from the NASA Prognostics Data Repository and place it in the root directory.
* **CUDA Device Allocation:** The code checks for GPU support automatically via `torch.cuda.is_available()`. If running on CPU only, memory pinning is disabled automatically.




```

```
