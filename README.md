# COMP0197 CW2 — Probabilistic Time Series Forecasting

Group project for COMP0197 Applied Deep Learning.
**Task**: Predict future values of a time series and quantify the associated uncertainty using a probability distribution output.

---

## Project Overview

This project implements and evaluates probabilistic forecasting models on the ETT (Electricity Transformer Temperature) datasets. Rather than producing single-point predictions, the main models output a Gaussian distribution (mean μ and standard deviation σ) at each forecast step, capturing both aleatoric and epistemic uncertainty.

**Models** (ablation hierarchy):
- `DeterministicLSTM` — deterministic baseline, trained with MSE loss
- `DeterministicTransformer` — encoder-only Transformer, trained with MSE loss
- `ProbabilisticTransformer` (Trans+A) — Transformer with Gaussian output head, trained with NLL loss
- `DecompDeterministicTransformer` (Trans+B) — Transformer with seasonal decomposition, trained with MSE loss
- `DecompProbabilisticTransformer` (Trans+A+B) — Transformer with Gaussian head + seasonal decomposition, trained with NLL loss; supports MC Dropout for epistemic uncertainty

**Datasets**: ETTh1 and ETTh2 (hourly, univariate OT column), auto-downloaded from GitHub.

---

## File Structure

```
CW2/
├── train.py                  # Entry point: download data, train all models, save weights
├── test.py                   # Entry point: load weights, evaluate, generate plots
├── run.sh                    # One-shot script: trains then evaluates with configurable flags
│
├── models/
│   ├── __init__.py           # Re-exports all model classes
│   ├── lstm.py               # DeterministicLSTM — 2-layer LSTM + linear head
│   ├── transformer.py        # Deterministic/Probabilistic Transformers + SeasonalDecomp variants
│   ├── gnn.py                # DeterministicGNN (optional baseline)
│   └── informer.py           # DeterministicInformer (optional baseline)
│
├── utils/
│   ├── __init__.py           # Re-exports all public functions
│   ├── data.py               # Data download, TimeSeriesDataset, get_dataloaders()
│   └── metrics.py            # gaussian_nll (loss), NLL, CRPS, coverage, compute_all_metrics()
│
├── data/                     # Auto-created; stores downloaded CSV files
├── saved_models/             # Auto-created by train.py; stores .pt weight files
├── results/                  # Auto-created by test.py; stores output plots and .npy arrays
│
├── environment.yml           # Micromamba environment spec
├── checklist.md              # Compliance checklist against assignment requirements
└── README.md                 # This file
```

---

## Environment Setup

Requires the `comp0197-pt` micromamba environment.

```bash
# Option A: create from spec (if environment does not exist)
micromamba env create -f environment.yml

# Option B: environment already exists — install the two extra packages only
micromamba run -n comp0197-pt pip install pandas==3.0.1 scikit-learn==1.8.0
```

**Extra packages used (2 of 3 allowed):**

| Package | Version | Use |
|---|---|---|
| `pandas` | 3.0.1 | CSV data loading |
| `scikit-learn` | 1.8.0 | `StandardScaler` normalisation |

All other dependencies (`torch`, `numpy`, `pillow`) are included in the base `comp0197-pt` environment.

---

## Quickstart

```bash
# Train all models and evaluate in one step (recommended)
micromamba run -n comp0197-pt bash run.sh

# Or run separately
micromamba run -n comp0197-pt python train.py
micromamba run -n comp0197-pt python test.py
```

No manual data download required. `train.py` fetches ETTh1 and ETTh2 via `urllib` on first run.

---

## Models

### `DeterministicLSTM`

Standard 2-layer LSTM for point forecasting. Used as the deterministic baseline.

- **Input**: `(B, input_len=96, 1)`
- **Output**: `(B, pred_len=24)` — point prediction per step
- **Loss**: MSE

### `DeterministicTransformer`

Encoder-only Transformer for point forecasting. Ablation baseline isolating the Transformer architecture from probabilistic components.

- **Input**: `(B, input_len=96, 1)`
- **Output**: `(B, pred_len=24)`
- **Loss**: MSE

### `ProbabilisticTransformer` (Trans+A)

Encoder-only Transformer with a Gaussian output head. Adds aleatoric uncertainty estimation over the deterministic Transformer.

- **Output**: `(mu, log_sigma)` each `(B, pred_len=24)`
- **Loss**: Gaussian NLL
- **Architecture**: Linear projection → learnable positional embedding → `nn.TransformerEncoder` → last-token pooling → two linear heads (μ, log σ)

### `DecompDeterministicTransformer` (Trans+B)

Transformer with trend-seasonal decomposition. Isolates the contribution of the decomposition module.

- **Decomposition**: moving average trend extraction; seasonal = input − trend
- **Loss**: MSE

### `DecompProbabilisticTransformer` (Trans+A+B)

Full model combining the Gaussian head (A) and seasonal decomposition (B).

- **Loss**: Gaussian NLL
- **MC Dropout**: call `model.train()` at inference time and run N=50 stochastic forward passes to decompose uncertainty into aleatoric (from σ) and epistemic (from MC variance) components

---

## Utilities

### `utils/data.py`

| Function / Class | Description |
|---|---|
| `download_data(dataset)` | Downloads CSV from GitHub if not cached locally |
| `TimeSeriesDataset` | Sliding-window `torch.utils.data.Dataset`; returns `(x: L×1, y: pred_len)` |
| `get_dataloaders(...)` | Splits data 70/10/20, fits `StandardScaler` on train, returns three `DataLoader`s + scaler |

### `utils/metrics.py`

| Function | Description |
|---|---|
| `gaussian_nll(mu, log_sigma, y)` | Training loss — Gaussian NLL (torch tensors) |
| `nll_gaussian(mu, sigma, y)` | Evaluation NLL in nats (numpy) |
| `crps_gaussian(mu, sigma, y)` | Closed-form CRPS for Gaussian distribution (numpy, no scipy) |
| `coverage(mu, sigma, y, alpha)` | Empirical coverage of two-sided prediction interval |
| `compute_all_metrics(mu, sigma, y)` | Returns dict with MSE, MAE, NLL, CRPS, Coverage_90 |

---

## Ablation Results

| Model | ETTh1 MSE | ETTh1 MAE | ETTh1 NLL | ETTh2 MSE | ETTh2 MAE | ETTh2 NLL |
|---|---|---|---|---|---|---|
| LSTM | 0.0467 | 0.1615 | — | 0.1303 | 0.2632 | — |
| Transformer | 0.0469 | 0.1630 | — | 0.1222 | 0.2542 | — |
| Trans+A (Gaussian) | 0.0470 | 0.1593 | -0.0631 | 0.1215 | 0.2484 | 0.1850 |
| Trans+B (SeasonalDecomp) | 0.0449 | 0.1557 | — | 0.1195 | 0.2520 | — |
| Trans+A+B (Full, MC Dropout) | 0.0451 | 0.1547 | -0.1944 | 0.1118 | 0.2357 | 0.1142 |

---

## Output Plots (`results/`)

`test.py` generates four plots per dataset:

| File | Description |
|---|---|
| `{dataset}_transA_pi.png` | Trans+A predicted mean ± 90% PI (deterministic inference) |
| `{dataset}_transAB_pi.png` | Trans+A+B predicted mean ± 90% PI (MC Dropout) |
| `{dataset}_calibration_transA.png` | Reliability diagram — Trans+A |
| `{dataset}_calibration_transAB.png` | Reliability diagram — Trans+A+B |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Input length | 96 steps (4 days) |
| Forecast horizon | 24 steps (1 day) |
| Batch size | 32 |
| Epochs | 30 (with early stopping, patience=15) |
| Learning rate | 1e-3 (with ReduceLROnPlateau, patience=7) |
| LSTM hidden size | 64 |
| Transformer d_model | 64 |
| Transformer heads | 4 |
| Transformer layers | 2 |
| Feedforward dim | 256 |
| Dropout | 0.2 |
| MC Dropout samples | 50 |

---

## Dependencies

| Package | Source | Use |
|---|---|---|
| `torch` | comp0197-pt base | Models, training, inference |
| `numpy` | comp0197-pt base | Metrics and array operations |
| `pillow` | comp0197-pt base | Plot generation (no matplotlib) |
| `pandas` | extra (1/3) | CSV data loading |
| `scikit-learn` | extra (2/3) | `StandardScaler` normalisation |
