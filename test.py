# GenAI usage: Claude (claude-sonnet-4-6) assisted in drafting this file.
# All evaluation logic, MC Dropout procedure, and visualisation choices were
# reviewed and verified manually by the authors.


"""
test.py — Evaluate saved models and generate plots.

Usage:
    python test.py [options]
    bash run.sh [--run_name NAME ...]

For each dataset, evaluates all 5 ablation models (MSE, MAE) and saves
prediction-interval plots and calibration curves to results/.
"""

import os
import argparse
import torch
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from models import (DeterministicGNN, DeterministicInformer,
                    DeterministicLSTM, DeterministicTransformer, ProbabilisticTransformer,
                    DecompDeterministicTransformer, DecompProbabilisticTransformer)
from utils import get_dataloaders, compute_all_metrics

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate saved probabilistic time series models.')
    p.add_argument('--input_len',       type=int,   default=96,    help='Look-back window length')
    p.add_argument('--pred_len',        type=int,   default=24,    help='Forecast horizon length')
    p.add_argument('--batch_size',      type=int,   default=64,    help='Evaluation batch size')
    p.add_argument('--hidden_size',     type=int,   default=64,    help='LSTM hidden size')
    p.add_argument('--d_model',         type=int,   default=64,    help='Transformer model dimension')
    p.add_argument('--nhead',           type=int,   default=4,     help='Transformer attention heads')
    p.add_argument('--num_layers',      type=int,   default=2,     help='Number of LSTM/Transformer layers')
    p.add_argument('--dim_feedforward', type=int,   default=256,   help='Transformer feedforward dimension')
    p.add_argument('--dropout',         type=float, default=0.4,   help='Dropout rate (must match training)')
    p.add_argument('--mc_samples',      type=int,   default=50,    help='MC Dropout forward passes')
    p.add_argument('--datasets',        type=str,   default='ETTh1 ETTh2',
                   help='Space-separated list of datasets to evaluate')
    p.add_argument('--target_col',      type=str,   default='OT',  help='Target column name in CSV')
    p.add_argument('--run_name',        type=str,   default='default',
                   help='Must match the --run_name used in train.py')
    p.add_argument('--run_gnn',         action='store_true', help='Evaluate GNN baseline')
    p.add_argument('--run_informer',    action='store_true', help='Evaluate Informer baseline')
    return p.parse_args()


args = parse_args()

INPUT_LEN       = args.input_len
PRED_LEN        = args.pred_len
BATCH_SIZE      = args.batch_size
HIDDEN_SIZE     = args.hidden_size
D_MODEL         = args.d_model
NHEAD           = args.nhead
NUM_LAYERS      = args.num_layers
DIM_FEEDFORWARD = args.dim_feedforward
DROPOUT         = args.dropout
MC_SAMPLES      = args.mc_samples
DATASETS        = args.datasets.split()
TARGET_COL      = args.target_col
RUN_NAME        = args.run_name
RUN_GNN         = args.run_gnn
RUN_INFORMER    = args.run_informer
SAVE_DIR        = 'saved_models'
RESULTS_DIR     = 'results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect_lstm_preds(model, loader):
    """Return (mu, y_true) as flat numpy arrays."""
    model.eval()
    mus, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mus.append(model(x).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(mus).ravel(), np.concatenate(ys).ravel()


def collect_transformer_preds(model, loader):
    """Deterministic forward pass → (mu, sigma, y_true) flat arrays."""
    model.eval()
    mus, sigmas, ys = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, log_sigma = model(x)
            sigma = torch.exp(torch.clamp(log_sigma, -6.0, 6.0))
            mus.append(mu.cpu().numpy())
            sigmas.append(sigma.cpu().numpy())
            ys.append(y.numpy())
    return (
        np.concatenate(mus).ravel(),
        np.concatenate(sigmas).ravel(),
        np.concatenate(ys).ravel(),
    )


def collect_mc_dropout_preds(model, loader, n_samples=MC_SAMPLES):
    """
    MC Dropout inference.
    Keeps dropout active during inference (model.train()) and runs
    n_samples stochastic forward passes.

    Returns:
        mu_total:    combined predictive mean (B*pred_len,)
        sigma_total: combined predictive std, decomposed as:
                     sigma_total^2 = epistemic_var + mean(aleatoric_var)
        y_true:      (B*pred_len,)
    """
    model.train()   # keep dropout active

    all_x, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            all_x.append(x.to(device))
            all_y.append(y.numpy())

    sample_mus    = []   # list of (N_total, pred_len) arrays
    sample_vars   = []   # aleatoric variance per sample

    for _ in range(n_samples):
        batch_mus, batch_vars = [], []
        with torch.no_grad():
            for x in all_x:
                mu, log_sigma = model(x)
                sigma = torch.exp(torch.clamp(log_sigma, -6.0, 6.0))
                batch_mus.append(mu.cpu().numpy())
                batch_vars.append((sigma ** 2).cpu().numpy())
        sample_mus.append(np.concatenate(batch_mus))    # (N, pred_len)
        sample_vars.append(np.concatenate(batch_vars))  # (N, pred_len)

    sample_mus  = np.stack(sample_mus,  axis=0)  # (S, N, pred_len)
    sample_vars = np.stack(sample_vars, axis=0)  # (S, N, pred_len)

    mu_total        = sample_mus.mean(axis=0)                  # (N, pred_len)
    epistemic_var   = sample_mus.var(axis=0)                   # (N, pred_len)
    aleatoric_var   = sample_vars.mean(axis=0)                 # (N, pred_len)
    sigma_total     = np.sqrt(epistemic_var + aleatoric_var)   # (N, pred_len)
    y_true          = np.concatenate(all_y)                    # (N, pred_len)

    return mu_total.ravel(), sigma_total.ravel(), y_true.ravel()


# Canvas layout constants
_W, _H   = 1200, 400
_PAD_L   = 60   # left padding (y-axis area)
_PAD_R   = 20
_PAD_T   = 40
_PAD_B   = 50
_PLOT_W  = _W - _PAD_L - _PAD_R
_PLOT_H  = _H - _PAD_T - _PAD_B

_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0  )
_GREY   = (180, 180, 180)
_BLUE   = (70,  130, 180)   # steelblue
_LBLUE  = (70,  130, 180, 80)   # translucent for PI band (RGBA)
_RED    = (200, 60,  60  )


def _data_to_px(values, vmin, vmax):
    """Map data values → pixel y-coordinates (top-down)."""
    normed = (values - vmin) / (vmax - vmin + 1e-8)
    return (_PAD_T + _PLOT_H - normed * _PLOT_H).astype(int)


def _x_to_px(indices, n):
    """Map time indices → pixel x-coordinates."""
    return (_PAD_L + indices / (n - 1) * _PLOT_W).astype(int)


def _draw_line(draw, xs, ys, color, width=1):
    for i in range(len(xs) - 1):
        draw.line([(xs[i], ys[i]), (xs[i+1], ys[i+1])], fill=color, width=width)


def plot_prediction_interval(mu, sigma, y_true, title, save_path, n_steps=200):
    """
    Plot predicted mean ± 90% interval vs ground truth and save as PNG.
    Uses Pillow (PIL) — no matplotlib dependency.
    """
    n = min(n_steps, len(mu))
    idx = np.arange(n)

    lo = mu[:n] - 1.645 * sigma[:n]
    hi = mu[:n] + 1.645 * sigma[:n]

    vmin = min(y_true[:n].min(), lo.min())
    vmax = max(y_true[:n].max(), hi.max())

    xs = _x_to_px(idx.astype(float), n)

    # Use RGBA for the shaded band, then composite onto white
    img  = Image.new('RGB',  (_W, _H), _WHITE)
    band = Image.new('RGBA', (_W, _H), (255, 255, 255, 0))
    draw_band = ImageDraw.Draw(band)

    # Draw shaded PI band as filled polygon
    ys_hi = _data_to_px(hi, vmin, vmax)
    ys_lo = _data_to_px(lo, vmin, vmax)
    poly = list(zip(xs, ys_hi)) + list(zip(reversed(xs), reversed(ys_lo)))
    draw_band.polygon(poly, fill=(70, 130, 180, 60))
    img = Image.alpha_composite(img.convert('RGBA'), band).convert('RGB')

    draw = ImageDraw.Draw(img)

    # Axes
    draw.rectangle([_PAD_L, _PAD_T, _PAD_L + _PLOT_W, _PAD_T + _PLOT_H], outline=_BLACK)

    # Ground truth (black) and predicted mean (blue)
    ys_true = _data_to_px(y_true[:n], vmin, vmax)
    ys_mu   = _data_to_px(mu[:n],     vmin, vmax)
    _draw_line(draw, xs, ys_true, _BLACK, width=1)
    _draw_line(draw, xs, ys_mu,   _BLUE,  width=1)

    # Legend
    draw.rectangle([_PAD_L + 10, _PAD_T + 8, _PAD_L + 25, _PAD_T + 18], fill=_BLACK)
    draw.text((_PAD_L + 30, _PAD_T + 5),  'Ground truth',   fill=_BLACK)
    draw.rectangle([_PAD_L + 130, _PAD_T + 8, _PAD_L + 145, _PAD_T + 18], fill=_BLUE)
    draw.text((_PAD_L + 150, _PAD_T + 5), 'Predicted mean', fill=_BLACK)
    draw.rectangle([_PAD_L + 280, _PAD_T + 8, _PAD_L + 295, _PAD_T + 18],
                   fill=(70, 130, 180))
    draw.text((_PAD_L + 300, _PAD_T + 5), '90% PI',         fill=_BLACK)

    # Title and axis labels
    draw.text((_W // 2 - len(title) * 3, 10), title, fill=_BLACK)
    draw.text((_W // 2 - 20, _H - 18), 'Time step', fill=_BLACK)
    draw.text((5, _H // 2 - 10), 'OT', fill=_BLACK)

    img.save(save_path)
    print(f'  Saved: {save_path}')


def plot_calibration(mu, sigma, y_true, title, save_path):
    """
    Reliability diagram: expected vs actual coverage at each confidence level.
    Uses Pillow (PIL) — no matplotlib dependency.
    """
    from utils import coverage as cov_fn
    alphas    = np.linspace(0.05, 0.95, 19)
    coverages = np.array([cov_fn(mu, sigma, y_true, alpha=a) for a in alphas])

    # Map both axes [0,1] onto pixel space
    def ax_to_px_x(v):
        return int(_PAD_L + v * _PLOT_W)

    def ax_to_px_y(v):
        return int(_PAD_T + _PLOT_H - v * _PLOT_H)

    sz = 500
    img  = Image.new('RGB', (sz, sz), _WHITE)
    draw = ImageDraw.Draw(img)

    pw = sz - _PAD_L - _PAD_R
    ph = sz - _PAD_T - _PAD_B

    def to_px(alpha_v, cov_v):
        x = int(_PAD_L + alpha_v * pw)
        y = int(_PAD_T + ph - cov_v * ph)
        return x, y

    # Perfect calibration diagonal
    _draw_line(draw,
               [_PAD_L, _PAD_L + pw],
               [_PAD_T + ph, _PAD_T],
               _GREY, width=1)

    # Model calibration curve
    xs_px = [to_px(a, c)[0] for a, c in zip(alphas, coverages)]
    ys_px = [to_px(a, c)[1] for a, c in zip(alphas, coverages)]
    _draw_line(draw, xs_px, ys_px, _BLUE, width=2)
    for x, y in zip(xs_px, ys_px):
        draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=_BLUE)

    # Axes border
    draw.rectangle([_PAD_L, _PAD_T, _PAD_L + pw, _PAD_T + ph], outline=_BLACK)

    # Title and axis labels
    draw.text((sz // 2 - len(title) * 3, 5), title, fill=_BLACK)
    draw.text((sz // 2 - 40, sz - 15), 'Expected coverage', fill=_BLACK)
    draw.text((5, sz // 2 - 10), 'Observed', fill=_BLACK)

    img.save(save_path)
    print(f'  Saved: {save_path}')


def evaluate_dataset(dataset: str, results: dict):
    print(f'\nEvaluating on {dataset}')

    _, _, test_loader, _ = get_dataloaders(
        dataset, target_col=TARGET_COL, input_len=INPUT_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    tag = RUN_NAME  # shorthand

    def _point_metrics(mu, y):
        return {'MSE': float(np.mean((y - mu) ** 2)),
                'MAE': float(np.mean(np.abs(y - mu)))}

    if RUN_GNN:
        gnn = DeterministicGNN(
            input_size=1, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
        ).to(device)
        gnn.load_state_dict(torch.load(
            os.path.join(SAVE_DIR, f'gnn_{dataset}_{tag}.pt'), map_location=device))
        mu_gnn, y_true_gnn = collect_lstm_preds(gnn, test_loader)
        m_gnn = _point_metrics(mu_gnn, y_true_gnn)
        results[f'{dataset}_GNN'] = m_gnn
        print(f'  [1] GNN              | MSE {m_gnn["MSE"]:.4f} | MAE {m_gnn["MAE"]:.4f}')

    if RUN_INFORMER:
        informer = DeterministicInformer(
            input_size=1, d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
            pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
        ).to(device)
        informer.load_state_dict(torch.load(
            os.path.join(SAVE_DIR, f'informer_{dataset}_{tag}.pt'), map_location=device))
        mu_inf, y_true_inf = collect_lstm_preds(informer, test_loader)
        m_inf = _point_metrics(mu_inf, y_true_inf)
        results[f'{dataset}_Informer'] = m_inf
        print(f'  [2] Informer         | MSE {m_inf["MSE"]:.4f} | MAE {m_inf["MAE"]:.4f}')

    lstm = DeterministicLSTM(
        input_size=1, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, pred_len=PRED_LEN, dropout=DROPOUT,
    ).to(device)
    lstm.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, f'lstm_{dataset}_{tag}.pt'), map_location=device))
    mu_lstm, y_true = collect_lstm_preds(lstm, test_loader)
    m_lstm = _point_metrics(mu_lstm, y_true)
    results[f'{dataset}_LSTM'] = m_lstm
    print(f'  [3] LSTM             | MSE {m_lstm["MSE"]:.4f} | MAE {m_lstm["MAE"]:.4f}')

    det_tf = DeterministicTransformer(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    det_tf.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, f'det_transformer_{dataset}_{tag}.pt'), map_location=device))
    mu_det, y_true_det = collect_lstm_preds(det_tf, test_loader)
    m_det = _point_metrics(mu_det, y_true_det)
    results[f'{dataset}_Transformer'] = m_det
    print(f'  [4] Transformer            | MSE {m_det["MSE"]:.4f} | MAE {m_det["MAE"]:.4f}')

    transformer = ProbabilisticTransformer(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    transformer.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, f'transformer_{dataset}_{tag}.pt'), map_location=device))
    mu_tf, sigma_tf, y_true_tf = collect_transformer_preds(transformer, test_loader)
    m_tf = compute_all_metrics(mu_tf, sigma_tf, y_true_tf)
    results[f'{dataset}_Transformer+GaussianHead'] = m_tf
    print(f'  [5] Transformer+GaussianHead | MSE {m_tf["MSE"]:.4f} | MAE {m_tf["MAE"]:.4f} | NLL {m_tf["NLL"]:.4f}')

    decomp_det = DecompDeterministicTransformer(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    decomp_det.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, f'decomp_transformer_{dataset}_{tag}.pt'), map_location=device))
    mu_decomp, y_true_decomp = collect_lstm_preds(decomp_det, test_loader)
    m_decomp = _point_metrics(mu_decomp, y_true_decomp)
    results[f'{dataset}_Transformer+SeasonalDecomp'] = m_decomp
    print(f'  [6] Transformer+SeasonalDecomp        | MSE {m_decomp["MSE"]:.4f} | MAE {m_decomp["MAE"]:.4f}')

    decomp_tf = DecompProbabilisticTransformer(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    decomp_tf.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, f'decomp_prob_transformer_{dataset}_{tag}.pt'), map_location=device))
    mu_dmc, sigma_dmc, y_true_dmc = collect_mc_dropout_preds(decomp_tf, test_loader)
    m_dmc = compute_all_metrics(mu_dmc, sigma_dmc, y_true_dmc)
    results[f'{dataset}_Transformer+GaussianHead+SeasonalDecomp'] = m_dmc
    print(f'  [7] Transformer+GaussianHead+SeasonalDecomp | MSE {m_dmc["MSE"]:.4f} | MAE {m_dmc["MAE"]:.4f} | NLL {m_dmc["NLL"]:.4f}')

    plot_prediction_interval(
        mu_tf, sigma_tf, y_true_tf,
        title=f'{dataset} — Trans+A prediction interval',
        save_path=os.path.join(RESULTS_DIR, f'{dataset}_transA_pi.png'),
    )
    plot_prediction_interval(
        mu_dmc, sigma_dmc, y_true_dmc,
        title=f'{dataset} — Trans+A+B (MC Dropout) prediction interval',
        save_path=os.path.join(RESULTS_DIR, f'{dataset}_transAB_pi.png'),
    )
    plot_calibration(
        mu_tf, sigma_tf, y_true_tf,
        title=f'{dataset} — Calibration (Trans+A)',
        save_path=os.path.join(RESULTS_DIR, f'{dataset}_calibration_transA.png'),
    )
    plot_calibration(
        mu_dmc, sigma_dmc, y_true_dmc,
        title=f'{dataset} — Calibration (Trans+A+B)',
        save_path=os.path.join(RESULTS_DIR, f'{dataset}_calibration_transAB.png'),
    )

    np.save(os.path.join(RESULTS_DIR, f'{dataset}_mu_transA.npy'),    mu_tf)
    np.save(os.path.join(RESULTS_DIR, f'{dataset}_sigma_transA.npy'), sigma_tf)
    np.save(os.path.join(RESULTS_DIR, f'{dataset}_mu_transAB.npy'),   mu_dmc)
    np.save(os.path.join(RESULTS_DIR, f'{dataset}_sigma_transAB.npy'),sigma_dmc)
    np.save(os.path.join(RESULTS_DIR, f'{dataset}_y_true.npy'),       y_true_tf)


def print_summary(results: dict):
    print(f'\nSUMMARY TABLE')
    header = f'{"Model":<40} {"MSE":>8} {"MAE":>8} {"NLL":>8}'
    print(header)
    print('-' * 68)
    for name, m in results.items():
        row = (f'{name:<40} '
               f'{m.get("MSE", float("nan")):>8.4f} '
               f'{m.get("MAE", float("nan")):>8.4f} '
               f'{m.get("NLL", float("nan")):>8.4f}')
        print(row)


if __name__ == '__main__':
    results = {}
    for ds in DATASETS:
        evaluate_dataset(ds, results)
    print_summary(results)
    print(f'\nPlots saved to ./{RESULTS_DIR}/')
