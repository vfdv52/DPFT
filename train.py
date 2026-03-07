# GenAI usage: AI used to supplement inline comments.

"""
train.py — Train all ablation models and save best checkpoints.

Usage:
    python train.py [options]
    bash run.sh [--epochs N --lr 0.001 ...]

Models trained on each dataset:
    1. GNN         — DeterministicGNN (temporal GCN), MSE loss
    2. Informer    — DeterministicInformer (ProbSparse attention), MSE loss
    3. LSTM        — DeterministicLSTM, MSE loss
    4. Transformer — DeterministicTransformer, MSE loss
    5. Transformer+Gaussian Head            — ProbabilisticTransformer, Gaussian NLL loss
    6. Transformer+SeasonalDecomp           — DecompDeterministicTransformer (seasonal decomposition), MSE loss
    7. Transformer+Gaussian Head+SeasonalDecomp — DecompProbabilisticTransformer, Gaussian NLL loss

Checkpoints saved to: saved_models/{model}_{dataset}_{run_name}.pt
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import (DeterministicGNN, DeterministicInformer,
                    DeterministicLSTM, DeterministicTransformer, ProbabilisticTransformer,
                    DecompDeterministicTransformer, DecompProbabilisticTransformer)
from utils import get_dataloaders, gaussian_nll

def parse_args():
    p = argparse.ArgumentParser(description='Train probabilistic time series models.')
    p.add_argument('--input_len',       type=int,   default=96,    help='Look-back window length')
    p.add_argument('--pred_len',        type=int,   default=24,    help='Forecast horizon length')
    p.add_argument('--batch_size',      type=int,   default=32,    help='Training batch size')
    p.add_argument('--epochs',          type=int,   default=50,    help='Number of training epochs')
    p.add_argument('--lr',              type=float, default=1e-3,  help='Initial learning rate')
    p.add_argument('--patience',        type=int,   default=7,     help='ReduceLROnPlateau patience')
    p.add_argument('--hidden_size',     type=int,   default=64,    help='LSTM hidden size')
    p.add_argument('--d_model',         type=int,   default=64,    help='Transformer model dimension')
    p.add_argument('--nhead',           type=int,   default=4,     help='Transformer attention heads')
    p.add_argument('--num_layers',      type=int,   default=2,     help='Number of LSTM/Transformer layers')
    p.add_argument('--dim_feedforward', type=int,   default=256,   help='Transformer feedforward dimension')
    p.add_argument('--dropout',         type=float, default=0.4,   help='Dropout rate')
    p.add_argument('--early_stop',      type=int,   default=15,    help='Early stopping patience (epochs without val improvement)')
    p.add_argument('--datasets',        type=str,   default='ETTh1 ETTh2',
                   help='Space-separated list of datasets to train on')
    p.add_argument('--target_col',      type=str,   default='OT',  help='Target column name in CSV')
    p.add_argument('--run_name',        type=str,   default='default',
                   help='Tag appended to saved model filenames (for hyperparameter experiments)')
    p.add_argument('--clip_grad',       type=float, default=1.0,
                   help='Max gradient norm for clipping (0 = disabled)')
    p.add_argument('--run_gnn',         action='store_true', help='Train GNN baseline')
    p.add_argument('--run_informer',    action='store_true', help='Train Informer baseline')
    return p.parse_args()


args = parse_args()

INPUT_LEN       = args.input_len
PRED_LEN        = args.pred_len
BATCH_SIZE      = args.batch_size
EPOCHS          = args.epochs
LR              = args.lr
PATIENCE        = args.patience
HIDDEN_SIZE     = args.hidden_size
D_MODEL         = args.d_model
NHEAD           = args.nhead
NUM_LAYERS      = args.num_layers
DIM_FEEDFORWARD = args.dim_feedforward
DROPOUT         = args.dropout
DATASETS        = args.datasets.split()
TARGET_COL      = args.target_col
RUN_NAME        = args.run_name
CLIP_GRAD       = args.clip_grad
EARLY_STOP      = args.early_stop
RUN_GNN         = args.run_gnn
RUN_INFORMER    = args.run_informer
SAVE_DIR        = 'saved_models'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def _train_epoch_mse(model, loader, optimizer, criterion):
    """Generic MSE training epoch (used by LSTM and DeterministicTransformer)."""
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if CLIP_GRAD > 0:
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def _val_epoch_mse(model, loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item()
    return total / len(loader)


def _train_epoch_nll(model, loader, optimizer):
    """NLL training epoch for ProbabilisticTransformer."""
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        mu, log_sigma = model(x)
        loss = gaussian_nll(mu, log_sigma, y)
        loss.backward()
        if CLIP_GRAD > 0:
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def _val_epoch_nll(model, loader):
    """Returns (nll, mse) on the validation set."""
    model.eval()
    total_nll, total_mse = 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mu, log_sigma = model(x)
            total_nll += gaussian_nll(mu, log_sigma, y).item()
            total_mse += nn.functional.mse_loss(mu, y).item()
    n = len(loader)
    return total_nll / n, total_mse / n


def _fit(model, train_loader, val_loader, optimizer, train_fn, val_fn, label, save_path):
    """Generic training loop with LR scheduling, best-val checkpointing, and early stopping."""
    scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=0.5)
    best_val, best_state = float('inf'), None
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_fn(model, train_loader, optimizer)
        val_loss   = val_fn(model, val_loader)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(f'  Epoch {epoch:3d}/{EPOCHS} | train {label} {train_loss:.4f} | val {label} {val_loss:.4f}', flush=True)
        if no_improve >= EARLY_STOP:
            print(f'  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP} epochs)')
            break
    torch.save(best_state, save_path)
    print(f'  Best val {label}: {best_val:.4f}  ->  saved to {save_path}')


def _fit_nll(model, train_loader, val_loader, optimizer, save_path):
    """Training loop for probabilistic models: schedules on NLL, prints NLL + MSE."""
    scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=0.5)
    best_val, best_state = float('inf'), None
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        train_nll = _train_epoch_nll(model, train_loader, optimizer)
        val_nll, val_mse = _val_epoch_nll(model, val_loader)
        scheduler.step(val_nll)
        if val_nll < best_val:
            best_val   = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(f'  Epoch {epoch:3d}/{EPOCHS} | train NLL {train_nll:.4f} | val NLL {val_nll:.4f} | val MSE {val_mse:.4f}', flush=True)
        if no_improve >= EARLY_STOP:
            print(f'  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP} epochs)')
            break
    torch.save(best_state, save_path)
    print(f'  Best val NLL: {best_val:.4f}  ->  saved to {save_path}')


def train_gnn(dataset: str):
    """Train DeterministicGNN (temporal GCN) with MSE loss."""
    print(f'\nTraining GNN on {dataset} [{RUN_NAME}]')
    train_loader, val_loader, _, _ = get_dataloaders(
        dataset, target_col=TARGET_COL, input_len=INPUT_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE)
    model = DeterministicGNN(
        input_size=1, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    criterion = nn.MSELoss()
    _fit(model, train_loader, val_loader,
         Adam(model.parameters(), lr=LR),
         lambda m, l, o: _train_epoch_mse(m, l, o, criterion),
         lambda m, l:    _val_epoch_mse(m, l, criterion),
         'MSE', os.path.join(SAVE_DIR, f'gnn_{dataset}_{RUN_NAME}.pt'))


def train_informer(dataset: str):
    """Train DeterministicInformer (ProbSparse attention) with MSE loss."""
    print(f'\nTraining Informer on {dataset} [{RUN_NAME}]')
    train_loader, val_loader, _, _ = get_dataloaders(
        dataset, target_col=TARGET_COL, input_len=INPUT_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE)
    model = DeterministicInformer(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    criterion = nn.MSELoss()
    _fit(model, train_loader, val_loader,
         Adam(model.parameters(), lr=LR),
         lambda m, l, o: _train_epoch_mse(m, l, o, criterion),
         lambda m, l:    _val_epoch_mse(m, l, criterion),
         'MSE', os.path.join(SAVE_DIR, f'informer_{dataset}_{RUN_NAME}.pt'))


def train_lstm(dataset: str):
    """Train DeterministicLSTM with MSE loss; saves best val checkpoint."""
    print(f'\nTraining LSTM on {dataset} [{RUN_NAME}]')
    train_loader, val_loader, _, _ = get_dataloaders(
        dataset, target_col=TARGET_COL, input_len=INPUT_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE)
    model = DeterministicLSTM(
        input_size=1, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, pred_len=PRED_LEN, dropout=DROPOUT,
    ).to(device)
    criterion = nn.MSELoss()
    _fit(model, train_loader, val_loader,
         Adam(model.parameters(), lr=LR),
         lambda m, l, o: _train_epoch_mse(m, l, o, criterion),
         lambda m, l:    _val_epoch_mse(m, l, criterion),
         'MSE', os.path.join(SAVE_DIR, f'lstm_{dataset}_{RUN_NAME}.pt'))


def train_det_transformer(dataset: str):
    """Train DeterministicTransformer (standard FFN) with MSE loss."""
    print(f'\nTraining Transformer on {dataset} [{RUN_NAME}]')
    train_loader, val_loader, _, _ = get_dataloaders(
        dataset, target_col=TARGET_COL, input_len=INPUT_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE)
    model = DeterministicTransformer(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    criterion = nn.MSELoss()
    _fit(model, train_loader, val_loader,
         Adam(model.parameters(), lr=LR),
         lambda m, l, o: _train_epoch_mse(m, l, o, criterion),
         lambda m, l:    _val_epoch_mse(m, l, criterion),
         'MSE', os.path.join(SAVE_DIR, f'det_transformer_{dataset}_{RUN_NAME}.pt'))


def train_transformer(dataset: str):
    """Train ProbabilisticTransformer (Gaussian head) with NLL loss."""
    print(f'\nTraining Transformer+GaussianHead on {dataset} [{RUN_NAME}]')
    train_loader, val_loader, _, _ = get_dataloaders(
        dataset, target_col=TARGET_COL, input_len=INPUT_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE)
    model = ProbabilisticTransformer(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    _fit_nll(model, train_loader, val_loader,
             Adam(model.parameters(), lr=LR),
             os.path.join(SAVE_DIR, f'transformer_{dataset}_{RUN_NAME}.pt'))


def train_decomp_transformer(dataset: str):
    """Train DecompDeterministicTransformer (seasonal decomposition) with MSE loss."""
    print(f'\nTraining Transformer+SeasonalDecomp on {dataset} [{RUN_NAME}]')
    train_loader, val_loader, _, _ = get_dataloaders(
        dataset, target_col=TARGET_COL, input_len=INPUT_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE)
    model = DecompDeterministicTransformer(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    criterion = nn.MSELoss()
    _fit(model, train_loader, val_loader,
         Adam(model.parameters(), lr=LR),
         lambda m, l, o: _train_epoch_mse(m, l, o, criterion),
         lambda m, l:    _val_epoch_mse(m, l, criterion),
         'MSE', os.path.join(SAVE_DIR, f'decomp_transformer_{dataset}_{RUN_NAME}.pt'))


def train_decomp_prob_transformer(dataset: str):
    """Train DecompProbabilisticTransformer (seasonal decomposition + Gaussian head) with NLL loss."""
    print(f'\nTraining Transformer+GaussianHead+SeasonalDecomp on {dataset} [{RUN_NAME}]')
    train_loader, val_loader, _, _ = get_dataloaders(
        dataset, target_col=TARGET_COL, input_len=INPUT_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE)
    model = DecompProbabilisticTransformer(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        pred_len=PRED_LEN, input_len=INPUT_LEN, dropout=DROPOUT,
    ).to(device)
    _fit_nll(model, train_loader, val_loader,
             Adam(model.parameters(), lr=LR),
             os.path.join(SAVE_DIR, f'decomp_prob_transformer_{dataset}_{RUN_NAME}.pt'))


if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)
    for ds in DATASETS:
        if RUN_GNN:
            train_gnn(ds)
        if RUN_INFORMER:
            train_informer(ds)
        train_lstm(ds)
        train_det_transformer(ds)
        train_transformer(ds)
        train_decomp_transformer(ds)
        train_decomp_prob_transformer(ds)
    print('\nAll models trained and saved.')
