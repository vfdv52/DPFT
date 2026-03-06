# GenAI usage: Claude (claude-sonnet-4-6) assisted in drafting this file.
# All data processing logic was reviewed and verified manually by the authors.

import os
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

DATASET_URLS = {
    'ETTh1': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
    'ETTh2': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv',
}


def download_data(dataset: str = 'ETTh1') -> str:
    """Download dataset CSV if not already present. Returns local file path."""
    assert dataset in DATASET_URLS, \
        f"Unknown dataset '{dataset}'. Choose from {list(DATASET_URLS)}"
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = os.path.join(_DATA_DIR, f'{dataset}.csv')
    if not os.path.exists(path):
        print(f'Downloading {dataset}...')
        urllib.request.urlretrieve(DATASET_URLS[dataset], path)
        print(f'Saved to {path}')
    return path


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for univariate time series forecasting."""

    def __init__(self, data: np.ndarray, input_len: int, pred_len: int):
        # data: (T, 1) float32 array
        self.data = torch.FloatTensor(data)
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_len]             # (input_len, 1)
        y = self.data[idx + self.input_len:
                      idx + self.input_len + self.pred_len]  # (pred_len, 1)
        return x, y.squeeze(-1)                              # y: (pred_len,)


def get_dataloaders(
    dataset: str = 'ETTh1',
    target_col: str = 'OT',
    input_len: int = 96,
    pred_len: int = 24,
    batch_size: int = 32,
):
    """
    Download data, split chronologically (70/10/20), fit scaler on train,
    and return (train_loader, val_loader, test_loader, scaler).
    """
    path = download_data(dataset)
    df = pd.read_csv(path)
    values = df[target_col].values.reshape(-1, 1).astype(np.float32)

    n = len(values)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.8)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(values[:train_end])
    val_data   = scaler.transform(values[train_end:val_end])
    test_data  = scaler.transform(values[val_end:])

    train_loader = DataLoader(
        TimeSeriesDataset(train_data, input_len, pred_len),
        batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TimeSeriesDataset(val_data, input_len, pred_len),
        batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        TimeSeriesDataset(test_data, input_len, pred_len),
        batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
