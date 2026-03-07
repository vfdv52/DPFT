# GenAI usage: AI used to supplement inline comments.

import torch
import torch.nn as nn


class DeterministicTransformer(nn.Module):
    """
    Encoder-only Transformer with a single MSE output head.
    Used as Ablation 2 baseline: same architecture as ProbabilisticTransformer
    but without the Gaussian head, to isolate the contribution of NLL training.
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        pred_len: int = 24,
        input_len: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj    = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Embedding(input_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_len, input_size)
        Returns:
            out: (B, pred_len) — point predictions
        """
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x) + self.pos_embedding(positions)
        h = self.encoder(h)
        return self.fc(h[:, -1, :])


class ProbabilisticTransformer(nn.Module):
    """
    Encoder-only Transformer with a Gaussian output head.

    Outputs (mu, log_sigma) per horizon step, enabling:
      - Aleatoric uncertainty via the learned sigma
      - Epistemic uncertainty via MC Dropout at inference time

    Architecture:
        Input → Linear projection → Learnable positional embedding
        → N x TransformerEncoderLayer (with dropout)
        → Last-token representation
        → Two independent linear heads: mu_head, log_sigma_head
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        pred_len: int = 24,
        input_len: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project raw input features into d_model dimensional space
        self.input_proj = nn.Linear(input_size, d_model)

        # Learnable positional embedding (better than fixed sinusoidal for time series)
        self.pos_embedding = nn.Embedding(input_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Gaussian output heads
        self.mu_head        = nn.Linear(d_model, pred_len)
        self.log_sigma_head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, input_len, input_size)
        Returns:
            mu:        (B, pred_len) — predicted mean
            log_sigma: (B, pred_len) — predicted log std (clamped in loss)
        """
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.input_proj(x) + self.pos_embedding(positions)  # (B, L, d_model)
        h = self.encoder(h)                                       # (B, L, d_model)

        feat = h[:, -1, :]                    # last-position token (B, d_model)
        mu        = self.mu_head(feat)        # (B, pred_len)
        log_sigma = self.log_sigma_head(feat) # (B, pred_len)
        return mu, log_sigma



class _MovingAvg(nn.Module):
    """
    Moving average for trend extraction (boundary-padded to preserve length).

    Pads the front and back with the first/last value respectively, then
    applies a 1-D average pooling with stride 1.  Output shape equals input.
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].expand(-1, pad, -1)
        back  = x[:, -1:, :].expand(-1, pad, -1)
        x_pad = torch.cat([front, x, back], dim=1)         # (B, L + 2*pad, C)
        return self.avg(x_pad.transpose(1, 2)).transpose(1, 2)  # (B, L, C)


class DecompDeterministicTransformer(nn.Module):
    """
    Encoder-only Transformer with trend-seasonal decomposition (Module B).

    Before encoding, the input is decomposed into:
        trend    = moving_avg(x)       — slow-varying component
        seasonal = x - trend           — residual oscillation

    The seasonal component is encoded by the standard Transformer encoder;
    the trend at the last time step is projected and added to the encoder
    output before the MSE prediction head.  This inductive bias improves
    modelling of non-stationary time series (e.g. ETTh datasets with
    strong daily/weekly periodicity).
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        pred_len: int = 24,
        input_len: int = 96,
        dropout: float = 0.1,
        moving_avg_kernel: int = 25,
    ):
        super().__init__()
        self.decomp        = _MovingAvg(moving_avg_kernel)
        self.input_proj    = nn.Linear(input_size, d_model)
        self.trend_proj    = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Embedding(input_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_len, input_size)
        Returns:
            out: (B, pred_len) — point predictions
        """
        trend    = self.decomp(x)       # (B, L, input_size)
        seasonal = x - trend            # (B, L, input_size)

        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h    = self.input_proj(seasonal) + self.pos_embedding(positions)
        h    = self.encoder(h)          # (B, L, d_model)
        feat = h[:, -1, :] + self.trend_proj(trend[:, -1, :])  # (B, d_model)
        return self.fc(feat)


class DecompProbabilisticTransformer(nn.Module):
    """
    Transformer with trend-seasonal decomposition (B) and Gaussian output head (A).
    Full model (Trans + A + B): decomposition + NLL training + MC Dropout at inference.

    Architecture:
        Input → decompose into trend + seasonal
        seasonal → Linear projection → Learnable positional embedding
        → N x TransformerEncoderLayer (dropout active in train mode)
        → Last-token + trend projection
        → Two independent linear heads: mu_head, log_sigma_head
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        pred_len: int = 24,
        input_len: int = 96,
        dropout: float = 0.1,
        moving_avg_kernel: int = 25,
    ):
        super().__init__()
        self.decomp        = _MovingAvg(moving_avg_kernel)
        self.input_proj    = nn.Linear(input_size, d_model)
        self.trend_proj    = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Embedding(input_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder        = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.mu_head        = nn.Linear(d_model, pred_len)
        self.log_sigma_head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, input_len, input_size)
        Returns:
            mu:        (B, pred_len) — predicted mean
            log_sigma: (B, pred_len) — predicted log std (clamped in loss)
        """
        trend    = self.decomp(x)
        seasonal = x - trend

        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h    = self.input_proj(seasonal) + self.pos_embedding(positions)
        h    = self.encoder(h)
        feat = h[:, -1, :] + self.trend_proj(trend[:, -1, :])
        return self.mu_head(feat), self.log_sigma_head(feat)
