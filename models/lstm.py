# GenAI usage: AI used to supplement inline comments.

import torch
import torch.nn as nn


class DeterministicLSTM(nn.Module):
    """
    Deterministic LSTM baseline for point forecasting.
    Trained with MSE loss; outputs a single value per horizon step.
    Used as Ablation 1 baseline against the probabilistic Transformer.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        pred_len: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_len, input_size)
        Returns:
            out: (B, pred_len) — point predictions
        """
        out, _ = self.lstm(x)          # (B, L, hidden_size)
        return self.fc(out[:, -1, :])  # use last hidden state
