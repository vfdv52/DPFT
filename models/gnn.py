# GenAI usage: Claude (claude-sonnet-4-6) assisted in drafting this file.
# All architectural choices were reviewed and verified manually by the authors.

import torch
import torch.nn as nn


class DeterministicGNN(nn.Module):
    """
    Temporal GCN baseline for time-series forecasting.

    Graph construction: each look-back time step is a node; edges connect every
    pair of steps within a ±K-hop window (self-loops included).  Two learnable
    GCN layers aggregate temporal context; the last node's representation is
    projected to pred_len outputs with MSE loss.
    """

    _K = 3  # neighbourhood radius

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        pred_len: int = 24,
        input_len: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, pred_len)

        # Precompute symmetrically normalised adjacency with self-loops
        A = torch.zeros(input_len, input_len)
        for i in range(input_len):
            for j in range(max(0, i - self._K), min(input_len, i + self._K + 1)):
                A[i, j] = 1.0          # diagonal = 1 (self-loop)
        deg        = A.sum(1)
        d_inv_sqrt = deg.pow(-0.5)
        A_norm     = d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)
        self.register_buffer('A_norm', A_norm)  # (input_len, input_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_len, input_size)
        Returns:
            out: (B, pred_len) — point predictions
        """
        h = self.input_proj(x)                      # (B, L, hidden)
        for layer in self.gcn_layers:
            h = torch.relu(layer(self.A_norm @ h))  # (B, L, hidden)
            h = self.drop(h)
        return self.fc(h[:, -1, :])                 # (B, pred_len)
