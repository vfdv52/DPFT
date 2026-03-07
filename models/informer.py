# GenAI usage: AI used to supplement inline comments.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention from Informer (Zhou et al., 2021).

    Selects top-u queries by a max–mean sparsity score; remaining positions
    receive mean(V) as a fallback, reducing complexity from O(L²) to O(L log L).
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead  = nhead
        self.d_k    = d_model // nhead
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out    = nn.Linear(d_model, d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, dk   = self.nhead, self.d_k

        Q = self.q_proj(x).view(B, L, H, dk).transpose(1, 2)  # (B,H,L,dk)
        K = self.k_proj(x).view(B, L, H, dk).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, dk).transpose(1, 2)

        u = max(1, min(self.factor * math.ceil(math.log(L + 1)), L))

        # Sample u random keys to estimate query sparsity
        idx_k  = torch.randperm(L, device=x.device)[:u]
        K_samp = K[:, :, idx_k, :]                                        # (B,H,u,dk)
        scores = torch.matmul(Q, K_samp.transpose(-1, -2)) / math.sqrt(dk) # (B,H,L,u)
        M      = scores.max(-1).values - scores.mean(-1)                   # (B,H,L)

        _, top_idx = M.topk(u, dim=-1)                                     # (B,H,u)
        Q_top      = Q.gather(2, top_idx.unsqueeze(-1).expand(-1,-1,-1,dk))# (B,H,u,dk)

        attn    = torch.softmax(
            torch.matmul(Q_top, K.transpose(-1,-2)) / math.sqrt(dk), dim=-1)
        attn    = self.drop(attn)
        ctx_top = torch.matmul(attn, V)                                    # (B,H,u,dk)

        # Fill all positions with mean(V), then overwrite the top-u positions
        ctx = V.mean(dim=2, keepdim=True).expand(-1, -1, L, -1).clone()
        ctx.scatter_(2, top_idx.unsqueeze(-1).expand(-1,-1,-1,dk), ctx_top)

        ctx = ctx.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out(ctx)


class _Distilling(nn.Module):
    """Conv1d + ELU + MaxPool1d — halves the sequence length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act  = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)               # (B, d_model, L)
        x = self.act(self.conv(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        return x.transpose(1, 2)            # (B, L//2, d_model)


class _InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, factor: int):
        super().__init__()
        self.attn  = _ProbSparseAttention(d_model, nhead, dropout, factor)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DeterministicInformer(nn.Module):
    """
    Informer encoder with ProbSparse self-attention and sequence distilling.
    Uses MSE point-prediction head for ablation comparison.

    Key differences vs. standard Transformer:
        - ProbSparse attention: O(L log L) instead of O(L²)
        - Distilling: Conv + MaxPool halves sequence length between layers
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
        factor: int = 5,
    ):
        super().__init__()
        self.input_proj    = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Embedding(input_len, d_model)
        self.encoder_layers = nn.ModuleList([
            _InformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, factor)
            for _ in range(num_encoder_layers)
        ])
        # One distilling operation between each pair of encoder layers
        self.distilling = nn.ModuleList([
            _Distilling(d_model) for _ in range(num_encoder_layers - 1)
        ])
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

        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if i < len(self.distilling):
                h = self.distilling[i](h)

        return self.fc(h[:, -1, :])
