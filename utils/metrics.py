# GenAI usage: Claude (claude-sonnet-4-6) assisted in drafting this file.
# All mathematical formulations were reviewed and verified manually by the authors.

import math
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss (mean over batch and horizon).

    Args:
        mu, log_sigma, y: (B, pred_len)
    Returns:
        scalar loss tensor
    """
    log_sigma = torch.clamp(log_sigma, min=-6.0, max=6.0)
    sigma2 = torch.exp(2 * log_sigma)
    loss = 0.5 * (torch.log(sigma2) + (y - mu) ** 2 / sigma2)
    return loss.mean()


# ---------------------------------------------------------------------------
# Evaluation metrics (pure numpy, no scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """
    Closed-form CRPS for a Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma.
    """
    z = (y - mu) / (sigma + 1e-8)
    crps = sigma * (z * (2 * _norm_cdf(z) - 1) + 2 * _norm_pdf(z) - 1.0 / math.sqrt(math.pi))
    return float(crps.mean())


def nll_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Gaussian NLL in nats."""
    nll = 0.5 * np.log(2 * math.pi * sigma ** 2 + 1e-8) + \
          (y - mu) ** 2 / (2 * sigma ** 2 + 1e-8)
    return float(nll.mean())


def coverage(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, alpha: float = 0.9) -> float:
    """Empirical coverage of a two-sided Gaussian prediction interval."""
    z_table = {0.9: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_table.get(alpha, 1.645)
    return float(np.mean((y >= mu - z * sigma) & (y <= mu + z * sigma)))


def compute_all_metrics(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute MSE, MAE, NLL, CRPS, and 90% coverage.
    All inputs are flat numpy arrays.
    """
    return {
        'MSE':         float(np.mean((y - mu) ** 2)),
        'MAE':         float(np.mean(np.abs(y - mu))),
        'NLL':         nll_gaussian(mu, sigma, y),
        'CRPS':        crps_gaussian(mu, sigma, y),
        'Coverage_90': coverage(mu, sigma, y, alpha=0.9),
    }
