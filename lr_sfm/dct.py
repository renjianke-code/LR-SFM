import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s: int = 42) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def dct_1d(x: torch.Tensor) -> torch.Tensor:
    """Type-II DCT along the time axis. x: [..., T, D] -> [..., T, D]."""
    N = x.shape[-2]
    n = torch.arange(N, device=x.device, dtype=x.dtype)
    k = n.unsqueeze(0)
    n = n.unsqueeze(1)
    cos_basis = torch.cos(np.pi * (n + 0.5) * k / N)
    orig = x.shape
    x_flat = x.reshape(-1, N, orig[-1])
    return torch.einsum("btn,tm->bmn", x_flat, cos_basis).reshape(orig)


def idct_1d(X: torch.Tensor) -> torch.Tensor:
    """Inverse Type-II DCT along the spectral axis."""
    N = X.shape[-2]
    n = torch.arange(N, device=X.device, dtype=X.dtype)
    k = n.unsqueeze(0)
    n = n.unsqueeze(1)
    cos_basis = torch.cos(np.pi * (n + 0.5) * k / N)
    cos_basis[:, 0] *= 0.5
    orig = X.shape
    X_flat = X.reshape(-1, N, orig[-1])
    return (torch.einsum("bkn,tk->btn", X_flat, cos_basis) * (2.0 / N)).reshape(orig)


def fit_dct_high_linear(low_dct: torch.Tensor, high_dct: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
    """Fit a train-only ridge map from retained DCT modes to omitted modes."""
    x = low_dct.reshape(-1, low_dct.shape[-2] * low_dct.shape[-1]).float()
    y = high_dct.reshape(-1, high_dct.shape[-2] * high_dct.shape[-1]).float()
    x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=-1)
    eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(x.T @ x + ridge * eye, x.T @ y).detach()


def complete_dct_high_modes(
    low_dct: torch.Tensor,
    mean_high: torch.Tensor | None = None,
    linear_weight: torch.Tensor | None = None,
    full_frames: int | None = None,
) -> torch.Tensor:
    """Append omitted DCT modes with zero, train-mean, or linear padding."""
    if full_frames is None:
        if mean_high is None:
            return low_dct
        full_frames = low_dct.shape[-2] + mean_high.shape[-2]
    high_frames = full_frames - low_dct.shape[-2]
    if high_frames <= 0:
        return low_dct

    if linear_weight is not None:
        weight = linear_weight.to(device=low_dct.device, dtype=low_dct.dtype)
        x = low_dct.reshape(-1, low_dct.shape[-2] * low_dct.shape[-1])
        x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=-1)
        high = (x @ weight).reshape(*low_dct.shape[:-2], high_frames, low_dct.shape[-1])
        return torch.cat([low_dct, high], dim=-2)

    if mean_high is None:
        high = low_dct.new_zeros(*low_dct.shape[:-2], high_frames, low_dct.shape[-1])
    else:
        high = mean_high.to(device=low_dct.device, dtype=low_dct.dtype)
        high = high.reshape(*([1] * (low_dct.dim() - 2)), high.shape[-2], high.shape[-1])
        high = high.expand(*low_dct.shape[:-2], high.shape[-2], high.shape[-1])
    return torch.cat([low_dct, high], dim=-2)
