"""Training and validation metrics."""

from __future__ import annotations

from typing import Optional

import torch


def psnr(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute PSNR for [B,V,3,H,W] images.

    If mask is provided, it must be [B,V,1,H,W] and MSE is computed on foreground.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes must match: {pred.shape} vs {gt.shape}")
    if mask is not None:
        if mask.shape[:2] != pred.shape[:2] or mask.shape[-2:] != pred.shape[-2:]:
            raise ValueError(f"mask shape does not match images: {mask.shape}")
        mse = ((pred - gt) ** 2 * mask).sum() / (mask.sum() * pred.shape[2] + eps)
    else:
        mse = ((pred - gt) ** 2).mean()
    return -10.0 * torch.log10(mse + eps)


# TODO: Add SSIM after the basic training loop is stable.
