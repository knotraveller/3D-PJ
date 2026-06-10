"""Losses for feed-forward Gaussian reconstruction."""

from __future__ import annotations

import warnings
from typing import Dict

import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    """RGB, alpha, and optional LPIPS reconstruction loss."""

    def __init__(
        self,
        lambda_rgb: float = 1.0,
        lambda_mask: float = 0.5,
        lambda_lpips: float = 0.1,
        use_lpips: bool = True,
        lpips_net: str = "vgg",
        lpips_chunk_size: int = 1,
        mask_rgb_loss: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_rgb = float(lambda_rgb)
        self.lambda_mask = float(lambda_mask)
        self.lambda_lpips = float(lambda_lpips)
        self.lpips_chunk_size = max(1, int(lpips_chunk_size))
        self.mask_rgb_loss = bool(mask_rgb_loss)
        self.eps = float(eps)

        self.lpips_model = None
        self.use_lpips = bool(use_lpips)
        if self.use_lpips:
            try:
                import lpips
            except ImportError:
                warnings.warn(
                    "lpips is not installed; ReconstructionLoss will disable LPIPS.",
                    RuntimeWarning,
                )
                self.use_lpips = False
            else:
                self.lpips_model = lpips.LPIPS(net=lpips_net)
                self.lpips_model.eval()
                for parameter in self.lpips_model.parameters():
                    parameter.requires_grad_(False)

    @staticmethod
    def _check_shapes(
        pred_rgb: torch.Tensor,
        pred_alpha: torch.Tensor,
        gt_rgb: torch.Tensor,
        gt_alpha: torch.Tensor,
    ) -> None:
        if pred_rgb.shape != gt_rgb.shape:
            raise ValueError(f"pred_rgb and gt_rgb shapes differ: {pred_rgb.shape} vs {gt_rgb.shape}")
        if pred_alpha.shape != gt_alpha.shape:
            raise ValueError(
                f"pred_alpha and gt_alpha shapes differ: {pred_alpha.shape} vs {gt_alpha.shape}"
            )
        if pred_rgb.ndim != 5 or pred_alpha.ndim != 5:
            raise ValueError("Expected RGB/alpha tensors shaped [B,V,C,H,W].")

    def forward(
        self,
        pred_rgb: torch.Tensor,
        pred_alpha: torch.Tensor,
        gt_rgb: torch.Tensor,
        gt_alpha: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._check_shapes(pred_rgb, pred_alpha, gt_rgb, gt_alpha)

        pred_rgb = pred_rgb.clamp(0.0, 1.0)
        pred_alpha = pred_alpha.clamp(0.0, 1.0)
        gt_rgb = gt_rgb.clamp(0.0, 1.0)
        gt_alpha = gt_alpha.clamp(0.0, 1.0)

        if self.mask_rgb_loss:
            rgb_loss = (gt_alpha * (pred_rgb - gt_rgb).abs()).sum()
            rgb_loss = rgb_loss / (gt_alpha.sum() * pred_rgb.shape[2] + self.eps)
        else:
            rgb_loss = (pred_rgb - gt_rgb).abs().mean()

        mask_loss = (pred_alpha - gt_alpha).abs().mean()

        if self.use_lpips and self.lpips_model is not None:
            batch_size, num_views, _, height, width = pred_rgb.shape
            pred_lpips = pred_rgb.reshape(batch_size * num_views, 3, height, width)
            gt_lpips = gt_rgb.reshape(batch_size * num_views, 3, height, width)
            pred_lpips = pred_lpips * 2.0 - 1.0
            gt_lpips = gt_lpips * 2.0 - 1.0
            self.lpips_model = self.lpips_model.to(device=pred_rgb.device)
            lpips_values = []
            for start in range(0, pred_lpips.shape[0], self.lpips_chunk_size):
                end = start + self.lpips_chunk_size
                lpips_values.append(self.lpips_model(pred_lpips[start:end], gt_lpips[start:end]))
            lpips_loss = torch.cat(lpips_values, dim=0).mean()
        else:
            lpips_loss = pred_rgb.new_tensor(0.0)

        total = (
            self.lambda_rgb * rgb_loss
            + self.lambda_mask * mask_loss
            + self.lambda_lpips * lpips_loss
        )

        return {
            "loss": total,
            "rgb_loss": rgb_loss.detach(),
            "mask_loss": mask_loss.detach(),
            "lpips_loss": lpips_loss.detach(),
        }
