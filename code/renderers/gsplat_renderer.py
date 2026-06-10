"""Thin wrapper around gsplat's differentiable Gaussian rasterizer."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


GSPLOT_INSTALL_MESSAGE = (
    "gsplat is required for differentiable 3DGS rendering. "
    "Install it with `pip install gsplat`, or follow the official build instructions."
)


class GSplatRenderer(nn.Module):
    """Render 3D Gaussian parameters into RGB and alpha images.

    Inputs:
        gaussians: [B, N, 14]
        K: [B, V, 3, 3]
        w2c: [B, V, 4, 4]

    Output:
        rgb: [B, V, 3, H, W]
        alpha: [B, V, 1, H, W]
    """

    def __init__(
        self,
        image_size: int = 256,
        background: str = "white",
        near_plane: float = 0.01,
        far_plane: float = 100.0,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        packed: bool = True,
        tile_size: int = 16,
        render_mode: str = "RGB",
        rasterize_mode: str = "classic",
    ) -> None:
        super().__init__()
        if background not in {"white", "black"}:
            raise ValueError('background must be "white" or "black".')
        if render_mode != "RGB":
            raise ValueError('Only render_mode="RGB" is currently supported.')

        self.image_size = int(image_size)
        self.background = background
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.radius_clip = float(radius_clip)
        self.eps2d = float(eps2d)
        self.packed = bool(packed)
        self.tile_size = int(tile_size)
        self.render_mode = render_mode
        self.rasterize_mode = rasterize_mode
        self._rasterization = self._import_rasterization()

    @staticmethod
    def _import_rasterization():
        try:
            from gsplat.rendering import rasterization
        except ImportError as exc:
            raise ImportError(GSPLOT_INSTALL_MESSAGE) from exc
        return rasterization

    def _backgrounds(
        self,
        batch_size: int,
        num_views: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.background == "white":
            return torch.ones(batch_size, num_views, 3, device=device, dtype=dtype)
        return torch.zeros(batch_size, num_views, 3, device=device, dtype=dtype)

    @staticmethod
    def _check_shapes(gaussians: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> None:
        if gaussians.ndim != 3 or gaussians.shape[-1] != 14:
            raise ValueError(
                f"gaussians must be [B, N, 14], got {tuple(gaussians.shape)}."
            )
        if K.ndim != 4 or K.shape[-2:] != (3, 3):
            raise ValueError(f"K must be [B, V, 3, 3], got {tuple(K.shape)}.")
        if w2c.ndim != 4 or w2c.shape[-2:] != (4, 4):
            raise ValueError(f"w2c must be [B, V, 4, 4], got {tuple(w2c.shape)}.")
        if K.shape[:2] != w2c.shape[:2] or K.shape[0] != gaussians.shape[0]:
            raise ValueError(
                "Batch/view dimensions must match: "
                f"gaussians={tuple(gaussians.shape)}, K={tuple(K.shape)}, w2c={tuple(w2c.shape)}."
            )

    def _split_gaussians(self, gaussians: torch.Tensor) -> Dict[str, torch.Tensor]:
        means = gaussians[..., 0:3]
        opacities = gaussians[..., 3].clamp(0.0, 1.0)
        scales = gaussians[..., 4:7].clamp_min(1e-4)
        quats = F.normalize(gaussians[..., 7:11], dim=-1, eps=1e-8)
        colors = gaussians[..., 11:14].clamp(0.0, 1.0)
        return {
            "means": means,
            "opacities": opacities,
            "scales": scales,
            "quats": quats,
            "colors": colors,
        }

    def _call_rasterization(
        self,
        parts: Dict[str, torch.Tensor],
        K: torch.Tensor,
        w2c: torch.Tensor,
        backgrounds: Optional[torch.Tensor],
    ):
        return self._rasterization(
            means=parts["means"],
            quats=parts["quats"],
            scales=parts["scales"],
            opacities=parts["opacities"],
            colors=parts["colors"],
            viewmats=w2c,
            Ks=K,
            width=self.image_size,
            height=self.image_size,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            radius_clip=self.radius_clip,
            eps2d=self.eps2d,
            packed=self.packed,
            tile_size=self.tile_size,
            backgrounds=None,
            render_mode=self.render_mode,
            rasterize_mode=self.rasterize_mode,
        )

    @staticmethod
    def _blender_to_opencv_viewmat(w2c: torch.Tensor) -> torch.Tensor:
        # Dataset cameras use Blender convention: +X right, +Y up, -Z forward.
        # gsplat expects OpenCV camera coordinates: +X right, +Y down, +Z forward.
        converted = w2c.clone()
        converted[..., 1, :] = -converted[..., 1, :]
        converted[..., 2, :] = -converted[..., 2, :]
        return converted

    @staticmethod
    def _to_channel_first(
        render_colors: torch.Tensor,
        render_alphas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if render_colors.ndim == 4:
            render_colors = render_colors.unsqueeze(0)
        if render_alphas.ndim == 3:
            render_alphas = render_alphas[..., None]
        if render_alphas.ndim == 4:
            render_alphas = render_alphas.unsqueeze(0)

        if render_colors.ndim != 5 or render_colors.shape[-1] != 3:
            raise RuntimeError(f"Unexpected gsplat RGB shape: {tuple(render_colors.shape)}")
        if render_alphas.ndim != 5:
            raise RuntimeError(f"Unexpected gsplat alpha shape: {tuple(render_alphas.shape)}")
        if render_alphas.shape[-1] != 1:
            render_alphas = render_alphas[..., :1]

        pred_rgb = render_colors.permute(0, 1, 4, 2, 3).contiguous()
        pred_alpha = render_alphas.permute(0, 1, 4, 2, 3).contiguous()
        return pred_rgb.clamp(0.0, 1.0), pred_alpha.clamp(0.0, 1.0)

    def _composite_background(
        self,
        pred_rgb: torch.Tensor,
        pred_alpha: torch.Tensor,
    ) -> torch.Tensor:
        if self.background == "black":
            return pred_rgb.clamp(0.0, 1.0)
        return (pred_rgb + (1.0 - pred_alpha)).clamp(0.0, 1.0)

    def _loop_batch_forward(
        self,
        parts: Dict[str, torch.Tensor],
        K: torch.Tensor,
        w2c: torch.Tensor,
        backgrounds: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        rgbs = []
        alphas = []
        metas = []
        batch_size = K.shape[0]
        for bidx in range(batch_size):
            one_parts = {key: value[bidx] for key, value in parts.items()}
            one_backgrounds = None if backgrounds is None else backgrounds[bidx]
            render_colors, render_alphas, meta = self._call_rasterization(
                one_parts,
                K[bidx],
                w2c[bidx],
                one_backgrounds,
            )
            pred_rgb, pred_alpha = self._to_channel_first(render_colors, render_alphas)
            rgbs.append(pred_rgb.squeeze(0))
            alphas.append(pred_alpha.squeeze(0))
            metas.append(meta)
        return torch.stack(rgbs, dim=0), torch.stack(alphas, dim=0), {"per_batch": metas}

    def forward(
        self,
        gaussians: torch.Tensor,
        K: torch.Tensor,
        w2c: torch.Tensor,
    ) -> Dict[str, torch.Tensor | object]:
        self._check_shapes(gaussians, K, w2c)
        batch_size, num_views = K.shape[:2]
        device = gaussians.device
        dtype = gaussians.dtype

        K = K.to(device=device, dtype=torch.float32)
        w2c = self._blender_to_opencv_viewmat(w2c.to(device=device, dtype=torch.float32))
        backgrounds = None
        parts = self._split_gaussians(gaussians.float())

        autocast_context = (
            torch.cuda.amp.autocast(enabled=False)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            try:
                render_colors, render_alphas, meta = self._call_rasterization(
                    parts,
                    K,
                    w2c,
                    backgrounds,
                )
                pred_rgb, pred_alpha = self._to_channel_first(render_colors, render_alphas)
            except (RuntimeError, ValueError, TypeError):
                pred_rgb, pred_alpha, meta = self._loop_batch_forward(
                    parts,
                    K,
                    w2c,
                    backgrounds,
                )
        pred_rgb = self._composite_background(pred_rgb, pred_alpha)

        return {
            "rgb": pred_rgb.to(dtype=dtype),
            "alpha": pred_alpha.to(dtype=dtype),
            "meta": meta,
        }
