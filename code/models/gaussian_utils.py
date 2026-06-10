"""Post-process raw ZeroGS-UNet predictions into 3D Gaussian parameters."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _check_camera_shapes(raw: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> None:
    if raw.ndim != 5:
        raise ValueError(f"raw must be [B, V, 16, S, S], got {tuple(raw.shape)}.")
    if raw.shape[2] != 16:
        raise ValueError(f"raw channel count must be 16, got {raw.shape[2]}.")
    if K.shape[:2] != raw.shape[:2] or K.shape[-2:] != (3, 3):
        raise ValueError(
            f"K must be [B, V, 3, 3] matching raw, got {tuple(K.shape)}."
        )
    if c2w.shape[:2] != raw.shape[:2] or c2w.shape[-2:] != (4, 4):
        raise ValueError(
            f"c2w must be [B, V, 4, 4] matching raw, got {tuple(c2w.shape)}."
        )


def _pixel_centers(
    image_size: int,
    splat_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return image-space pixel centers for each splat cell.

    u and v shapes are [1, 1, S, S], where S=splat_size.
    """
    scale = float(image_size) / float(splat_size)
    rows = torch.arange(splat_size, device=device, dtype=dtype)
    cols = torch.arange(splat_size, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(rows, cols, indexing="ij")
    u = (uu + 0.5) * scale
    v = (vv + 0.5) * scale
    return u[None, None], v[None, None]


def _normalize_quaternion(quat_raw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternion channels and fall back to identity for zero vectors."""
    norm = torch.linalg.norm(quat_raw, dim=2, keepdim=True)
    quat = quat_raw / (norm + eps)

    identity = torch.zeros_like(quat_raw)
    identity[:, :, 0:1] = 1.0
    return torch.where(norm > eps, quat, identity)


def raw_to_gaussians(
    raw: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    image_size: int = 256,
    splat_size: int = 64,
    depth_min: float = 2.5,
    depth_max: float = 5.5,
    offset_scale: float = 0.05,
    opacity_bias: float = 4.0,
) -> Dict[str, torch.Tensor]:
    """Convert raw network output to 3D Gaussian parameters.

    Args:
        raw: [B, V, 16, S, S] channels are depth/offset/scale/quat/opacity/rgb/conf.
        K: [B, V, 3, 3] camera intrinsics.
        c2w: [B, V, 4, 4] camera-to-world matrices.

    Returns:
        gaussians: [B, V*S*S, 14]
        gaussian_map: [B, V, 14, S, S]
        confidence: [B, V, 1, S, S]
    """
    _check_camera_shapes(raw, K, c2w)
    if image_size <= 0 or splat_size <= 0:
        raise ValueError("image_size and splat_size must be positive.")
    if raw.shape[-2:] != (splat_size, splat_size):
        raise ValueError(
            f"raw spatial shape must be [{splat_size}, {splat_size}], "
            f"got {tuple(raw.shape[-2:])}."
        )

    batch_size, num_views, _, height, width = raw.shape
    dtype = raw.dtype
    device = raw.device
    K = K.to(device=device, dtype=dtype)
    c2w = c2w.to(device=device, dtype=dtype)

    # Split raw map:
    # raw [B, V, 16, S, S] -> each component [B, V, C_i, S, S].
    depth_raw = raw[:, :, 0:1]
    offset_raw = raw[:, :, 1:4]
    scale_raw = raw[:, :, 4:7]
    quat_raw = raw[:, :, 7:11]
    opacity_raw = raw[:, :, 11:12]
    rgb_raw = raw[:, :, 12:15]
    confidence_raw = raw[:, :, 15:16]

    # depth [B, V, 1, S, S] in the camera forward direction.
    depth = depth_min + (depth_max - depth_min) * torch.sigmoid(depth_raw)

    # Feature-cell centers in original image pixels:
    # u/v [1, 1, S, S] broadcast against [B, V, S, S].
    u, v = _pixel_centers(image_size, splat_size, device, dtype)
    fx = K[:, :, 0, 0][:, :, None, None]
    fy = K[:, :, 1, 1][:, :, None, None]
    cx = K[:, :, 0, 2][:, :, None, None]
    cy = K[:, :, 1, 2][:, :, None, None]
    depth_hw = depth.squeeze(2)  # [B, V, S, S]

    # Blender camera convention: +X right, +Y up, -Z forward.
    x_cam = (u - cx) / fx * depth_hw
    y_cam = -(v - cy) / fy * depth_hw
    z_cam = -depth_hw
    p_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, V, S, S, 3]

    # p_world = p_cam @ R_c2w.T + t_c2w.
    R_c2w = c2w[:, :, :3, :3]  # [B, V, 3, 3]
    t_c2w = c2w[:, :, :3, 3]   # [B, V, 3]
    p_world = torch.matmul(
        p_cam.unsqueeze(-2),
        R_c2w[:, :, None, None].transpose(-1, -2),
    ).squeeze(-2)
    p_world = p_world + t_c2w[:, :, None, None, :]  # [B, V, S, S, 3]

    # offset_raw [B, V, 3, S, S] -> offset [B, V, S, S, 3].
    offset = offset_scale * torch.tanh(offset_raw).permute(0, 1, 3, 4, 2)
    center_hw = p_world + offset
    center = center_hw.permute(0, 1, 4, 2, 3).contiguous()  # [B, V, 3, S, S]

    scale = 0.005 + 0.05 * F.softplus(scale_raw)
    scale = torch.clamp(scale, min=0.002, max=0.15)  # [B, V, 3, S, S]
    quat = _normalize_quaternion(quat_raw)            # [B, V, 4, S, S]
    opacity = torch.sigmoid(opacity_raw - float(opacity_bias))  # [B, V, 1, S, S]
    rgb = torch.sigmoid(rgb_raw)                      # [B, V, 3, S, S]
    confidence = torch.sigmoid(confidence_raw)        # [B, V, 1, S, S]

    # Channel order:
    # [x, y, z, opacity, sx, sy, sz, qw, qx, qy, qz, r, g, b]
    gaussian_map = torch.cat(
        [center, opacity, scale, quat, rgb],
        dim=2,
    )
    if gaussian_map.shape != (batch_size, num_views, 14, height, width):
        raise RuntimeError(f"Unexpected gaussian_map shape: {tuple(gaussian_map.shape)}")

    # [B, V, 14, S, S] -> [B, V, S, S, 14] -> [B, V*S*S, 14].
    gaussians = (
        gaussian_map.permute(0, 1, 3, 4, 2)
        .contiguous()
        .reshape(batch_size, num_views * splat_size * splat_size, 14)
    )

    return {
        "gaussians": gaussians,
        "gaussian_map": gaussian_map,
        "confidence": confidence,
    }
