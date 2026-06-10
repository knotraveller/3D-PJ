"""Ray generation and ray embedding utilities.

The default Pluecker embedding order is [direction, moment], where
moment = origin x direction. Training inputs are formed by concatenating RGB
and ray embedding along the channel dimension:
    RGB [B, V, 3, H, W] + rays [B, V, 6, H, W] -> [B, V, 9, H, W].

Example:
    from ray_utils import get_embedding

    images = batch["images"]  # [B, V, 3, H, W]
    K = batch["K"]            # [B, V, 3, 3]
    c2w = batch["c2w"]        # [B, V, 4, 4]
    ray_emb = get_embedding(K, c2w, resolution=256)
    model_input = torch.cat([images, ray_emb], dim=2)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch


def _normalize_np(values: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norm, eps)


def get_rays_np(
    K: np.ndarray,
    c2w: np.ndarray,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ray origins and directions with Blender camera convention.

    Args:
        K: [3, 3] intrinsic matrix.
        c2w: [4, 4] camera-to-world matrix.
        resolution: square image size.

    Returns:
        rays_o: [H, W, 3]
        rays_d: [H, W, 3]
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    K = np.asarray(K, dtype=np.float32)
    c2w = np.asarray(c2w, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must have shape [3, 3], got {K.shape}.")
    if c2w.shape != (4, 4):
        raise ValueError(f"c2w must have shape [4, 4], got {c2w.shape}.")

    height = width = int(resolution)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
        indexing="xy",
    )
    x_cam = (u - cx) / fx
    y_cam = -(v - cy) / fy
    z_cam = -np.ones_like(x_cam)

    dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    dirs_cam = _normalize_np(dirs_cam)

    R_c2w = c2w[:3, :3]
    origin_world = c2w[:3, 3]
    rays_d = dirs_cam @ R_c2w.T
    rays_d = _normalize_np(rays_d).astype(np.float32)
    rays_o = np.broadcast_to(origin_world.reshape(1, 1, 3), rays_d.shape)

    return rays_o.astype(np.float32), rays_d


def get_plucker_np(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    order: str = "dm",
) -> np.ndarray:
    """Return Pluecker ray embedding [H, W, 6].

    order="dm" returns [direction, moment].
    order="md" returns [moment, direction].
    """
    if order not in {"dm", "md"}:
        raise ValueError('order must be "dm" or "md".')

    rays_o = np.asarray(rays_o, dtype=np.float32)
    rays_d = np.asarray(rays_d, dtype=np.float32)
    if rays_o.shape != rays_d.shape or rays_o.shape[-1] != 3:
        raise ValueError(
            f"rays_o and rays_d must share shape [..., 3], got {rays_o.shape} and {rays_d.shape}."
        )

    moment = np.cross(rays_o, rays_d)
    if order == "dm":
        return np.concatenate([rays_d, moment], axis=-1).astype(np.float32)
    return np.concatenate([moment, rays_d], axis=-1).astype(np.float32)


def _torch_meshgrid(height: int, width: int, device: torch.device, dtype: torch.dtype):
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    return torch.meshgrid(ys, xs, indexing="ij")


def _promoted_float_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32
    return dtype


def get_rays_torch(
    K: torch.Tensor,
    c2w: torch.Tensor,
    resolution: int,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate torch ray origins and directions for single or batched views.

    K can be [3, 3] or [B, V, 3, 3].
    c2w can be [4, 4] or [B, V, 4, 4].
    Leading dimensions are broadcast when possible.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K must end with [3, 3], got {tuple(K.shape)}.")
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"c2w must end with [4, 4], got {tuple(c2w.shape)}.")

    if device is None:
        device = K.device
    device = torch.device(device)
    dtype = _promoted_float_dtype(K, c2w)
    K = K.to(device=device, dtype=dtype)
    c2w = c2w.to(device=device, dtype=dtype)

    leading_shape = torch.broadcast_shapes(K.shape[:-2], c2w.shape[:-2])
    K = K.expand(*leading_shape, 3, 3)
    c2w = c2w.expand(*leading_shape, 4, 4)

    height = width = int(resolution)
    v_grid, u_grid = _torch_meshgrid(height, width, device, dtype)

    fx = K[..., 0, 0][..., None, None]
    fy = K[..., 1, 1][..., None, None]
    cx = K[..., 0, 2][..., None, None]
    cy = K[..., 1, 2][..., None, None]

    x_cam = (u_grid - cx) / fx
    y_cam = -(v_grid - cy) / fy
    z_cam = -torch.ones_like(x_cam)
    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    dirs_cam = torch.nn.functional.normalize(dirs_cam, dim=-1)

    R_c2w = c2w[..., :3, :3]
    rays_d = torch.matmul(
        dirs_cam.unsqueeze(-2),
        R_c2w[..., None, None, :, :].transpose(-1, -2),
    ).squeeze(-2)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

    origin_world = c2w[..., :3, 3]
    rays_o = origin_world[..., None, None, :].expand_as(rays_d)
    return rays_o, rays_d


def get_embedding(
    K: torch.Tensor,
    c2w: torch.Tensor,
    resolution: int = 256,
    embedding_type: str = "plucker",
    order: str = "dm",
    channel_first: bool = True,
) -> torch.Tensor:
    """Return online ray embedding for training.

    Args:
        K: [B, V, 3, 3] camera intrinsics.
        c2w: [B, V, 4, 4] camera-to-world matrices.
        resolution: square image size.
        embedding_type: "plucker" for [d, m] or [m, d], or "ray_dir".
        order: "dm" or "md" for Pluecker embedding.
        channel_first: if True, return [B, V, C, H, W]; otherwise [B, V, H, W, C].
    """
    if embedding_type not in {"plucker", "ray_dir"}:
        raise ValueError('embedding_type must be "plucker" or "ray_dir".')
    if order not in {"dm", "md"}:
        raise ValueError('order must be "dm" or "md".')

    rays_o, rays_d = get_rays_torch(K=K, c2w=c2w, resolution=resolution)

    if embedding_type == "ray_dir":
        embedding = rays_d
    else:
        moment = torch.cross(rays_o, rays_d, dim=-1)
        if order == "dm":
            embedding = torch.cat([rays_d, moment], dim=-1)
        else:
            embedding = torch.cat([moment, rays_d], dim=-1)

    if channel_first:
        embedding = embedding.movedim(-1, -3)
    return embedding
