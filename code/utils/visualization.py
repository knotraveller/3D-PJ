"""Visualization helpers for training and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 3:
        array = tensor.permute(1, 2, 0).numpy()
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
    elif tensor.ndim == 2:
        array = tensor.numpy()[..., None].repeat(3, axis=-1)
    else:
        raise ValueError(f"Expected [C,H,W] or [H,W], got {tuple(tensor.shape)}")
    return (array * 255.0 + 0.5).astype(np.uint8)


def save_training_visualization(
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    gt_rgb: torch.Tensor,
    gt_alpha: torch.Tensor,
    save_path: str,
    max_views: int = 7,
) -> None:
    """Save GT vs prediction grid for the first item in a batch."""
    if pred_rgb.ndim != 5 or gt_rgb.ndim != 5:
        raise ValueError("RGB tensors must be [B,V,3,H,W].")
    if pred_alpha.ndim != 5 or gt_alpha.ndim != 5:
        raise ValueError("Alpha tensors must be [B,V,1,H,W].")

    views = min(max_views, pred_rgb.shape[1])
    rows: List[List[np.ndarray]] = []
    error = (pred_rgb - gt_rgb).abs().mean(dim=2, keepdim=True)
    for tensor in (gt_rgb, pred_rgb, gt_alpha, pred_alpha, error):
        rows.append([_to_numpy_image(tensor[0, vidx]) for vidx in range(views)])

    tile_h, tile_w = rows[0][0].shape[:2]
    canvas = np.ones((len(rows) * tile_h, views * tile_w, 3), dtype=np.uint8) * 255
    for row_idx, row in enumerate(rows):
        for col_idx, tile in enumerate(row):
            y0 = row_idx * tile_h
            x0 = col_idx * tile_w
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def save_loss_curves(log_jsonl_path: str, save_path: str) -> None:
    """Plot epoch-level loss and PSNR curves from train_log.jsonl."""
    log_path = Path(log_jsonl_path)
    if not log_path.is_file():
        return

    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    keys = ("loss", "rgb_loss", "mask_loss", "lpips_loss")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    loss_ax, psnr_ax = axes
    for key in keys:
        xs = [int(row["epoch"]) for row in rows if isinstance(row.get(key), dict)]
        means = [float(row[key]["mean"]) for row in rows if isinstance(row.get(key), dict)]
        mins = [float(row[key]["min"]) for row in rows if isinstance(row.get(key), dict)]
        maxs = [float(row[key]["max"]) for row in rows if isinstance(row.get(key), dict)]
        if means:
            loss_ax.plot(xs, means, marker="o", markersize=2, linewidth=1.5, label=key)
            loss_ax.fill_between(xs, mins, maxs, alpha=0.12)
    loss_ax.set_ylabel("loss")
    loss_ax.set_yscale("log")
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    psnr_xs = [int(row["epoch"]) for row in rows if isinstance(row.get("psnr"), dict)]
    psnr_means = [
        float(row["psnr"]["mean"]) for row in rows if isinstance(row.get("psnr"), dict)
    ]
    psnr_mins = [
        float(row["psnr"]["min"]) for row in rows if isinstance(row.get("psnr"), dict)
    ]
    psnr_maxs = [
        float(row["psnr"]["max"]) for row in rows if isinstance(row.get("psnr"), dict)
    ]
    if psnr_means:
        psnr_ax.plot(
            psnr_xs,
            psnr_means,
            marker="o",
            markersize=2,
            linewidth=1.5,
            label="psnr",
            color="tab:green",
        )
        psnr_ax.fill_between(psnr_xs, psnr_mins, psnr_maxs, alpha=0.12, color="tab:green")
        psnr_ax.legend()
    psnr_ax.set_xlabel("epoch")
    psnr_ax.set_ylabel("PSNR")
    psnr_ax.grid(True, alpha=0.3)
    if min(epochs) != max(epochs):
        psnr_ax.set_xlim(min(epochs), max(epochs))

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _stats(tensor: torch.Tensor) -> Dict[str, float]:
    tensor = tensor.detach().float().cpu()
    return {
        "min": float(tensor.min()),
        "mean": float(tensor.mean()),
        "max": float(tensor.max()),
    }


def save_gaussian_stats(gaussians: torch.Tensor, save_path: str) -> None:
    """Save simple Gaussian parameter statistics as JSON."""
    opacity = gaussians[..., 3]
    scale = gaussians[..., 4:7]
    xyz = gaussians[..., 0:3]
    rgb = gaussians[..., 11:14]
    payload = {
        "opacity": _stats(opacity),
        "scale": _stats(scale),
        "xyz": _stats(xyz),
        "rgb": _stats(rgb),
        "num_opacity_gt_0p01": int((opacity > 0.01).sum().item()),
        "num_opacity_gt_0p05": int((opacity > 0.05).sum().item()),
        "num_gaussians": int(opacity.numel()),
    }
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
