"""Image tensor helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert [C,H,W] or [H,W] float tensor in [0,1] to uint8 image array."""
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 3:
        array = tensor.permute(1, 2, 0).numpy()
        if array.shape[-1] == 1:
            array = array[..., 0]
    elif tensor.ndim == 2:
        array = tensor.numpy()
    else:
        raise ValueError(f"Expected [C,H,W] or [H,W], got {tuple(tensor.shape)}.")
    return (array * 255.0 + 0.5).astype(np.uint8)


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    """Save a [C,H,W] tensor as an image."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(save_path)
