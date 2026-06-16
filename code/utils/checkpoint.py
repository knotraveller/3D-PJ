"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    payload: Dict[str, Any],
    path: str,
    *,
    overwrite: bool,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {checkpoint_path}"
        )
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location)
