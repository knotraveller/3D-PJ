"""YAML config loading and small override helpers."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def apply_debug_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    return deep_update(
        config,
        {
            "data": {
                "max_train_samples": 8,
                "max_val_samples": 4,
                "num_workers": 0,
            },
            "train": {
                "epochs": 2,
                "batch_size": 1,
                "log_every": 1,
                "vis_every": 1,
                "save_every": 1,
                "val_every": 1,
            },
            "performance": {
                "write_every": 1,
                "system_sample_every": 1,
            },
        },
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
