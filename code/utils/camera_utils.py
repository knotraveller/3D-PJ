"""Camera utilities for Blender-style Zero123++ multi-view data.

Coordinate convention:
    - c2w is camera-to-world.
    - w2c is world-to-camera.
    - Blender camera local +X is right, +Y is up, and -Z is forward.
    - World +Z is up.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


RELATIVE_AZIMUTHS = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
TARGET_ELEVATIONS = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0]


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def get_intrinsics_from_fov(resolution: int, fov_deg: float) -> np.ndarray:
    """Return a 3x3 pinhole intrinsic matrix for a square image."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if fov_deg <= 0.0 or fov_deg >= 180.0:
        raise ValueError("fov_deg must be in (0, 180).")

    width = float(resolution)
    height = float(resolution)
    fov = math.radians(float(fov_deg))
    fx = fy = 0.5 * width / math.tan(0.5 * fov)
    cx = width / 2.0
    cy = height / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def camera_position_from_spherical(
    azimuth_deg: float,
    elevation_deg: float,
    radius: float,
) -> np.ndarray:
    """Return the camera world position from z-up spherical coordinates."""
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    cos_elev = math.cos(elevation)

    x = float(radius) * cos_elev * math.sin(azimuth)
    y = -float(radius) * cos_elev * math.cos(azimuth)
    z = float(radius) * math.sin(elevation)
    return np.array([x, y, z], dtype=np.float32)


def look_at_c2w(
    camera_position: np.ndarray,
    target: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),
    up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
) -> np.ndarray:
    """Return a Blender-convention camera-to-world matrix.

    Blender cameras look along local -Z, so the local +Z column in c2w is
    -forward. The rotation columns are [right, true_up, -forward].
    """
    camera_position = np.asarray(camera_position, dtype=np.float32).reshape(3)
    target = np.asarray(target, dtype=np.float32).reshape(3)
    up = _normalize(np.asarray(up, dtype=np.float32).reshape(3))

    forward = _normalize(target - camera_position)
    right = np.cross(forward, up)

    if np.linalg.norm(right) < 1e-8:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, fallback_up)

    right = _normalize(right)
    true_up = _normalize(np.cross(right, forward))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = camera_position
    return c2w


def _make_view(
    name: str,
    index: int,
    azimuth: float,
    elevation: float,
    relative_azimuth: float,
    radius: float,
    K: np.ndarray,
) -> Dict[str, Any]:
    camera_position = camera_position_from_spherical(azimuth, elevation, radius)
    c2w = look_at_c2w(camera_position)
    w2c = np.linalg.inv(c2w).astype(np.float32)

    return {
        "name": name,
        "index": int(index),
        "azimuth": float(azimuth),
        "elevation": float(elevation),
        "relative_azimuth": float(relative_azimuth),
        "K": K.copy(),
        "c2w": c2w,
        "w2c": w2c,
    }


def get_zero123pp_camera_specs(
    ref_azimuth: float = 0.0,
    input_elevation: float = 0.0,
    radius: float = 4.0,
    fov_deg: float = 30.0,
    resolution: int = 256,
) -> Dict[str, Any]:
    """Return camera intrinsics/extrinsics for cond plus 6 Zero123++ targets."""
    K = get_intrinsics_from_fov(resolution, fov_deg)
    views: List[Dict[str, Any]] = [
        _make_view(
            name="cond",
            index=0,
            azimuth=float(ref_azimuth),
            elevation=float(input_elevation),
            relative_azimuth=0.0,
            radius=radius,
            K=K,
        )
    ]

    for target_index, (relative_azimuth, elevation) in enumerate(
        zip(RELATIVE_AZIMUTHS, TARGET_ELEVATIONS),
        start=1,
    ):
        views.append(
            _make_view(
                name=f"target_{target_index - 1:03d}",
                index=target_index,
                azimuth=float(ref_azimuth) + relative_azimuth,
                elevation=elevation,
                relative_azimuth=relative_azimuth,
                radius=radius,
                K=K,
            )
        )

    return {
        "resolution": int(resolution),
        "fov": float(fov_deg),
        "radius": float(radius),
        "views": views,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def save_cameras_json(camera_specs: dict, json_path: str) -> None:
    """Save camera specs as JSON, converting numpy arrays into lists."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(camera_specs), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _matrix_fields_to_numpy(view: Dict[str, Any]) -> Dict[str, Any]:
    converted = dict(view)
    for key in ("K", "c2w", "w2c"):
        if key in converted:
            converted[key] = np.asarray(converted[key], dtype=np.float32)
    return converted


def load_cameras_json(json_path: str) -> Dict[str, Any]:
    """Load camera specs from JSON and convert matrix fields to numpy arrays."""
    with Path(json_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "views" in data:
        data["views"] = [_matrix_fields_to_numpy(view) for view in data["views"]]
    else:
        # Compatibility path for older cameras.json files with input/targets.
        if "input" in data:
            data["input"] = _matrix_fields_to_numpy(data["input"])
        if "targets" in data:
            data["targets"] = [
                _matrix_fields_to_numpy(view) for view in data["targets"]
            ]
    return data
