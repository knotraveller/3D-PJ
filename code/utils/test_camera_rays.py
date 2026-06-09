"""Smoke tests for camera_utils.py and ray_utils.py.

Run directly:
    python code/utils/test_camera_rays.py

Or with pytest:
    pytest code/utils/test_camera_rays.py
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from camera_utils import (
    camera_position_from_spherical,
    get_intrinsics_from_fov,
    get_zero123pp_camera_specs,
)
from ray_utils import get_embedding, get_rays_np


RESOLUTION = 256
FOV = 30.0
RADIUS = 4.0


def _stack_camera_specs():
    specs = get_zero123pp_camera_specs(
        ref_azimuth=0.0,
        input_elevation=0.0,
        radius=RADIUS,
        fov_deg=FOV,
        resolution=RESOLUTION,
    )
    K = np.stack([view["K"] for view in specs["views"]], axis=0)
    c2w = np.stack([view["c2w"] for view in specs["views"]], axis=0)
    return specs, K, c2w


def test_intrinsics_from_fov():
    K = get_intrinsics_from_fov(RESOLUTION, FOV)
    assert K.shape == (3, 3)
    assert np.isclose(K[0, 0], 477.70, atol=0.1)
    assert np.isclose(K[1, 1], 477.70, atol=0.1)
    assert np.isclose(K[0, 2], 128.0)
    assert np.isclose(K[1, 2], 128.0)


def test_camera_position_from_spherical():
    position = camera_position_from_spherical(0.0, 0.0, RADIUS)
    assert position.shape == (3,)
    assert np.allclose(position, np.array([0.0, -4.0, 0.0]), atol=1e-6)


def test_zero123pp_camera_specs_shapes_and_inverse():
    specs, _, _ = _stack_camera_specs()
    assert specs["resolution"] == RESOLUTION
    assert len(specs["views"]) == 7

    for view in specs["views"]:
        assert view["K"].shape == (3, 3)
        assert view["c2w"].shape == (4, 4)
        assert view["w2c"].shape == (4, 4)
        assert np.allclose(view["c2w"] @ view["w2c"], np.eye(4), atol=1e-5)


def test_get_rays_np_shape_and_center_direction():
    specs, _, _ = _stack_camera_specs()
    cond = specs["views"][0]
    rays_o, rays_d = get_rays_np(cond["K"], cond["c2w"], RESOLUTION)

    assert rays_o.shape == (RESOLUTION, RESOLUTION, 3)
    assert rays_d.shape == (RESOLUTION, RESOLUTION, 3)

    camera_position = cond["c2w"][:3, 3]
    expected_direction = -camera_position / np.linalg.norm(camera_position)
    center_ray_d = rays_d[RESOLUTION // 2, RESOLUTION // 2]
    assert np.allclose(center_ray_d, expected_direction, atol=5e-3)


def test_get_embedding_and_rgb_concat_shape():
    if torch is None:
        print("Skipping torch embedding test because PyTorch is not installed.")
        return

    _, K, c2w = _stack_camera_specs()
    batch_size = 2
    K_torch = torch.from_numpy(K).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    c2w_torch = torch.from_numpy(c2w).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    ray_emb = get_embedding(
        K=K_torch,
        c2w=c2w_torch,
        resolution=RESOLUTION,
        embedding_type="plucker",
        order="dm",
        channel_first=True,
    )
    assert tuple(ray_emb.shape) == (batch_size, 7, 6, RESOLUTION, RESOLUTION)

    images = torch.zeros(batch_size, 7, 3, RESOLUTION, RESOLUTION)
    model_input = torch.cat([images, ray_emb], dim=2)
    assert tuple(model_input.shape) == (batch_size, 7, 9, RESOLUTION, RESOLUTION)


def run_all_tests() -> None:
    test_intrinsics_from_fov()
    test_camera_position_from_spherical()
    test_zero123pp_camera_specs_shapes_and_inverse()
    test_get_rays_np_shape_and_center_direction()
    test_get_embedding_and_rgb_concat_shape()


if __name__ == "__main__":
    run_all_tests()
    print("All camera/ray tests passed.")
