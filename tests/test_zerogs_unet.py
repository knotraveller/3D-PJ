"""Minimal tests for ZeroGSUNet.

Run directly:
    python tests/test_zerogs_unet.py

Or with pytest:
    pytest tests/test_zerogs_unet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models import ZeroGSUNet


def make_dummy_cameras(batch_size: int, num_views: int, image_size: int):
    fx = fy = 0.5 * image_size / torch.tan(torch.tensor(0.5 * torch.pi / 6.0))
    cx = cy = image_size / 2.0
    K = torch.tensor(
        [
            [float(fx), 0.0, cx],
            [0.0, float(fy), cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    K = K[None, None].repeat(batch_size, num_views, 1, 1)
    c2w = torch.eye(4, dtype=torch.float32)[None, None].repeat(
        batch_size,
        num_views,
        1,
        1,
    )
    return K, c2w


@torch.no_grad()
def test_raw_output_shape_256():
    B, V = 2, 7
    x = torch.randn(B, V, 9, 256, 256)
    model = ZeroGSUNet(
        image_size=256,
        splat_size=64,
        base_channels=8,
        use_view_attention=False,
        use_global_attention=False,
    ).eval()

    out = model(x)
    assert out["raw"].shape == (B, V, 16, 64, 64)


@torch.no_grad()
def test_gaussian_output_shapes_and_ranges():
    B, V = 2, 7
    image_size, splat_size = 128, 32
    x = torch.randn(B, V, 9, image_size, image_size)
    K, c2w = make_dummy_cameras(B, V, image_size)
    model = ZeroGSUNet(
        image_size=image_size,
        splat_size=splat_size,
        base_channels=8,
        use_view_attention=False,
        use_global_attention=False,
    ).eval()

    out = model(x, K=K, c2w=c2w)

    assert out["raw"].shape == (B, V, 16, splat_size, splat_size)
    assert out["gaussian_map"].shape == (B, V, 14, splat_size, splat_size)
    assert out["gaussians"].shape == (B, V * splat_size * splat_size, 14)
    assert out["confidence"].shape == (B, V, 1, splat_size, splat_size)

    gaussians = out["gaussians"]
    opacity = gaussians[..., 3]
    scale = gaussians[..., 4:7]
    quat = gaussians[..., 7:11]
    rgb = gaussians[..., 11:14]

    assert opacity.min() >= 0.0 and opacity.max() <= 1.0
    assert scale.min() >= 0.0
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0

    quat_norm = torch.linalg.norm(quat, dim=-1)
    assert torch.allclose(
        quat_norm.mean(),
        torch.tensor(1.0, dtype=quat_norm.dtype),
        atol=1e-2,
    )


@torch.no_grad()
def test_small_size_with_attention():
    B, V = 1, 7
    image_size, splat_size = 128, 32
    x = torch.randn(B, V, 9, image_size, image_size)
    K, c2w = make_dummy_cameras(B, V, image_size)
    model = ZeroGSUNet(
        image_size=image_size,
        splat_size=splat_size,
        base_channels=8,
        use_view_attention=True,
        use_global_attention=True,
    ).eval()

    out = model(x, K=K, c2w=c2w)
    assert out["raw"].shape == (B, V, 16, splat_size, splat_size)
    assert out["gaussians"].shape == (B, V * splat_size * splat_size, 14)


def run_all_tests() -> None:
    test_raw_output_shape_256()
    test_gaussian_output_shapes_and_ranges()
    test_small_size_with_attention()


if __name__ == "__main__":
    run_all_tests()
    print("All ZeroGSUNet tests passed.")
