from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
MINI_ROOT = ROOT.parent / "mini123"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(MINI_ROOT) not in sys.path:
    sys.path.insert(1, str(MINI_ROOT))

from mini123.diffusion import GaussianDiffusion
from mini123.model import MiniViewUNet
from models import ZeroGSUNet
from renderers import GSplatRenderer
from utils.camera_utils import get_zero123pp_camera_specs
from utils.ray_utils import get_embedding


MINI_IMAGE_SIZE = 128
GS_IMAGE_SIZE = 256
OUTPUT_DIR = ROOT / "outputs" / "fuse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single image -> mini123 multi-view -> ZeroGS Gaussian render."
    )
    parser.add_argument("image", help="Path to one input image.")
    parser.add_argument("--123_model", dest="model_123", required=True, help="mini123 .pt path.")
    parser.add_argument("--gs_model", required=True, help="ZeroGS .pt path.")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' field: {path}")
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'config' field: {path}")
    return checkpoint


def load_rgba(path: Path, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(path).convert("RGBA")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    rgba = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    alpha = rgba[3:4]
    rgb = rgba[:3] * alpha + (1.0 - alpha)
    return rgb, alpha


def pose_codes(config: dict) -> torch.Tensor:
    pose_dim = int(config["model"].get("pose_dim", 4))
    if pose_dim == 6 or str(config["data"].get("pose_code_mode", "fixed4")) == "camera6":
        relative_azimuths = (30.0, 90.0, 150.0, 210.0, 270.0, 330.0)
        target_elevations = (20.0, -10.0, 20.0, -10.0, 20.0, -10.0)
        rows = []
        for azimuth, elevation in zip(relative_azimuths, target_elevations):
            da = math.radians(azimuth)
            de = math.radians(elevation)
            rows.append([math.sin(da), math.cos(da), math.sin(de), math.cos(de), 0.0, 0.0])
        return torch.tensor(rows, dtype=torch.float32)

    from mini123.datasets import all_target_pose_codes

    return all_target_pose_codes(mode="fixed4")


def mini_generate_views(
    image_path: Path,
    model_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint = load_checkpoint(model_path, device)
    config = deepcopy(checkpoint["config"])
    config["data"]["image_size"] = MINI_IMAGE_SIZE

    cond_rgb, cond_alpha = load_rgba(image_path, MINI_IMAGE_SIZE)
    cond = cond_rgb.mul(2.0).sub(1.0)[None].repeat(6, 1, 1, 1).to(device)
    cond_for_model = cond
    if bool(config["data"].get("use_alpha_condition", False)):
        cond_for_model = torch.cat(
            [cond, cond_alpha[None].repeat(6, 1, 1, 1).to(device)],
            dim=1,
        )

    model = MiniViewUNet(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    diffusion = GaussianDiffusion(**config["diffusion"]).to(device)
    with torch.no_grad():
        samples, _ = diffusion.sample(
            model,
            cond_for_model,
            pose_codes(config).to(device),
            image_size=MINI_IMAGE_SIZE,
            trajectory_steps=1,
            out_channels=int(config["model"].get("out_channels", 3)),
        )

    rgb = samples[:, :3].add(1.0).mul(0.5).clamp(0.0, 1.0)
    if samples.shape[1] >= 4:
        alpha = samples[:, 3:4].add(1.0).mul(0.5).clamp(0.0, 1.0)
        rgb = rgb * alpha + (1.0 - alpha)
    else:
        alpha = torch.ones(samples.shape[0], 1, MINI_IMAGE_SIZE, MINI_IMAGE_SIZE, device=device)
    return rgb.detach(), alpha.detach()


def camera_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cameras = get_zero123pp_camera_specs(resolution=GS_IMAGE_SIZE)
    views = sorted(cameras["views"], key=lambda view: int(view["index"]))
    K = torch.from_numpy(np.stack([view["K"] for view in views], axis=0)).float()
    c2w = torch.from_numpy(np.stack([view["c2w"] for view in views], axis=0)).float()
    w2c = torch.from_numpy(np.stack([view["w2c"] for view in views], axis=0)).float()
    return K[None].to(device), c2w[None].to(device), w2c[None].to(device)


def resize_views(views: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(views, size=(size, size), mode="bilinear", align_corners=False)


def zerogs_render(
    cond_rgb_256: torch.Tensor,
    target_rgb_128: torch.Tensor,
    gs_model_path: Path,
    device: torch.device,
) -> torch.Tensor:
    checkpoint = load_checkpoint(gs_model_path, device)
    config = deepcopy(checkpoint["config"])
    config["model"]["image_size"] = GS_IMAGE_SIZE
    config["model"]["splat_size"] = GS_IMAGE_SIZE // 4
    config["renderer"]["image_size"] = GS_IMAGE_SIZE

    model = ZeroGSUNet(**config["model"]).to(device)
    renderer = GSplatRenderer(**config["renderer"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    renderer.eval()

    target_rgb_256 = resize_views(target_rgb_128, GS_IMAGE_SIZE)
    images = torch.cat([cond_rgb_256[None].to(device), target_rgb_256], dim=0)[None]
    images = images.clamp(0.0, 1.0)

    K, c2w, w2c = camera_tensors(device)
    with torch.no_grad():
        rays = get_embedding(
            K=K,
            c2w=c2w,
            resolution=GS_IMAGE_SIZE,
            embedding_type="plucker",
            order="dm",
            channel_first=True,
        )
        model_out = model(torch.cat([images, rays], dim=2), K=K, c2w=c2w)
        render_out = renderer(model_out["gaussians"], K=K, w2c=w2c)
    return render_out["rgb"][0].detach().clamp(0.0, 1.0)


def save_two_row_grid(top: torch.Tensor, bottom: torch.Tensor, output_path: Path) -> None:
    if top.shape != bottom.shape or top.shape[0] != 7:
        raise ValueError(f"Expected two [7,3,H,W] tensors, got {top.shape} and {bottom.shape}")
    top = top.detach().float().cpu().clamp(0.0, 1.0)
    bottom = bottom.detach().float().cpu().clamp(0.0, 1.0)
    tiles = torch.cat([top, bottom], dim=0)
    height, width = int(tiles.shape[-2]), int(tiles.shape[-1])
    canvas = Image.new("RGB", (7 * width, 2 * height), "white")
    for idx, tensor in enumerate(tiles):
        array = (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        row, col = divmod(idx, 7)
        canvas.paste(Image.fromarray(array), (col * width, row * height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    model_123_path = Path(args.model_123)
    gs_model_path = Path(args.gs_model)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_rgb_256, _ = load_rgba(image_path, GS_IMAGE_SIZE)
    target_rgb_128, _ = mini_generate_views(image_path, model_123_path, device)
    recon_rgb_256 = zerogs_render(cond_rgb_256, target_rgb_128, gs_model_path, device)

    target_rgb_256 = resize_views(target_rgb_128, GS_IMAGE_SIZE)
    row_top = torch.cat([cond_rgb_256[None].to(target_rgb_256.device), target_rgb_256], dim=0)

    output_path = OUTPUT_DIR / f"{image_path.stem}_fuse.png"
    save_two_row_grid(row_top, recon_rgb_256, output_path)
    print(f"123 model: {model_123_path}")
    print(f"GS model: {gs_model_path}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
