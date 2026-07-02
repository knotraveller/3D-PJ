from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models import ZeroGSUNet
from renderers import GSplatRenderer
from utils.camera_utils import get_zero123pp_camera_specs
from utils.ray_utils import get_embedding


IMAGE_SIZE = 256
DEFAULT_IMAGE_DIR = ROOT / "tests" / "targets"
OUTPUT_DIR = ROOT / "outputs" / "fuse2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Zero123++-ordered 7 images through a ZeroGS checkpoint."
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        default=str(DEFAULT_IMAGE_DIR),
        help="Folder containing cond.png and 000_rgb.png..005_rgb.png.",
    )
    parser.add_argument("--gs_model", required=True, help="ZeroGS .pt checkpoint path.")
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


def load_rgb(path: Path, size: int = IMAGE_SIZE) -> torch.Tensor:
    image = Image.open(path).convert("RGBA")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    rgba = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    alpha = rgba[3:4]
    return (rgba[:3] * alpha + (1.0 - alpha)).clamp(0.0, 1.0)


def image_paths(image_dir: Path) -> list[Path]:
    paths = [image_dir / "cond.png"] + [image_dir / f"{idx:03d}_rgb.png" for idx in range(6)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required Zero123++ input images: {names}")
    return paths


def load_views(image_dir: Path) -> torch.Tensor:
    return torch.stack([load_rgb(path) for path in image_paths(image_dir)], dim=0)


def camera_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cameras = get_zero123pp_camera_specs(resolution=IMAGE_SIZE)
    views = sorted(cameras["views"], key=lambda view: int(view["index"]))
    K = torch.from_numpy(np.stack([view["K"] for view in views], axis=0)).float()
    c2w = torch.from_numpy(np.stack([view["c2w"] for view in views], axis=0)).float()
    w2c = torch.from_numpy(np.stack([view["w2c"] for view in views], axis=0)).float()
    return K[None].to(device), c2w[None].to(device), w2c[None].to(device)


def render_views(images: torch.Tensor, gs_model: Path, device: torch.device) -> torch.Tensor:
    checkpoint = load_checkpoint(gs_model, device)
    config = deepcopy(checkpoint["config"])
    config["model"]["image_size"] = IMAGE_SIZE
    config["model"]["splat_size"] = IMAGE_SIZE // 4
    config["renderer"]["image_size"] = IMAGE_SIZE

    model = ZeroGSUNet(**config["model"]).to(device)
    renderer = GSplatRenderer(**config["renderer"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    renderer.eval()

    images = F.interpolate(
        images,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    images = images[None].to(device).clamp(0.0, 1.0)
    K, c2w, w2c = camera_tensors(device)

    with torch.no_grad():
        rays = get_embedding(
            K=K,
            c2w=c2w,
            resolution=IMAGE_SIZE,
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
    image_dir = Path(args.image_dir)
    gs_model = Path(args.gs_model)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = load_views(image_dir)
    recon = render_views(images, gs_model, device)
    output_path = OUTPUT_DIR / f"{image_dir.name}_gs.png"
    save_two_row_grid(images, recon, output_path)
    print(f"input images: {image_dir}")
    print(f"GS model: {gs_model}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
