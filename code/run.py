"""Minimal data-loading demo for rendered Objaverse multi-view samples.

This file intentionally skips model/training code. It only shows how to:
  1. find rendered sample folders,
  2. load RGB images and cameras.json,
  3. stack K/c2w for 7 views,
  4. build ray embeddings online,
  5. concatenate RGB and ray embedding into [B, V, 9, H, W].

Run:
    conda run -n 3d python code/run.py --renders_root dataset/renders_256 --batch_size 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


UTILS_DIR = Path(__file__).resolve().parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from camera_utils import load_cameras_json
from ray_utils import get_embedding


TARGET_COUNT = 6


def find_sample_dirs(renders_root: Path) -> List[Path]:
    """Return folders that look like one rendered ref sample."""
    sample_dirs = []
    for cameras_json in renders_root.rglob("cameras.json"):
        sample_dir = cameras_json.parent
        if (sample_dir / "cond" / "rgb.png").is_file() and (
            sample_dir / "targets" / "000_rgb.png"
        ).is_file():
            sample_dirs.append(sample_dir)
    return sorted(sample_dirs)


def load_rgb(path: Path, resolution: int) -> torch.Tensor:
    """Load one RGB image as float tensor [3, H, W] in [0, 1]."""
    image = Image.open(path).convert("RGB")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def camera_views_from_json(cameras: Dict) -> List[Dict]:
    """Normalize new views-format and old input/targets-format cameras.json."""
    if "views" in cameras:
        return sorted(cameras["views"], key=lambda view: int(view["index"]))

    views = []
    cond = dict(cameras["input"])
    cond.setdefault("name", "cond")
    cond.setdefault("index", 0)
    cond.setdefault("relative_azimuth", 0.0)
    views.append(cond)

    for idx, target in enumerate(cameras["targets"], start=1):
        view = dict(target)
        view.setdefault("name", f"target_{idx - 1:03d}")
        view["index"] = idx
        views.append(view)
    return views


class RenderedObjaverseDataset(Dataset):
    """Dataset returning only images and camera matrices for one 7-view sample."""

    def __init__(self, renders_root: str, resolution: int = 256) -> None:
        self.renders_root = Path(renders_root)
        self.resolution = int(resolution)
        self.sample_dirs = find_sample_dirs(self.renders_root)
        if not self.sample_dirs:
            raise FileNotFoundError(f"No rendered samples found under {self.renders_root}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample_dir = self.sample_dirs[index]

        image_paths = [sample_dir / "cond" / "rgb.png"]
        image_paths.extend(
            sample_dir / "targets" / f"{target_idx:03d}_rgb.png"
            for target_idx in range(TARGET_COUNT)
        )
        images = torch.stack(
            [load_rgb(path, self.resolution) for path in image_paths],
            dim=0,
        )  # [V, 3, H, W]
        

        cameras = load_cameras_json(str(sample_dir / "cameras.json"))
        views = camera_views_from_json(cameras)
        if len(views) != TARGET_COUNT + 1:
            raise ValueError(f"Expected 7 camera views in {sample_dir}, got {len(views)}")

        K = torch.from_numpy(np.stack([view["K"] for view in views], axis=0)).float()
        c2w = torch.from_numpy(np.stack([view["c2w"] for view in views], axis=0)).float()

        return {
            "sample_dir": str(sample_dir),
            "images": images,
            "K": K,
            "c2w": c2w,
            "mask": torch.ones_like(images[:, :1]), # TODO: load actual masks
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect rendered data loading.")
    parser.add_argument("--renders_root", default="dataset/renders_256")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    dataset = RenderedObjaverseDataset(
        renders_root=args.renders_root,
        resolution=args.resolution,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))
    images = batch["images"]  # [B, V, 3, H, W]
    K = batch["K"]            # [B, V, 3, 3]
    c2w = batch["c2w"]        # [B, V, 4, 4]
    mask = batch["mask"]      # [B, V, 1, H, W]

    ray_emb = get_embedding(
        K=K,
        c2w=c2w,
        resolution=args.resolution,
        embedding_type="plucker",
        order="dm",
        channel_first=True,
    )  # [B, V, 6, H, W]

    model_input = torch.cat([images, ray_emb], dim=2)

    print(f"dataset size: {len(dataset)}")
    print(f"first sample: {batch['sample_dir'][0]}")
    print(f"images:      {tuple(images.shape)}")
    print(f"K:           {tuple(K.shape)}")
    print(f"c2w:         {tuple(c2w.shape)}")
    print(f"ray_emb:     {tuple(ray_emb.shape)}")
    print(f"model_input: {tuple(model_input.shape)}")
    print(f"mask:        TOBE finished")


    


if __name__ == "__main__":
    main()
