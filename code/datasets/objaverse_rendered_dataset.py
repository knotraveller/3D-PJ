"""Dataset for Blender-rendered Objaverse multi-view samples."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CODE_DIR = Path(__file__).resolve().parents[1]
UTILS_DIR = CODE_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from camera_utils import load_cameras_json


TARGET_COUNT = 6


def _load_rgb(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_alpha(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).contiguous()


def _views_from_cameras_json(cameras: Dict) -> List[Dict]:
    """Return 7 camera views in cond, target_000..target_005 order."""
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


def _infer_camera_resolution(cameras: Dict, views: List[Dict]) -> float:
    if "resolution" in cameras:
        return float(cameras["resolution"])
    K0 = np.asarray(views[0]["K"], dtype=np.float32)
    # For centered square renders, cx = cy = resolution / 2.
    return float(K0[0, 2] * 2.0)


def _scale_intrinsics(K: np.ndarray, source_size: float, target_size: int) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32).copy()
    if source_size <= 0:
        raise ValueError(f"Invalid camera source resolution: {source_size}")
    scale = float(target_size) / float(source_size)
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale
    return K


def _sample_id_from_dir(sample_dir: Path) -> str:
    parent = sample_dir.parent.name
    return f"{parent}_{sample_dir.name}"


class ObjaverseRenderedDataset(Dataset):
    """Load rendered 7-view Objaverse samples.

    Each item contains:
        images: [7, 3, H, W]
        alphas: [7, 1, H, W]
        K: [7, 3, 3]
        c2w: [7, 4, 4]
        w2c: [7, 4, 4]
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        split_file: Optional[str] = None,
        max_samples: Optional[int] = None,
        require_all_views: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = int(image_size)
        self.require_all_views = bool(require_all_views)

        if split_file:
            self.sample_dirs = self._load_split_file(Path(split_file))
        else:
            self.sample_dirs = self._discover_samples()

        if max_samples is not None:
            self.sample_dirs = self.sample_dirs[: int(max_samples)]
        if not self.sample_dirs:
            raise FileNotFoundError(f"No rendered samples found under {self.root_dir}.")

    def _load_split_file(self, split_file: Path) -> List[Path]:
        if not split_file.is_file():
            raise FileNotFoundError(f"split_file does not exist: {split_file}")

        sample_dirs = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = self.root_dir / path
            if self._is_valid_sample(path):
                sample_dirs.append(path)
            elif self.require_all_views:
                raise FileNotFoundError(f"Invalid sample listed in split file: {path}")
        return sorted(sample_dirs)

    def _discover_samples(self) -> List[Path]:
        sample_dirs = []
        for cameras_json in self.root_dir.rglob("cameras.json"):
            sample_dir = cameras_json.parent
            if self._is_valid_sample(sample_dir):
                sample_dirs.append(sample_dir)
        return sorted(sample_dirs)

    def _is_valid_sample(self, sample_dir: Path) -> bool:
        required = [
            sample_dir / "cameras.json",
            sample_dir / "cond" / "rgb.png",
            sample_dir / "cond" / "alpha.png",
        ]
        for idx in range(TARGET_COUNT):
            required.append(sample_dir / "targets" / f"{idx:03d}_rgb.png")
            required.append(sample_dir / "targets" / f"{idx:03d}_alpha.png")
        if self.require_all_views:
            return all(path.is_file() for path in required)
        return (sample_dir / "cameras.json").is_file()

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample_dir = self.sample_dirs[index]
        rgb_paths = [sample_dir / "cond" / "rgb.png"] + [
            sample_dir / "targets" / f"{idx:03d}_rgb.png"
            for idx in range(TARGET_COUNT)
        ]
        alpha_paths = [sample_dir / "cond" / "alpha.png"] + [
            sample_dir / "targets" / f"{idx:03d}_alpha.png"
            for idx in range(TARGET_COUNT)
        ]

        images = torch.stack(
            [_load_rgb(path, self.image_size) for path in rgb_paths],
            dim=0,
        )
        alphas = torch.stack(
            [_load_alpha(path, self.image_size) for path in alpha_paths],
            dim=0,
        )

        cameras = load_cameras_json(str(sample_dir / "cameras.json"))
        views = _views_from_cameras_json(cameras)
        if len(views) != TARGET_COUNT + 1:
            raise ValueError(f"Expected 7 camera views in {sample_dir}, got {len(views)}.")

        source_size = _infer_camera_resolution(cameras, views)
        K = torch.from_numpy(
            np.stack(
                [
                    _scale_intrinsics(view["K"], source_size, self.image_size)
                    for view in views
                ],
                axis=0,
            )
        ).float()
        c2w = torch.from_numpy(np.stack([view["c2w"] for view in views], axis=0)).float()
        w2c = torch.from_numpy(np.stack([view["w2c"] for view in views], axis=0)).float()

        return {
            "images": images,
            "alphas": alphas,
            "K": K,
            "c2w": c2w,
            "w2c": w2c,
            "sample_id": _sample_id_from_dir(sample_dir),
            "sample_dir": str(sample_dir),
        }


if __name__ == "__main__":
    dataset = ObjaverseRenderedDataset(
        root_dir="dataset/renders_256",
        image_size=256,
        max_samples=3,
    )
    item = dataset[0]
    print(item["images"].shape)
    print(item["alphas"].shape)
    print(item["K"].shape)
