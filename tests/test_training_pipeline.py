"""Manual end-to-end differentiable training pipeline test.

This test requires gsplat. Run:
    $env:PYTHONPATH="code"
    conda run -n 3d python tests/test_training_pipeline.py --root dataset/renders_256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from datasets import ObjaverseRenderedDataset
from models import ZeroGSUNet
from renderers import GSplatRenderer
from training.losses import ReconstructionLoss
from utils.ray_utils import get_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one training pipeline backward pass.")
    parser.add_argument("--root", default="dataset/renders_256")
    parser.add_argument("--image_size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ObjaverseRenderedDataset(
        root_dir=args.root,
        image_size=args.image_size,
        max_samples=1,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = batch["images"].to(device)
    alphas = batch["alphas"].to(device)
    K = batch["K"].to(device)
    c2w = batch["c2w"].to(device)
    w2c = batch["w2c"].to(device)

    ray_emb = get_embedding(K, c2w, resolution=args.image_size)
    x = torch.cat([images, ray_emb], dim=2)

    model = ZeroGSUNet(
        image_size=args.image_size,
        splat_size=args.image_size // 4,
        base_channels=8,
        use_view_attention=False,
        use_global_attention=False,
    ).to(device)
    renderer = GSplatRenderer(image_size=args.image_size).to(device)
    criterion = ReconstructionLoss(use_lpips=False).to(device)

    out = model(x, K=K, c2w=c2w)
    render_out = renderer(out["gaussians"], K=K, w2c=w2c)
    loss_dict = criterion(
        pred_rgb=render_out["rgb"],
        pred_alpha=render_out["alpha"],
        gt_rgb=images,
        gt_alpha=alphas,
    )
    loss = loss_dict["loss"]
    loss.backward()
    print("pipeline ok")


if __name__ == "__main__":
    main()
