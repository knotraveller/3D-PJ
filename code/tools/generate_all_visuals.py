"""Generate reconstruction comparison grids for every rendered ref_* sample."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from datasets import ObjaverseRenderedDataset
from models import ZeroGSUNet
from renderers import GSplatRenderer
from utils.checkpoint import load_checkpoint
from utils.ray_utils import get_embedding
from utils.visualization import save_training_visualization


REF_DIR_PATTERN = re.compile(r"^ref_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct every ref_XXX sample below an image root."
    )
    parser.add_argument("--model", required=True, help="Path to a model checkpoint.")
    parser.add_argument(
        "--image",
        required=True,
        help="Root directory recursively containing ref_XXX folders.",
    )
    return parser.parse_args()


def is_ref_directory(path: Path) -> bool:
    """Return whether a path has the expected ref_XXX directory name."""
    return path.is_dir() and REF_DIR_PATTERN.fullmatch(path.name) is not None


def output_path_for_sample(
    sample_dir: Path,
    image_root: Path,
    output_root: Path,
) -> Path:
    """Map a sample directory to a collision-free PNG path."""
    try:
        relative_dir = sample_dir.resolve().relative_to(image_root.resolve())
    except ValueError:
        relative_dir = Path(sample_dir.name)
    return output_root / relative_dir.parent / f"{relative_dir.name}.png"


def _load_model_components(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ZeroGSUNet, GSplatRenderer, Dict]:
    checkpoint = load_checkpoint(str(checkpoint_path), map_location=device)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise KeyError("Checkpoint must contain both 'model' and 'config'.")

    config = checkpoint["config"]
    model = ZeroGSUNet(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    renderer = GSplatRenderer(**config["renderer"]).to(device)
    renderer.eval()
    return model, renderer, config


def _batch_to_device(
    batch: Dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> Dict[str, torch.Tensor | list[str]]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.model)
    image_root = Path(args.image)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, config = _load_model_components(checkpoint_path, device)
    image_size = int(config["model"]["image_size"])

    dataset = ObjaverseRenderedDataset(
        root_dir=str(image_root),
        image_size=image_size,
    )
    ref_indices = [
        index
        for index, sample_dir in enumerate(dataset.sample_dirs)
        if is_ref_directory(sample_dir)
    ]
    if not ref_indices:
        raise FileNotFoundError(
            f"No complete ref_XXX samples were found below: {image_root}"
        )

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_root = Path("outputs") / "all_visuals" / timestamp
    output_root.mkdir(parents=True, exist_ok=False)

    loader = DataLoader(
        Subset(dataset, ref_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(f"Found {len(ref_indices)} ref_XXX sample(s) under {image_root.resolve()}")
    print(f"Using model checkpoint: {checkpoint_path.resolve()}")
    print(f"Visualizations will be saved to {output_root.resolve()}")

    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        progress = tqdm(loader, desc="generate visuals", ascii=True)
        for batch in progress:
            batch = _batch_to_device(batch, device)
            images = batch["images"]
            alphas = batch["alphas"]
            K = batch["K"]
            c2w = batch["c2w"]
            w2c = batch["w2c"]
            sample_dirs = batch["sample_dir"]
            assert torch.is_tensor(images)
            assert torch.is_tensor(alphas)
            assert torch.is_tensor(K)
            assert torch.is_tensor(c2w)
            assert torch.is_tensor(w2c)
            assert isinstance(sample_dirs, list)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                ray_embedding = get_embedding(
                    K=K,
                    c2w=c2w,
                    resolution=image_size,
                    embedding_type="plucker",
                    order="dm",
                    channel_first=True,
                )
                model_input = torch.cat([images, ray_embedding], dim=2)
                model_output = model(model_input, K=K, c2w=c2w)
                render_output = renderer(model_output["gaussians"], K=K, w2c=w2c)

            sample_dir = Path(sample_dirs[0])
            save_path = output_path_for_sample(sample_dir, image_root, output_root)
            save_training_visualization(
                pred_rgb=render_output["rgb"],
                pred_alpha=render_output["alpha"],
                gt_rgb=images,
                gt_alpha=alphas,
                save_path=str(save_path),
            )
            tqdm.write(f"Visualization saved to {save_path.resolve()}")

    print(
        f"Generated {len(ref_indices)} visualization(s) in "
        f"{output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
