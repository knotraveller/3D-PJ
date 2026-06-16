"""Command-line training entrypoint.

Run from the repository root with:
    $env:PYTHONPATH="code"; python -m training.train --config configs/zerogs_default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from training.trainer import Trainer
from utils.config import apply_debug_overrides, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ZeroGS-UNet.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint whose model weights should be loaded.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer and epoch state from --checkpoint instead of finetuning.",
    )
    parser.add_argument("--debug", action="store_true", help="Use tiny debug settings.")
    parser.add_argument(
        "--overfit_one_batch",
        action="store_true",
        help="Repeat one batch to test differentiability and loss decrease.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and not args.checkpoint:
        raise SystemExit("--resume requires --checkpoint <path>.")

    config = load_config(args.config)
    if args.debug:
        config = apply_debug_overrides(config)
    if args.overfit_one_batch:
        config["train"]["overfit_one_batch"] = True
        config["data"]["num_workers"] = 0
        config["train"].setdefault("overfit_steps", 300)
        config["train"]["log_every"] = min(int(config["train"].get("log_every", 20)), 20)
        config["train"]["vis_every"] = min(int(config["train"].get("vis_every", 20)), 20)

    seed_everything(int(config["experiment"].get("seed", 42)))
    trainer = Trainer(config)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint, resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
