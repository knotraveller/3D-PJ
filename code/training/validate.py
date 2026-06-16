"""Validation entrypoint for ZeroGS-UNet checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from training.trainer import Trainer
from utils.config import load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a ZeroGS-UNet checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to load.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"].get("seed", 42)))
    trainer = Trainer(config)
    checkpoint_epoch = trainer.load_checkpoint(args.checkpoint, resume=False)
    val_loss = trainer.validate(epoch=checkpoint_epoch, save_outputs=True)
    trainer.writer.close()
    print(f"validation loss: {val_loss:.6f}")


if __name__ == "__main__":
    main()
