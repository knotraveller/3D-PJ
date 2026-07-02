from __future__ import annotations

import json
from contextlib import nullcontext
import sys
from types import MethodType
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from training.losses import ReconstructionLoss
from training.trainer import Trainer, summarize_metric_rows, validation_visual_filename


def test_summarize_metric_rows_computes_mean_max_min() -> None:
    summary = summarize_metric_rows(
        [
            {
                "loss": 0.4,
                "rgb_loss": 0.3,
                "mask_loss": 0.2,
                "lpips_loss": 0.1,
                "psnr": 20.0,
            },
            {
                "loss": 0.2,
                "rgb_loss": 0.1,
                "mask_loss": 0.4,
                "lpips_loss": 0.3,
                "psnr": 24.0,
            },
        ]
    )

    assert summary["loss"] == pytest.approx(
        {"mean": 0.3, "max": 0.4, "min": 0.2}
    )
    assert summary["mask_loss"] == pytest.approx(
        {"mean": 0.3, "max": 0.4, "min": 0.2}
    )
    assert summary["psnr"] == pytest.approx(
        {"mean": 22.0, "max": 24.0, "min": 20.0}
    )


def test_validation_visual_filename_is_filesystem_safe() -> None:
    assert validation_visual_filename("asset_ref_000") == "asset_ref_000.png"
    assert validation_visual_filename("asset/ref:000") == "asset_ref_000.png"


def test_summarize_metric_rows_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="empty metric"):
        summarize_metric_rows([])


def test_train_epoch_log_uses_epoch_metric_summaries(tmp_path: Path) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.log_dir = tmp_path / "logs"
    trainer.log_dir.mkdir(parents=True)
    trainer.global_step = 7
    trainer.writer = _ScalarWriter()
    trainer.optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=1.0e-4)

    trainer._log_train_epoch(
        3,
        [
            {
                "loss": 0.4,
                "rgb_loss": 0.2,
                "mask_loss": 0.1,
                "lpips_loss": 0.05,
                "psnr": 20.0,
            },
            {
                "loss": 0.2,
                "rgb_loss": 0.1,
                "mask_loss": 0.3,
                "lpips_loss": 0.15,
                "psnr": 24.0,
            },
        ],
    )

    row = json.loads((trainer.log_dir / "train_log.jsonl").read_text(encoding="utf-8"))
    assert "step" not in row
    assert row["epoch"] == 3
    assert row["num_batches"] == 2
    assert row["loss"] == pytest.approx({"mean": 0.3, "max": 0.4, "min": 0.2})
    assert row["psnr"] == pytest.approx({"mean": 22.0, "max": 24.0, "min": 20.0})
    assert any(
        name == "train/loss" and value == pytest.approx(0.3) and step == 7
        for name, value, step in trainer.writer.scalars
    )


class _DisabledProfiler:
    enabled = False

    @staticmethod
    def track(_name: str, **_kwargs: object):
        return nullcontext()


class _ScalarWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars.append((name, value, step))


def test_train_one_epoch_accumulates_gradients(tmp_path: Path) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
    trainer.amp_enabled = False
    trainer.device = torch.device("cpu")
    trainer.profiler = _DisabledProfiler()
    trainer.writer = _ScalarWriter()
    trainer.global_step = 0
    trainer.performance_write_every = 0
    trainer.config = {
        "train": {
            "grad_clip": None,
            "gradient_accumulation_steps": 2,
            "epoch_visuals": False,
        }
    }
    trainer.log_dir = tmp_path / "logs"
    trainer.plot_dir = tmp_path / "plots"
    trainer.train_loader = [
        {"x": torch.ones(1, 1)},
        {"x": torch.ones(1, 1)},
        {"x": torch.ones(1, 1)},
    ]

    def _forward_batch(self, moved_batch, stage_prefix):
        loss = self.model(moved_batch["x"]).sum()
        zero = loss.detach() * 0.0
        return {
            "loss": loss,
            "rgb_loss": loss.detach(),
            "mask_loss": zero,
            "lpips_loss": zero,
            "psnr": loss.detach() + 20.0,
        }

    trainer._forward_batch = MethodType(_forward_batch, trainer)
    trainer.train_one_epoch(epoch=1, total_epochs=1)

    assert trainer.global_step == 3
    assert trainer.model.weight.item() == pytest.approx(-1.0)
    row = json.loads((trainer.log_dir / "train_log.jsonl").read_text(encoding="utf-8"))
    assert "step" not in row
    assert row["epoch"] == 1
    assert row["num_batches"] == 3


def test_validate_saves_all_visuals_and_loss_yaml(tmp_path: Path) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.model = nn.Identity()
    trainer.device = torch.device("cpu")
    trainer.criterion = ReconstructionLoss(
        use_lpips=False,
        lambda_rgb=1.0,
        lambda_mask=0.5,
    )
    trainer.profiler = _DisabledProfiler()
    trainer.writer = _ScalarWriter()
    trainer.global_step = 12
    trainer.performance_write_every = 0
    trainer.validate_dir = tmp_path / "validate"
    trainer.validate_visual_dir = trainer.validate_dir / "all_visuals"
    trainer.validate_epoch_visual_dir = trainer.validate_dir / "epoch_visuals"
    trainer.validate_log_dir = trainer.validate_dir / "logs"
    trainer.validate_visual_dir.mkdir(parents=True)
    trainer.validate_epoch_visual_dir.mkdir(parents=True)
    trainer.validate_log_dir.mkdir(parents=True)

    images = torch.zeros(2, 1, 3, 4, 4)
    alphas = torch.ones(2, 1, 1, 4, 4)
    pred_rgb = torch.stack(
        [
            torch.full((1, 3, 4, 4), 0.2),
            torch.full((1, 3, 4, 4), 0.4),
        ]
    )
    pred_alpha = torch.stack(
        [
            torch.full((1, 1, 4, 4), 0.8),
            torch.full((1, 1, 4, 4), 0.6),
        ]
    )
    batch = {
        "images": images,
        "alphas": alphas,
        "sample_id": ["asset_a_ref_000", "asset_b/ref:090"],
    }
    trainer.val_loader = [batch]

    def _forward_batch(self, moved_batch, stage_prefix):
        losses = self.criterion(
            pred_rgb=pred_rgb,
            pred_alpha=pred_alpha,
            gt_rgb=moved_batch["images"],
            gt_alpha=moved_batch["alphas"],
        )
        return {
            "images": moved_batch["images"],
            "alphas": moved_batch["alphas"],
            "pred_rgb": pred_rgb,
            "pred_alpha": pred_alpha,
            "psnr": torch.tensor(0.0),
            **losses,
        }

    trainer._forward_batch = MethodType(_forward_batch, trainer)
    mean_loss = trainer.validate(epoch=5, save_outputs=True)

    assert mean_loss == pytest.approx(0.45)
    assert (trainer.validate_visual_dir / "asset_a_ref_000.png").is_file()
    assert (trainer.validate_visual_dir / "asset_b_ref_090.png").is_file()

    payload = yaml.safe_load(
        (trainer.validate_dir / "loss.yaml").read_text(encoding="utf-8")
    )
    assert payload["epoch"] == 5
    assert payload["num_samples"] == 2
    assert payload["loss"] == pytest.approx(
        {"mean": 0.45, "max": 0.6, "min": 0.3}
    )
    assert "psnr" in payload
    validate_log = trainer.validate_log_dir / "validate_log.jsonl"
    log_row = json.loads(validate_log.read_text(encoding="utf-8").splitlines()[0])
    assert "step" not in log_row
    assert log_row["epoch"] == 5
    assert log_row["loss"] == pytest.approx(
        {"mean": 0.45, "max": 0.6, "min": 0.3}
    )
