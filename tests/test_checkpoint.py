from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from training.trainer import Trainer
from utils.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_save_can_refuse_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "epoch_0001.pt"
    payload = {
        "completed_epoch": 1,
        "model": {"weight": torch.ones(1)},
    }

    saved_path = save_checkpoint(payload, str(path), overwrite=False)
    assert saved_path == path
    assert load_checkpoint(str(path))["completed_epoch"] == 1

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_checkpoint(payload, str(path), overwrite=False)


def _make_minimal_trainer(
    tmp_path: Path,
    *,
    epochs: int = 10,
    scheduler: str = "cosine",
    lr: float = 1.0e-3,
    lr_min: float = 0.0,
    plateau_patience: int = 10,
) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.config = {
        "train": {
            "epochs": epochs,
            "scheduler": scheduler,
            "lr": lr,
            "lr_min": lr_min,
            "plateau_patience": plateau_patience,
        }
    }
    trainer.device = torch.device("cpu")
    trainer.model = nn.Linear(2, 1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=lr)
    scheduler_name = Trainer._normalize_scheduler_name(scheduler)
    if scheduler_name == "cosine":
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=epochs,
            eta_min=lr_min,
        )
    elif scheduler_name == "plateau":
        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer,
            mode="min",
            factor=0.5,
            patience=plateau_patience,
            min_lr=lr_min,
        )
    else:
        trainer.scheduler = None
    trainer.amp_enabled = False
    trainer.start_epoch = 0
    trainer.global_step = 0
    trainer.best_val_loss = float("inf")
    trainer.log_dir = tmp_path / "logs"
    trainer.log_dir.mkdir(parents=True)
    return trainer


def _advance_scheduler(trainer: Trainer, epochs: int) -> None:
    for _ in range(epochs):
        trainer.optimizer.step()
        trainer.scheduler.step()


def test_trainer_resume_restores_progress_and_logs_event(tmp_path: Path) -> None:
    source = _make_minimal_trainer(tmp_path / "source")
    _advance_scheduler(source, 4)
    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(
        {
            "completed_epoch": 4,
            "global_step": 123,
            "model": source.model.state_dict(),
            "optimizer": source.optimizer.state_dict(),
            "scheduler": source.scheduler.state_dict(),
            "scaler": None,
            "config": {},
            "best_val_loss": 0.25,
        },
        str(checkpoint_path),
        overwrite=False,
    )

    trainer = _make_minimal_trainer(tmp_path / "target")
    loaded_epoch = trainer.load_checkpoint(str(checkpoint_path), resume=True)

    assert loaded_epoch == 4
    assert trainer.start_epoch == 4
    assert trainer.global_step == 123
    assert trainer.best_val_loss == 0.25
    assert trainer.scheduler.last_epoch == 4
    event_log = trainer.log_dir / "train_events.jsonl"
    assert '"event": "resume"' in event_log.read_text(encoding="utf-8")


def test_trainer_finetune_resets_progress(tmp_path: Path) -> None:
    source = _make_minimal_trainer(tmp_path / "source")
    checkpoint_path = tmp_path / "finetune.pt"
    save_checkpoint(
        {
            "completed_epoch": 9,
            "global_step": 321,
            "model": source.model.state_dict(),
            "optimizer": source.optimizer.state_dict(),
            "scheduler": source.scheduler.state_dict(),
            "scaler": None,
            "config": {},
            "best_val_loss": 0.1,
        },
        str(checkpoint_path),
        overwrite=False,
    )

    trainer = _make_minimal_trainer(tmp_path / "target")
    trainer.global_step = 7
    trainer.best_val_loss = 1.0
    loaded_epoch = trainer.load_checkpoint(str(checkpoint_path), resume=False)

    assert loaded_epoch == 9
    assert trainer.start_epoch == 0
    assert trainer.global_step == 0
    assert trainer.best_val_loss == float("inf")
    assert not (trainer.log_dir / "train_events.jsonl").exists()


def test_resume_extends_cosine_scheduler_to_new_total_epochs(
    tmp_path: Path,
) -> None:
    source = _make_minimal_trainer(tmp_path / "source", epochs=10)
    _advance_scheduler(source, 10)
    assert source.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    checkpoint_path = tmp_path / "extended.pt"
    save_checkpoint(
        {
            "completed_epoch": 10,
            "global_step": 100,
            "model": source.model.state_dict(),
            "optimizer": source.optimizer.state_dict(),
            "scheduler": source.scheduler.state_dict(),
            "scaler": None,
            "config": source.config,
            "best_val_loss": 0.5,
        },
        str(checkpoint_path),
        overwrite=False,
    )

    trainer = _make_minimal_trainer(tmp_path / "target", epochs=20)
    trainer.load_checkpoint(str(checkpoint_path), resume=True)

    assert trainer.scheduler.T_max == 20
    assert trainer.scheduler.last_epoch == 10
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-4)
    assert trainer.scheduler.get_last_lr() == pytest.approx([5.0e-4])

    trainer.optimizer.step()
    trainer.scheduler.step()
    expected_next_lr = 1.0e-3 * (1.0 + math.cos(math.pi * 11 / 20)) / 2.0
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected_next_lr)

    event_log = (trainer.log_dir / "train_events.jsonl").read_text(encoding="utf-8")
    assert '"rescaled": true' in event_log
    assert '"previous_t_max": 10' in event_log
    assert '"target_t_max": 20' in event_log


def test_resume_rebuilds_scheduler_when_yaml_switches_strategy(
    tmp_path: Path,
) -> None:
    source = _make_minimal_trainer(tmp_path / "source", scheduler="cosine")
    _advance_scheduler(source, 4)
    checkpoint_path = tmp_path / "switch_to_plateau.pt"
    save_checkpoint(
        {
            "completed_epoch": 4,
            "global_step": 40,
            "model": source.model.state_dict(),
            "optimizer": source.optimizer.state_dict(),
            "scheduler": source.scheduler.state_dict(),
            "scaler": None,
            "config": source.config,
            "best_val_loss": 0.5,
        },
        str(checkpoint_path),
        overwrite=False,
    )

    trainer = _make_minimal_trainer(
        tmp_path / "target",
        epochs=20,
        scheduler="plateau",
        lr=2.0e-3,
        lr_min=1.0e-5,
    )
    trainer.load_checkpoint(str(checkpoint_path), resume=True)

    assert isinstance(
        trainer.scheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)
    assert trainer.scheduler.state_dict()["min_lrs"] == pytest.approx([1.0e-5])

    event_log = (trainer.log_dir / "train_events.jsonl").read_text(encoding="utf-8")
    assert '"configured": "plateau"' in event_log
    assert '"checkpoint_type": "cosine"' in event_log
    assert '"reinitialized": true' in event_log


def test_resume_cosine_uses_current_yaml_lr_and_lr_min(
    tmp_path: Path,
) -> None:
    source = _make_minimal_trainer(
        tmp_path / "source",
        epochs=10,
        scheduler="cosine",
        lr=1.0e-3,
        lr_min=0.0,
    )
    _advance_scheduler(source, 10)
    checkpoint_path = tmp_path / "current_lr_min.pt"
    save_checkpoint(
        {
            "completed_epoch": 10,
            "global_step": 100,
            "model": source.model.state_dict(),
            "optimizer": source.optimizer.state_dict(),
            "scheduler": source.scheduler.state_dict(),
            "scaler": None,
            "config": source.config,
            "best_val_loss": 0.5,
        },
        str(checkpoint_path),
        overwrite=False,
    )

    trainer = _make_minimal_trainer(
        tmp_path / "target",
        epochs=20,
        scheduler="cosine",
        lr=2.0e-3,
        lr_min=1.0e-4,
    )
    trainer.load_checkpoint(str(checkpoint_path), resume=True)

    expected_lr = 1.0e-4 + (2.0e-3 - 1.0e-4) * (
        1.0 + math.cos(math.pi * 10 / 20)
    ) / 2.0
    assert trainer.scheduler.T_max == 20
    assert trainer.scheduler.eta_min == pytest.approx(1.0e-4)
    assert trainer.scheduler.base_lrs == pytest.approx([2.0e-3])
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_trainer_requires_completed_epoch(tmp_path: Path) -> None:
    source = _make_minimal_trainer(tmp_path / "source")
    checkpoint_path = tmp_path / "invalid.pt"
    save_checkpoint(
        {
            "model": source.model.state_dict(),
            "global_step": 0,
            "optimizer": source.optimizer.state_dict(),
            "scheduler": source.scheduler.state_dict(),
            "scaler": None,
            "config": {},
            "best_val_loss": 1.0,
        },
        str(checkpoint_path),
        overwrite=False,
    )

    trainer = _make_minimal_trainer(tmp_path / "target")
    with pytest.raises(KeyError, match="completed_epoch"):
        trainer.load_checkpoint(str(checkpoint_path), resume=True)
