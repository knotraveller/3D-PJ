"""Training loop for ZeroGS feed-forward Gaussian reconstruction."""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from datasets import ObjaverseRenderedDataset
from models import ZeroGSUNet
from renderers import GSplatRenderer
from training.losses import ReconstructionLoss
from utils.checkpoint import load_checkpoint as load_checkpoint_file
from utils.checkpoint import save_checkpoint as save_checkpoint_file
from utils.metrics import psnr
from utils.performance import PerformanceMonitor
from utils.ray_utils import get_embedding
from utils.visualization import (
    save_gaussian_stats,
    save_loss_curves,
    save_training_visualization,
)


LOSS_KEYS = ("loss", "rgb_loss", "mask_loss", "lpips_loss")
METRIC_KEYS = (*LOSS_KEYS, "psnr")


def summarize_metric_rows(
    rows: list[dict[str, float]],
    metric_keys: Iterable[str] = METRIC_KEYS,
) -> dict[str, dict[str, float]]:
    """Compute mean, maximum, and minimum for each scalar metric."""
    if not rows:
        raise ValueError("Cannot summarize an empty metric result.")
    return {
        key: {
            "mean": sum(row[key] for row in rows) / len(rows),
            "max": max(row[key] for row in rows),
            "min": min(row[key] for row in rows),
        }
        for key in metric_keys
    }


def validation_visual_filename(sample_id: str) -> str:
    """Return a filesystem-safe visualization filename for one sample."""
    safe_id = re.sub(r'[<>:"/\\|?*]+', "_", sample_id).strip(" .")
    if not safe_id:
        raise ValueError(f"Invalid validation sample id: {sample_id!r}")
    return f"{safe_id}.png"


class Trainer:
    """Owns model, renderer, data, optimization, logging, and validation."""

    def __init__(self, config: dict) -> None:
        self.config = config
        exp_cfg = config["experiment"]
        train_cfg = config["train"]
        data_cfg = config["data"]

        requested_device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.output_dir = Path(exp_cfg["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.visual_dir = self.output_dir / "visuals"
        self.validate_dir = self.output_dir / "validate"
        self.validate_visual_dir = self.validate_dir / "all_visuals"
        self.validate_epoch_visual_dir = self.validate_dir / "epoch_visuals"
        self.validate_log_dir = self.validate_dir / "logs"
        self.plot_dir = self.output_dir / "plots"
        self.stats_dir = self.output_dir / "stats"
        self.log_dir = self.output_dir / "logs"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        for path in (
            self.checkpoint_dir,
            self.visual_dir,
            self.validate_visual_dir,
            self.validate_epoch_visual_dir,
            self.validate_log_dir,
            self.plot_dir,
            self.stats_dir,
            self.log_dir,
            self.tensorboard_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "tensorboard is required for training logs. Install it with `pip install tensorboard`."
            ) from exc
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        self.train_dataset = ObjaverseRenderedDataset(
            root_dir=data_cfg["train_root"],
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get("max_train_samples"),
        )
        self.val_dataset = ObjaverseRenderedDataset(
            root_dir=data_cfg["val_root"],
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get("max_val_samples"),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=data_cfg.get("num_workers", 0),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 0),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        self.model = ZeroGSUNet(**config["model"]).to(self.device)
        self.renderer = GSplatRenderer(**config["renderer"]).to(self.device)
        self.criterion = ReconstructionLoss(**config["loss"]).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = self._build_scheduler()

        self.amp_enabled = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = math.inf
        perf_cfg = config.get("performance", {})
        self.profiler = PerformanceMonitor(
            enabled=bool(perf_cfg.get("enabled", False)),
            device=self.device,
            ema_momentum=float(perf_cfg.get("ema_momentum", 0.8)),
            sync_cuda=bool(perf_cfg.get("sync_cuda", True)),
            sample_system=bool(perf_cfg.get("sample_system", True)),
            sample_gpu_utilization=bool(perf_cfg.get("sample_gpu_utilization", True)),
            profile_gpu_modules=bool(perf_cfg.get("profile_gpu_modules", True)),
        )
        self.performance_write_every = int(perf_cfg.get("write_every", 10))
        self.performance_system_every = int(
            perf_cfg.get("system_sample_every", self.performance_write_every)
        )

    @staticmethod
    def _report_saved(label: str, path: Path) -> None:
        tqdm.write(f"{label} saved to {path.resolve()}")

    def _log_event(self, event: str, **details: object) -> Path:
        event_path = self.log_dir / "train_events.jsonl"
        row = {
            "event": event,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            **details,
        }
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return event_path

    @staticmethod
    def _normalize_scheduler_name(value: object) -> str:
        if value is None or value is False:
            return "none"
        name = str(value).strip().lower()
        if name in {"", "none", "off", "false", "null", "disabled"}:
            return "none"
        if name in {"cosine", "cosineannealing", "cosineannealinglr", "cosine_annealing"}:
            return "cosine"
        if name in {
            "plateau",
            "reduce_lr_on_plateau",
            "reduce_on_plateau",
            "reducelronplateau",
        }:
            return "plateau"
        raise ValueError(
            "train.scheduler must be one of: cosine, plateau, or none; "
            f"got {value!r}."
        )

    def _configured_scheduler_name(self) -> str:
        return self._normalize_scheduler_name(
            self.config["train"].get("scheduler", "cosine")
        )

    @staticmethod
    def _lr_bounds(train_cfg: dict, fallback_lr: float | None = None) -> tuple[float, float]:
        if "lr" in train_cfg:
            lr = float(train_cfg["lr"])
        elif fallback_lr is not None:
            lr = float(fallback_lr)
        else:
            raise KeyError("train.lr is required when building the optimizer.")
        lr_min = float(train_cfg.get("lr_min", 0.0))
        if lr < 0.0:
            raise ValueError("train.lr must be >= 0.")
        if lr_min < 0.0:
            raise ValueError("train.lr_min must be >= 0.")
        if lr_min > lr:
            raise ValueError("train.lr_min must be <= train.lr.")
        return lr, lr_min

    def _configured_lr_bounds(self) -> tuple[float, float]:
        fallback_lr = float(self.optimizer.param_groups[0]["lr"])
        return self._lr_bounds(self.config["train"], fallback_lr=fallback_lr)

    def _base_lrs_from_config(self) -> list[float]:
        lr, _ = self._configured_lr_bounds()
        return [lr for _ in self.optimizer.param_groups]

    def _build_scheduler(self):
        train_cfg = self.config["train"]
        scheduler_name = self._configured_scheduler_name()
        if scheduler_name == "none":
            return None

        _, lr_min = self._configured_lr_bounds()
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(train_cfg["epochs"])),
                eta_min=lr_min,
            )
        if scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(train_cfg.get("plateau_factor", 0.5)),
                patience=int(train_cfg.get("plateau_patience", 10)),
                threshold=float(train_cfg.get("plateau_threshold", 1.0e-4)),
                cooldown=int(train_cfg.get("plateau_cooldown", 0)),
                min_lr=lr_min,
                eps=float(train_cfg.get("plateau_eps", 1.0e-8)),
            )
        raise AssertionError(f"Unhandled scheduler: {scheduler_name}")

    @staticmethod
    def _checkpoint_scheduler_name(
        scheduler_state: dict | None,
        scheduler_type: object | None = None,
        checkpoint_config: object | None = None,
    ) -> str:
        if scheduler_type is not None:
            return Trainer._normalize_scheduler_name(scheduler_type)
        if isinstance(scheduler_state, dict):
            if "T_max" in scheduler_state:
                return "cosine"
            if "num_bad_epochs" in scheduler_state and "patience" in scheduler_state:
                return "plateau"
        if isinstance(checkpoint_config, dict):
            train_cfg = checkpoint_config.get("train", {})
            if isinstance(train_cfg, dict) and "scheduler" in train_cfg:
                return Trainer._normalize_scheduler_name(train_cfg["scheduler"])
        return "none"

    def _set_optimizer_learning_rates(
        self,
        learning_rates: list[float],
        base_lrs: list[float] | None = None,
    ) -> None:
        if len(learning_rates) == 1 and len(self.optimizer.param_groups) > 1:
            learning_rates = learning_rates * len(self.optimizer.param_groups)
        if len(learning_rates) != len(self.optimizer.param_groups):
            raise ValueError(
                "Learning-rate count does not match optimizer param groups: "
                f"{len(learning_rates)} vs {len(self.optimizer.param_groups)}."
            )
        if base_lrs is None:
            base_lrs = learning_rates
        if len(base_lrs) == 1 and len(self.optimizer.param_groups) > 1:
            base_lrs = base_lrs * len(self.optimizer.param_groups)
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Base-learning-rate count does not match optimizer param groups: "
                f"{len(base_lrs)} vs {len(self.optimizer.param_groups)}."
            )
        for param_group, learning_rate, base_lr in zip(
            self.optimizer.param_groups,
            learning_rates,
            base_lrs,
        ):
            param_group["lr"] = float(learning_rate)
            param_group["initial_lr"] = float(base_lr)

    @staticmethod
    def _cosine_learning_rates(
        completed_epoch: int,
        target_t_max: int,
        base_lrs: list[float],
        lr_min: float,
    ) -> list[float]:
        t_max = max(1, int(target_t_max))
        return [
            lr_min
            + (base_lr - lr_min)
            * (1.0 + math.cos(math.pi * completed_epoch / t_max))
            / 2.0
            for base_lr in base_lrs
        ]

    def _restore_scheduler(
        self,
        scheduler_state: dict | None,
        completed_epoch: int,
        *,
        checkpoint_scheduler_type: object | None = None,
        checkpoint_config: object | None = None,
    ) -> dict[str, object]:
        current_name = self._configured_scheduler_name()
        checkpoint_name = self._checkpoint_scheduler_name(
            scheduler_state,
            scheduler_type=checkpoint_scheduler_type,
            checkpoint_config=checkpoint_config,
        )
        base_lrs = self._base_lrs_from_config()
        _, lr_min = self._configured_lr_bounds()

        if self.scheduler is None:
            self._set_optimizer_learning_rates(base_lrs, base_lrs=base_lrs)
            return {
                "enabled": False,
                "configured": current_name,
                "checkpoint_type": checkpoint_name,
                "learning_rates": base_lrs,
            }

        if not isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        ):
            state = self.scheduler.state_dict()
            reinitialized = checkpoint_name != current_name or not isinstance(
                scheduler_state,
                dict,
            )
            if not reinitialized:
                for key in ("best", "cooldown_counter", "num_bad_epochs", "last_epoch"):
                    if key in scheduler_state:
                        state[key] = scheduler_state[key]
                restored_lrs = [
                    float(param_group["lr"])
                    for param_group in self.optimizer.param_groups
                ]
                learning_rates = [
                    min(max(learning_rate, lr_min), base_lr)
                    for learning_rate, base_lr in zip(restored_lrs, base_lrs)
                ]
            else:
                state["last_epoch"] = completed_epoch
                learning_rates = base_lrs
            state["_last_lr"] = learning_rates
            self.scheduler.load_state_dict(state)
            self._set_optimizer_learning_rates(learning_rates, base_lrs=base_lrs)
            if reinitialized:
                tqdm.write(
                    "Scheduler rebuilt from current config as "
                    f"{type(self.scheduler).__name__}; checkpoint scheduler "
                    f"{checkpoint_name!r} was ignored."
                )
            return {
                "enabled": True,
                "type": type(self.scheduler).__name__,
                "configured": current_name,
                "checkpoint_type": checkpoint_name,
                "reinitialized": reinitialized,
                "lr_min": lr_min,
                "learning_rates": learning_rates,
            }

        restored_state = dict(scheduler_state) if isinstance(scheduler_state, dict) else {}
        previous_t_max = restored_state.get("T_max")
        previous_lr_min = restored_state.get("eta_min")
        previous_base_lrs = restored_state.get("base_lrs")
        target_t_max = max(1, int(self.config["train"]["epochs"]))
        learning_rates = self._cosine_learning_rates(
            completed_epoch,
            target_t_max,
            base_lrs,
            lr_min,
        )
        scheduler_state_current = self.scheduler.state_dict()
        scheduler_state_current["T_max"] = target_t_max
        scheduler_state_current["eta_min"] = lr_min
        scheduler_state_current["base_lrs"] = base_lrs
        scheduler_state_current["last_epoch"] = completed_epoch
        scheduler_state_current["_step_count"] = completed_epoch + 1
        scheduler_state_current["_last_lr"] = learning_rates
        self.scheduler.load_state_dict(scheduler_state_current)
        self._set_optimizer_learning_rates(learning_rates, base_lrs=base_lrs)

        reinitialized = checkpoint_name != current_name or not isinstance(scheduler_state, dict)
        rescaled = (
            reinitialized
            or previous_t_max != target_t_max
            or previous_lr_min != lr_min
            or previous_base_lrs != base_lrs
        )
        if rescaled:
            tqdm.write(
                "Cosine scheduler restored from current config: "
                f"checkpoint scheduler={checkpoint_name!r}, "
                f"T_max={target_t_max}, lr_min={lr_min}, "
                f"epoch={completed_epoch}, lr={learning_rates}."
            )
        return {
            "enabled": True,
            "type": type(self.scheduler).__name__,
            "configured": current_name,
            "checkpoint_type": checkpoint_name,
            "reinitialized": reinitialized,
            "rescaled": rescaled,
            "previous_t_max": previous_t_max,
            "target_t_max": target_t_max,
            "lr_min": lr_min,
            "learning_rates": learning_rates,
        }

    def _batch_to_device(self, batch: Dict[str, torch.Tensor | str]) -> Dict[str, torch.Tensor | str]:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _forward_batch(
        self,
        batch: Dict[str, torch.Tensor | str],
        stage_prefix: str,
    ) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        alphas = batch["alphas"]
        K = batch["K"]
        c2w = batch["c2w"]
        w2c = batch["w2c"]
        assert torch.is_tensor(images)
        assert torch.is_tensor(alphas)
        assert torch.is_tensor(K)
        assert torch.is_tensor(c2w)
        assert torch.is_tensor(w2c)

        # [B,V,3,H,W] + [B,V,6,H,W] -> [B,V,9,H,W].
        with self.profiler.track(f"{stage_prefix}/ray_embedding", profile_gpu=True):
            ray_emb = get_embedding(
                K=K,
                c2w=c2w,
                resolution=images.shape[-1],
                embedding_type="plucker",
                order="dm",
                channel_first=True,
            )
            model_input = torch.cat([images, ray_emb], dim=2)
        with self.profiler.track(f"{stage_prefix}/model_forward", profile_gpu=True):
            model_out = self.model(model_input, K=K, c2w=c2w)
        with self.profiler.track(f"{stage_prefix}/render", profile_gpu=True):
            render_out = self.renderer(model_out["gaussians"], K=K, w2c=w2c)
        with self.profiler.track(f"{stage_prefix}/loss", profile_gpu=True):
            loss_dict = self.criterion(
                pred_rgb=render_out["rgb"],
                pred_alpha=render_out["alpha"],
                gt_rgb=images,
                gt_alpha=alphas,
            )
        with self.profiler.track(f"{stage_prefix}/metrics", profile_gpu=True):
            psnr_value = psnr(render_out["rgb"], images, mask=alphas)
        return {
            "images": images,
            "alphas": alphas,
            "gaussians": model_out["gaussians"],
            "pred_rgb": render_out["rgb"],
            "pred_alpha": render_out["alpha"],
            "psnr": psnr_value,
            **loss_dict,
        }

    @staticmethod
    def _metric_row(outputs: Dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            key: float(outputs[key].detach().cpu())
            for key in METRIC_KEYS
        }

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    @staticmethod
    def _gradient_accumulation_steps(train_cfg: dict) -> int:
        steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        if steps < 1:
            raise ValueError("train.gradient_accumulation_steps must be >= 1.")
        return steps

    @staticmethod
    def _accumulation_window_size(
        batch_index: int,
        total_batches: int,
        accumulation_steps: int,
    ) -> int:
        window_start = (batch_index // accumulation_steps) * accumulation_steps
        return min(accumulation_steps, total_batches - window_start)

    def _log_tensorboard_metrics(
        self,
        phase: str,
        summary: dict[str, dict[str, float]],
    ) -> None:
        for key, stats in summary.items():
            self.writer.add_scalar(f"{phase}/{key}", stats["mean"], self.global_step)
            self.writer.add_scalar(f"{phase}/{key}_mean", stats["mean"], self.global_step)
            self.writer.add_scalar(f"{phase}/{key}_max", stats["max"], self.global_step)
            self.writer.add_scalar(f"{phase}/{key}_min", stats["min"], self.global_step)

    def _log_train_epoch(
        self,
        epoch: int,
        metric_rows: list[dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        summary = summarize_metric_rows(metric_rows)
        lr = float(self.optimizer.param_groups[0]["lr"])
        row: dict[str, object] = {
            "epoch": int(epoch),
            "num_batches": len(metric_rows),
            "lr": lr,
            **summary,
        }
        log_path = self.log_dir / "train_log.jsonl"
        self._append_jsonl(log_path, row)
        self._report_saved("Training metrics log", log_path)
        self._log_tensorboard_metrics("train", summary)
        self.writer.add_scalar("lr", lr, self.global_step)
        return summary

    def _save_visuals(self, outputs: Dict[str, torch.Tensor], stem: str) -> None:
        save_path = self.visual_dir / f"{stem}.png"
        save_training_visualization(
            pred_rgb=outputs["pred_rgb"],
            pred_alpha=outputs["pred_alpha"],
            gt_rgb=outputs["images"],
            gt_alpha=outputs["alphas"],
            save_path=str(save_path),
        )
        self._report_saved("Training visualization", save_path)
        self.writer.add_image(
            "train/visual",
            torch.as_tensor(plt_image_to_chw(save_path)),
            self.global_step,
        )
        stats_path = self.stats_dir / f"gaussian_stats_{stem}.json"
        save_gaussian_stats(
            outputs["gaussians"],
            str(stats_path),
        )
        self._report_saved("Gaussian statistics", stats_path)

    def _save_training_curves(self) -> None:
        plot_path = self.plot_dir / "loss_curve.png"
        save_loss_curves(
            str(self.log_dir / "train_log.jsonl"),
            str(plot_path),
        )
        if plot_path.is_file():
            self._report_saved("Loss curve", plot_path)

    def _maybe_log_performance(self, epoch: int, phase: str, force: bool = False) -> None:
        if not self.profiler.enabled or self.performance_write_every <= 0:
            return

        sample_every = max(1, self.performance_system_every)
        should_sample = force or self.global_step % sample_every == 0
        if should_sample:
            self.profiler.sample_system()

        should_write = force or self.global_step % self.performance_write_every == 0
        if not should_write:
            return

        snapshot = self.profiler.snapshot(
            step=self.global_step,
            epoch=epoch,
            phase=phase,
        )
        snapshot["gpu_static"] = self._gpu_static_snapshot()
        saved_paths = self.profiler.write_snapshot(self.stats_dir, snapshot)
        if saved_paths is not None:
            latest_path, log_path = saved_paths
            tqdm.write(
                "Performance statistics saved to "
                f"{latest_path.resolve()} and appended to {log_path.resolve()}"
            )
        self.profiler.log_tensorboard(self.writer, self.global_step)

    def _gpu_static_snapshot(self) -> dict[str, object]:
        modules = {
            "model": self.model,
            "renderer": self.renderer,
            "criterion": self.criterion,
        }
        return {
            "device": str(self.device),
            "unit": "MB",
            "modules": {
                name: self._module_gpu_memory(module)
                for name, module in modules.items()
            },
            "optimizer_state": self._optimizer_gpu_memory(),
        }

    @staticmethod
    def _tensor_gpu_memory(tensors: list[torch.Tensor]) -> dict[str, float | int]:
        gpu_tensors = [tensor for tensor in tensors if tensor.is_cuda]
        bytes_total = sum(tensor.numel() * tensor.element_size() for tensor in gpu_tensors)
        return {
            "gpu_mb": bytes_total / (1024.0 * 1024.0),
            "gpu_tensors": len(gpu_tensors),
            "tensors": len(tensors),
        }

    def _module_gpu_memory(self, module: torch.nn.Module) -> dict[str, float | int]:
        parameters = list(module.parameters(recurse=True))
        buffers = list(module.buffers(recurse=True))
        parameter_memory = self._tensor_gpu_memory(parameters)
        buffer_memory = self._tensor_gpu_memory(buffers)
        trainable_parameters = sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        )
        total_mb = float(parameter_memory["gpu_mb"]) + float(buffer_memory["gpu_mb"])
        return {
            "parameters_mb": parameter_memory["gpu_mb"],
            "buffers_mb": buffer_memory["gpu_mb"],
            "total_mb": total_mb,
            "parameter_tensors": parameter_memory["tensors"],
            "gpu_parameter_tensors": parameter_memory["gpu_tensors"],
            "buffer_tensors": buffer_memory["tensors"],
            "gpu_buffer_tensors": buffer_memory["gpu_tensors"],
            "trainable_parameters": trainable_parameters,
        }

    def _optimizer_gpu_memory(self) -> dict[str, float | int]:
        tensors: list[torch.Tensor] = []
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    tensors.append(value)
        memory = self._tensor_gpu_memory(tensors)
        return {
            "state_mb": memory["gpu_mb"],
            "state_tensors": memory["tensors"],
            "gpu_state_tensors": memory["gpu_tensors"],
        }

    def _check_finite(self, outputs: Dict[str, torch.Tensor]) -> None:
        loss = outputs["loss"]
        if torch.isfinite(loss):
            return
        diagnostics_path = self.stats_dir / f"nonfinite_step_{self.global_step:06d}.json"
        save_gaussian_stats(
            outputs["gaussians"],
            str(diagnostics_path),
        )
        self._report_saved("Non-finite Gaussian diagnostics", diagnostics_path)
        diagnostics = {
            "loss": float(loss.detach().cpu()),
            "rgb_loss": float(outputs["rgb_loss"].detach().cpu()),
            "mask_loss": float(outputs["mask_loss"].detach().cpu()),
            "lpips_loss": float(outputs["lpips_loss"].detach().cpu()),
            "gaussians_min": float(outputs["gaussians"].min().detach().cpu()),
            "gaussians_max": float(outputs["gaussians"].max().detach().cpu()),
            "opacity_min": float(outputs["gaussians"][..., 3].min().detach().cpu()),
            "opacity_max": float(outputs["gaussians"][..., 3].max().detach().cpu()),
            "scale_min": float(outputs["gaussians"][..., 4:7].min().detach().cpu()),
            "scale_max": float(outputs["gaussians"][..., 4:7].max().detach().cpu()),
            "pred_rgb_min": float(outputs["pred_rgb"].min().detach().cpu()),
            "pred_rgb_max": float(outputs["pred_rgb"].max().detach().cpu()),
            "pred_alpha_min": float(outputs["pred_alpha"].min().detach().cpu()),
            "pred_alpha_max": float(outputs["pred_alpha"].max().detach().cpu()),
        }
        raise RuntimeError(f"Non-finite loss detected: {diagnostics}")

    def train(self) -> None:
        train_cfg = self.config["train"]
        if train_cfg.get("overfit_one_batch", False):
            self._overfit_one_batch()
            self.writer.close()
            return

        total_epochs = int(train_cfg["epochs"])
        if self.start_epoch >= total_epochs:
            tqdm.write(
                f"Checkpoint already completed {self.start_epoch} epoch(s); "
                f"configured total is {total_epochs}. Nothing to train."
            )
            self.writer.close()
            return

        try:
            for epoch_index in range(self.start_epoch, total_epochs):
                epoch = epoch_index + 1
                save_every = int(train_cfg.get("save_every", 1))
                numbered_checkpoint = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
                if save_every > 0 and epoch % save_every == 0 and numbered_checkpoint.exists():
                    raise FileExistsError(
                        "Refusing to overwrite existing checkpoint before training epoch "
                        f"{epoch}: {numbered_checkpoint}"
                    )

                self.train_one_epoch(epoch, total_epochs)
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                ):
                    self.scheduler.step()
                val_every = int(train_cfg.get("val_every", 1))
                val_loss = None
                if val_every > 0 and epoch % val_every == 0:
                    val_loss = self.validate(epoch)
                    if isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        self.scheduler.step(val_loss)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(
                            "best.pt",
                            completed_epoch=epoch,
                            overwrite=True,
                        )
                if save_every > 0 and epoch % save_every == 0:
                    self.save_checkpoint(
                        f"epoch_{epoch:04d}.pt",
                        completed_epoch=epoch,
                        overwrite=False,
                    )
                    self.save_checkpoint(
                        "latest.pt",
                        completed_epoch=epoch,
                        overwrite=True,
                    )
        finally:
            self.writer.close()

    def train_one_epoch(self, epoch: int, total_epochs: int) -> None:
        self.model.train()
        train_cfg = self.config["train"]
        progress = tqdm(
            total=len(self.train_loader),
            desc=f"train epoch {epoch}/{total_epochs}",
            ascii=True,
        )
        data_iter = iter(self.train_loader)
        last_outputs = None
        metric_rows: list[dict[str, float]] = []
        total_batches = len(self.train_loader)
        accumulation_steps = self._gradient_accumulation_steps(train_cfg)
        for batch_index in range(total_batches):
            with self.profiler.track("train/iteration_total"):
                with self.profiler.track("train/data_load"):
                    batch = next(data_iter)
                with self.profiler.track("train/to_device", profile_gpu=True):
                    batch = self._batch_to_device(batch)
                if batch_index % accumulation_steps == 0:
                    with self.profiler.track("train/zero_grad", profile_gpu=True):
                        self.optimizer.zero_grad(set_to_none=True)
                with self.profiler.track("train/forward_total"):
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        outputs = self._forward_batch(batch, stage_prefix="train")
                        loss = outputs["loss"]
                with self.profiler.track("train/check_finite"):
                    self._check_finite(outputs)

                window_size = self._accumulation_window_size(
                    batch_index,
                    total_batches,
                    accumulation_steps,
                )
                backward_loss = loss / float(window_size)
                with self.profiler.track("train/backward", profile_gpu=True):
                    self.scaler.scale(backward_loss).backward()
                should_update = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == total_batches
                )
                if should_update:
                    with self.profiler.track("train/grad_unscale", profile_gpu=True):
                        self.scaler.unscale_(self.optimizer)
                    grad_clip = train_cfg.get("grad_clip")
                    if grad_clip:
                        with self.profiler.track("train/grad_clip", profile_gpu=True):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                float(grad_clip),
                            )
                    with self.profiler.track("train/optimizer_step", profile_gpu=True):
                        self.scaler.step(self.optimizer)
                    with self.profiler.track("train/scaler_update", profile_gpu=True):
                        self.scaler.update()
                with self.profiler.track("train/collect_metrics"):
                    metric_rows.append(self._metric_row(outputs))

            self._maybe_log_performance(epoch, phase="train")
            progress.set_postfix(loss=float(loss.detach().cpu()))
            progress.update(1)
            self.global_step += 1
            last_outputs = outputs
        progress.close()
        with self.profiler.track("train/epoch_log"):
            self._log_train_epoch(epoch, metric_rows)
        if bool(train_cfg["epoch_visuals"]) and last_outputs is not None:
            with self.profiler.track("train/epoch_visuals"):
                self._save_visuals(last_outputs, stem=f"epoch_{epoch:04d}")
        with self.profiler.track("train/epoch_curves"):
            self._save_training_curves()

    def _overfit_one_batch(self) -> None:
        self.model.train()
        train_cfg = self.config["train"]
        with self.profiler.track("overfit/data_load"):
            batch = next(iter(self.train_loader))
        with self.profiler.track("overfit/to_device", profile_gpu=True):
            batch = self._batch_to_device(batch)
        steps = int(train_cfg.get("overfit_steps", 300))
        last_outputs = None
        metric_rows: list[dict[str, float]] = []
        accumulation_steps = self._gradient_accumulation_steps(train_cfg)
        for step in tqdm(range(steps), desc="overfit one batch", ascii=True):
            with self.profiler.track("overfit/iteration_total"):
                if step % accumulation_steps == 0:
                    with self.profiler.track("overfit/zero_grad", profile_gpu=True):
                        self.optimizer.zero_grad(set_to_none=True)
                with self.profiler.track("overfit/forward_total"):
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        outputs = self._forward_batch(batch, stage_prefix="overfit")
                        loss = outputs["loss"]
                with self.profiler.track("overfit/check_finite"):
                    self._check_finite(outputs)
                window_size = self._accumulation_window_size(
                    step,
                    steps,
                    accumulation_steps,
                )
                backward_loss = loss / float(window_size)
                with self.profiler.track("overfit/backward", profile_gpu=True):
                    self.scaler.scale(backward_loss).backward()
                should_update = (
                    (step + 1) % accumulation_steps == 0
                    or step + 1 == steps
                )
                if should_update:
                    with self.profiler.track("overfit/grad_unscale", profile_gpu=True):
                        self.scaler.unscale_(self.optimizer)
                    grad_clip = train_cfg.get("grad_clip")
                    if grad_clip:
                        with self.profiler.track("overfit/grad_clip", profile_gpu=True):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                float(grad_clip),
                            )
                    with self.profiler.track("overfit/optimizer_step", profile_gpu=True):
                        self.scaler.step(self.optimizer)
                    with self.profiler.track("overfit/scaler_update", profile_gpu=True):
                        self.scaler.update()
                with self.profiler.track("overfit/collect_metrics"):
                    metric_rows.append(self._metric_row(outputs))
            self._maybe_log_performance(epoch=0, phase="overfit")
            self.global_step += 1
            last_outputs = outputs
        with self.profiler.track("overfit/log"):
            self._log_train_epoch(0, metric_rows)
        if last_outputs is not None:
            with self.profiler.track("overfit/visuals"):
                self._save_visuals(last_outputs, stem="epoch_0000")
        with self.profiler.track("overfit/curves"):
            self._save_training_curves()
        self.save_checkpoint("latest.pt", completed_epoch=0, overwrite=True)

    @torch.no_grad()
    def validate(self, epoch: int, *, save_outputs: bool = False) -> float:
        self.model.eval()
        num_samples = 0
        metric_rows: list[dict[str, float]] = []
        progress = tqdm(total=len(self.val_loader), desc=f"val epoch {epoch}", ascii=True)
        data_iter = iter(self.val_loader)
        for batch_idx in range(len(self.val_loader)):
            with self.profiler.track("val/iteration_total"):
                with self.profiler.track("val/data_load"):
                    batch = next(data_iter)
                with self.profiler.track("val/to_device", profile_gpu=True):
                    batch = self._batch_to_device(batch)
                with self.profiler.track("val/forward_total"):
                    outputs = self._forward_batch(batch, stage_prefix="val")
                with self.profiler.track("val/reduce_metrics", profile_gpu=True):
                    batch_size = int(outputs["images"].shape[0])
                    sample_ids = batch["sample_id"]
                    if not isinstance(sample_ids, list):
                        raise TypeError("Validation batch sample_id must be a list.")

                    for sample_idx in range(batch_size):
                        if batch_size == 1:
                            sample_losses = outputs
                        else:
                            sample_losses = self.criterion(
                                pred_rgb=outputs["pred_rgb"][sample_idx : sample_idx + 1],
                                pred_alpha=outputs["pred_alpha"][sample_idx : sample_idx + 1],
                                gt_rgb=outputs["images"][sample_idx : sample_idx + 1],
                                gt_alpha=outputs["alphas"][sample_idx : sample_idx + 1],
                            )
                        sample_psnr = psnr(
                            outputs["pred_rgb"][sample_idx : sample_idx + 1],
                            outputs["images"][sample_idx : sample_idx + 1],
                            mask=outputs["alphas"][sample_idx : sample_idx + 1],
                        )
                        metric_rows.append(
                            {
                                **{
                                    key: float(sample_losses[key].detach().cpu())
                                    for key in LOSS_KEYS
                                },
                                "psnr": float(sample_psnr.detach().cpu()),
                            }
                        )
                        num_samples += 1

                        if save_outputs:
                            visual_path = (
                                self.validate_visual_dir
                                / validation_visual_filename(sample_ids[sample_idx])
                            )
                            save_training_visualization(
                                pred_rgb=outputs["pred_rgb"][sample_idx : sample_idx + 1],
                                pred_alpha=outputs["pred_alpha"][sample_idx : sample_idx + 1],
                                gt_rgb=outputs["images"][sample_idx : sample_idx + 1],
                                gt_alpha=outputs["alphas"][sample_idx : sample_idx + 1],
                                save_path=str(visual_path),
                            )
                            self._report_saved("Validation visualization", visual_path)

                if batch_idx == 0 and not save_outputs:
                    with self.profiler.track("val/visuals"):
                        visual_path = (
                            self.validate_epoch_visual_dir / f"epoch_{epoch:04d}.png"
                        )
                        save_training_visualization(
                            pred_rgb=outputs["pred_rgb"],
                            pred_alpha=outputs["pred_alpha"],
                            gt_rgb=outputs["images"],
                            gt_alpha=outputs["alphas"],
                            save_path=str(visual_path),
                        )
                        self._report_saved("Validation visualization", visual_path)
            progress.update(1)
        progress.close()
        metric_summary = summarize_metric_rows(metric_rows)
        mean_loss = metric_summary["loss"]["mean"]
        validate_log_path = self.validate_log_dir / "validate_log.jsonl"
        self._append_jsonl(
            validate_log_path,
            {
                "epoch": int(epoch),
                "num_samples": num_samples,
                **metric_summary,
            },
        )
        self._report_saved("Validation metrics log", validate_log_path)
        if save_outputs:
            loss_path = self.validate_dir / "loss.yaml"
            payload = {
                "epoch": int(epoch),
                "num_samples": num_samples,
                **metric_summary,
            }
            loss_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            self._report_saved("Validation loss summary", loss_path)
        with self.profiler.track("val/log"):
            self._log_tensorboard_metrics("val", metric_summary)
        self._maybe_log_performance(epoch, phase="val", force=True)
        return mean_loss

    def save_checkpoint(
        self,
        name: str,
        completed_epoch: int,
        *,
        overwrite: bool,
    ) -> Path:
        payload = {
            "completed_epoch": completed_epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scheduler_type": self._configured_scheduler_name(),
            "scaler": self.scaler.state_dict() if self.amp_enabled else None,
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }
        checkpoint_path = save_checkpoint_file(
            payload,
            str(self.checkpoint_dir / name),
            overwrite=overwrite,
        )
        self._report_saved("Checkpoint", checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str, *, resume: bool) -> int:
        checkpoint = load_checkpoint_file(path, map_location=self.device)
        required_fields = {
            "completed_epoch",
            "global_step",
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "config",
            "best_val_loss",
        }
        missing_fields = sorted(required_fields.difference(checkpoint))
        if missing_fields:
            raise KeyError(
                "Checkpoint is missing required fields: "
                + ", ".join(missing_fields)
            )

        self.model.load_state_dict(checkpoint["model"])
        completed_epoch = int(checkpoint["completed_epoch"])
        checkpoint_path = Path(path).resolve()

        if not resume:
            self.start_epoch = 0
            self.global_step = 0
            self.best_val_loss = math.inf
            tqdm.write(
                f"Model weights loaded from {checkpoint_path}; "
                "training state was not restored."
            )
            return completed_epoch

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.amp_enabled and checkpoint["scaler"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        scheduler_resume = self._restore_scheduler(
            checkpoint["scheduler"],
            completed_epoch,
            checkpoint_scheduler_type=checkpoint.get("scheduler_type"),
            checkpoint_config=checkpoint["config"],
        )

        self.start_epoch = completed_epoch
        self.global_step = int(checkpoint["global_step"])
        self.best_val_loss = float(checkpoint["best_val_loss"])
        event_path = self._log_event(
            "resume",
            checkpoint=str(checkpoint_path),
            completed_epoch=completed_epoch,
            next_epoch=completed_epoch + 1,
            global_step=self.global_step,
            scheduler=scheduler_resume,
        )
        tqdm.write(
            f"Training resumed from {checkpoint_path}: completed epoch "
            f"{completed_epoch}, next epoch {completed_epoch + 1}, "
            f"global step {self.global_step}."
        )
        self._report_saved("Resume event log", event_path)
        return completed_epoch


def plt_image_to_chw(path: Path) -> torch.Tensor:
    """Load saved visualization PNG as [3,H,W] tensor for TensorBoard."""
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
