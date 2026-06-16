"""Training loop for ZeroGS feed-forward Gaussian reconstruction."""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

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


VALIDATION_LOSS_KEYS = ("loss", "rgb_loss", "mask_loss", "lpips_loss")


def summarize_validation_losses(
    rows: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute mean, maximum, and minimum for each validation loss."""
    if not rows:
        raise ValueError("Cannot summarize an empty validation result.")
    return {
        key: {
            "mean": sum(row[key] for row in rows) / len(rows),
            "max": max(row[key] for row in rows),
            "min": min(row[key] for row in rows),
        }
        for key in VALIDATION_LOSS_KEYS
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
        self.plot_dir = self.output_dir / "plots"
        self.stats_dir = self.output_dir / "stats"
        self.log_dir = self.output_dir / "logs"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        for path in (
            self.checkpoint_dir,
            self.visual_dir,
            self.validate_visual_dir,
            self.validate_epoch_visual_dir,
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
        self.scheduler = None
        if train_cfg.get("scheduler", "cosine") == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(train_cfg["epochs"])),
            )

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
        )
        self.performance_write_every = int(
            perf_cfg.get("write_every", train_cfg.get("log_every", 10))
        )
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

    def _restore_scheduler(
        self,
        scheduler_state: dict | None,
        completed_epoch: int,
    ) -> dict[str, object]:
        if self.scheduler is None:
            return {"enabled": False}
        if scheduler_state is None:
            raise ValueError(
                "Checkpoint has no scheduler state, but the current config enables one."
            )

        if not isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        ):
            self.scheduler.load_state_dict(scheduler_state)
            return {
                "enabled": True,
                "type": type(self.scheduler).__name__,
                "rescaled": False,
            }

        restored_state = dict(scheduler_state)
        previous_t_max = int(restored_state["T_max"])
        target_t_max = int(self.config["train"]["epochs"])
        if previous_t_max == target_t_max:
            self.scheduler.load_state_dict(restored_state)
            return {
                "enabled": True,
                "type": type(self.scheduler).__name__,
                "rescaled": False,
                "previous_t_max": previous_t_max,
                "target_t_max": target_t_max,
                "learning_rates": self.scheduler.get_last_lr(),
            }

        base_lrs = [float(lr) for lr in restored_state["base_lrs"]]
        eta_min = float(restored_state["eta_min"])
        learning_rates = [
            eta_min
            + (base_lr - eta_min)
            * (1.0 + math.cos(math.pi * completed_epoch / target_t_max))
            / 2.0
            for base_lr in base_lrs
        ]
        restored_state["T_max"] = target_t_max
        restored_state["last_epoch"] = completed_epoch
        restored_state["_step_count"] = completed_epoch + 1
        restored_state["_last_lr"] = learning_rates
        self.scheduler.load_state_dict(restored_state)
        for param_group, base_lr, learning_rate in zip(
            self.optimizer.param_groups,
            base_lrs,
            learning_rates,
        ):
            param_group["initial_lr"] = base_lr
            param_group["lr"] = learning_rate

        tqdm.write(
            "Cosine scheduler extended from "
            f"T_max={previous_t_max} to T_max={target_t_max}; "
            f"restored at epoch {completed_epoch} with lr={learning_rates}."
        )
        return {
            "enabled": True,
            "type": type(self.scheduler).__name__,
            "rescaled": True,
            "previous_t_max": previous_t_max,
            "target_t_max": target_t_max,
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
        with self.profiler.track(f"{stage_prefix}/ray_embedding"):
            ray_emb = get_embedding(
                K=K,
                c2w=c2w,
                resolution=images.shape[-1],
                embedding_type="plucker",
                order="dm",
                channel_first=True,
            )
            model_input = torch.cat([images, ray_emb], dim=2)
        with self.profiler.track(f"{stage_prefix}/model_forward"):
            model_out = self.model(model_input, K=K, c2w=c2w)
        with self.profiler.track(f"{stage_prefix}/render"):
            render_out = self.renderer(model_out["gaussians"], K=K, w2c=w2c)
        with self.profiler.track(f"{stage_prefix}/loss"):
            loss_dict = self.criterion(
                pred_rgb=render_out["rgb"],
                pred_alpha=render_out["alpha"],
                gt_rgb=images,
                gt_alpha=alphas,
            )
        with self.profiler.track(f"{stage_prefix}/metrics"):
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

    def _log_step(self, outputs: Dict[str, torch.Tensor], epoch: int) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        row = {
            "step": self.global_step,
            "epoch": epoch,
            "loss": float(outputs["loss"].detach().cpu()),
            "rgb_loss": float(outputs["rgb_loss"].detach().cpu()),
            "mask_loss": float(outputs["mask_loss"].detach().cpu()),
            "lpips_loss": float(outputs["lpips_loss"].detach().cpu()),
            "psnr": float(outputs["psnr"].detach().cpu()),
            "lr": float(lr),
        }
        log_path = self.log_dir / "train_log.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        self.writer.add_scalar("train/loss", row["loss"], self.global_step)
        self.writer.add_scalar("train/rgb_loss", row["rgb_loss"], self.global_step)
        self.writer.add_scalar("train/mask_loss", row["mask_loss"], self.global_step)
        self.writer.add_scalar("train/lpips_loss", row["lpips_loss"], self.global_step)
        self.writer.add_scalar("train/psnr", row["psnr"], self.global_step)
        self.writer.add_scalar("lr", row["lr"], self.global_step)

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
        saved_paths = self.profiler.write_snapshot(self.stats_dir, snapshot)
        if saved_paths is not None:
            latest_path, log_path = saved_paths
            tqdm.write(
                "Performance statistics saved to "
                f"{latest_path.resolve()} and appended to {log_path.resolve()}"
            )
        self.profiler.log_tensorboard(self.writer, self.global_step)

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
                if self.scheduler is not None:
                    self.scheduler.step()
                val_every = int(train_cfg.get("val_every", 1))
                if val_every > 0 and epoch % val_every == 0:
                    val_loss = self.validate(epoch)
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
        for _ in range(len(self.train_loader)):
            with self.profiler.track("train/iteration_total"):
                with self.profiler.track("train/data_load"):
                    batch = next(data_iter)
                with self.profiler.track("train/to_device"):
                    batch = self._batch_to_device(batch)
                with self.profiler.track("train/zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
                with self.profiler.track("train/forward_total"):
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        outputs = self._forward_batch(batch, stage_prefix="train")
                        loss = outputs["loss"]
                with self.profiler.track("train/check_finite"):
                    self._check_finite(outputs)

                with self.profiler.track("train/backward"):
                    self.scaler.scale(loss).backward()
                with self.profiler.track("train/grad_unscale"):
                    self.scaler.unscale_(self.optimizer)
                grad_clip = train_cfg.get("grad_clip")
                if grad_clip:
                    with self.profiler.track("train/grad_clip"):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                with self.profiler.track("train/optimizer_step"):
                    self.scaler.step(self.optimizer)
                with self.profiler.track("train/scaler_update"):
                    self.scaler.update()

                log_every = int(train_cfg.get("log_every", 10))
                if log_every > 0 and self.global_step % log_every == 0:
                    with self.profiler.track("train/log"):
                        self._log_step(outputs, epoch)
                vis_every = int(train_cfg.get("vis_every", 200))
                if vis_every > 0 and self.global_step % vis_every == 0:
                    with self.profiler.track("train/visuals"):
                        self._save_visuals(
                            outputs,
                            stem=f"step_{self.global_step:06d}",
                        )

            self._maybe_log_performance(epoch, phase="train")
            progress.set_postfix(loss=float(loss.detach().cpu()))
            progress.update(1)
            self.global_step += 1
            last_outputs = outputs
        progress.close()
        if bool(train_cfg["epoch_visuals"]) and last_outputs is not None:
            with self.profiler.track("train/epoch_visuals"):
                self._save_visuals(last_outputs, stem=f"epoch_{epoch:04d}")

    def _overfit_one_batch(self) -> None:
        self.model.train()
        train_cfg = self.config["train"]
        with self.profiler.track("overfit/data_load"):
            batch = next(iter(self.train_loader))
        with self.profiler.track("overfit/to_device"):
            batch = self._batch_to_device(batch)
        steps = int(train_cfg.get("overfit_steps", 300))
        for step in tqdm(range(steps), desc="overfit one batch", ascii=True):
            with self.profiler.track("overfit/iteration_total"):
                with self.profiler.track("overfit/zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
                with self.profiler.track("overfit/forward_total"):
                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        outputs = self._forward_batch(batch, stage_prefix="overfit")
                        loss = outputs["loss"]
                with self.profiler.track("overfit/check_finite"):
                    self._check_finite(outputs)
                with self.profiler.track("overfit/backward"):
                    self.scaler.scale(loss).backward()
                with self.profiler.track("overfit/grad_unscale"):
                    self.scaler.unscale_(self.optimizer)
                grad_clip = train_cfg.get("grad_clip")
                if grad_clip:
                    with self.profiler.track("overfit/grad_clip"):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                with self.profiler.track("overfit/optimizer_step"):
                    self.scaler.step(self.optimizer)
                with self.profiler.track("overfit/scaler_update"):
                    self.scaler.update()

                if step % int(train_cfg.get("log_every", 20)) == 0:
                    with self.profiler.track("overfit/log"):
                        self._log_step(outputs, epoch=0)
                if step % int(train_cfg.get("vis_every", 20)) == 0:
                    with self.profiler.track("overfit/visuals"):
                        self._save_visuals(
                            outputs,
                            stem=f"step_{self.global_step:06d}",
                        )
            self._maybe_log_performance(epoch=0, phase="overfit")
            self.global_step += 1
        self.save_checkpoint("latest.pt", completed_epoch=0, overwrite=True)

    @torch.no_grad()
    def validate(self, epoch: int, *, save_outputs: bool = False) -> float:
        self.model.eval()
        total_psnr = 0.0
        num_samples = 0
        loss_rows: list[dict[str, float]] = []
        progress = tqdm(total=len(self.val_loader), desc=f"val epoch {epoch}", ascii=True)
        data_iter = iter(self.val_loader)
        for batch_idx in range(len(self.val_loader)):
            with self.profiler.track("val/iteration_total"):
                with self.profiler.track("val/data_load"):
                    batch = next(data_iter)
                with self.profiler.track("val/to_device"):
                    batch = self._batch_to_device(batch)
                with self.profiler.track("val/forward_total"):
                    outputs = self._forward_batch(batch, stage_prefix="val")
                with self.profiler.track("val/reduce_metrics"):
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
                        loss_rows.append(
                            {
                                key: float(sample_losses[key].detach().cpu())
                                for key in VALIDATION_LOSS_KEYS
                            }
                        )
                        sample_psnr = psnr(
                            outputs["pred_rgb"][sample_idx : sample_idx + 1],
                            outputs["images"][sample_idx : sample_idx + 1],
                            mask=outputs["alphas"][sample_idx : sample_idx + 1],
                        )
                        total_psnr += float(sample_psnr.detach().cpu())
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
        loss_summary = summarize_validation_losses(loss_rows)
        mean_loss = loss_summary["loss"]["mean"]
        mean_psnr = total_psnr / max(1, num_samples)
        if save_outputs:
            loss_path = self.validate_dir / "loss.yaml"
            payload = {
                "epoch": int(epoch),
                "num_samples": num_samples,
                **loss_summary,
            }
            loss_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            self._report_saved("Validation loss summary", loss_path)
        with self.profiler.track("val/log"):
            self.writer.add_scalar("val/loss", mean_loss, epoch)
            self.writer.add_scalar("val/psnr", mean_psnr, epoch)
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
