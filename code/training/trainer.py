"""Training loop for ZeroGS feed-forward Gaussian reconstruction."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict

import torch
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
        self.plot_dir = self.output_dir / "plots"
        self.stats_dir = self.output_dir / "stats"
        self.log_dir = self.output_dir / "logs"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        for path in (
            self.checkpoint_dir,
            self.visual_dir,
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

    def _save_visuals(self, outputs: Dict[str, torch.Tensor]) -> None:
        save_path = self.visual_dir / f"step_{self.global_step:06d}.png"
        save_training_visualization(
            pred_rgb=outputs["pred_rgb"],
            pred_alpha=outputs["pred_alpha"],
            gt_rgb=outputs["images"],
            gt_alpha=outputs["alphas"],
            save_path=str(save_path),
        )
        self.writer.add_image(
            "train/visual",
            torch.as_tensor(plt_image_to_chw(save_path)),
            self.global_step,
        )
        save_gaussian_stats(
            outputs["gaussians"],
            str(self.stats_dir / f"gaussian_stats_step_{self.global_step:06d}.json"),
        )
        save_loss_curves(
            str(self.log_dir / "train_log.jsonl"),
            str(self.plot_dir / "loss_curve.png"),
        )

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
        self.profiler.write_snapshot(self.stats_dir, snapshot)
        self.profiler.log_tensorboard(self.writer, self.global_step)

    def _check_finite(self, outputs: Dict[str, torch.Tensor]) -> None:
        loss = outputs["loss"]
        if torch.isfinite(loss):
            return
        save_gaussian_stats(
            outputs["gaussians"],
            str(self.stats_dir / f"nonfinite_step_{self.global_step:06d}.json"),
        )
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
            return

        for epoch in range(int(train_cfg["epochs"])):
            self.train_one_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            if (epoch + 1) % int(train_cfg.get("val_every", 1)) == 0:
                val_loss = self.validate(epoch)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt", epoch=epoch)
            if (epoch + 1) % int(train_cfg.get("save_every", 1)) == 0:
                self.save_checkpoint(f"epoch_{epoch + 1:04d}.pt", epoch=epoch)
                self.save_checkpoint("latest.pt", epoch=epoch)
        self.writer.close()

    def train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        train_cfg = self.config["train"]
        progress = tqdm(total=len(self.train_loader), desc=f"train epoch {epoch}", ascii=True)
        data_iter = iter(self.train_loader)
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

                if self.global_step % int(train_cfg.get("log_every", 10)) == 0:
                    with self.profiler.track("train/log"):
                        self._log_step(outputs, epoch)
                if self.global_step % int(train_cfg.get("vis_every", 200)) == 0:
                    with self.profiler.track("train/visuals"):
                        self._save_visuals(outputs)

            self._maybe_log_performance(epoch, phase="train")
            progress.set_postfix(loss=float(loss.detach().cpu()))
            progress.update(1)
            self.global_step += 1
        progress.close()

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
                        self._save_visuals(outputs)
            self._maybe_log_performance(epoch=0, phase="overfit")
            self.global_step += 1
        self.save_checkpoint("latest.pt", epoch=0)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0
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
                    total_loss += float(outputs["loss"].detach().cpu())
                    total_psnr += float(outputs["psnr"].detach().cpu())
                    num_batches += 1
                if batch_idx == 0:
                    with self.profiler.track("val/visuals"):
                        save_training_visualization(
                            pred_rgb=outputs["pred_rgb"],
                            pred_alpha=outputs["pred_alpha"],
                            gt_rgb=outputs["images"],
                            gt_alpha=outputs["alphas"],
                            save_path=str(
                                self.output_dir / "val_visuals" / f"epoch_{epoch:04d}.png"
                            ),
                        )
            progress.update(1)
        progress.close()
        mean_loss = total_loss / max(1, num_batches)
        mean_psnr = total_psnr / max(1, num_batches)
        with self.profiler.track("val/log"):
            self.writer.add_scalar("val/loss", mean_loss, epoch)
            self.writer.add_scalar("val/psnr", mean_psnr, epoch)
        self._maybe_log_performance(epoch, phase="val", force=True)
        return mean_loss

    def save_checkpoint(self, name: str, epoch: int = 0) -> None:
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.amp_enabled else None,
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }
        save_checkpoint_file(payload, str(self.checkpoint_dir / name))

    def load_checkpoint(self, path: str) -> None:
        checkpoint = load_checkpoint_file(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.amp_enabled and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", math.inf))


def plt_image_to_chw(path: Path) -> torch.Tensor:
    """Load saved visualization PNG as [3,H,W] tensor for TensorBoard."""
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
