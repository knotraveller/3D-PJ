"""Lightweight performance monitoring for training loops."""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch


class RunningStat:
    """Track latest, EMA, min, max, and mean for one scalar metric."""

    def __init__(self, ema_momentum: float = 0.8, unit: str = "") -> None:
        self.ema_momentum = float(ema_momentum)
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum must be in [0, 1).")
        self.unit = unit
        self.count = 0
        self.latest = 0.0
        self.ema = 0.0
        self.total = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, value: float) -> None:
        value = float(value)
        self.latest = value
        self.total += value
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        if self.count == 1:
            self.ema = value
        else:
            self.ema = self.ema_momentum * self.ema + (1.0 - self.ema_momentum) * value

    def as_dict(self) -> Dict[str, float | int | str]:
        mean = self.total / max(1, self.count)
        return {
            "latest": self.latest,
            "ema": self.ema,
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "count": self.count,
            "unit": self.unit,
        }


class PerformanceMonitor:
    """Collect stage timings plus optional CPU/GPU resource samples."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        device: Optional[torch.device] = None,
        ema_momentum: float = 0.8,
        sync_cuda: bool = True,
        sample_system: bool = True,
        sample_gpu_utilization: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device or torch.device("cpu")
        self.ema_momentum = float(ema_momentum)
        self.sync_cuda = bool(sync_cuda)
        self.sample_system_enabled = bool(sample_system)
        self.sample_gpu_utilization = bool(sample_gpu_utilization)
        self.timings_ms: Dict[str, RunningStat] = {}
        self.system: Dict[str, RunningStat] = {}
        self._psutil = None
        self._process = None
        self._nvidia_smi_available: Optional[bool] = None
        if self.enabled and self.sample_system_enabled:
            self._init_psutil()

    def _init_psutil(self) -> None:
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            return
        self._psutil = psutil
        self._process = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)

    def _cuda_timing_active(self) -> bool:
        return (
            self.enabled
            and self.sync_cuda
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

    def _synchronize(self) -> None:
        if self._cuda_timing_active():
            torch.cuda.synchronize(self._cuda_device_index())

    def _cuda_device_index(self) -> int:
        if self.device.index is not None:
            return int(self.device.index)
        return int(torch.cuda.current_device())

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        """Measure a stage in milliseconds."""
        if not self.enabled:
            yield
            return

        self._synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.update_timing(name, elapsed_ms)

    def update_timing(self, name: str, elapsed_ms: float) -> None:
        self._update(self.timings_ms, name, elapsed_ms, unit="ms")

    def update_system(self, name: str, value: float, unit: str) -> None:
        self._update(self.system, name, value, unit=unit)

    def _update(
        self,
        target: Dict[str, RunningStat],
        name: str,
        value: float,
        *,
        unit: str,
    ) -> None:
        stat = target.get(name)
        if stat is None:
            stat = RunningStat(ema_momentum=self.ema_momentum, unit=unit)
            target[name] = stat
        stat.update(float(value))

    def sample_system(self) -> None:
        """Sample CPU, process memory, CUDA memory, and nvidia-smi utilization when available."""
        if not self.enabled or not self.sample_system_enabled:
            return

        if self._psutil is not None and self._process is not None:
            self.update_system(
                "cpu_percent",
                float(self._psutil.cpu_percent(interval=None)),
                unit="percent",
            )
            self.update_system(
                "process_cpu_percent",
                float(self._process.cpu_percent(interval=None)),
                unit="percent",
            )
            rss_mb = self._process.memory_info().rss / (1024.0 * 1024.0)
            self.update_system("process_rss_mb", rss_mb, unit="MB")

        if self.device.type == "cuda" and torch.cuda.is_available():
            device = self._cuda_device_index()
            self.update_system(
                "cuda_memory_allocated_mb",
                torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
                unit="MB",
            )
            self.update_system(
                "cuda_memory_reserved_mb",
                torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
                unit="MB",
            )
            self.update_system(
                "cuda_max_memory_allocated_mb",
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                unit="MB",
            )
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            except RuntimeError:
                pass
            else:
                used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0)
                self.update_system("cuda_device_memory_used_mb", used_mb, unit="MB")
                self.update_system(
                    "cuda_device_memory_total_mb",
                    total_bytes / (1024.0 * 1024.0),
                    unit="MB",
                )

            if self.sample_gpu_utilization:
                self._sample_nvidia_smi()

    def _sample_nvidia_smi(self) -> None:
        if self._nvidia_smi_available is False:
            return
        index = self.device.index
        if index is None:
            index = self._cuda_device_index()
        command = [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            self._nvidia_smi_available = False
            return
        if result.returncode != 0 or not result.stdout.strip():
            self._nvidia_smi_available = False
            return
        self._nvidia_smi_available = True
        first_line = result.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) < 4:
            return
        try:
            gpu_util, mem_util, mem_used, mem_total = [float(part) for part in parts[:4]]
        except ValueError:
            return
        self.update_system("gpu_utilization_percent", gpu_util, unit="percent")
        self.update_system("gpu_memory_utilization_percent", mem_util, unit="percent")
        self.update_system("nvidia_smi_memory_used_mb", mem_used, unit="MB")
        self.update_system("nvidia_smi_memory_total_mb", mem_total, unit="MB")

    def snapshot(
        self,
        *,
        step: int,
        epoch: int,
        phase: str,
    ) -> Dict[str, object]:
        return {
            "step": int(step),
            "epoch": int(epoch),
            "phase": phase,
            "timestamp": time.time(),
            "enabled": self.enabled,
            "ema_momentum": self.ema_momentum,
            "ema_formula": "ema = ema_momentum * previous_ema + (1 - ema_momentum) * current",
            "sync_cuda": self.sync_cuda,
            "timings_ms": {
                name: stat.as_dict() for name, stat in sorted(self.timings_ms.items())
            },
            "system": {
                name: stat.as_dict() for name, stat in sorted(self.system.items())
            },
        }

    def write_snapshot(self, stats_dir: Path, snapshot: Dict[str, object]) -> None:
        if not self.enabled:
            return
        stats_dir.mkdir(parents=True, exist_ok=True)
        latest_path = stats_dir / "performance_latest.json"
        latest_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        log_path = stats_dir / "performance_log.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot) + "\n")

    def log_tensorboard(self, writer: object, step: int) -> None:
        if not self.enabled:
            return
        for name, stat in self.timings_ms.items():
            writer.add_scalar(f"perf/timing_ms/{name}/latest", stat.latest, step)
            writer.add_scalar(f"perf/timing_ms/{name}/ema", stat.ema, step)
        for name, stat in self.system.items():
            writer.add_scalar(f"perf/system/{name}/latest", stat.latest, step)
            writer.add_scalar(f"perf/system/{name}/ema", stat.ema, step)
