from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.plot_GPU import (
    collect_gpu_module_memory,
    collect_gpu_static_memory,
    collect_system_gpu,
    collect_timing_ema,
    output_dir_for_json,
    save_gpu_plots,
)


def _sample_payload() -> dict[str, object]:
    return {
        "step": 12,
        "epoch": 3,
        "phase": "train",
        "system": {
            "gpu_utilization_percent": {"ema": 75.0},
            "gpu_memory_utilization_percent": {"ema": 50.0},
            "nvidia_smi_memory_used_mb": {"ema": 4000.0},
            "nvidia_smi_memory_total_mb": {"latest": 8000.0, "ema": 8000.0},
            "cuda_memory_allocated_mb": {"ema": 1000.0},
            "cuda_memory_reserved_mb": {"ema": 3000.0},
        },
        "timings_ms": {
            "train/iteration_total": {"ema": 100.0},
            "train/forward_total": {"ema": 80.0},
            "train/model_forward": {"ema": 30.0},
            "train/render": {"ema": 10.0},
            "train/backward": {"ema": 40.0},
            "val/model_forward": {"ema": 20.0},
            "val/render": {"ema": 5.0},
        },
        "gpu_modules": {
            "train/model_forward": {
                "peak_allocated_delta_mb": {"ema": 256.0},
            },
            "train/render": {
                "peak_allocated_delta_mb": {"ema": 32.0},
            },
            "train/zero_grad": {
                "peak_allocated_delta_mb": {"ema": 0.0},
            },
        },
        "gpu_static": {
            "modules": {
                "model": {"total_mb": 214.0},
                "renderer": {"total_mb": 0.0},
                "criterion": {"total_mb": 56.0},
            },
            "optimizer_state": {"state_mb": 428.0},
        },
    }


def test_collect_system_gpu_uses_ema_and_total() -> None:
    result = collect_system_gpu(_sample_payload())

    assert result["gpu_utilization_percent"] == 75.0
    assert result["memory_used_mb"] == 4000.0
    assert result["memory_total_mb"] == 8000.0
    assert result["memory_free_mb"] == 4000.0


def test_collect_timing_ema_excludes_overlapping_totals() -> None:
    result = collect_timing_ema(_sample_payload(), "train")

    assert result == {
        "backward": 40.0,
        "model_forward": 30.0,
        "render": 10.0,
    }


def test_collect_gpu_memory_sections() -> None:
    assert collect_gpu_module_memory(_sample_payload()) == {
        "train/model_forward": 256.0,
        "train/render": 32.0,
    }
    assert collect_gpu_static_memory(_sample_payload()) == {
        "optimizer_state": 428.0,
        "model": 214.0,
        "criterion": 56.0,
    }


def test_output_dir_for_json_derives_experiment_name() -> None:
    json_path = Path("outputs") / "zerogs_finetune" / "stats" / "performance_latest.json"

    assert output_dir_for_json(json_path, None) == (
        Path("outputs") / "plots" / "zerogs_finetune_gpu"
    )
    assert output_dir_for_json(json_path, "manual") == Path("outputs") / "plots" / "manual"


def test_save_gpu_plots_writes_expected_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = Path("outputs") / "plots" / "demo_gpu"

    saved_paths = save_gpu_plots(_sample_payload(), output_dir)

    assert output_dir.joinpath("gpu_performance_ema.json").is_file()
    assert output_dir.joinpath("gpu_utilization_pie.png").is_file()
    assert output_dir.joinpath("gpu_memory_pie.png").is_file()
    assert output_dir.joinpath("gpu_system_bar.png").is_file()
    assert output_dir.joinpath("train_timing_pie.png").is_file()
    assert output_dir.joinpath("train_timing_bar.png").is_file()
    assert output_dir.joinpath("val_timing_pie.png").is_file()
    assert output_dir.joinpath("val_timing_bar.png").is_file()
    assert output_dir.joinpath("gpu_module_peak_memory_bar.png").is_file()
    assert output_dir.joinpath("gpu_static_memory_bar.png").is_file()
    assert len(saved_paths) >= 10
