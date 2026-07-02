"""Plot GPU utilization, memory, and timing charts from performance_latest.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXCLUDED_TOTAL_TIMINGS = {"iteration_total", "forward_total"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GPU and timing charts from performance_latest.json."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to performance_latest.json.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output folder name under outputs/plots/. Defaults to <experiment>_gpu.",
    )
    return parser.parse_args()


def read_performance_json(path: Path) -> dict[str, object]:
    """Read one performance snapshot JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Performance JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def output_dir_for_json(json_path: Path, name: str | None) -> Path:
    """Return outputs/plots/<name>, deriving a name from outputs/<experiment>/stats."""
    if name is None:
        parts = json_path.resolve().parts
        derived = f"{json_path.stem}_gpu"
        if "outputs" in parts:
            output_index = parts.index("outputs")
            if output_index + 1 < len(parts):
                derived = f"{parts[output_index + 1]}_gpu"
        name = derived

    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("--name must be a single output folder name.")
    return Path("outputs") / "plots" / name


def _ema(section: dict[str, object], key: str) -> float | None:
    value = section.get(key)
    if not isinstance(value, dict) or "ema" not in value:
        return None
    return float(value["ema"])


def _latest_or_ema(section: dict[str, object], key: str) -> float | None:
    value = section.get(key)
    if not isinstance(value, dict):
        return None
    if "latest" in value:
        return float(value["latest"])
    if "ema" in value:
        return float(value["ema"])
    return None


def clamp_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


def collect_system_gpu(payload: dict[str, object]) -> dict[str, float]:
    """Collect system-level GPU EMA metrics."""
    system = payload.get("system", {})
    if not isinstance(system, dict):
        return {}

    result: dict[str, float] = {}
    for key in (
        "gpu_utilization_percent",
        "gpu_memory_utilization_percent",
        "nvidia_smi_memory_used_mb",
        "nvidia_smi_memory_total_mb",
        "cuda_memory_allocated_mb",
        "cuda_memory_reserved_mb",
        "cuda_max_memory_allocated_mb",
    ):
        value = _ema(system, key)
        if value is not None:
            result[key] = value

    total = _latest_or_ema(system, "nvidia_smi_memory_total_mb")
    if total is None:
        total = _latest_or_ema(system, "cuda_device_memory_total_mb")
    if total is not None:
        result["memory_total_mb"] = total

    used = result.get("nvidia_smi_memory_used_mb")
    if used is None:
        used = _ema(system, "cuda_device_memory_used_mb")
    if used is not None:
        result["memory_used_mb"] = used
    if used is not None and total is not None:
        result["memory_free_mb"] = max(0.0, total - used)
    return result


def collect_timing_ema(
    payload: dict[str, object],
    phase: str,
) -> dict[str, float]:
    """Collect non-overlapping timing EMA values for one phase."""
    timings = payload.get("timings_ms", {})
    if not isinstance(timings, dict):
        return {}

    prefix = f"{phase}/"
    result: dict[str, float] = {}
    for name, stat in timings.items():
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        short_name = name[len(prefix) :]
        if short_name in EXCLUDED_TOTAL_TIMINGS:
            continue
        if not isinstance(stat, dict) or "ema" not in stat:
            continue
        value = float(stat["ema"])
        if value > 0.0:
            result[short_name] = value
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def collect_gpu_module_memory(
    payload: dict[str, object],
    metric: str = "peak_allocated_delta_mb",
) -> dict[str, float]:
    """Collect positive module-level GPU memory EMA values."""
    modules = payload.get("gpu_modules", {})
    if not isinstance(modules, dict):
        return {}

    result: dict[str, float] = {}
    for name, metrics in modules.items():
        if not isinstance(name, str) or not isinstance(metrics, dict):
            continue
        stat = metrics.get(metric)
        if not isinstance(stat, dict) or "ema" not in stat:
            continue
        value = float(stat["ema"])
        if value > 1.0e-9:
            result[name] = value
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def collect_gpu_static_memory(payload: dict[str, object]) -> dict[str, float]:
    """Collect static GPU memory for model, criterion, renderer, and optimizer state."""
    gpu_static = payload.get("gpu_static", {})
    if not isinstance(gpu_static, dict):
        return {}

    result: dict[str, float] = {}
    modules = gpu_static.get("modules", {})
    if isinstance(modules, dict):
        for name, stat in modules.items():
            if not isinstance(name, str) or not isinstance(stat, dict):
                continue
            value = stat.get("total_mb")
            if value is not None and float(value) > 0.0:
                result[name] = float(value)

    optimizer_state = gpu_static.get("optimizer_state", {})
    if isinstance(optimizer_state, dict):
        value = optimizer_state.get("state_mb")
        if value is not None and float(value) > 0.0:
            result["optimizer_state"] = float(value)
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def _top_items(
    values: dict[str, float],
    limit: int,
) -> dict[str, float]:
    items = [(key, value) for key, value in values.items() if value > 0.0]
    items.sort(key=lambda item: item[1], reverse=True)
    if len(items) <= limit:
        return dict(items)

    top = items[: limit - 1]
    other = sum(value for _, value in items[limit - 1 :])
    return {**dict(top), "other": other}


def plot_pie(
    values: dict[str, float],
    save_path: Path,
    title: str,
    *,
    limit: int = 10,
) -> bool:
    """Save a pie chart, returning whether a file was written."""
    values = _top_items(values, limit)
    values = {key: value for key, value in values.items() if value > 0.0}
    if not values:
        return False

    plt.figure(figsize=(7, 5))
    plt.pie(
        list(values.values()),
        labels=list(values.keys()),
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return True


def plot_bar(
    values: dict[str, float],
    save_path: Path,
    title: str,
    ylabel: str,
    *,
    limit: int = 20,
) -> bool:
    """Save a vertical bar chart, returning whether a file was written."""
    values = _top_items(values, limit)
    values = {key: value for key, value in values.items() if value > 0.0}
    if not values:
        return False

    labels = list(values.keys())
    numbers = list(values.values())
    width = max(8.0, min(18.0, 0.45 * len(labels) + 4.0))
    plt.figure(figsize=(width, 5))
    plt.bar(labels, numbers)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return True


def save_gpu_plots(
    payload: dict[str, object],
    output_dir: Path,
) -> list[Path]:
    """Save GPU utilization, memory, and timing plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    system_gpu = collect_system_gpu(payload)
    train_timing = collect_timing_ema(payload, "train")
    val_timing = collect_timing_ema(payload, "val")
    module_memory = collect_gpu_module_memory(payload)
    static_memory = collect_gpu_static_memory(payload)

    summary = {
        "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        "phase": payload.get("phase"),
        "system_gpu": system_gpu,
        "train_timing_ms": train_timing,
        "val_timing_ms": val_timing,
        "gpu_module_peak_allocated_delta_mb": module_memory,
        "gpu_static_memory_mb": static_memory,
    }
    summary_path = output_dir / "gpu_performance_ema.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    saved_paths.append(summary_path)

    gpu_util = clamp_percent(system_gpu.get("gpu_utilization_percent"))
    if gpu_util is not None:
        path = output_dir / "gpu_utilization_pie.png"
        if plot_pie(
            {"gpu_busy_percent": gpu_util, "gpu_idle_percent": 100.0 - gpu_util},
            path,
            "GPU utilization EMA",
        ):
            saved_paths.append(path)

    memory_used = system_gpu.get("memory_used_mb")
    memory_free = system_gpu.get("memory_free_mb")
    if memory_used is not None and memory_free is not None:
        path = output_dir / "gpu_memory_pie.png"
        if plot_pie(
            {"memory_used_mb": memory_used, "memory_free_mb": memory_free},
            path,
            "GPU memory EMA",
        ):
            saved_paths.append(path)

    system_bar = {
        key: value
        for key, value in {
            "gpu_util_percent": clamp_percent(system_gpu.get("gpu_utilization_percent")),
            "gpu_mem_util_percent": clamp_percent(
                system_gpu.get("gpu_memory_utilization_percent")
            ),
        }.items()
        if value is not None
    }
    path = output_dir / "gpu_system_bar.png"
    if plot_bar(system_bar, path, "GPU utilization EMA", "percent"):
        saved_paths.append(path)

    for phase, timing in (("train", train_timing), ("val", val_timing)):
        pie_path = output_dir / f"{phase}_timing_pie.png"
        if plot_pie(timing, pie_path, f"{phase} timing EMA", limit=12):
            saved_paths.append(pie_path)
        bar_path = output_dir / f"{phase}_timing_bar.png"
        if plot_bar(timing, bar_path, f"{phase} timing EMA", "milliseconds", limit=20):
            saved_paths.append(bar_path)

    module_pie_path = output_dir / "gpu_module_peak_memory_pie.png"
    if plot_pie(
        module_memory,
        module_pie_path,
        "GPU module peak allocated delta EMA",
        limit=12,
    ):
        saved_paths.append(module_pie_path)
    module_bar_path = output_dir / "gpu_module_peak_memory_bar.png"
    if plot_bar(
        module_memory,
        module_bar_path,
        "GPU module peak allocated delta EMA",
        "MB",
        limit=20,
    ):
        saved_paths.append(module_bar_path)

    static_pie_path = output_dir / "gpu_static_memory_pie.png"
    if plot_pie(static_memory, static_pie_path, "Static GPU memory", limit=10):
        saved_paths.append(static_pie_path)
    static_bar_path = output_dir / "gpu_static_memory_bar.png"
    if plot_bar(static_memory, static_bar_path, "Static GPU memory", "MB", limit=10):
        saved_paths.append(static_bar_path)

    return saved_paths


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args_from(argv)
    json_path = Path(args.json)
    payload = read_performance_json(json_path)
    output_dir = output_dir_for_json(json_path, args.name)
    saved_paths = save_gpu_plots(payload, output_dir)
    for path in saved_paths:
        print(f"GPU performance plot saved to {path.resolve()}")


def parse_args_from(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GPU and timing charts from performance_latest.json."
    )
    parser.add_argument("--json", required=True)
    parser.add_argument("--name", default=None)
    return parser.parse_args(list(argv))


if __name__ == "__main__":
    main()
