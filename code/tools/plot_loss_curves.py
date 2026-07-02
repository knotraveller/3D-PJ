"""Plot epoch-level loss and PSNR curves from train_log.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_KEYS = ("loss", "rgb_loss", "mask_loss", "lpips_loss")
METRIC_KEYS = (*LOSS_KEYS, "psnr")
STAT_KEYS = ("mean", "max", "min")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot epoch-level training loss and PSNR curves from train_log.jsonl."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to train_log.jsonl.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Output folder name under outputs/plots/.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read non-empty JSONL rows from an epoch-level training log."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL log does not exist: {path}")

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object row at {path}:{line_number}")
        rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in JSONL log: {path}")
    return rows


def validate_epoch_metrics(
    rows: Iterable[dict[str, object]],
    metric_keys: Iterable[str] = METRIC_KEYS,
) -> list[dict[str, object]]:
    """Validate and sort epoch-level metric rows."""
    keys = tuple(metric_keys)
    validated: list[dict[str, object]] = []
    seen_epochs: set[int] = set()
    for row_index, row in enumerate(rows):
        if "epoch" not in row:
            raise KeyError(f"Missing 'epoch' in row {row_index}")
        if "step" in row:
            raise KeyError(
                f"Unexpected 'step' in row {row_index}; train_log.jsonl is epoch-level."
            )
        epoch = int(row["epoch"])
        if epoch in seen_epochs:
            raise ValueError(f"Duplicate epoch {epoch} in epoch-level log.")
        seen_epochs.add(epoch)

        item = dict(row)
        item["epoch"] = epoch
        for key in keys:
            value = item.get(key)
            if not isinstance(value, dict):
                raise KeyError(f"Missing metric summary '{key}' in row {row_index}")
            for stat in STAT_KEYS:
                if stat not in value:
                    raise KeyError(
                        f"Missing '{key}.{stat}' in row {row_index}"
                    )
            item[key] = {
                stat: float(value[stat])
                for stat in STAT_KEYS
            }
        validated.append(item)

    if not validated:
        raise ValueError("No epoch metric rows found.")
    return sorted(validated, key=lambda row: int(row["epoch"]))


def output_dir_for_name(name: str) -> Path:
    """Return the output directory for a safe plot set name."""
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("--name must be a single output folder name.")
    return Path("outputs") / "plots" / name


def _metric_series(
    rows: list[dict[str, object]],
    key: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    xs: list[int] = []
    means: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, dict):
            continue
        xs.append(int(row["epoch"]))
        means.append(float(value["mean"]))
        mins.append(float(value["min"]))
        maxs.append(float(value["max"]))
    return xs, means, mins, maxs


def _plot_metric_with_range(
    rows: list[dict[str, object]],
    key: str,
    save_path: Path,
    *,
    ylabel: str,
) -> bool:
    xs, means, mins, maxs = _metric_series(rows, key)
    if not means:
        return False

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, means, marker="o", markersize=2, linewidth=1.5, label=f"{key} mean")
    plt.fill_between(xs, mins, maxs, alpha=0.15, label="min-max")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return True


def _plot_combined_losses(
    rows: list[dict[str, object]],
    loss_keys: Iterable[str],
    save_path: Path,
    *,
    log_scale: bool,
) -> bool:
    epochs = [int(row["epoch"]) for row in rows]
    plotted = False
    plt.figure(figsize=(9, 5))
    for key in loss_keys:
        xs, means, mins, maxs = _metric_series(rows, key)
        if not means:
            continue
        plotted = True
        plt.plot(xs, means, marker="o", markersize=2, linewidth=1.5, label=f"{key} mean")
        plt.fill_between(xs, mins, maxs, alpha=0.10)
    if not plotted:
        plt.close()
        return False

    plt.xlabel("epoch")
    plt.ylabel("loss")
    if log_scale:
        plt.yscale("log")
    if min(epochs) != max(epochs):
        plt.xlim(min(epochs), max(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return True


def save_epoch_metric_plots(
    rows: list[dict[str, object]],
    output_dir: Path,
    loss_keys: Iterable[str] = LOSS_KEYS,
    metric_keys: Iterable[str] = METRIC_KEYS,
) -> list[Path]:
    """Save combined loss plots and per-metric epoch stat plots."""
    loss_keys = tuple(loss_keys)
    metric_keys = tuple(metric_keys)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "epoch_metrics.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    saved_paths = [summary_path]
    combined_path = output_dir / "loss_epoch_stats.png"
    if _plot_combined_losses(rows, loss_keys, combined_path, log_scale=False):
        saved_paths.append(combined_path)

    combined_log_path = output_dir / "loss_epoch_stats_log.png"
    if _plot_combined_losses(rows, loss_keys, combined_log_path, log_scale=True):
        saved_paths.append(combined_log_path)

    for key in metric_keys:
        save_path = output_dir / f"{key}_epoch_stats.png"
        if _plot_metric_with_range(rows, key, save_path, ylabel=key):
            saved_paths.append(save_path)
    return saved_paths


def main() -> None:
    args = parse_args()
    rows = validate_epoch_metrics(read_jsonl(Path(args.json)))
    output_dir = output_dir_for_name(args.name)
    saved_paths = save_epoch_metric_plots(rows, output_dir)
    for path in saved_paths:
        print(f"Metric plot saved to {path.resolve()}")


if __name__ == "__main__":
    main()
