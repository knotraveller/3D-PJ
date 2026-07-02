from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.plot_loss_curves import (
    output_dir_for_name,
    read_jsonl,
    save_epoch_metric_plots,
    validate_epoch_metrics,
)


def _metric(mean: float, max_value: float, min_value: float) -> dict[str, float]:
    return {"mean": mean, "max": max_value, "min": min_value}


def _sample_row(epoch: int, loss: float, psnr: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "num_batches": 2,
        "lr": 1.0e-4,
        "loss": _metric(loss, loss + 0.2, loss - 0.1),
        "rgb_loss": _metric(loss * 0.5, loss * 0.6, loss * 0.4),
        "mask_loss": _metric(loss * 0.2, loss * 0.3, loss * 0.1),
        "lpips_loss": _metric(loss * 0.1, loss * 0.2, 0.0),
        "psnr": _metric(psnr, psnr + 1.0, psnr - 1.0),
    }


def test_validate_epoch_metrics_rejects_step_rows() -> None:
    with pytest.raises(KeyError, match="Unexpected 'step'"):
        validate_epoch_metrics([{**_sample_row(1, 0.3, 22.0), "step": 10}])


def test_validate_epoch_metrics_sorts_and_normalizes() -> None:
    rows = validate_epoch_metrics(
        [
            _sample_row(2, 0.2, 24.0),
            _sample_row(1, 0.3, 22.0),
        ]
    )

    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[0]["loss"] == pytest.approx({"mean": 0.3, "max": 0.5, "min": 0.2})
    assert rows[1]["psnr"] == pytest.approx({"mean": 24.0, "max": 25.0, "min": 23.0})


def test_output_dir_for_name_rejects_paths() -> None:
    assert output_dir_for_name("zerogs") == Path("outputs") / "plots" / "zerogs"

    with pytest.raises(ValueError, match="single output folder"):
        output_dir_for_name("../zerogs")


def test_read_jsonl_and_save_epoch_metric_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(_sample_row(1, 0.3, 22.0)),
                json.dumps(_sample_row(2, 0.2, 24.0)),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    rows = validate_epoch_metrics(read_jsonl(log_path))
    saved_paths = save_epoch_metric_plots(rows, output_dir_for_name("demo"))

    payload = json.loads(
        (tmp_path / "outputs" / "plots" / "demo" / "epoch_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload[0]["psnr"]["mean"] == 22.0
    assert (tmp_path / "outputs" / "plots" / "demo" / "epoch_metrics.json").is_file()
    assert (tmp_path / "outputs" / "plots" / "demo" / "loss_epoch_stats.png").is_file()
    assert (tmp_path / "outputs" / "plots" / "demo" / "loss_epoch_stats_log.png").is_file()
    assert (tmp_path / "outputs" / "plots" / "demo" / "psnr_epoch_stats.png").is_file()
    assert len(saved_paths) == 8
