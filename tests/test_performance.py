from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.performance import PerformanceMonitor, RunningStat


def test_running_stat_tracks_ema_min_max_mean() -> None:
    stat = RunningStat(ema_momentum=0.8, unit="ms")
    stat.update(10.0)
    stat.update(20.0)

    payload = stat.as_dict()
    assert payload["latest"] == 20.0
    assert payload["ema"] == 12.0
    assert payload["min"] == 10.0
    assert payload["max"] == 20.0
    assert payload["mean"] == 15.0
    assert payload["count"] == 2
    assert payload["unit"] == "ms"


def test_performance_monitor_writes_snapshot(tmp_path: Path) -> None:
    monitor = PerformanceMonitor(enabled=True, sync_cuda=False, ema_momentum=0.5)

    with monitor.track("train/example"):
        pass
    monitor.update_system("cpu_percent", 25.0, unit="percent")
    snapshot = monitor.snapshot(step=3, epoch=1, phase="train")
    monitor.write_snapshot(tmp_path, snapshot)

    latest = json.loads((tmp_path / "performance_latest.json").read_text(encoding="utf-8"))
    assert latest["step"] == 3
    assert latest["epoch"] == 1
    assert latest["phase"] == "train"
    assert "train/example" in latest["timings_ms"]
    assert latest["system"]["cpu_percent"]["latest"] == 25.0

    lines = (tmp_path / "performance_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
