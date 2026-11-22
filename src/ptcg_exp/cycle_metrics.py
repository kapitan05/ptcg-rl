"""Flatten one finished PPO cycle (training.json, health.json, canary.json) into scalar metrics.

Shared by the live W&B hook in `run.py` and by `wandb_backfill.py`, so live and backfilled runs
log the same keys: `train/*` (mean over updates), `heldout/*`, `health/<check>` (numeric values,
dict values averaged), `health/status` (0 pass, 1 warn, 2 fail) and `canary/*`.
"""

from __future__ import annotations

import json
from pathlib import Path

STATUS_CODE = {"pass": 0, "warn": 1, "fail": 2}


def _number(value: object) -> float | None:
    """Scalar for a metric value: numbers as-is, dicts of numbers averaged, else None."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        nums = [float(v) for v in value.values() if isinstance(v, int | float)]
        return sum(nums) / len(nums) if nums else None
    return None


def cycle_metrics(cycle: Path) -> dict[str, float]:
    """Scalar metrics of `cycle` (a `ppo/cycle-NNN` directory); missing files are skipped."""
    out: dict[str, float] = {"cycle": float(int(cycle.name.split("-")[-1]))}
    training = cycle / "models/training.json"
    if training.exists():
        t = json.loads(training.read_text())
        for k, v in (t.get("mean_metrics") or {}).items():
            if (x := _number(v)) is not None:
                out[f"train/{k}"] = x
        for k in ("heldout_global_approximate_kl", "heldout_sampled_forward_kl"):
            if (x := _number(t.get(k))) is not None:
                out[f"heldout/{k}"] = x
        if (x := _number(t.get("parameter_delta_l2"))) is not None:
            out["train/parameter_delta_l2"] = x
        out["train/early_stopped"] = float(bool(t.get("early_stopped")))
    health = cycle / "health.json"
    if health.exists():
        h = json.loads(health.read_text())
        out["health/status"] = float(STATUS_CODE.get(h.get("status", ""), -1))
        for check in h.get("checks", []):
            if (x := _number(check.get("value"))) is not None:
                out[f"health/{check['check']}"] = x
    canary = cycle / "canary.json"
    if canary.exists():
        for k, v in json.loads(canary.read_text()).items():
            if (x := _number(v)) is not None:
                out[f"canary/{k}"] = x
    return out
