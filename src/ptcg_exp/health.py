"""PPO health checks: pass / warn / fail per check for one cycle → health.json.

Inputs (all produced by the existing pipeline plus the additive hooks in ppo/train.py):
  <cycle>/rollout/manifest.json   arena_results (opponent, learner_seat, result, steps), audit
  <cycle>/update/training.json    updates_log, heldout KL, health_inputs, parameter_delta_l2
  <cycle>/validation.json         C++ export parity (optional; gate)
  <cycle>/canary.json             drift metrics from ptcg_exp.canary (optional)
A previous cycle (if given) enables drift checks. Any FAIL means: do not promote the cycle.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np

from ptcg_exp.config import HealthThresholds
from ptcg_exp.tracking import JSON

Status = Literal["pass", "warn", "fail", "skip"]
ORDER = {"pass": 0, "skip": 0, "warn": 1, "fail": 2}


def _check(name: str, status: Status, value: object, rule: str) -> JSON:
    """One health-check record."""
    return dict(check=name, status=status, value=value, rule=rule)


def _find(cycle: Path, name: str) -> Path | None:
    """Locate a pipeline output file anywhere under the cycle directory."""
    hits = sorted(cycle.rglob(name))
    return hits[0] if hits else None


def rollout_checks(
    rollout: JSON,
    t: HealthThresholds,
    cfg_temperature: float | None,
    previous: JSON | None,
) -> list[JSON]:
    """Pre-update data checks on the collected rollout."""
    out = []
    games = rollout.get("arena_results", [])
    by: dict[tuple[str, int], list[float]] = defaultdict(list)
    for g in games:
        if "opponent" not in g:
            continue
        seat = g["learner_seat"]
        score = 0.5 if g["result"] == 2 else float(g["result"] == seat)
        by[(g["opponent"], seat)].append(score)
    if by:
        lo, hi = t.seat_win_band
        bad = {
            f"{o}/seat{s}": round(float(np.mean(v)), 3)
            for (o, s), v in by.items()
            if not lo <= np.mean(v) <= hi and len(v) >= 20
        }
        rates = {f"{o}/seat{s}": round(float(np.mean(v)), 3) for (o, s), v in sorted(by.items())}
        out.append(
            _check(
                "win_rate_per_opponent_seat",
                "warn" if bad else "pass",
                rates,
                f"each in [{lo}, {hi}] when ≥20 games; outside: {bad or 'none'}",
            )
        )
    else:
        out.append(
            _check("win_rate_per_opponent_seat", "skip", None, "rollout lacks opponent tags")
        )
    steps = [g["steps"] for g in games]
    if steps and previous and previous.get("arena_results"):
        prev = np.mean([g["steps"] for g in previous["arena_results"]])
        shift = abs(np.mean(steps) - prev) / max(prev, 1e-9)
        out.append(
            _check(
                "game_length_shift",
                "warn" if shift > t.length_shift_warn else "pass",
                dict(mean=float(np.mean(steps)), previous=float(prev), shift=float(shift)),
                f"relative shift ≤ {t.length_shift_warn}",
            )
        )
    audit = rollout.get("audit", {})
    errs = audit.get("max_abs_errors", {}) if isinstance(audit, dict) else {}
    worst = max(errs.values()) if errs else None
    out.append(
        _check(
            "behavior_audit",
            "pass" if worst is not None and worst <= 2e-5 else "fail",
            errs,
            "logp/value/entropy recomputation ≤ 2e-5 (r = 1 at update start)",
        )
    )
    temp = rollout.get("temperature")
    if cfg_temperature is not None:
        out.append(
            _check(
                "rollout_temperature",
                "pass" if temp == cfg_temperature else "fail",
                temp,
                f"equals configured {cfg_temperature}",
            )
        )
    coll = rollout.get("collection", {})
    out.append(
        _check(
            "seeds_recorded",
            "pass"
            if isinstance(coll, dict)
            and coll.get("seed") is not None
            and coll.get("sample_seed") is not None
            else "fail",
            dict(seed=coll.get("seed"), sample_seed=coll.get("sample_seed"))
            if isinstance(coll, dict)
            else None,
            "arena and sampling seeds recorded",
        )
    )
    return out


def update_checks(training: JSON, t: HealthThresholds, previous: JSON | None) -> list[JSON]:
    """Per-update streamed checks and post-update checks."""
    out = []
    hi = training.get("health_inputs")
    log = training.get("updates_log") or []
    if not hi or not log:
        out.append(
            _check(
                "health_hooks_present",
                "fail",
                None,
                "training.json lacks updates_log/health_inputs (old ppo/train.py?)",
            )
        )
        return out
    adv = hi["raw_advantage"]
    rr = hi["return_range"]
    ok = adv["nonfinite"] == 0 and -1 - 1e-6 <= rr[0] and rr[1] <= 1 + 1e-6
    out.append(
        _check(
            "gae_sanity",
            "pass" if ok else "fail",
            dict(adv=adv, return_range=rr),
            "no non-finite advantages; returns within [-1, 1]",
        )
    )
    ev0 = hi["explained_variance_before"]
    out.append(
        _check(
            "explained_variance_before",
            "warn" if ev0 < 0 else "pass",
            round(ev0, 4),
            "critic better than a constant (EV ≥ 0)",
        )
    )
    kls = [r["approximate_kl"] for r in log]
    kmax = max(kls)
    fail_at = t.kl_fail_multiple * t.kl_target
    out.append(
        _check(
            "approx_kl_per_update",
            "fail" if kmax > fail_at else "pass",
            dict(
                mean=float(np.mean(kls)),
                max=float(kmax),
                early_stopped=training.get("early_stopped"),
            ),
            f"max ≤ {t.kl_fail_multiple}×kl_target = {fail_at:.4g}",
        )
    )
    clip = max(r["clip_fraction"] for r in log)
    out.append(
        _check(
            "clip_fraction",
            "warn" if clip > t.clip_fraction_warn else "pass",
            round(clip, 4),
            f"max ≤ {t.clip_fraction_warn}",
        )
    )
    p99 = max(r.get("ratio_p99", 1.0) for r in log)
    out.append(
        _check(
            "ratio_p99",
            "warn" if p99 > t.ratio_p99_warn else "pass",
            round(p99, 4),
            f"max p99 ≤ {t.ratio_p99_warn}",
        )
    )
    g = [r["gradient_norm"] for r in log]
    spike = max(g) / max(float(np.median(g)), 1e-12)
    out.append(
        _check(
            "gradient_spike",
            "warn" if spike > t.grad_spike_warn else "pass",
            dict(max=max(g), median=float(np.median(g)), ratio=spike),
            f"max/median pre-clip norm ≤ {t.grad_spike_warn}",
        )
    )
    finite = all(np.isfinite(v) for r in log for v in r.values() if isinstance(v, int | float))
    out.append(
        _check(
            "finite_metrics", "pass" if finite else "fail", finite, "all per-update metrics finite"
        )
    )
    ur = [r.get("update_ratio") for r in log if r.get("update_ratio") is not None]
    if ur:
        lo, hi_b = t.update_ratio_band
        med = float(np.median(ur))
        out.append(
            _check(
                "update_ratio",
                "pass" if lo <= med <= hi_b else "warn",
                med,
                f"median ‖Δθ‖/‖θ‖ in [{lo}, {hi_b}]",
            )
        )
    delta = training.get("parameter_delta_l2", 0)
    out.append(
        _check("parameters_moved", "pass" if delta and delta > 0 else "fail", delta, "Δθ L2 > 0")
    )
    hkl = training.get("heldout_global_approximate_kl")
    band = (t.kl_low_multiple * t.kl_target, t.kl_fail_multiple * t.kl_target)
    status: Status = "fail" if hkl is None or hkl > band[1] else "warn" if hkl < band[0] else "pass"
    out.append(
        _check(
            "heldout_kl", status, hkl, f"in [{band[0]:.4g}, {band[1]:.4g}] (low = step too small)"
        )
    )
    ev1 = hi["explained_variance_after_heldout"]
    out.append(
        _check(
            "explained_variance_after",
            "warn" if ev1 < 0 else "pass",
            round(ev1, 4),
            "EV ≥ 0 on held-out",
        )
    )
    ece = hi["value_ece_heldout"]
    out.append(
        _check(
            "value_calibration_ece",
            "warn" if ece > t.ece_warn else "pass",
            round(ece, 4),
            f"10-bin ECE ≤ {t.ece_warn}",
        )
    )
    e0, e1 = hi["entropy_behavior_heldout"], hi["entropy_updated_heldout"]
    drop = (e0 - e1) / max(e0, 1e-9)
    out.append(
        _check(
            "entropy_drop",
            "warn" if drop > t.entropy_drop_warn else "pass",
            dict(behavior=e0, updated=e1, drop=drop),
            f"relative drop ≤ {t.entropy_drop_warn}",
        )
    )
    return out


def canary_checks(
    canary: dict[str, float] | None, previous: dict[str, float] | None, t: HealthThresholds
) -> list[JSON]:
    """Drift on the fixed probe set."""
    if canary is None:
        return [_check("canary", "skip", None, "no canary.json for this cycle")]
    out = [
        _check(
            "illegal_picks",
            "pass" if canary["illegal_pick_rate"] == 0 else "fail",
            canary["illegal_pick_rate"],
            "argmax always on a legal option",
        )
    ]
    if previous is not None:
        drop = previous["teacher_agreement"] - canary["teacher_agreement"]
        out.append(
            _check(
                "teacher_agreement_drift",
                "warn" if drop > t.canary_agreement_drop_warn else "pass",
                dict(now=canary["teacher_agreement"], previous=previous["teacher_agreement"]),
                f"drop ≤ {t.canary_agreement_drop_warn}",
            )
        )
    if "kl_vs_previous" in canary:
        out.append(
            _check("canary_kl_vs_previous", "pass", canary["kl_vs_previous"], "informational")
        )
    return out


def check_cycle(
    cycle: Path,
    t: HealthThresholds,
    previous_cycle: Path | None = None,
    temperature: float | None = 1.0,
) -> JSON:
    """Run every available check for one cycle and write <cycle>/health.json."""

    def load(base: Path | None, name: str) -> JSON | None:
        """Load a JSON output if present."""
        if base is None:
            return None
        path = _find(base, name)
        return json.loads(path.read_text()) if path else None

    checks: list[JSON] = []
    rollout_path = cycle / "rollout/manifest.json"
    rollout = json.loads(rollout_path.read_text()) if rollout_path.exists() else None
    prev_rollout = None
    if previous_cycle and (previous_cycle / "rollout/manifest.json").exists():
        prev_rollout = json.loads((previous_cycle / "rollout/manifest.json").read_text())
    if rollout:
        checks += rollout_checks(rollout, t, temperature, prev_rollout)
    else:
        checks.append(_check("rollout_manifest", "fail", None, "missing rollout/manifest.json"))
    training = load(cycle, "training.json")
    if training:
        checks += update_checks(training, t, load(previous_cycle, "training.json"))
    else:
        checks.append(_check("training_json", "fail", None, "missing training.json"))
    # validate() raises on any C++ ≢ PyTorch mismatch, so a written report means the gate passed.
    validated = (cycle / "validation").is_dir() and any((cycle / "validation").glob("*.json"))
    checks.append(
        _check(
            "cpp_export_parity",
            "pass" if validated else "fail",
            validated,
            "new checkpoint exported to C++ and validated (2e-5) before next rollout",
        )
    )
    checks += canary_checks(load(cycle, "canary.json"), load(previous_cycle, "canary.json"), t)
    worst = max((c["status"] for c in checks), key=lambda s: ORDER[s])
    report = dict(
        cycle=str(cycle),
        status=worst,
        promote_allowed=worst != "fail",
        thresholds=t.model_dump(mode="json"),
        checks=checks,
    )
    (cycle / "health.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def main() -> None:
    """Check one PPO cycle directory and print a compact table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycle", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--temperature", type=float, default=1.0)
    a = p.parse_args()
    r = check_cycle(a.cycle, HealthThresholds(), a.previous, a.temperature)
    for c in r["checks"]:
        print(f"{c['status']:>5}  {c['check']:<30} {json.dumps(c['value'], default=float)[:70]}")
    print(f"overall: {r['status']}  promote_allowed={r['promote_allowed']}")
    raise SystemExit(2 if r["status"] == "fail" else 0)


if __name__ == "__main__":
    main()
