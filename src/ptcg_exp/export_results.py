"""Export saved run results (`reports/runs`) to three tidy CSVs for dashboards and the report.

    PYTHONPATH=src uv run python -m ptcg_exp.export_results   # → reports/results/*.csv

- `pairs.csv`: one row per (experiment arm, seed, opponent) for the arm's own final agent:
  games, wins, draws, losses, win rate and its Wilson 95% CI from the eval summary.
- `ratings.csv`: every agent's BT rating per run, with its role (own checkpoint, start,
  teacher, rule bot, held-out) and, for own PPO checkpoints, the global PPO cycle.
- `cycles.csv`: per PPO cycle health and training metrics (`cycle_metrics`).

Only finished evaluations are exported; pipechecks and smoke runs are left out.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from ptcg_exp.cycle_metrics import cycle_metrics
from ptcg_exp.tracking import ROOT

# run-name prefix → (experiment id, arm label, global cycle of the run's start checkpoint)
ARMS: dict[str, tuple[str, str, int]] = {
    "C-ppo_baseline_klstop": ("C/01", "C/01 PPO baseline", 0),
    "C-ppo_baseline": ("C/01", "C/01 PPO baseline", 0),
    "C-ppo_continue": ("C/03", "C/03 continue (control)", 12),
    "C-ppo_league_pfsp": ("C/05", "C/05 league + PFSP", 12),
    "C-ppo_league": ("C/04", "C/04 snapshot league", 12),
    "C-critic_value_weight": ("C/02", "C/02 critic weight 1.0", 0),
    "A-distill_ppo_smoke": ("A/02", "A/02 distilled Transformer", 0),
}
DAGGER = {"dagger_student": "A/04 DAgger student", "dagger_teacher_bc": "A/04 BC control"}
OPPONENT = {
    "start_transformer": "start Transformer",
    "a02_transformer": "start Transformer",
    "c01_final": "C/01 final",
    "gbdt_1600_s0": "GBDT teacher",
    "lucario_rule": "Lucario rule bot",
    "dragapult_rule": "Dragapult rule bot",
    "abomasnow_rule": "Abomasnow rule bot",
    "iono_rule": "Iono rule bot",
    "heldout_dragapult_gbdt": "held-out Dragapult GBDT",
    "heldout_abomasnow_gbdt": "held-out Abomasnow GBDT",
    "heldout_iono_gbdt": "held-out Iono GBDT",
}
CYCLE_FIELDS = {
    "health/status": "health_status",
    "train/approximate_kl": "approx_kl",
    "train/clip_fraction": "clip_fraction",
    "train/entropy": "entropy",
    "train/value_loss": "value_loss",
    "health/heldout_kl": "heldout_kl",
    "health/value_calibration_ece": "value_ece",
    "health/explained_variance_after": "explained_variance",
    "canary/teacher_agreement": "teacher_agreement",
    "train/early_stopped": "early_stopped",
}
STATUS = {0.0: "pass", 1.0: "warn", 2.0: "fail"}


def _arm(run: str) -> tuple[str, str, int] | None:
    for prefix, arm in ARMS.items():
        if run.startswith(prefix + "-"):
            return arm
    return None


def _seed(run: str) -> int:
    return int(re.search(r"-s(\d+)-", run).group(1))  # type: ignore[union-attr]


def _own(agent: str, run: str) -> tuple[str, str, int | None] | None:
    """(experiment, arm, local cycle or None for final) if `agent` is the run's own model."""
    if run.startswith("A-dagger_r1_eval-"):
        return ("A/04", DAGGER[agent], None) if agent in DAGGER else None
    arm = _arm(run)
    if arm is None:
        return None
    if m := re.search(r"_ppo_c(\d+)$", agent):
        return arm[0], arm[1], int(m.group(1))
    if (agent.endswith("_ppo") and arm[0] != "A/02") or (
        arm[0] in ("C/02", "A/02") and agent.endswith(f"_s{_seed(run)}_transformer")
    ):
        return arm[0], arm[1], None
    return None


def _role(agent: str) -> str:
    if agent in ("start_transformer", "a02_transformer"):
        return "start"
    if agent == "c01_final":
        return "C/01 final"
    if agent.startswith("gbdt_"):
        return "teacher"
    if agent.startswith("heldout_"):
        return "held-out"
    if agent.endswith("_rule"):
        return "rule bot"
    return "other"


def export(runs_dir: Path, out: Path) -> dict[str, int]:
    """Write pairs.csv, ratings.csv and cycles.csv; return row counts."""
    pairs: list[dict[str, Any]] = []
    ratings: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    for run_dir in sorted(d for d in runs_dir.iterdir() if d.is_dir()):
        run = run_dir.name
        summary = run_dir / "eval/summary.json"
        if summary.exists() and (run.startswith("A-dagger_r1_eval-") or _arm(run)):
            s = json.loads(summary.read_text())
            seed = _seed(run)
            experiment, arm_label, start = _arm(run) or ("A/04", "A/04 eval", 0)
            cycles_file = run_dir / "ppo/cycles.json"
            final_cycle = (
                start + int(json.loads(cycles_file.read_text())[-1]["cycle"])
                if cycles_file.exists()
                else None
            )
            for r in s["ratings"]:
                own = _own(r["agent"], run)
                role, cycle = _role(r["agent"]), None
                if own is not None:
                    experiment_r, arm_r = own[0], own[1]
                    role = "own final" if own[2] is None else "own checkpoint"
                    cycle = final_cycle if own[2] is None else start + own[2]
                else:
                    experiment_r, arm_r = experiment, arm_label
                    if r["agent"] == "c01_final":
                        cycle = 11
                    elif role == "start":
                        cycle = -1
                ratings.append(
                    dict(
                        run=run,
                        experiment=experiment_r,
                        arm=arm_r,
                        seed=seed,
                        agent=r["agent"],
                        role=role,
                        cycle="" if cycle is None else cycle,
                        mu=round(r["mu"], 2),
                        lcb=round(r["lcb"], 2),
                        ucb=round(r["interval95"][1], 2),
                        games=r["games"],
                    )
                )
            for p in s["pairs"]:
                own = _own(p["agent"], run)
                if own is None or own[2] is not None:
                    continue
                opponent = p["opponent"]
                if _own(opponent, run) or opponent in DAGGER:
                    continue  # own checkpoints / the other DAgger arm
                pairs.append(
                    dict(
                        run=run,
                        experiment=own[0],
                        arm=own[1],
                        seed=seed,
                        opponent=OPPONENT.get(opponent, opponent),
                        games=p["games"],
                        wins=p["wins"],
                        draws=p["draws"],
                        losses=p["losses"],
                        win_rate=round(p["win_rate"], 4),
                        ci_low=round(p["ci95"][0], 4),
                        ci_high=round(p["ci95"][1], 4),
                    )
                )
        arm = _arm(run)
        if arm and (run_dir / "ppo").exists():
            for cycle_dir in sorted((run_dir / "ppo").glob("cycle-*")):
                m = cycle_metrics(cycle_dir)
                if "health/status" not in m:
                    continue
                row: dict[str, Any] = dict(
                    run=run,
                    experiment=arm[0],
                    arm=arm[1],
                    seed=_seed(run),
                    cycle=int(m["cycle"]),
                    global_cycle=arm[2] + int(m["cycle"]),
                )
                for key, name in CYCLE_FIELDS.items():
                    v = m.get(key)
                    row[name] = (
                        STATUS.get(v, "")
                        if name == "health_status" and v is not None
                        else ("" if v is None else round(v, 6))
                    )
                cycles.append(row)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pairs", pairs), ("ratings", ratings), ("cycles", cycles)):
        with (out / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return dict(pairs=len(pairs), ratings=len(ratings), cycles=len(cycles))


def main() -> None:
    """Export `reports/runs` to `reports/results`."""
    print(export(ROOT / "reports/runs", ROOT / "reports/results"))


if __name__ == "__main__":
    main()
