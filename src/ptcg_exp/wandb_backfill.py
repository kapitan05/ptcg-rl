"""Upload saved run results to W&B: past runs, and offline runs written on the cluster.

Reads the small per-run files kept in `reports/runs/<run_id>/` (pulled from `runs/`):
manifest.json, ppo/cycle-*/{models/training.json,health.json,canary.json}, eval/summary.json.
Each run becomes one W&B run with the same stable id as live logging (`tracking.wandb_id`).

    uv run wandb login                                  # once; stores the key outside the repo
    PYTHONPATH=src uv run python -m ptcg_exp.wandb_backfill            # all runs in reports/runs
    PYTHONPATH=src uv run python -m ptcg_exp.wandb_backfill --runs C-ppo_baseline-s0-ba1b0384
    PYTHONPATH=src uv run python -m ptcg_exp.wandb_backfill --sync-offline reports/runs

Logged: config (+ seed, git, contract ids), PPO per-cycle metrics with step = cycle (the same
keys as live logging, `ppo_cycle/*`), gates, and evaluation tables (BT ratings, pairwise win
rates with Wilson CIs) with the run's own final rating in the summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ptcg_exp.cycle_metrics import cycle_metrics
from ptcg_exp.tracking import ROOT, wandb_id


def _own_agent(ratings: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The run's final learned agent in its eval roster (`*_ppo`, else the first `*_s<seed>`)."""
    for r in ratings:
        if str(r["agent"]).endswith("_ppo"):
            return r
    return None


def upload(run_dir: Path, project: str, entity: str | None) -> None:
    """Create or resume the W&B run for one saved run directory and log everything in it."""
    import wandb

    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest.get("config", {})
    run = wandb.init(
        project=project,
        entity=entity,
        id=wandb_id(run_dir.name),
        name=run_dir.name,
        resume="allow",
        group=cfg.get("question"),
        tags=[t for t in (cfg.get("owner"), cfg.get("profile"), "backfill") if t],
        config=cfg
        | dict(
            seed=manifest.get("seed"),
            config_hash=manifest.get("config_hash"),
            git_sha=(manifest.get("git") or {}).get("sha"),
            contract_id=manifest.get("contract_id"),
            schema_id=manifest.get("schema_id"),
        ),
        reinit=True,
    )
    cycles = sorted((run_dir / "ppo").glob("cycle-*")) if (run_dir / "ppo").exists() else []
    for cycle in cycles:
        metrics = cycle_metrics(cycle)
        step = int(metrics["cycle"])
        run.log({f"ppo_cycle/{k}": v for k, v in metrics.items()}, step=step)
    for gate, g in (manifest.get("gates") or {}).items():
        run.summary[f"gate/{gate}"] = bool(g.get("passed"))
    summary = run_dir / "eval/summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        ratings = s.get("ratings", [])
        run.log(
            {
                "eval/ratings": wandb.Table(
                    columns=["agent", "mu", "sigma", "lcb", "games"],
                    data=[[r["agent"], r["mu"], r["sigma"], r["lcb"], r["games"]] for r in ratings],
                ),
                "eval/pairs": wandb.Table(
                    columns=["agent", "opponent", "games", "win_rate", "ci_low", "ci_high"],
                    data=[
                        [p["agent"], p["opponent"], p["games"], p["win_rate"], *p["ci95"]]
                        for p in s.get("pairs", [])
                    ],
                ),
            },
            step=len(cycles),
        )
        if (own := _own_agent(ratings)) is not None:
            run.summary["eval/mu"] = own["mu"]
            run.summary["eval/lcb"] = own["lcb"]
            for p in s.get("pairs", []):
                if p["agent"] == own["agent"]:
                    run.summary[f"eval/win_rate/{p['opponent']}"] = p["win_rate"]
    run.finish()


def main() -> None:
    """Upload every saved run (or the named ones), or sync offline cluster runs."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=ROOT / "reports/runs")
    p.add_argument("--runs", nargs="*", help="run IDs (default: all with a manifest)")
    p.add_argument("--project", default="ptcg-rl")
    p.add_argument("--entity", default=None, help="W&B team; default: your personal account")
    p.add_argument(
        "--sync-offline",
        type=Path,
        help="sync every wandb/offline-run-* directory found below this path, then exit",
    )
    a = p.parse_args()
    if a.sync_offline:
        dirs = sorted(str(d) for d in a.sync_offline.rglob("offline-run-*") if d.is_dir())
        if dirs:
            cmd = [sys.executable, "-m", "wandb", "sync", "-p", a.project]
            cmd += ["-e", a.entity] if a.entity else []
            subprocess.run([*cmd, *dirs], check=True)
        print(f"synced {len(dirs)} offline runs")
        return
    names = a.runs or sorted(d.name for d in a.runs_dir.iterdir() if (d / "manifest.json").exists())
    for name in names:
        print(f"[backfill] {name}", flush=True)
        upload(a.runs_dir / name, a.project, a.entity)


if __name__ == "__main__":
    main()
