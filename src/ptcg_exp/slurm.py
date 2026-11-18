"""Render and submit Slurm jobs for `ptcg run` (Eden: account 22-stud-2627-z, partition student)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ptcg_exp.config import ExperimentConfig

ACCOUNT = os.environ.get("PTCG_SLURM_ACCOUNT", "22-stud-2627-z")
PARTITION = os.environ.get("PTCG_SLURM_PARTITION", "student")
TEMPLATE = """#!/bin/bash
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH -J {name}
#SBATCH -o {logs}/%x-%A_%a.out
{gres}{array}set -euo pipefail
cd {root}
export PYTHONPATH=src OMP_NUM_THREADS={cpus} CUBLAS_WORKSPACE_CONFIG=:4096:8
# Python is called directly: srun is not on PATH inside Eden batch jobs.
SEED=${{SLURM_ARRAY_TASK_ID:-{default_seed}}}
.venv/bin/python -m ptcg_exp.run {config} --seed "$SEED" {stages}
"""


def render(
    cfg: ExperimentConfig,
    config_path: Path,
    root: Path,
    seeds: list[int],
    stages: list[str] | None = None,
    max_parallel: int | None = None,
) -> str:
    """Return the sbatch script for a config; seeds become a job array (optionally throttled)."""
    r = cfg.resources
    return TEMPLATE.format(
        account=ACCOUNT,
        partition=PARTITION,
        cpus=r.cpus,
        mem=r.mem,
        time=r.time,
        name=f"{cfg.owner}-{cfg.name}"[:40],
        logs=root / "logs",
        gres=f"#SBATCH --gres=gpu:{r.gpus}\n" if r.gpus else "",
        array=(
            f"#SBATCH --array={','.join(map(str, seeds))}"
            f"{f'%{max_parallel}' if max_parallel else ''}\n"
            if len(seeds) > 1
            else ""
        ),
        root=root,
        default_seed=seeds[0],
        config=config_path,
        stages=f"--stages {','.join(stages)}" if stages else "",
    )


def submit(
    cfg: ExperimentConfig,
    config_path: Path,
    root: Path,
    seeds: list[int],
    stages: list[str] | None = None,
    dependency: str | None = None,
    max_parallel: int | None = None,
) -> str:
    """Write jobs/<owner>-<name>.sbatch, submit it, and return the Slurm job ID."""
    (root / "jobs").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    script = root / "jobs" / f"{cfg.owner}-{cfg.name}.sbatch"
    script.write_text(render(cfg, config_path, root, seeds, stages, max_parallel))
    cmd = (
        ["sbatch", "--parsable"]
        + ([f"--dependency={dependency}"] if dependency else [])
        + [str(script)]
    )
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout.strip().split(";")[0]
