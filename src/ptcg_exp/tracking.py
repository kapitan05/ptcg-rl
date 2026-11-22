"""Run provenance: manifest.json (git, config, seeds, hashes, hardware) and metrics.jsonl."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ptcg_exp.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[2]

# A parsed JSON document (run outputs, summaries). Values are untyped by nature; Any is
# deliberate here and confined to this alias.
JSON = dict[str, Any]


def sha256(path: Path) -> str | None:
    """Return a file's sha256, or None when it does not exist."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


CODE_DIRS = ("src", "configs")


def code_sha256(root: Path = ROOT) -> str:
    """Hash every .py/.yaml/.cpp/.h file under src/ and configs/ (path + content, sorted).

    Unlike the git SHA, this identifies the exact code that ran even when it was synced
    with uncommitted changes (rsync to the cluster).
    """
    h = hashlib.sha256()
    for d in CODE_DIRS:
        for f in sorted((root / d).rglob("*")):
            if f.suffix in {".py", ".yaml", ".cpp", ".h", ".hpp"} and "__pycache__" not in f.parts:
                h.update(str(f.relative_to(root)).encode())
                h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def git_state() -> JSON:
    """Return the commit SHA and whether tracked source files differ from it."""

    def git(*args: str) -> str:
        """Run git in the repository root and return stripped stdout ('' on failure)."""
        try:
            out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        except OSError:
            return ""
        return out.stdout.strip() if out.returncode == 0 else ""

    return dict(
        sha=git("rev-parse", "HEAD") or None,
        branch=git("rev-parse", "--abbrev-ref", "HEAD") or None,
        dirty=bool(git("status", "--porcelain", "--", "src", "configs", "pyproject.toml")),
        code_sha256=code_sha256(),
    )


def contract_ids() -> dict[str, str | None]:
    """Return the feature/model contract hashes without importing heavy modules."""
    ids: dict[str, str | None] = {}
    try:
        from agents.mega_lucario.gbdt.features import SCHEMA_ID
        from agents.mega_lucario.transformer.features import CONTRACT_ID

        ids.update(schema_id=SCHEMA_ID, contract_id=CONTRACT_ID)
    except Exception:  # noqa: BLE001 - provenance must never crash a run
        ids.update(schema_id=None, contract_id=None)
    return ids


def hardware() -> JSON:
    """Describe host, Slurm allocation and visible GPUs."""
    gpus = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpus = [g.strip() for g in out.stdout.splitlines() if g.strip()] or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return dict(
        host=socket.gethostname(),
        platform=platform.platform(),
        python=platform.python_version(),
        cpus=os.cpu_count(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
        slurm_cpus=os.environ.get("SLURM_CPUS_PER_TASK"),
        gpus=gpus,
    )


def run_id(cfg: ExperimentConfig, seed: int) -> str:
    """Return `<owner>-<name>-s<seed>-<confighash8>`; stable across reruns of the same config."""
    return f"{cfg.owner}-{cfg.name}-s{seed}-{cfg.config_hash()[:8]}"


class Run:
    """Directory-backed record of one (config, seed) run."""

    def __init__(self, cfg: ExperimentConfig, seed: int, root: Path | None = None) -> None:
        """Create or reopen the run directory and its manifest."""
        self.cfg = cfg
        self.seed = seed
        self.id = run_id(cfg, seed)
        self.dir = (root or ROOT / "runs") / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.metrics_path = self.dir / "metrics.jsonl"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            self.manifest = dict(
                run_id=self.id,
                config=cfg.model_dump(mode="json"),
                config_hash=cfg.config_hash(),
                seed=seed,
                created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                stages={},
                gates={},
            )
        self.manifest.update(
            git=git_state(),
            uv_lock_sha256=sha256(ROOT / "uv.lock"),
            arena_sha256=sha256(ROOT / "build/arena_cpp"),
            engine_manifest_sha256=sha256(ROOT / "src/tools/fetch_cg_engine.py"),
            hardware=hardware(),
            **contract_ids(),
        )
        self.save()

    def save(self) -> None:
        """Atomically write the manifest."""
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, default=str) + "\n")
        tmp.replace(self.manifest_path)

    def log(self, stage: str, **metrics: object) -> None:
        """Append one metrics record (and mirror it to W&B when enabled)."""
        record = dict(t=time.time(), stage=stage, **metrics)
        with self.metrics_path.open("a") as out:
            out.write(json.dumps(record, default=str) + "\n")
        _wandb_log(self, stage, metrics)

    def stage_done(self, stage: str, inputs_hash: str) -> bool:
        """Return True when a stage already completed with identical inputs."""
        s = self.manifest["stages"].get(stage, {})
        return s.get("status") == "done" and s.get("inputs_hash") == inputs_hash

    def record_stage(self, stage: str, inputs_hash: str, **info: object) -> None:
        """Mark a stage complete with resource usage and outputs."""
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.manifest["stages"][stage] = dict(
            status="done",
            inputs_hash=inputs_hash,
            finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            children_max_rss_mb=round(
                usage.ru_maxrss / (1024 if os.uname().sysname != "Darwin" else 1024**2), 1
            ),
            **info,
        )
        self.save()

    def record_gate(self, gate: str, passed: bool, **details: object) -> None:
        """Record a correctness gate result; failed gates block downstream stages."""
        self.manifest["gates"][gate] = dict(passed=passed, **details)
        self.save()
        self.log("gate", gate=gate, passed=passed, **details)

    def gates_ok(self) -> bool:
        """Return True when every recorded gate passed."""
        return all(g["passed"] for g in self.manifest["gates"].values())


_WANDB: JSON = {}


def _wandb_log(run: Run, stage: str, metrics: JSON) -> None:
    """Log numeric metrics to W&B if PTCG_WANDB=1 (offline by default on the cluster)."""
    if os.environ.get("PTCG_WANDB") != "1":
        return
    # Tracking must never break an experiment: any W&B error is reported once and logging stops.
    if _WANDB.get(run.id) is False:
        return
    try:
        import wandb

        if run.id not in _WANDB:
            os.environ.setdefault("WANDB_MODE", "offline")
            _WANDB[run.id] = wandb.init(
                project="ptcg-rl",
                name=run.id,
                id=wandb_id(run.id),
                resume="allow",
                group=run.cfg.question,
                tags=[run.cfg.owner, run.cfg.profile],
                config=run.manifest["config"]
                | dict(seed=run.seed, config_hash=run.cfg.config_hash()),
                dir=str(run.dir),
                reinit=True,
            )
        numeric = {f"{stage}/{k}": v for k, v in metrics.items() if isinstance(v, int | float)}
        if numeric:
            _WANDB[run.id].log(numeric)
    except Exception as e:  # noqa: BLE001
        print(f"[ptcg] W&B disabled for {run.id}: {type(e).__name__}: {e}", flush=True)
        _WANDB[run.id] = False


def wandb_id(run_id: str) -> str:
    """Stable W&B run id for a run ID, so live logging and backfill write to the same run."""
    return hashlib.sha256(run_id.encode()).hexdigest()[:16]
