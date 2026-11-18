"""`ptcg run <config> --seed N`: run pipeline stages by wrapping the existing, validated scripts.

Stages (in order, each optional): data → gbdt → distill → ppo → eval.
Every stage records inputs hash, outputs, seeds and resources in runs/<id>/manifest.json,
is skipped when already done with identical inputs, and failed correctness gates stop
all downstream stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ptcg_exp import config as config_mod
from ptcg_exp import slurm
from ptcg_exp.evaluate import evaluate
from ptcg_exp.seeds import derive
from ptcg_exp.subsets import make_subset
from ptcg_exp.tracking import JSON, ROOT, Run, sha256

L = "src/agents/mega_lucario"
STAGES = ("data", "gbdt", "distill", "ppo", "eval")


class GateError(RuntimeError):
    """A correctness gate failed; downstream stages must not run."""


def _py(script: str, *args: object) -> list[str]:
    """Build a python command for a repository script."""
    return [sys.executable, str(ROOT / script), *map(str, args)]


def _sh(cmd: list[str], log: Path) -> None:
    """Run a command, tee its output to a log file, and raise on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        out.write(f"$ {' '.join(cmd)}\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}); see {log}")


def archive(directory: Path) -> Path:
    """Replace a directory by a verified <dir>.tar.zst (file count checked) and return it."""
    target = directory.with_name(directory.name + ".tar.zst")
    files = sum(1 for p in directory.rglob("*") if p.is_file())
    subprocess.run(
        f'tar -C "{directory.parent}" -cf - "{directory.name}" | zstd -q -T0 -9 -f -o "{target}"',
        shell=True,
        check=True,
    )
    listed = subprocess.run(
        f'zstd -dc "{target}" | tar -tf -', shell=True, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    if sum(1 for x in listed if not x.endswith("/")) != files:
        raise RuntimeError(f"Archive file count mismatch for {directory}")
    target.with_name(target.name + ".sha256").write_text(f"{sha256(target)}  {target.name}\n")
    shutil.rmtree(directory)
    return target


def _hash(*parts: object) -> str:
    """Stable hash of stage inputs."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class Pipeline:
    """Stage implementations for one (config, seed) run."""

    def __init__(self, run: Run) -> None:
        """Bind the run and its config."""
        self.run = run
        self.cfg = run.cfg
        self.seed = run.seed
        self.dir = run.dir
        self.out: dict[str, str] = dict(run.manifest.get("outputs", {}))

    def seed_of(self, stage: str, name: str) -> int:
        """Derived, recorded seed for (stage, name)."""
        s = derive(self.seed, stage, name)
        self.run.manifest.setdefault("seeds", {}).setdefault(stage, {})[name] = s
        return s

    def done(self, stage: str, inputs: str, **outputs: str) -> None:
        """Record outputs and completion of a stage."""
        self.out.update(outputs)
        self.run.manifest["outputs"] = self.out
        self.run.record_stage(stage, inputs, outputs=outputs)

    # ---- stages -----------------------------------------------------------------
    def data(self) -> None:
        """Balanced subset of existing replays, or fresh teacher replays."""
        d = self.cfg.stages.data
        assert d is not None
        inputs = _hash(d.model_dump(), self.seed)
        if self.run.stage_done("data", inputs):
            return
        if d.collect_games_per_opponent:
            replays = self.dir / "data/replays"
            cmd = _py(
                f"{L}/scripts/collect_replays.py",
                "--output-dir",
                replays,
                "--games-per-opponent",
                d.collect_games_per_opponent,
                "--teacher",
                d.teacher,
                "--seed",
                self.seed_of("data", "collect"),
                "--workers",
                self.cfg.resources.cpus,
                "--resume",
            )
            _sh(cmd, self.dir / "data/collect.log")
        elif d.per_dir:
            replays = ROOT / Path(d.replay_dir).parent / "scale" / f"games_{d.per_dir * 8}"
            if not (replays / "subset.json").exists():
                make_subset(ROOT / d.replay_dir, replays, d.per_dir)
        else:
            replays = ROOT / d.replay_dir
        subset = replays / "subset.json"
        self.done(
            "data",
            inputs,
            replays=str(replays),
            replays_manifest_sha256=str(sha256(subset) or sha256(replays / "manifest.json")),
        )

    def gbdt(self) -> None:
        """GBDT extract → train → audit gate (C++ scores ≡ Python within 1e-10)."""
        g = self.cfg.stages.gbdt
        assert g is not None
        inputs = _hash(g.model_dump(), self.out.get("replays_manifest_sha256"), self.seed)
        if self.run.stage_done("gbdt", inputs):
            return
        base = self.dir / "gbdt"
        feats, model = base / "features", base / "model"
        log = base / "gbdt.log"
        _sh(
            _py(
                f"{L}/gbdt/scripts/extract.py",
                "--replay-dir",
                self.out["replays"],
                "--output-dir",
                feats,
                "--validation-fraction",
                g.validation_fraction,
                "--test-fraction",
                g.test_fraction,
                "--seed",
                self.seed_of("gbdt", "split"),
            ),
            log,
        )
        _sh(
            _py(
                f"{L}/gbdt/scripts/train.py",
                "--feature-dir",
                feats,
                "--output-dir",
                model,
                "--threads",
                g.threads,
                "--seed",
                self.seed_of("gbdt", "train"),
                "--profile",
                g.profile,
            ),
            log,
        )
        model_file = model / "model.gbdt"
        # train.py prints one JSON summary line (model_id, best_iteration, metrics) to stdout.
        summary: JSON = next(
            (
                json.loads(x)
                for x in reversed(log.read_text().splitlines())
                if x.startswith('{"model_id"')
            ),
            {},
        )
        self.run.log(
            "gbdt", **{k: v for k, v in _flat(summary).items() if isinstance(v, int | float)}
        )
        if g.audit:
            played = base / "audit_replays"
            _sh(
                _py(
                    f"{L}/scripts/collect_replays.py",
                    "--output-dir",
                    played,
                    "--games",
                    1,
                    "--teacher",
                    "mega_lucario_gbdt",
                    "--model",
                    model_file,
                    "--seed",
                    self.seed_of("gbdt", "audit"),
                    "--workers",
                    8,
                ),
                log,
            )
            report = base / "audit.json"
            try:
                _sh(
                    _py(
                        f"{L}/gbdt/scripts/audit.py",
                        "--model",
                        model_file,
                        "--replay-dir",
                        played,
                        "--output",
                        report,
                        "--feature-dir",
                        feats,
                        "--probe",
                        ROOT / "build/gbdt_probe",
                    ),
                    log,
                )
                passed = True
            except RuntimeError:
                passed = False
            self.run.record_gate("gbdt_audit_1e-10", passed, report=str(report))
            if not passed:
                raise GateError("GBDT audit failed")
        self.done(
            "gbdt",
            inputs,
            gbdt_model=str(model_file),
            gbdt_model_sha256=str(sha256(model_file)),
            gbdt_features=str(feats),
        )

    def collect_teacher_games(
        self, replays: Path, teacher: str, log: Path, workers: int = 0
    ) -> None:
        """Collect GBDT-teacher games for distillation (resumable: --resume continues a prefix)."""
        c = self.cfg.stages.distill
        assert c is not None
        _sh(
            _py(
                f"{L}/scripts/collect_replays.py",
                "--output-dir",
                replays,
                "--games-per-opponent",
                c.teacher_games_per_opponent,
                "--teacher",
                "mega_lucario_gbdt",
                "--model",
                teacher,
                "--seed",
                self.seed_of("distill", "teacher_games"),
                "--workers",
                workers or self.cfg.resources.cpus,
                "--resume",
            ),
            log,
        )

    def prefetch(self) -> None:
        """CPU-only: collect distillation teacher games ahead of a GPU job (same seeds and path)."""
        c = self.cfg.stages.distill
        assert c is not None
        teacher = str(ROOT / c.teacher_model) if c.teacher_model else self.out["gbdt_model"]
        base = self.dir / "distill"
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", self.cfg.resources.cpus))
        self.collect_teacher_games(base / "teacher_replays", teacher, base / "distill.log", workers)
        self.run.save()

    def distill(self) -> None:
        """GBDT-teacher replays → Transformer extract/train/export → validate gate → canary set."""
        c = self.cfg.stages.distill
        assert c is not None
        teacher = str(ROOT / c.teacher_model) if c.teacher_model else self.out["gbdt_model"]
        inputs = _hash(c.model_dump(), sha256(Path(teacher)), self.seed)
        if self.run.stage_done("distill", inputs):
            return
        base = self.dir / "distill"
        log = base / "distill.log"
        replays, feats, model = base / "teacher_replays", base / "features", base / "model"
        if c.features_from:
            feats = ROOT / c.features_from.format(seed=self.seed)
            if not (feats / "manifest.json").exists():
                raise FileNotFoundError(f"features_from has no manifest.json: {feats}")
            self.run.manifest["features_from_manifest_sha256"] = sha256(feats / "manifest.json")
        else:
            self.collect_teacher_games(replays, teacher, log)
            _sh(
                _py(
                    f"{L}/transformer/scripts/extract.py",
                    "--replay-dir",
                    replays,
                    "--model",
                    teacher,
                    "--output-dir",
                    feats,
                    "--workers",
                    self.cfg.resources.cpus,
                    "--seed",
                    self.seed_of("distill", "split"),
                ),
                log,
            )
            if not c.keep_teacher_replays:
                archive(replays)
        train = [
            "--feature-dir",
            feats,
            "--output-dir",
            model,
            "--epochs",
            c.epochs,
            "--batch-size",
            c.batch_size,
            "--learning-rate",
            c.learning_rate,
            "--temperature",
            c.temperature,
            "--hard-weight",
            c.hard_weight,
            "--value-weight",
            c.value_weight,
            "--device",
            c.device,
            "--seed",
            self.seed_of("distill", "train"),
        ]
        if c.max_updates:
            train += ["--max-updates", c.max_updates]
        _sh(_py(f"{L}/transformer/scripts/train.py", *train), log)
        _sh(
            _py(
                f"{L}/transformer/scripts/export.py",
                "--checkpoint",
                model / "model.pt",
                "--output",
                model / "model.bin",
            ),
            log,
        )
        try:
            _sh(
                _py(
                    f"{L}/transformer/scripts/validate.py",
                    "--checkpoint",
                    model / "model.pt",
                    "--model",
                    model / "model.bin",
                    "--teacher-model",
                    teacher,
                    "--output-dir",
                    base / "validation",
                    "--feature-dir",
                    feats,
                    "--workers",
                    min(8, self.cfg.resources.cpus),
                ),
                log,
            )
            passed = True
        except RuntimeError:
            passed = False
        self.run.record_gate("transformer_validate_2e-5", passed)
        if not passed:
            raise GateError("Transformer validation failed")
        _sh(
            [
                sys.executable,
                "-m",
                "ptcg_exp.canary",
                "build",
                "--feature-dir",
                str(feats),
                "--output",
                str(base / "canary.npz"),
            ],
            log,
        )
        self.done(
            "distill",
            inputs,
            transformer_checkpoint=str(model / "model.pt"),
            transformer_bin=str(model / "model.bin"),
            canary=str(base / "canary.npz"),
            transformer_sha256=str(sha256(model / "model.pt")),
        )

    def ppo(self) -> None:
        """PPO cycles; each cycle gets canary drift metrics and a health.json gate."""
        p = self.cfg.stages.ppo
        assert p is not None
        ckpt = (
            str(ROOT / p.checkpoint.format(seed=self.seed))
            if p.checkpoint
            else self.out["transformer_checkpoint"]
        )
        gbdt = str(ROOT / p.gbdt_model) if p.gbdt_model else self.out["gbdt_model"]
        inputs = _hash(p.model_dump(), sha256(Path(ckpt)), sha256(Path(gbdt)), self.seed)
        if self.run.stage_done("ppo", inputs):
            return
        base = self.dir / "ppo"
        log = self.dir / "ppo.log"  # outside base: run_cycle.py requires an empty output dir
        from ptcg_exp import health  # numpy: not importable on the Eden login node

        h = self.cfg.health
        canary = str(ROOT / p.canary.format(seed=self.seed)) if p.canary else self.out.get("canary")
        current, prev_cycle = ckpt, None
        statuses: list[str] = []
        # One cycle per run_cycle.py call: a health gate after every cycle, per-cycle cleanup of
        # the raw rollouts, and resumption at cycle granularity within the 1-day job limit.
        for i in range(p.cycles):
            cycle = base / f"cycle-{i:03d}"
            finished = (cycle / "health.json").exists() and (cycle / "models/model.pt").exists()
            if not finished:
                if cycle.exists():
                    shutil.rmtree(cycle)  # partial cycle from an interrupted job
                self._ppo_cycle(i, current, base, gbdt, log)
                if canary:
                    args = [
                        "--canary",
                        canary,
                        "--checkpoint",
                        str(cycle / "models/model.pt"),
                        "--output",
                        str(cycle / "canary.json"),
                    ]
                    if prev_cycle is not None:
                        args += ["--previous-logits", str(prev_cycle / "canary.npy")]
                    _sh([sys.executable, "-m", "ptcg_exp.canary", "score", *args], log)
                report = health.check_cycle(cycle, h, prev_cycle)
                self.run.log("ppo_health", cycle=cycle.name, status=report["status"])
                if not p.keep_rollouts:
                    self._drop_rollout(cycle / "rollout")
            status = json.loads((cycle / "health.json").read_text())["status"]
            statuses.append(status)
            if status == "fail":
                self.run.record_gate("ppo_health", False, statuses=statuses)
                raise GateError(f"PPO health failed at {cycle.name}: {statuses}")
            current, prev_cycle = str(cycle / "models/model.pt"), cycle
        self.run.record_gate("ppo_health", True, statuses=statuses)
        last = base / f"cycle-{p.cycles - 1:03d}"
        curve = {
            f"ppo_c{k}_bin": str(base / f"cycle-{k:03d}/models/model.bin")
            for k in p.eval_cycles
            if k < p.cycles - 1
        }
        self.done(
            "ppo",
            inputs,
            ppo_bin=str(last / "models/model.bin"),
            ppo_checkpoint=str(last / "models/model.pt"),
            **curve,
        )

    def _ppo_cycle(self, i: int, checkpoint: str, base: Path, gbdt: str, log: Path) -> None:
        """Run exactly one PPO cycle (index i) with run_cycle.py, starting from `checkpoint`."""
        p = self.cfg.stages.ppo
        assert p is not None
        _sh(
            _py(
                f"{L}/ppo/scripts/run_cycle.py",
                "--checkpoint",
                checkpoint,
                "--output-dir",
                base,
                "--start-cycle",
                i,
                "--cycles",
                1,
                "--gbdt-model",
                gbdt,
                "--games",
                p.games,
                "--updates",
                p.updates,
                "--batch-size",
                p.batch_size,
                "--workers",
                p.workers,
                "--extract-workers",
                p.workers,
                "--audit-threads",
                min(8, p.workers),
                "--device",
                p.device,
                "--seed",
                self.seed_of("ppo", "arena"),
                "--sample-seed",
                self.seed_of("ppo", "sample"),
                "--update-seed",
                self.seed_of("ppo", "update") % 1_000_000,
                "--kl-target",
                self.cfg.health.kl_target,
            ),
            log,
        )

    @staticmethod
    def _drop_rollout(rollout: Path) -> None:
        """Archive raw replays; delete regenerable arrays (sha256 kept in rollout/manifest.json)."""
        if (rollout / "replays").is_dir():
            archive(rollout / "replays")
        for name in ("rollout.npz", "decisions.json"):
            (rollout / name).unlink(missing_ok=True)

    def eval(self) -> None:
        """Fixed protocol over this run's models plus configured extras."""
        e = self.cfg.stages.eval
        assert e is not None
        if not self.run.gates_ok():
            raise GateError("Refusing to evaluate: a correctness gate failed")
        models = {
            f"{self.cfg.name}_s{self.seed}_{k.rsplit('_', 1)[0]}": v
            for k, v in self.out.items()
            if k in ("gbdt_model", "transformer_bin", "ppo_bin")
            or (k.startswith("ppo_c") and k.endswith("_bin"))
        }
        models.update({k: str(ROOT / v.format(seed=self.seed)) for k, v in e.extra.items()})
        inputs = _hash(e.model_dump(), {k: sha256(Path(v)) for k, v in models.items()})
        if self.run.stage_done("eval", inputs):
            return
        summary = evaluate(
            models, self.dir / "eval", e.games_per_pair, e.workers, e.protocol, e.include_rule_bots
        )
        for r in summary["ratings"]:
            self.run.log("eval", agent=r["agent"], mu=r["mu"], sigma=r["sigma"], lcb=r["lcb"])
        self.done("eval", inputs, eval_summary=str(self.dir / "eval/summary.json"))


def _flat(d: object, prefix: str = "") -> JSON:
    """Flatten nested dicts for metric logging."""
    if not isinstance(d, dict):
        return {prefix.rstrip("."): d}
    out: JSON = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}."))
    return out


def run(config_path: Path, seed: int, stages: list[str] | None = None) -> Run:
    """Run the selected (or all configured) stages for one seed."""
    cfg = config_mod.load(config_path)
    r = Run(cfg, seed)
    pipe = Pipeline(r)
    wanted = [
        s for s in STAGES if getattr(cfg.stages, s) is not None and (not stages or s in stages)
    ]
    if stages and "prefetch" in stages:
        wanted = ["prefetch"]
    for stage in wanted:
        print(f"[ptcg] {r.id}: stage {stage}", flush=True)
        getattr(pipe, stage)()
    print(f"[ptcg] {r.id}: done → {r.dir}", flush=True)
    return r


def main() -> None:
    """CLI: run locally, or submit one Slurm array job per config (seeds = array tasks)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument(
        "--seed", type=int, help="Run one seed locally (default: all seeds, sequentially)"
    )
    p.add_argument("--stages", type=lambda s: s.split(","), help=f"Subset of {','.join(STAGES)}")
    p.add_argument("--submit", action="store_true", help="Submit to Slurm instead of running")
    p.add_argument("--dependency", help="Slurm dependency, e.g. afterok:123")
    a = p.parse_args()
    cfg = config_mod.load(a.config)
    seeds = [a.seed] if a.seed is not None else cfg.seeds
    if a.submit:
        job = slurm.submit(cfg, a.config.resolve(), ROOT, seeds, a.stages, a.dependency)
        print(f"Submitted {cfg.owner}/{cfg.name} seeds={seeds} as job {job}")
        return
    for s in seeds:
        run(a.config, s, a.stages)


if __name__ == "__main__":
    main()
