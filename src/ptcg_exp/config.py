"""Pydantic experiment configs: YAML with `extends:` inheritance and a stable config hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Profile = Literal["smoke", "full"]


class Strict(BaseModel):
    """Base model that rejects unknown keys, so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class Resources(Strict):
    """Slurm resources for one run (one seed)."""

    cpus: int = 16
    mem: str = "32G"
    time: str = "12:00:00"
    gpus: int = 0


class DataStage(Strict):
    """Teacher replay source, optionally reduced to a balanced subset."""

    replay_dir: str = "outputs/mega_lucario/rule_replays"
    per_dir: int | None = None  # games per opponent/seat dir; None = use replay_dir as is
    teacher: Literal["mega_lucario", "mega_lucario_gbdt"] = "mega_lucario"
    collect_games_per_opponent: int | None = None  # collect fresh replays when set


class GbdtStage(Strict):
    """GBDT extraction and LambdaRank training."""

    profile: Literal["quick", "standard", "search"] = "standard"
    threads: int = 16
    validation_fraction: float = 0.25
    test_fraction: float = 0.1
    audit: bool = True  # gate 4: C++ vs Python GBDT scores within 1e-10


class DaggerStage(Strict):
    """One DAgger round: new games labeled by the distill teacher, merged with base features.

    player=student: the student Transformer plays and the teacher relabels its states (DAgger).
    player=teacher: the teacher plays the same number of games (equal-label-budget BC control).
    Both arms use the same game seeds, so they are paired.
    """

    base_features: str  # feature dir this round adds to ("{seed}" substituted)
    player: Literal["student", "teacher"]
    student_model: str | None = None  # student .bin ("{seed}" substituted); required for student
    games_per_opponent: int = 625  # x 4 rule-bot opponents, both seats


class DistillStage(Strict):
    """Transformer distillation from a GBDT teacher."""

    teacher_model: str | None = None  # path to model.gbdt; None = this run's GBDT
    teacher_games_per_opponent: int = 1250
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 3e-4
    temperature: float = 1.0
    hard_weight: float = 0.1
    value_weight: float = 0.25
    device: Literal["cpu", "cuda"] = "cuda"
    max_updates: int | None = None
    # Reuse an extracted feature dir instead of collecting/extracting ("{seed}" is substituted).
    # Pairs an ablation with its baseline: same games, same split, only the changed knob differs.
    features_from: str | None = None
    keep_teacher_replays: bool = False  # False: archive raw replays as .tar.zst after extraction


class League(Strict):
    """Opponent pool for PPO beyond the repo default (plan P2 snapshot league, P3 PFSP).

    Each cycle trains vs. the default opponents plus frozen Transformer snapshots: the `recent`
    newest and `older` evenly spaced older ones, drawn from `seed_pool` (earlier runs' .bin files,
    "{seed}" substituted) followed by this run's own cycle checkpoints.
    """

    recent: int = 3
    older: int = 2
    seed_pool: list[str] = Field(default_factory=list)
    snapshot_weight: float = 1.0  # sampling weight of each snapshot (defaults weigh 1.0 each)
    # PFSP: weight_i *= max(floor, 1 - win_rate_i) ** power, from the previous cycle's
    # per-opponent win rate (opponents not seen yet: win rate 0.5).
    pfsp: bool = False
    pfsp_power: float = 1.0
    pfsp_floor: float = 0.2


class PpoStage(Strict):
    """PPO collect/train cycles started from a distilled checkpoint."""

    checkpoint: str | None = None  # None = this run's distilled model; "{seed}" is substituted
    gbdt_model: str | None = None
    cycles: int = 1
    games: int = 5000
    updates: int = 128
    batch_size: int = 1024
    workers: int = 16
    device: Literal["cpu", "cuda"] = "cuda"
    # Raw rollouts are ~43 GB per 5,000-game cycle (Eden quota: 200 GB). False: after a cycle's
    # health check, archive replays as .tar.zst and drop rollout.npz/decisions.json (their sha256
    # stays in rollout/manifest.json; collection is seeded, so they are regenerable).
    keep_rollouts: bool = False
    eval_cycles: list[int] = Field(default_factory=list)  # extra cycles to rate (learning curve)
    canary: str | None = None  # canary.npz when no distill stage ran ("{seed}" substituted)
    # In-loop early stop: end the epoch when a minibatch's approx KL > kl_early_stop × kl_target
    # (0 = off, the repo default).
    kl_early_stop: float = 0.0
    league: League | None = None


class EvalStage(Strict):
    """Arena evaluation protocol (docs/evaluation.md)."""

    protocol: str = "rating_v1"
    games_per_pair: int = 200
    workers: int = 32
    include_rule_bots: bool = True
    extra: dict[str, str] = Field(default_factory=dict)  # label -> model path ("{seed}" ok)


class HealthThresholds(Strict):
    """Pass/warn/fail thresholds for PPO health checks (src/ptcg_exp/health.py)."""

    kl_target: float = 0.004
    kl_fail_multiple: float = 4.0
    kl_low_multiple: float = 0.25
    clip_fraction_warn: float = 0.3
    ratio_p99_warn: float = 2.0
    grad_spike_warn: float = 20.0
    update_ratio_band: tuple[float, float] = (1e-5, 1e-2)
    entropy_drop_warn: float = 0.3
    ece_warn: float = 0.15
    seat_win_band: tuple[float, float] = (0.02, 0.98)
    length_shift_warn: float = 0.3
    canary_agreement_drop_warn: float = 0.10


class Stages(Strict):
    """Optional stage sections; a stage runs only if present (and selected)."""

    data: DataStage | None = None
    gbdt: GbdtStage | None = None
    dagger: DaggerStage | None = None
    distill: DistillStage | None = None
    ppo: PpoStage | None = None
    eval: EvalStage | None = None


class ExperimentConfig(Strict):
    """One experiment: hypothesis owner, stages, seeds and resources."""

    name: str
    owner: str
    question: str
    hypothesis: str = ""
    profile: Profile = "full"
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    resources: Resources = Field(default_factory=Resources)
    stages: Stages = Field(default_factory=Stages)
    health: HealthThresholds = Field(default_factory=HealthThresholds)

    def config_hash(self) -> str:
        """Return the sha256 of the canonical JSON form (independent of key order)."""
        data = self.model_dump(mode="json")
        ppo = data["stages"].get("ppo")
        if ppo is not None:  # fields added after C/01: omit at default so earlier run IDs hold
            for key, default in _PPO_LATE_DEFAULTS.items():
                if ppo.get(key) == default:
                    del ppo[key]
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


_PPO_LATE_DEFAULTS: dict[str, object] = {"kl_early_stop": 0.0, "league": None}


def _merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """Deep-merge two YAML mappings; override wins, nested dicts merge."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def load_raw(path: Path) -> dict[str, object]:
    """Load YAML and resolve the `extends:` chain relative to each file."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _merge(load_raw((path.parent / str(parent)).resolve()), data)


def load(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment config."""
    return ExperimentConfig.model_validate(load_raw(Path(path)))
