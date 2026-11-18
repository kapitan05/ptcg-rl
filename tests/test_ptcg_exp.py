"""Tests for the experiment layer: configs, seeds, subsets, statistics, Slurm, health checks."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_exp import config, health, seeds, slurm, subsets  # noqa: E402
from ptcg_exp.evaluate import build_roster, pair_table, promote, wilson  # noqa: E402

# ---------------------------------------------------------------------------- config


def test_all_repo_configs_validate() -> None:
    """Every committed config must load (catches typos: unknown keys are rejected)."""
    paths = sorted((ROOT / "configs").rglob("*.yaml"))
    assert paths
    for p in paths:
        if p.parent.name == "base" and p.name == "full.yaml":
            continue  # partial base, only meaningful when extended
        config.load(p)


def test_extends_merges_and_hash_is_stable(tmp_path: Path) -> None:
    """Child overrides win, nested dicts merge, and key order doesn't change the hash."""
    (tmp_path / "base.yaml").write_text("resources: {cpus: 4, mem: 8G}\nseeds: [0]\n")
    (tmp_path / "a.yaml").write_text(
        "extends: base.yaml\nname: x\nowner: A\nquestion: q\nresources: {cpus: 16}\n"
    )
    (tmp_path / "b.yaml").write_text(
        "question: q\nowner: A\nname: x\nextends: base.yaml\nresources: {cpus: 16}\n"
    )
    a, b = config.load(tmp_path / "a.yaml"), config.load(tmp_path / "b.yaml")
    assert a.resources.cpus == 16 and a.resources.mem == "8G"
    assert a.config_hash() == b.config_hash()
    c = a.model_copy(update=dict(seeds=[1]))
    assert c.config_hash() != a.config_hash()


def test_unknown_key_rejected(tmp_path: Path) -> None:
    """A misspelled option must fail loudly instead of silently using a default."""
    (tmp_path / "bad.yaml").write_text(
        "name: x\nowner: A\nquestion: q\nstages: {gbdt: {thread: 4}}\n"
    )
    with pytest.raises(Exception):
        config.load(tmp_path / "bad.yaml")


# ---------------------------------------------------------------------------- seeds


def test_seed_derivation_stable_distinct_and_disjoint() -> None:
    """Derived seeds are reproducible, differ per name, and never meet the eval range."""
    assert seeds.derive(0, "gbdt", "split") == seeds.derive(0, "gbdt", "split")
    tree = seeds.seed_tree(0, {"gbdt": ["split", "train"], "ppo": ["arena", "sample"]})
    flat = [v for s in tree.values() for v in s.values()]
    assert len(set(flat)) == len(flat)
    assert all(seeds.TRAIN_RANGE[0] <= v < seeds.TRAIN_RANGE[1] for v in flat)
    e = seeds.eval_seed("rating_v1")
    assert seeds.EVAL_RANGE[0] <= e < seeds.EVAL_RANGE[1]
    assert e + 100 * 1_000_000 < 2**32
    seeds.assert_disjoint(flat, [e], span=10_000)
    with pytest.raises(ValueError):
        seeds.assert_disjoint([e - 5], [e], span=10)


# ---------------------------------------------------------------------------- subsets


def test_subset_is_balanced_nested_and_hashed(tmp_path: Path) -> None:
    """First-k per opponent/seat dir, numeric order, card metadata copied, hashes recorded."""
    src = tmp_path / "src"
    for d in ("iono-seat0", "iono-seat1"):
        (src / d).mkdir(parents=True)
        (src / d / "card_metadata.json").write_text("[]")
        for i in (0, 1, 2, 10):
            (src / d / f"game-{i}.jsonl").write_text(f"{d}{i}")
    m = subsets.make_subset(src, tmp_path / "k2", 2)
    assert m["games"] == 4
    names = sorted(p.name for p in (tmp_path / "k2/iono-seat0").glob("game-*"))
    assert names == ["game-0.jsonl", "game-1.jsonl"]  # numeric, not lexicographic (10 excluded)
    assert (tmp_path / "k2/iono-seat1/card_metadata.json").exists()
    assert all(len(e["sha256"]) == 64 for e in m["entries"])  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        subsets.make_subset(src, tmp_path / "k9", 9)


# ---------------------------------------------------------------------------- statistics


def test_wilson_interval() -> None:
    """Known values and edge cases of the Wilson score interval."""
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0


def test_pair_table_counts_both_seats(tmp_path: Path) -> None:
    """Results are attributed to the right agent regardless of seat; draws count half."""
    games = [
        dict(a="x", b="y", result=0, agent_errors=0, engine_errors=0),  # x (seat 0) wins
        dict(a="y", b="x", result=1, agent_errors=0, engine_errors=0),  # x (seat 1) wins
        dict(a="x", b="y", result=2, agent_errors=0, engine_errors=0),  # draw
    ]
    path = tmp_path / "games.jsonl"
    path.write_text("\n".join(json.dumps(g) for g in games))
    t = {(r["agent"], r["opponent"]): r for r in pair_table(path)}
    assert t[("x", "y")]["wins"] == 2 and t[("x", "y")]["draws"] == 1
    assert t[("x", "y")]["win_rate"] == pytest.approx(2.5 / 3)
    assert t[("y", "x")]["win_rate"] == pytest.approx(0.5 / 3)


def test_roster_and_promotion_rule() -> None:
    """Model files map to agents by suffix; promotion needs LCB ≥ champion mean, no regressions."""
    roster = build_roster({"g": "m.gbdt", "t": "m.bin"})
    assert (
        roster["g"]["agent"] == "mega_lucario_gbdt"
        and roster["t"]["agent"] == "mega_lucario_transformer"
    )
    assert "lucario_rule" in roster
    summary = dict(
        ratings=[dict(agent="new", mu=1100, lcb=1060), dict(agent="old", mu=1050, lcb=1030)],
        pairs=[
            dict(agent="new", opponent="iono", win_rate=0.70, ci95=[0.62, 0.77]),
            dict(agent="old", opponent="iono", win_rate=0.60, ci95=[0.52, 0.68]),
        ],
    )
    assert promote(summary, "new", "old")["promote"] is True
    summary["pairs"][0].update(win_rate=0.40, ci95=[0.33, 0.47])  # type: ignore[index]
    assert promote(summary, "new", "old")["promote"] is False


# ---------------------------------------------------------------------------- slurm


def test_slurm_render_uses_own_account_and_no_srun() -> None:
    """Rendered jobs use the user's account/partition, arrays for seeds, and call python directly."""
    cfg = config.load(ROOT / "configs/exp/A/01_gbdt_data_scale.yaml")
    text = slurm.render(cfg, Path("configs/x.yaml"), Path("/home/u/ptcg-rl"), [0, 1, 2])
    assert "#SBATCH -A 22-stud-2627-z" in text and "#SBATCH -p student" in text
    assert "#SBATCH --array=0,1,2" in text and "--mem=192G" in text
    assert "srun " not in text.replace("# Python is called directly: srun", "")
    assert "--export" not in text


# ---------------------------------------------------------------------------- health


def _cycle(tmp_path: Path, **over: object) -> Path:
    """Write a synthetic, healthy PPO cycle; keyword overrides inject faults."""
    c = tmp_path / "cycle-000"
    (c / "rollout").mkdir(parents=True)
    (c / "models").mkdir()
    (c / "validation").mkdir()
    (c / "validation/report.json").write_text("{}")
    rng = np.random.default_rng(0)
    games = [
        dict(opponent=o, learner_seat=s, result=int(rng.integers(0, 2)), steps=100)
        for o in ("iono", "dragapult")
        for s in (0, 1)
        for _ in range(30)
    ]
    rollout = dict(
        arena_results=games,
        temperature=over.get("temperature", 1.0),
        audit=dict(max_abs_errors=dict(logp=1e-7, value=1e-8, entropy=1e-7)),
        collection=dict(seed=1, sample_seed=2),
    )
    (c / "rollout/manifest.json").write_text(json.dumps(rollout))
    kl = float(over.get("kl", 0.004))  # type: ignore[arg-type]
    log = [
        dict(
            approximate_kl=kl,
            clip_fraction=0.08,
            ratio_p99=1.3,
            gradient_norm=1.0 + 0.1 * i,
            update_ratio=1e-4,
            loss=0.1,
            entropy=0.9,
            value_loss=0.3,
        )
        for i in range(8)
    ]
    training = dict(
        updates_log=log,
        early_stopped=False,
        parameter_delta_l2=0.5,
        heldout_global_approximate_kl=float(over.get("heldout_kl", 0.004)),  # type: ignore[arg-type]
        health_inputs=dict(
            raw_advantage=dict(mean=0.0, std=0.5, nonfinite=int(over.get("nan", 0))),  # type: ignore[arg-type]
            return_range=[-1.0, 1.0],
            explained_variance_before=0.3,
            explained_variance_after_heldout=0.35,
            value_ece_heldout=0.05,
            entropy_behavior_heldout=0.9,
            entropy_updated_heldout=0.85,
        ),
    )
    if over.get("no_hooks"):
        training.pop("updates_log")
    if over.get("no_heldout_kl"):
        training.pop("heldout_global_approximate_kl")
    (c / "models/training.json").write_text(json.dumps(training))
    (c / "canary.json").write_text(
        json.dumps(dict(teacher_agreement=0.8, illegal_pick_rate=float(over.get("illegal", 0.0))))
    )  # type: ignore[arg-type]
    return c


def _status(report: dict[str, object], name: str) -> str:
    """Status of one named check."""
    return next(c["status"] for c in report["checks"] if c["check"] == name)  # type: ignore[union-attr, index]


def test_health_healthy_cycle_passes(tmp_path: Path) -> None:
    """A clean synthetic cycle passes every check and allows promotion."""
    r = health.check_cycle(_cycle(tmp_path), config.HealthThresholds())
    assert r["status"] == "pass", [c for c in r["checks"] if c["status"] != "pass"]  # type: ignore[union-attr]
    assert (tmp_path / "cycle-000/health.json").exists()


@pytest.mark.parametrize(
    ("fault", "check"),
    [
        (dict(temperature=0.8), "rollout_temperature"),
        (dict(kl=0.05), "approx_kl_per_update"),
        (dict(heldout_kl=0.03), "heldout_kl"),
        (dict(nan=3), "gae_sanity"),
        (dict(illegal=0.01), "illegal_picks"),
        (dict(no_hooks=True), "health_hooks_present"),
        (dict(no_heldout_kl=True), "heldout_kl"),
    ],
)
def test_health_injected_faults_fail(tmp_path: Path, fault: dict[str, object], check: str) -> None:
    """Each injected fault is caught by its check and blocks promotion."""
    r = health.check_cycle(_cycle(tmp_path, **fault), config.HealthThresholds())
    assert _status(r, check) == "fail"
    assert r["promote_allowed"] is False


def test_health_small_step_warns(tmp_path: Path) -> None:
    """A held-out KL far below target is a warning (LR probably too small), not a failure."""
    r = health.check_cycle(_cycle(tmp_path, heldout_kl=1e-5), config.HealthThresholds())
    assert _status(r, "heldout_kl") == "warn" and r["promote_allowed"] is True


def test_code_sha256_tracks_content(tmp_path: Path) -> None:
    """The code hash changes with file content, not with mtime or unrelated files."""
    from ptcg_exp.tracking import code_sha256

    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    f = tmp_path / "src" / "a.py"
    f.write_text("x = 1\n")
    h1 = code_sha256(tmp_path)
    (tmp_path / "src" / "notes.txt").write_text("ignored")
    assert code_sha256(tmp_path) == h1
    f.write_text("x = 2\n")
    assert code_sha256(tmp_path) != h1


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd not installed")
def test_archive_replaces_dir_with_verified_tarball(tmp_path: Path) -> None:
    """archive() writes <dir>.tar.zst + .sha256, keeps every file, and removes the dir."""
    from ptcg_exp.run import archive

    d = tmp_path / "replays" / "iono-seat0"
    d.mkdir(parents=True)
    for i in range(5):
        (d / f"game-{i}.json").write_text(json.dumps({"game": i}))
    out = archive(tmp_path / "replays")
    assert out.name == "replays.tar.zst" and out.exists()
    assert (tmp_path / "replays.tar.zst.sha256").exists()
    assert not (tmp_path / "replays").exists()


def test_eval_labels_are_unique() -> None:
    """Learning-curve checkpoints get distinct roster labels from the final PPO model."""
    keys = ["gbdt_model", "transformer_bin", "ppo_bin", "ppo_c3_bin", "ppo_c7_bin"]
    labels = [k.rsplit("_", 1)[0] for k in keys]
    assert labels == ["gbdt", "transformer", "ppo", "ppo_c3", "ppo_c7"]


def _feature_dir(root: Path, games: list[str]) -> Path:
    """Minimal distillation-v2 feature dir with one train shard."""
    root.mkdir(parents=True)
    (root / "train-0000.npz").write_bytes(b"npz" + root.name.encode())
    (root / "train-0000.json").write_text("[]")
    manifest = dict(
        format="ptcg-distillation-v2",
        contract_id="c",
        teacher_model_id="t",
        teacher_model_sha256="s",
        feature_names=["f"],
        split_seed=1,
        splits=dict(
            train=dict(
                games=[dict(sha256=g) for g in games],
                shards=[dict(path="train-0000.npz", sha256="x")],
                forced_queries=2,
            )
        ),
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_merge_links_shards_and_rejects_shared_games(tmp_path: Path) -> None:
    """Merged dir has tagged hard links and summed counts; a shared game is refused."""
    from ptcg_exp.features_merge import merge

    a = _feature_dir(tmp_path / "a", ["g1", "g2"])
    b = _feature_dir(tmp_path / "b", ["g3"])
    m = merge({"base": a, "dagger_student": b}, tmp_path / "m")
    train = m["splits"]["train"]
    assert len(train["games"]) == 3 and train["forced_queries"] == 4
    assert {s["path"] for s in train["shards"]} == {
        "base-train-0000.npz",
        "dagger_student-train-0000.npz",
    }
    assert (tmp_path / "m/base-train-0000.npz").stat().st_ino == (
        a / "train-0000.npz"
    ).stat().st_ino
    c = _feature_dir(tmp_path / "c", ["g2"])
    with pytest.raises(ValueError, match="two sources"):
        merge({"base": a, "dup": c}, tmp_path / "m2")


def test_agent_for_infers_deck_and_kind() -> None:
    """Held-out opponents of other decks map to their own arena agents."""
    from ptcg_exp.evaluate import agent_for

    assert agent_for("outputs/mega_lucario/scale/model.gbdt") == "mega_lucario_gbdt"
    assert agent_for("runs/x/distill/model/model.bin") == "mega_lucario_transformer"
    assert agent_for("outputs/dragapult/heldout/model/model.gbdt") == "dragapult_gbdt"
    assert agent_for("outputs/mega_abomasnow/heldout/model/model.gbdt") == "mega_abomasnow_gbdt"
