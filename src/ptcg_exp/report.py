"""Aggregate results into tables and figures for the report and slides.

ptcg-report data-scale --runs outputs/mega_lucario/scale/runs --eval <dir>/summary.json --out reports/A
ptcg-report ppo-health --ppo <run>/ppo --out reports/C
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABEL = re.compile(r"games_(\d+)_s(\d+)")


def _top1(train_log: Path) -> dict[str, float]:
    """Read the validation/test top-1 metrics from GBDT train.py's JSON summary line."""
    for line in reversed(train_log.read_text().splitlines()):
        if line.startswith('{"model_id"'):
            metrics = json.loads(line)["metrics"]
            flat = {}
            # metrics[bank][split] = {query_top1, queries, rows}
            for bank, splits in metrics.items():
                for split, m in splits.items():
                    for k, v in m.items():
                        if isinstance(v, int | float):
                            flat[f"{bank}.{split}.{k}"] = float(v)
            return flat
    return {}


def _peak_rss_gb(path: Path) -> float | None:
    """Peak RSS (GB) from a `/usr/bin/time -v` log."""
    if not path.exists():
        return None
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return int(m.group(1)) / 1024**2 if m else None


def data_scale(runs: Path, eval_summary: Path | None, out: Path) -> list[dict[str, object]]:
    """Table + figure: GBDT strength vs. number of teacher games, with seed spread and CIs."""
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    ratings = {}
    pairs = {}
    if eval_summary and eval_summary.exists():
        s = json.loads(eval_summary.read_text())
        ratings = {r["agent"]: r for r in s["ratings"]}
        pairs = {(p["agent"], p["opponent"]): p for p in s["pairs"]}
    for d in sorted(runs.glob("games_*_s*")):
        m = LABEL.search(d.name)
        if not m or not (d / "train.log").exists():
            continue
        size, seed = int(m.group(1)), int(m.group(2))
        label = f"gbdt_{size}_s{seed}"
        row: dict[str, object] = dict(size=size, seed=seed, label=label, **_top1(d / "train.log"))
        row["extract_peak_gb"] = _peak_rss_gb(d / "extract.time")
        row["train_peak_gb"] = _peak_rss_gb(d / "train.time")
        if label in ratings:
            row.update(mu=ratings[label]["mu"], sigma=ratings[label]["sigma"])
        if (label, "lucario_rule") in pairs:
            p = pairs[(label, "lucario_rule")]
            row.update(
                vs_teacher=p["win_rate"], vs_teacher_lo=p["ci95"][0], vs_teacher_hi=p["ci95"][1]
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"No finished runs under {runs}")
    keys = sorted({k for r in rows for k in r})
    with (out / "data_scale.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    by: dict[int, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by[int(r["size"])].append(r)  # type: ignore[arg-type]
    sizes = sorted(by)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    preferred = ("main.test.query_top1", "main.validation.query_top1")
    acc_key = next((k for k in preferred if k in keys), None)
    jitter = {0: 0.97, 1: 1.0, 2: 1.03}
    if acc_key:
        for bank, color in (("main", "#2a74c9"), ("general", "#8a63d2")):
            key = acc_key.replace("main.", f"{bank}.")
            vals = [[float(r[key]) for r in by[s] if key in r] for s in sizes]
            axes[0].scatter(
                [s for s, v in zip(sizes, vals) for _ in v],
                [x for v in vals for x in v],
                color=color,
                alpha=0.45,
                s=18,
            )
            axes[0].plot(sizes, [np.mean(v) for v in vals], color=color, marker="o", label=bank)
        axes[0].set_ylabel("test top-1 agreement with teacher")
        axes[0].legend(fontsize=8)
    axes[0].set_title("Per-decision imitation accuracy")
    if any("mu" in r for r in rows):
        for s in sizes:
            for r in by[s]:
                if "mu" in r:
                    mu, half = float(r["mu"]), 1.96 * float(r["sigma"])  # type: ignore[arg-type]
                    axes[1].errorbar(
                        s * jitter[int(r["seed"])],
                        mu,
                        yerr=half,
                        fmt="o",  # type: ignore[arg-type]
                        color="#2a74c9",
                        capsize=3,
                    )
        if ratings and "lucario_rule" in ratings:
            t = ratings["lucario_rule"]
            axes[1].axhspan(
                t["mu"] - 1.96 * t["sigma"],
                t["mu"] + 1.96 * t["sigma"],
                color="#e8a400",
                alpha=0.18,
            )
            axes[1].axhline(t["mu"], color="#e8a400", lw=1.5, label="rule-bot teacher")
            axes[1].legend(fontsize=8, loc="lower right")
        axes[1].set_ylabel("Bradley–Terry rating (95% CI)")
    axes[1].set_title("Playing strength (all games)")
    if any("vs_teacher" in r for r in rows):
        for s in sizes:
            for r in by[s]:
                if "vs_teacher" in r:
                    lo, hi = float(r["vs_teacher_lo"]), float(r["vs_teacher_hi"])  # type: ignore[arg-type]
                    v = float(r["vs_teacher"])  # type: ignore[arg-type]
                    axes[2].errorbar(
                        s * jitter[int(r["seed"])],
                        v,
                        yerr=[[v - lo], [hi - v]],  # type: ignore[arg-type]
                        fmt="o",
                        color="#1e9e5a",
                        capsize=3,
                    )
        axes[2].axhline(0.5, color="#9fb0c3", ls="--", lw=1)
        axes[2].set_ylabel("win rate vs teacher (95% Wilson CI)")
    axes[2].set_title("Head-to-head vs teacher (200 games)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("teacher games (log scale)")
    fig.tight_layout()
    fig.savefig(out / "data_scale.png", dpi=160)
    return rows


def ppo_health(ppo_dir: Path, out: Path) -> None:
    """Per-cycle dashboard of the main health metrics plus a status table."""
    out.mkdir(parents=True, exist_ok=True)
    cycles = sorted(ppo_dir.glob("cycle-*"))
    if not cycles:
        raise ValueError(f"No cycles under {ppo_dir}")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    keys = [
        "approximate_kl",
        "clip_fraction",
        "entropy",
        "value_loss",
        "gradient_norm",
        "update_ratio",
    ]
    status_rows = []
    for c in cycles:
        t = json.loads(next(c.rglob("training.json")).read_text())
        log = t.get("updates_log", [])
        for ax, k in zip(axes.flat, keys, strict=True):
            ax.plot([r.get(k, np.nan) for r in log], label=c.name)
            ax.set_title(k)
        if (c / "health.json").exists():
            h = json.loads((c / "health.json").read_text())
            status_rows.append(
                dict(
                    cycle=c.name,
                    status=h["status"],
                    warn=[x["check"] for x in h["checks"] if x["status"] == "warn"],
                    fail=[x["check"] for x in h["checks"] if x["status"] == "fail"],
                )
            )
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ppo_health.png", dpi=160)
    (out / "ppo_health.json").write_text(json.dumps(status_rows, indent=2) + "\n")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data-scale")
    d.add_argument("--runs", type=Path, required=True)
    d.add_argument("--eval", type=Path)
    d.add_argument("--out", type=Path, required=True)
    h = sub.add_parser("ppo-health")
    h.add_argument("--ppo", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "data-scale":
        for r in data_scale(a.runs, a.eval, a.out):
            print(r)
    else:
        ppo_health(a.ppo, a.out)


if __name__ == "__main__":
    main()
