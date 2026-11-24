"""Per-experiment result figures from `reports/results/*.csv` (see `export_results`).

    PYTHONPATH=src uv run python -m ptcg_exp.figures   # → reports/<A|C>/<exp>/*.png + reports/overview.png

For every experiment arm with a finished evaluation:
- `winrates.png`: win rate of the arm's final model against each opponent: one hollow marker
  per seed, the pooled rate (draws count half) as a filled dot with its Wilson 95% interval.
For PPO experiments additionally:
- `learning_curve.png`: BT rating minus the Lucario rule bot's rating in the same roster, by
  global PPO cycle, one line per seed.
- `health.png`: six per-cycle health metrics, one line per seed; warned cycles are triangles.
And `reports/overview.png`: every arm's pooled win rate against the key opponents.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ptcg_exp.tracking import ROOT  # noqa: E402

# Validated categorical palette (light surface), fixed order: seed k → slot k.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, WARN, BAD = "#1f1f1e", "#6b6a64", "#e6e5e0", "#c98500", "#c4342b"
EXP_DIR = {
    "A/02": "A/02_distill_ppo_smoke",
    "A/04": "A/04_dagger_r1",
    "C/01": "C/01_ppo_baseline",
    "C/02": "C/02_critic_value_weight",
    "C/03": "C/03_ppo_continue",
    "C/04": "C/04_ppo_league",
    "C/05": "C/05_ppo_league_pfsp",
    "C/06": "C/06_sprint3_eval",
    "C/06b": "C/06_sprint3_eval/2k_games",
}
OPPONENTS = [
    "start Transformer",
    "C/01 final",
    "C/03 continue",
    "C/04 league",
    "C/05 league + PFSP",
    "GBDT teacher",
    "Lucario rule bot",
    "Dragapult rule bot",
    "Abomasnow rule bot",
    "Iono rule bot",
    "held-out Dragapult GBDT",
    "held-out Abomasnow GBDT",
    "held-out Iono GBDT",
]
HEALTH = [
    ("approx_kl", "Approx. KL per update (mean)"),
    ("heldout_kl", "Held-out KL after update"),
    ("clip_fraction", "Clip fraction"),
    ("entropy", "Policy entropy"),
    ("value_ece", "Value calibration error"),
    ("teacher_agreement", "Agreement with GBDT (canary)"),
]
Row = dict[str, str]


def _read(path: Path) -> list[Row]:
    with path.open() as f:
        return list(csv.DictReader(f))


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Rate and Wilson 95% interval of `wins` (draws already counted as half) in `n` games."""
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def winrates(pairs: list[Row], arm: str, ax: plt.Axes) -> None:
    """Dot plot of one arm: per-seed markers + pooled rate with its interval, per opponent."""
    rows = [r for r in pairs if r["arm"] == arm]
    opps = [o for o in OPPONENTS if any(r["opponent"] == o for r in rows)][::-1]
    seeds = sorted({int(r["seed"]) for r in rows})
    for i, o in enumerate(opps):
        rs = [r for r in rows if r["opponent"] == o]
        n = sum(int(r["games"]) for r in rs)
        w = sum(int(r["wins"]) + 0.5 * int(r["draws"]) for r in rs)
        p, lo, hi = wilson(w, n)
        ax.plot([lo, hi], [i, i], color=INK, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([p], [i], s=40, color=INK, zorder=3)
        ax.annotate(
            f"{p:.3f} [{lo:.2f}, {hi:.2f}]",
            (hi, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK,
        )
        for r in rs:
            ax.scatter(
                [float(r["win_rate"])],
                [i + 0.22],
                s=22,
                facecolors="none",
                edgecolors=SERIES[int(r["seed"]) % len(SERIES)],
                linewidths=1.6,
                zorder=3,
            )
    ax.axvline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_yticks(range(len(opps)), opps, fontsize=8.5, color=INK)
    ax.set_xlim(0.2, 1.05)
    ax.set_xlabel("win rate of the final model (draw = ½)", fontsize=8.5, color=MUTED)
    ax.set_title(arm, fontsize=10, color=INK, loc="left")
    _style(ax)
    for s in seeds:
        ax.scatter(
            [], [], s=22, facecolors="none", edgecolors=SERIES[s % len(SERIES)], label=f"seed {s}"
        )
    ax.scatter([], [], s=40, color=INK, label="pooled ± 95% CI")
    ax.legend(
        fontsize=7.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=len(seeds) + 1,
    )


def learning_curve(ratings: list[Row], experiment: str, path: Path) -> bool:
    """Rating above the Lucario rule bot by global cycle, one line per seed."""
    by_run: dict[str, list[Row]] = defaultdict(list)
    for r in ratings:
        by_run[r["run"]].append(r)
    lines: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for rows in by_run.values():
        luc = next((r for r in rows if r["agent"] == "lucario_rule"), None)
        if luc is None:
            continue
        for r in rows:
            own = r["experiment"] == experiment and r["role"] in ("own final", "own checkpoint")
            anchor = r["role"] in ("start", "C/01 final") and r["experiment"] == experiment
            anchor = anchor and (experiment == "C/01") == (r["role"] == "start")
            if (own or anchor) and r["cycle"] != "":
                lines[int(r["seed"])].append((int(r["cycle"]), float(r["mu"]) - float(luc["mu"])))
    if not lines:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    ends: list[tuple[float, int]] = []
    for s, pts in sorted(lines.items()):
        pts.sort()
        c = SERIES[s % len(SERIES)]
        ax.plot(
            [x for x, _ in pts],
            [y for _, y in pts],
            color=c,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"seed {s}",
        )
        ends.append((pts[-1][1], s))
    # End labels, nudged apart in data units so close values don't overlap.
    last_x = max(x for pts in lines.values() for x, _ in pts)
    all_y = [y for pts in lines.values() for _, y in pts]
    gap = 0.07 * (max(all_y) - min(all_y) or 1.0)
    prev = -math.inf
    for y, s in sorted(ends):
        ly = max(y, prev + gap)
        ax.annotate(
            f"s{s} {y:+.0f}",
            (last_x, ly),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK,
        )
        prev = ly
    ax.axhline(0, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.annotate(
        "Lucario rule bot",
        (ax.get_xlim()[0], 0),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7.5,
        color=MUTED,
    )
    ax.set_xlabel(
        "PPO cycle (−1 = distilled start; C/03+ start at C/01 cycle 11)", fontsize=8.5, color=MUTED
    )
    ax.set_ylabel("BT rating − Lucario rule bot", fontsize=8.5, color=MUTED)
    ax.set_title(f"{experiment}: learning curve", fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def health(cycles: list[Row], experiment: str, path: Path) -> bool:
    """Small multiples of per-cycle health metrics, one line per seed; warn/fail marked."""
    rows = [r for r in cycles if r["experiment"] == experiment]
    if len({r["cycle"] for r in rows}) < 2:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0), dpi=150)
    for ax, (key, title) in zip(axes.flat, HEALTH, strict=True):
        by_seed: dict[tuple[str, int], list[Row]] = defaultdict(list)
        for r in rows:
            if r[key] != "":
                by_seed[(r["run"], int(r["seed"]))].append(r)
        for (_, s), rs in sorted(by_seed.items(), key=lambda kv: kv[0][1]):
            rs.sort(key=lambda r: int(r["global_cycle"]))
            xs = [int(r["global_cycle"]) for r in rs]
            ys = [float(r[key]) for r in rs]
            ax.plot(xs, ys, color=SERIES[s % len(SERIES)], linewidth=1.6, label=f"seed {s}")
            for r, x, y in zip(rs, xs, ys, strict=True):
                if r["health_status"] in ("warn", "fail"):
                    ax.scatter(
                        [x],
                        [y],
                        marker="^",
                        s=36,
                        color=WARN if r["health_status"] == "warn" else BAD,
                        zorder=3,
                    )
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        _style(ax)
    axes.flat[0].scatter([], [], marker="^", s=36, color=WARN, label="health warn")
    axes.flat[0].scatter([], [], marker="^", s=36, color=BAD, label="health fail")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=True))
    fig.legend(
        uniq.values(), uniq.keys(), loc="lower center", ncol=len(uniq), fontsize=7.5, frameon=False
    )
    fig.suptitle(
        f"{experiment}: PPO health per cycle (x = global PPO cycle)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)
    return True


def overview(pairs: list[Row], path: Path) -> None:
    """Every arm's pooled win rate (with CI) against five key opponents, as small multiples."""
    arms = [
        a for a in dict.fromkeys(r["arm"] for r in pairs) if not a.startswith("C/06")
    ]  # C/06, C/06b
    order = ["A/02", "A/04", "C/02", "C/01", "C/03", "C/04", "C/05"]
    arms.sort(key=lambda a: (order.index(a[:4]) if a[:4] in order else 99, a))
    panels = [
        (
            "vs. its start",
            lambda r: (
                r["opponent"]
                == ("GBDT teacher" if r["arm"].startswith("A/02") else "start Transformer")
            ),
        ),
        ("vs. Lucario rule bot", lambda r: r["opponent"] == "Lucario rule bot"),
        ("vs. Dragapult rule bot", lambda r: r["opponent"] == "Dragapult rule bot"),
        ("vs. Abomasnow rule bot", lambda r: r["opponent"] == "Abomasnow rule bot"),
        ("vs. held-out GBDTs", lambda r: r["opponent"].startswith("held-out")),
    ]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(12.5, 0.45 * len(arms) + 1.4), dpi=150, sharey=True
    )
    for ax, (title, keep) in zip(axes, panels, strict=True):
        for i, arm in enumerate(arms[::-1]):
            rs = [r for r in pairs if r["arm"] == arm and keep(r)]
            if not rs:
                continue
            n = sum(int(r["games"]) for r in rs)
            p, lo, hi = wilson(sum(int(r["wins"]) + 0.5 * int(r["draws"]) for r in rs), n)
            c = SERIES[0] if arm.startswith("C/0") and not arm.startswith("C/02") else MUTED
            ax.plot([lo, hi], [i, i], color=c, linewidth=2, solid_capstyle="round")
            ax.scatter([p], [i], s=30, color=c, zorder=3)
            ax.annotate(
                f"{p:.2f}",
                (p, i),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=INK,
            )
        ax.axvline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
        ax.set_xlim(0.3, 0.85)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        _style(ax)
    axes[0].set_yticks(range(len(arms)), arms[::-1], fontsize=8, color=INK)
    fig.suptitle(
        "Pooled win rate of each arm's final model (95% Wilson CI; blue = PPO arms)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """Write every figure from `reports/results`."""
    res, reports = ROOT / "reports/results", ROOT / "reports"
    pairs, ratings, cycles = (_read(res / f"{n}.csv") for n in ("pairs", "ratings", "cycles"))
    written: list[str] = []
    for exp, sub in EXP_DIR.items():
        arms = list(dict.fromkeys(r["arm"] for r in pairs if r["experiment"] == exp))
        if not arms:
            continue
        out = reports / sub
        out.mkdir(parents=True, exist_ok=True)
        cols = min(len(arms), 2)
        nrows = math.ceil(len(arms) / cols)
        height = 4.8 if len(OPPONENTS) <= 10 else 5.6
        fig, axes = plt.subplots(
            nrows, cols, figsize=(6.8 * cols, height * nrows), dpi=150, squeeze=False
        )
        for ax in axes.flat[len(arms) :]:
            ax.set_visible(False)
        for ax, arm in zip(axes.flat, arms, strict=False):
            winrates(pairs, arm, ax)
        fig.tight_layout()
        fig.savefig(out / "winrates.png")
        plt.close(fig)
        written.append(f"{sub}/winrates.png")
        if exp.startswith("C/") and exp not in ("C/02", "C/06", "C/06b"):
            if learning_curve(ratings, exp, out / "learning_curve.png"):
                written.append(f"{sub}/learning_curve.png")
            if health(cycles, exp, out / "health.png"):
                written.append(f"{sub}/health.png")
    overview(pairs, reports / "overview.png")
    written.append("overview.png")
    print("\n".join(written))


if __name__ == "__main__":
    main()
