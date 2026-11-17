"""Fixed arena evaluation protocol: roster → rating_arena.py → per-pair Wilson CIs → summary."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from ptcg_exp.seeds import eval_seed
from ptcg_exp.tracking import JSON

ROOT = Path(__file__).resolve().parents[2]
RULE_BOTS = {
    "lucario_rule": "mega_lucario",
    "dragapult_rule": "dragapult",
    "iono_rule": "iono",
    "abomasnow_rule": "mega_abomasnow",
}
AGENT_FOR_SUFFIX = {".gbdt": "mega_lucario_gbdt", ".bin": "mega_lucario_transformer"}


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a win rate (draws should be counted as half wins)."""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_roster(
    models: dict[str, str], include_rule_bots: bool = True
) -> dict[str, dict[str, str]]:
    """Map labels to arena agents; model files pick their agent by suffix."""
    roster: dict[str, dict[str, str]] = {}
    if include_rule_bots:
        roster.update({label: dict(agent=agent) for label, agent in RULE_BOTS.items()})
    for label, path in models.items():
        suffix = Path(path).suffix
        if suffix not in AGENT_FOR_SUFFIX:
            raise ValueError(f"Unknown model type for {label}: {path}")
        roster[label] = dict(agent=AGENT_FOR_SUFFIX[suffix], model=str(Path(path).resolve()))
    return roster


def pair_table(games_path: Path) -> list[JSON]:
    """Per ordered pair (both seats pooled): wins/draws/losses, win rate and Wilson CI."""
    stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: dict(w=0, d=0, l=0, s0=0, s0n=0)
    )
    for line in games_path.read_text().splitlines():
        r = json.loads(line)
        if r["agent_errors"] or r["engine_errors"]:
            raise ValueError(f"Arena error in evaluation game: {r}")
        a, b, res = r["a"], r["b"], r["result"]
        for me, other, my_seat in ((a, b, 0), (b, a, 1)):
            s = stats[(me, other)]
            if res == 2:
                s["d"] += 1
            elif res == my_seat:
                s["w"] += 1
            else:
                s["l"] += 1
            if my_seat == 0:
                s["s0n"] += 1
                s["s0"] += 1 if res == 0 else 0.5 if res == 2 else 0
    rows = []
    for (me, other), s in sorted(stats.items()):
        n = int(s["w"] + s["d"] + s["l"])
        score = s["w"] + 0.5 * s["d"]
        lo, hi = wilson(score, n)
        rows.append(
            dict(
                agent=me,
                opponent=other,
                games=n,
                wins=int(s["w"]),
                draws=int(s["d"]),
                losses=int(s["l"]),
                win_rate=score / n,
                ci95=[lo, hi],
                seat0_win_rate=s["s0"] / s["s0n"] if s["s0n"] else None,
            )
        )
    return rows


def evaluate(
    models: dict[str, str],
    output: Path,
    games_per_pair: int = 200,
    workers: int = 32,
    protocol: str = "rating_v1",
    include_rule_bots: bool = True,
) -> JSON:
    """Run the protocol and write `summary.json` (ratings + per-pair table)."""
    output.mkdir(parents=True, exist_ok=True)
    roster = build_roster(models, include_rule_bots)
    (output / "roster.json").write_text(json.dumps(roster, indent=2) + "\n")
    seed = eval_seed(protocol)
    pairs = len(roster) * (len(roster) - 1) // 2
    if seed + pairs * 1_000_000 + games_per_pair >= 2**32:
        raise ValueError("Evaluation seed range overflows uint32")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/tools/rating_arena.py"),
            "--roster",
            str(output / "roster.json"),
            "--output-dir",
            str(output / "ratings"),
            "--games-per-pair",
            str(games_per_pair),
            "--seed",
            str(seed),
            "--workers",
            str(workers),
            "--resume",
        ],
        check=True,
        cwd=ROOT,
    )
    ratings = json.loads((output / "ratings/ratings.json").read_text())
    summary = dict(
        protocol=protocol,
        eval_seed=seed,
        games_per_pair=games_per_pair,
        roster=roster,
        ratings=ratings["ratings"],
        pairs=pair_table(output / "ratings/games.jsonl"),
        corpus_sha256=ratings.get("corpus_sha256"),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def promote(summary: JSON, candidate: str, champion: str) -> JSON:
    """Promotion rule: candidate LCB ≥ champion mean and no opponent drop beyond its CI."""
    r = {x["agent"]: x for x in summary["ratings"]}
    if candidate not in r or champion not in r:
        raise ValueError("Candidate and champion must both be in the evaluated roster")
    pairs = {(p["agent"], p["opponent"]): p for p in summary["pairs"]}
    regressions = []
    for (agent, opp), p in pairs.items():
        if agent != candidate or opp == champion or (champion, opp) not in pairs:
            continue
        c = pairs[(champion, opp)]
        if p["ci95"][1] < c["win_rate"]:
            regressions.append(dict(opponent=opp, candidate=p["win_rate"], champion=c["win_rate"]))
    ok = r[candidate]["lcb"] >= r[champion]["mu"] and not regressions
    return dict(
        promote=ok,
        candidate_lcb=r[candidate]["lcb"],
        champion_mu=r[champion]["mu"],
        regressions=regressions,
    )


def main() -> None:
    """Evaluate `label=path` models against the rule bots with the fixed protocol."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", default=[], help="label=path (.gbdt or .bin)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--games-per-pair", type=int, default=200)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--protocol", default="rating_v1")
    p.add_argument("--no-rule-bots", action="store_true")
    a = p.parse_args()
    models = dict(m.split("=", 1) for m in a.model)
    s = evaluate(models, a.output_dir, a.games_per_pair, a.workers, a.protocol, not a.no_rule_bots)
    for x in s["ratings"]:
        print(f"{x['agent']:>24} {x['mu']:7.1f} ± {1.96 * x['sigma']:5.1f}")


if __name__ == "__main__":
    main()
