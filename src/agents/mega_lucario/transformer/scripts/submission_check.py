"""Gate a packaged Transformer submission before it goes to Kaggle.

1. Scores: NumPy model ≡ the C++ scores logged in C++-played replays (`transformer_queries`),
   within 2e-5 on every query, using the replay's own observations.
2. Actions: the submission's `decide` reproduces every Transformer action in those replays.
3. Card table: the cg library's all-card JSON (what main.py uses) ≡ the replays' card_metadata.
4. Real games: the packaged main.py plays the 4 rule bots through the cg Python simulator
   (the Kaggle code path), with per-decision time (p50/p99/max).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/tools"))
import argparse
import json
import time

import numpy as np

from agents.mega_lucario.transformer.features import token_indices, tokenize, with_prefix
from agents.mega_lucario.transformer.numpy_model import NumpyTransformer, decide

NAME = "mega_lucario_transformer"


def replay_parity(model: NumpyTransformer, replays: list[Path]) -> dict[str, object]:
    """Compare scores, actions and per-decision time against C++-logged replays."""
    max_error = 0.0
    queries = decisions = mismatches = 0
    times: list[float] = []
    table: dict[int, dict[str, object]] = {}
    for path in replays:
        table = {
            c["cardId"]: c for c in json.loads((path.parent / "card_metadata.json").read_text())
        }
        for line in path.open():
            record = json.loads(line)
            if record.get("type") != "decision" or record.get("agent") != NAME:
                continue
            obs = record["observation"]
            begin = time.perf_counter()
            action = decide(model, obs, table)
            times.append(time.perf_counter() - begin)
            decisions += 1
            mismatches += action != record["action"]
            tokens = None
            for q in record.get("transformer_queries", []):
                if len(q["options"]) < 2:
                    continue
                tokens = tokenize(obs, [], table) if tokens is None else tokens
                hidden = model.encode(tokens)
                logits, _ = model.head(hidden, with_prefix(tokens, obs, q["prefix"]))
                got = logits[token_indices(obs, q["options"])]
                max_error = max(max_error, float(np.abs(got - np.asarray(q["scores"])).max()))
                queries += 1
    return dict(
        replays=len(replays),
        decisions=decisions,
        queries=queries,
        score_max_abs_error=max_error,
        action_mismatches=mismatches,
        decide_seconds=_percentiles(times),
        card_table=table,
    )


def _percentiles(times: list[float]) -> dict[str, float]:
    arr = np.asarray(times or [0.0])
    return dict(
        p50=float(np.percentile(arr, 50)),
        p99=float(np.percentile(arr, 99)),
        max=float(arr.max()),
    )


def main() -> None:
    """Run all checks; exit non-zero if any gate fails."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", type=Path, required=True, help="built submission directory")
    p.add_argument("--replay-dir", type=Path, required=True, help="C++ replays of this model")
    p.add_argument("--games", type=int, default=5, help="cg games per rule bot and seat")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=2e-5)
    a = p.parse_args()
    sub = a.submission.resolve()
    model = NumpyTransformer.load(sub / "model.npz")
    replays = sorted(a.replay_dir.rglob("game-*.jsonl"))
    parity = replay_parity(model, replays)
    table = parity.pop("card_table")

    sys.path.insert(0, str(sub))
    from cg.api import lib

    cg_table = {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}
    used = set(table)
    card_table_equal = all(cg_table.get(k) == table[k] for k in used)

    import local_match

    me = local_match.load_submission(sub)
    games: dict[str, dict[str, int]] = {}
    timing: list[float] = []
    for name in local_match.OPPONENTS:
        rival = local_match.load_rule(name)
        tally = dict(wins=0, losses=0, draws=0)
        for seat in (0, 1):
            for _ in range(a.games):
                agents = [me, rival] if seat == 0 else [rival, me]
                result = local_match.play(agents, timing if seat == 0 else [])
                key = "draws" if result == 2 else "wins" if result == seat else "losses"
                tally[key] += 1
        games[name] = tally
        print(name, tally, flush=True)
    report = dict(
        **parity,
        card_table_equal=card_table_equal,
        cards_compared=len(used),
        cg_games=games,
        cg_decision_seconds=_percentiles(timing),
        tolerance=a.tolerance,
    )
    report["passed"] = bool(
        parity["queries"]
        and parity["score_max_abs_error"] <= a.tolerance
        and parity["action_mismatches"] == 0
        and card_table_equal
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Submission check FAILED")


if __name__ == "__main__":
    main()
