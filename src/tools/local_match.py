"""Play a packaged submission against official rule bots in the Python simulator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[2]
OPPONENTS = ("mega_lucario", "dragapult", "iono", "mega_abomasnow")

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def load_submission(directory: Path) -> Callable[[dict[str, Any]], list[int]]:
    """Exec main.py without __file__ from its own directory, like a sandboxed runner."""
    module = types.ModuleType("submission_main")
    previous = Path.cwd()
    os.chdir(directory)
    try:
        exec(compile((directory / "main.py").read_text(), "main.py", "exec"), module.__dict__)
    finally:
        os.chdir(previous)
    return module.agent


def load_rule(name: str) -> Callable[[dict[str, Any]], list[int]]:
    """Import an official rule policy from the repository."""
    path = ROOT / "src/agents" / name / "main.py"
    spec = importlib.util.spec_from_file_location(f"rule_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def play(agents: list[Callable[[dict[str, Any]], list[int]]], timing: list[float]) -> int:
    """Play one game; return the winning seat (0/1) or 2 for a draw."""
    from cg.game import battle_finish, battle_select, battle_start

    deck_obs = {"select": None, "logs": [], "current": None}
    obs, start = battle_start(agents[0](deck_obs), agents[1](deck_obs))
    if obs is None:
        raise RuntimeError(f"Start failed: player={start.errorPlayer} type={start.errorType}")
    try:
        for _ in range(10000):
            current = obs.get("current") or {}
            if obs.get("select") is None or current.get("result", -1) != -1:
                return current.get("result", 2)
            actor = current.get("yourIndex", 0)
            begin = time.perf_counter()
            action = agents[actor](obs)
            if actor == 0:
                timing.append(time.perf_counter() - begin)
            obs = battle_select(action)
        raise RuntimeError("Step limit reached")
    finally:
        battle_finish()


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", type=Path, default=ROOT / "submissions/mega_lucario_rule")
    p.add_argument("--games", type=int, default=10, help="Games per opponent and seat")
    a = p.parse_args()
    submission = a.submission.resolve()
    sys.path.insert(0, str(submission))
    me = load_submission(submission)
    report = {}
    timing: list[float] = []
    for name in OPPONENTS:
        rival = load_rule(name)
        wins = losses = draws = 0
        for seat in (0, 1):
            for _ in range(a.games):
                agents = [me, rival] if seat == 0 else [rival, me]
                # Seat 0 timing only measures the submission when it sits in seat 0.
                result = play(agents, timing if seat == 0 else [])
                if result == 2:
                    draws += 1
                elif result == seat:
                    wins += 1
                else:
                    losses += 1
        report[name] = dict(wins=wins, losses=losses, draws=draws)
        print(name, report[name], flush=True)
    worst = max(timing) if timing else 0.0
    print(json.dumps(dict(report=report, decisions_timed=len(timing), max_decision_s=worst)))


if __name__ == "__main__":
    main()
