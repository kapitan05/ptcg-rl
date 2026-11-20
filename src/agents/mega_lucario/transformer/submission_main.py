"""Kaggle entry point: Mega Lucario Transformer, NumPy inference (no torch).

Packaged as main.py by `src/tools/build_submission.py --transformer-checkpoint`, next to
model.npz, deck.csv, the cg package and the feature code under agents/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _home() -> Path:
    """Locate the submission directory: next to main.py, cwd, or the Kaggle agent dir."""
    candidates = [Path.cwd(), Path("/kaggle_simulations/agent")]
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).resolve().parent)
    for candidate in candidates:
        if (candidate / "model.npz").is_file():
            return candidate
    raise FileNotFoundError("model.npz not found")


HOME = _home()
sys.path.insert(0, str(HOME))

from cg.api import lib  # noqa: E402

from agents.mega_lucario.transformer.numpy_model import NumpyTransformer, decide  # noqa: E402

MODEL = NumpyTransformer.load(HOME / "model.npz")
DECK = [int(card) for card in (HOME / "deck.csv").read_text().split()]
TABLE: dict[int, dict[str, Any]] = {}


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Return the deck at setup, else the Transformer's greedy selection."""
    if obs_dict.get("select") is None:
        return DECK
    if not TABLE:
        TABLE.update({c["cardId"]: c for c in json.loads(lib.AllCard().decode())})
    return decide(MODEL, obs_dict, TABLE)
