"""Deterministic seed tree: every stochastic source derives from one master seed."""

from __future__ import annotations

import hashlib

# Evaluation seeds live in their own range, disjoint from every derived training seed.
TRAIN_RANGE = (1, 2_000_000_000)
EVAL_RANGE = (3_000_000_000, 3_500_000_000)


def derive(master: int, stage: str, name: str) -> int:
    """Return a stable uint32-range seed for (master, stage, name) inside TRAIN_RANGE."""
    digest = hashlib.sha256(f"{master}/{stage}/{name}".encode()).digest()
    low, high = TRAIN_RANGE
    return low + int.from_bytes(digest[:8], "big") % (high - low)


def eval_seed(protocol: str, index: int = 0) -> int:
    """Return the fixed evaluation seed for a named protocol, disjoint from training seeds."""
    digest = hashlib.sha256(f"eval/{protocol}/{index}".encode()).digest()
    low, high = EVAL_RANGE
    return low + int.from_bytes(digest[:8], "big") % (high - low - 10_000_000)


def seed_tree(master: int, names: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Derive all named seeds per stage, e.g. {"gbdt": ["split", "train"]}."""
    return {stage: {n: derive(master, stage, n) for n in ns} for stage, ns in names.items()}


def assert_disjoint(train_seeds: list[int], eval_seeds: list[int], span: int) -> None:
    """Fail if any train seed range [s, s+span) overlaps an eval seed range."""
    for e in eval_seeds:
        for t in train_seeds:
            if t < e + span and e < t + span:
                raise ValueError(f"Seed ranges overlap: train {t} vs eval {e} (span {span})")
