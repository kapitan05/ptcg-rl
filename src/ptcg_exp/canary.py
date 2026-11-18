"""Canary probe set: fixed held-out decisions scored by every checkpoint to detect drift.

Built once from the distillation validation split (first query of each decision, GBDT teacher
scores attached). For a checkpoint we store its masked logits; comparing two checkpoints gives
KL(new ‖ prev), top-1 agreement with the teacher, and the illegal-pick rate (must be 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ptcg_exp.tracking import JSON, sha256


def build(feature_dir: Path, output: Path, decisions: int = 2000, seed: int = 7) -> JSON:
    """Sample fixed first-queries from the validation split into canary.npz."""
    from agents.mega_lucario.transformer.train import load_split

    manifest = json.loads((feature_dir / "manifest.json").read_text())
    data = load_split(feature_dir, manifest, "validation")
    n_dec = len(data["outcomes"])
    pick = np.sort(
        np.random.default_rng(seed).choice(n_dec, size=min(decisions, n_dec), replace=False)
    )
    first = data["decision_offsets"][pick]
    width = int(data["sizes"][first].max())
    x = np.zeros((len(first), width, data["x"].shape[1]), np.float32)
    mask = np.zeros((len(first), width), bool)
    teacher = np.full((len(first), width), -np.inf, np.float32)
    for row, q in enumerate(first):
        b, e = data["offsets"][q : q + 2]
        x[row, : e - b] = data["x"][b:e]
        mask[row, : e - b] = True
        teacher[row, : e - b] = data["scores"][b:e]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, mask=mask, teacher=teacher, decisions=pick)
    info = dict(
        path=str(output),
        sha256=sha256(output),
        decisions=len(first),
        seed=seed,
        source_manifest_sha256=sha256(feature_dir / "manifest.json"),
    )
    output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    return info


@torch.no_grad()
def score(canary: Path, checkpoint: Path, batch_size: int = 256) -> np.ndarray:
    """Return float64 masked logits of a checkpoint on the canary set (CPU, deterministic)."""
    from agents.mega_lucario.gbdt.features import DECK_COUNTS
    from agents.mega_lucario.transformer.checkpoint import load_checkpoint
    from agents.mega_lucario.transformer.features import CONTRACT_ID, NAMES

    model, _ = load_checkpoint(checkpoint, CONTRACT_ID, DECK_COUNTS, NAMES)
    model = model.double().eval()
    with np.load(canary) as d:
        x, mask = d["x"], d["mask"]
    out = []
    for s in range(0, len(x), batch_size):
        logits = model(
            torch.as_tensor(x[s : s + batch_size], dtype=torch.float64),
            torch.as_tensor(mask[s : s + batch_size]),
        )
        out.append(logits.numpy())
    return np.concatenate(out)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax that tolerates -inf entries."""
    m = np.max(z, axis=1, keepdims=True)
    e = np.where(np.isfinite(z), np.exp(z - m), 0.0)
    return z - m - np.log(e.sum(axis=1, keepdims=True))


def compare(canary: Path, logits: np.ndarray, previous: np.ndarray | None) -> dict[str, float]:
    """Drift metrics: teacher agreement, illegal-pick rate, and KL(new ‖ previous)."""
    with np.load(canary) as d:
        teacher, mask = d["teacher"], d["mask"]
    pick = logits.argmax(1)
    legal = np.isfinite(logits[np.arange(len(pick)), pick]) & mask[np.arange(len(pick)), pick]
    result = dict(
        teacher_agreement=float((pick == teacher.argmax(1)).mean()),
        illegal_pick_rate=float(1 - legal.mean()),
    )
    if previous is not None:
        lp, lq = _log_softmax(logits), _log_softmax(previous)
        p = np.exp(lp)
        terms = np.where(p > 0, p * (lp - np.where(np.isfinite(lq), lq, 0.0)), 0.0)
        result["kl_vs_previous"] = float(terms.sum(1).mean())
        result["agreement_vs_previous"] = float((pick == previous.argmax(1)).mean())
    return result


def main() -> None:
    """Build a canary set, or score a checkpoint against it."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--feature-dir", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--decisions", type=int, default=2000)
    s = sub.add_parser("score")
    s.add_argument("--canary", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--previous-logits", type=Path)
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "build":
        print(json.dumps(build(a.feature_dir, a.output, a.decisions)))
        return
    logits = score(a.canary, a.checkpoint)
    np.save(a.output.with_suffix(".npy"), logits)
    prev = np.load(a.previous_logits) if a.previous_logits else None
    metrics = compare(a.canary, logits, prev)
    a.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
