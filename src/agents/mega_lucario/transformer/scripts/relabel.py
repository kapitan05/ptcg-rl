"""DAgger: label student-Transformer replays with GBDT-teacher scores (distillation-v2 shards).

Same game split, sharding, duplicate checks and manifest as `extract.py`; only the per-game
labeling differs (see `agents/mega_lucario/transformer/relabel.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json

from agents.mega_lucario.transformer import relabel
from agents.mega_lucario.transformer.scripts import extract as base


def main() -> None:
    """Parse command-line arguments, relabel, and add DAgger fields to the manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="GBDT teacher (labeler)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=204)
    p.add_argument("--shard-games", type=int, default=50)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--validation-fraction", type=float, default=0.05)
    a = p.parse_args()
    base.extract_shard = relabel.relabel_shard  # type: ignore[assignment]
    report = base.extract(
        a.replay_dir,
        a.model,
        a.output_dir,
        a.seed,
        a.shard_games,
        a.workers,
        a.validation_fraction,
    )
    report["labels"] = "dagger-student-states"
    report["student_teacher_agreement"] = relabel.agreement(a.output_dir)
    (a.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(student_teacher_agreement=report["student_teacher_agreement"])))


if __name__ == "__main__":
    main()
