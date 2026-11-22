"""Nested, opponent/seat-balanced replay subsets built from hard links."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from ptcg_exp.tracking import JSON

GAME = re.compile(r"game-(\d+)\.jsonl$")


def sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def games_in(directory: Path) -> list[Path]:
    """Return a directory's replays in numeric game order."""
    found = [(int(m.group(1)), p) for p in directory.iterdir() if (m := GAME.search(p.name))]
    return [p for _, p in sorted(found)]


def make_subset(source: Path, output: Path, per_dir: int) -> JSON:
    """Hard-link the first `per_dir` games of every opponent/seat dir; return the manifest."""
    dirs = sorted(d for d in source.iterdir() if d.is_dir())
    if not dirs:
        raise ValueError(f"No opponent/seat directories in {source}")
    if output.exists():
        raise ValueError(f"Subset output exists: {output}")
    entries = []
    for d in dirs:
        games = games_in(d)
        if len(games) < per_dir:
            raise ValueError(f"{d.name} has {len(games)} games < {per_dir}")
        target = output / d.name
        target.mkdir(parents=True)
        shutil.copy2(d / "card_metadata.json", target / "card_metadata.json")
        for g in games[:per_dir]:
            os.link(g, target / g.name)
            entries.append(dict(path=f"{d.name}/{g.name}", sha256=sha256(g)))
    manifest = dict(
        format="ptcg-replay-subset-v1",
        source=str(source),
        per_dir=per_dir,
        dirs=[d.name for d in dirs],
        games=len(entries),
        entries=entries,
    )
    (output / "subset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Parse command-line arguments and build one subset."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-dir", type=int, required=True)
    a = p.parse_args()
    m = make_subset(a.source, a.output, a.per_dir)
    print(json.dumps({k: v for k, v in m.items() if k != "entries"}))


if __name__ == "__main__":
    main()
