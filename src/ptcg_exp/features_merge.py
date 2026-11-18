"""Merge distillation feature dirs (e.g. teacher states + DAgger student states) into one.

Shards are hard-linked (no copy) under `<tag>-<split>-NNNN`; the merged manifest has the
`ptcg-distillation-v2` layout `train.py` expects. Sources must share contract and teacher, and
their games must be disjoint (checked by sha256), so no game can leak across splits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ptcg_exp.tracking import JSON, sha256

SAME = ("format", "contract_id", "teacher_model_id", "teacher_model_sha256", "feature_names")


def merge(sources: dict[str, Path], output: Path) -> JSON:
    """Merge `{tag: feature_dir}` into `output` and return the merged manifest."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    manifests = {t: json.loads((d / "manifest.json").read_text()) for t, d in sources.items()}
    first = next(iter(manifests.values()))
    for tag, m in manifests.items():
        for key in SAME:
            if m.get(key) != first.get(key):
                raise ValueError(f"{tag}: {key} differs from the first source")
    output.mkdir(parents=True, exist_ok=True)
    merged: JSON = {k: v for k, v in first.items() if k not in ("splits", "split_seed")}
    merged["splits"] = {}
    merged["sources"] = {
        t: dict(
            path=str(d), manifest_sha256=sha256(d / "manifest.json"), split_seed=m.get("split_seed")
        )
        for (t, d), m in zip(sources.items(), manifests.values(), strict=True)
    }
    seen: set[str] = set()
    for tag, m in manifests.items():
        src = sources[tag]
        for split, info in m["splits"].items():
            dst = merged["splits"].setdefault(split, dict(games=[], shards=[], forced_queries=0))
            for g in info["games"]:
                if g["sha256"] in seen:
                    raise ValueError(f"Game {g['sha256'][:12]} appears in two sources/splits")
                seen.add(g["sha256"])
                dst["games"].append({**g, "source": tag})
            for shard in info["shards"]:
                name = f"{tag}-{shard['path']}"
                for suffix in (".npz", ".json"):
                    os.link(
                        (src / shard["path"]).with_suffix(suffix),
                        (output / name).with_suffix(suffix),
                    )
                dst["shards"].append({**shard, "path": name, "source": tag})
            dst["forced_queries"] += info.get("forced_queries", 0)
    (output / "manifest.json").write_text(json.dumps(merged, indent=2) + "\n")
    return merged
