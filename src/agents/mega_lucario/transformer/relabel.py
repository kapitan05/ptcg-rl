"""DAgger relabeling: GBDT-teacher score targets on states visited by the *student* Transformer.

`scripts/extract.py` only accepts GBDT-played replays (it checks the teacher's logged C++
scores). Here the student played, so the teacher scores each student query in Python
(`predict_banks`; Python ≡ C++ GBDT parity is audited at 1e-10 by `gbdt/scripts/audit.py`).

Per decision, queries follow the student's own selection prefix (the states it actually
reached); the policy target is the teacher's argmax, the value target is the student game's
outcome. Shards have the exact `ptcg-distillation-v2` layout, so `train.py` reads them as is.
Metadata keeps both the student's choice (`target`) and the teacher's (`teacher_target`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from agents.mega_lucario.gbdt.bundle import load_bundle
from agents.mega_lucario.gbdt.features import NAMES, SCHEMA_ID, action_queries, check_deck
from agents.mega_lucario.gbdt.train import predict_banks
from agents.mega_lucario.replay import decisions, inspect_replay, replay_schema, sha256
from agents.mega_lucario.transformer.features import token_indices, tokenize, with_prefix

# Whose states to relabel. Self-test: PTCG_RELABEL_PLAYER=mega_lucario_gbdt on GBDT-played
# replays must give 100% student-teacher agreement (the GBDT played its own argmax).
# An environment variable, so spawned extraction workers inherit it.
STUDENT = os.environ.get("PTCG_RELABEL_PLAYER", "mega_lucario_transformer")


def relabel_shard(job: tuple[Path, Path, Path, str, int, list[Path]]) -> dict[str, Any]:
    """Write one shard of student states labeled with GBDT-teacher scores."""
    replay_dir, model, output, split, index, selected = job
    _, boosters = load_bundle(model)
    context = NAMES.index("context_0")
    features: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    sizes: list[int] = []
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    outcomes: list[float] = []
    decision_sizes: list[int] = []
    forced = 0
    for path in selected:
        header, terminal = inspect_replay(path)
        if replay_schema(header) != SCHEMA_ID or STUDENT not in header["agents"]:
            raise ValueError("Wrong student/schema")
        for seat, name in enumerate(header["agents"]):
            if name == STUDENT:
                check_deck(header["decks"][seat])
        digest = sha256(path)
        table = {
            c["cardId"]: c for c in json.loads((path.parent / "card_metadata.json").read_text())
        }
        games.append(
            dict(
                path=str(path.relative_to(replay_dir)),
                sha256=digest,
                seed=header["seed"],
                agents=header["agents"],
                student_models=header["models"],
                metadata_sha256=sha256(path.parent / "card_metadata.json"),
                **terminal,
            )
        )
        for record in decisions(path):
            if record["agent"] != STUDENT:
                continue
            queries = list(action_queries(record["observation"], record["action"], table))
            if not queries:
                continue
            rows = [np.asarray(q["features"], dtype=np.float64) for q in queries]
            flat = predict_banks(boosters, np.concatenate(rows), context, threads=1)
            bounds = np.cumsum([0, *map(len, rows)])
            rows_by_option = dict(zip(queries[0]["options"], queries[0]["features"], strict=True))
            initial = tokenize(record["observation"], [], table, rows_by_option)
            for k, query in enumerate(queries):
                score = np.asarray(flat[bounds[k] : bounds[k + 1]], dtype=np.float64)
                if not np.isfinite(score).all():
                    raise ValueError("Non-finite teacher scores")
                teacher = int(np.argmax(score))
                if len(score) == 1:
                    forced += 1
                tokens = with_prefix(initial, record["observation"], query["prefix"])
                indices = token_indices(record["observation"], query["options"])
                labels = np.full(len(tokens), -1e9, np.float32)
                labels[indices] = score
                features.append(tokens)
                scores.append(labels)
                sizes.append(len(tokens))
                targets.append(indices[teacher])
                metadata.append(
                    dict(
                        game=digest,
                        step=record["step"],
                        seat=record["seat"],
                        context=record["observation"]["select"]["context"],
                        prefix=query["prefix"],
                        options=query["options"],
                        target=query["target"],
                        teacher_target=query["options"][teacher],
                    )
                )
            decision_sizes.append(len(queries))
            outcomes.append(
                0.0
                if terminal["result"] == 2
                else (1.0 if terminal["result"] == record["seat"] else -1.0)
            )
    if not features:
        raise ValueError("No trainable queries in shard")
    name = f"{split}-{index:04d}"
    np.savez_compressed(
        output / f"{name}.npz",
        x=np.concatenate(features),
        scores=np.concatenate(scores),
        sizes=np.asarray(sizes, dtype=np.int32),
        targets=np.asarray(targets, dtype=np.int32),
        decision_sizes=np.asarray(decision_sizes, dtype=np.int32),
        outcomes=np.asarray(outcomes, dtype=np.float32),
    )
    (output / f"{name}.json").write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
    shard = dict(
        path=f"{name}.npz",
        sha256=sha256(output / f"{name}.npz"),
        metadata_sha256=sha256(output / f"{name}.json"),
        decisions=len(outcomes),
        queries=len(sizes),
        rows=sum(sizes),
        max_tokens=max(sizes),
    )
    print(json.dumps(dict(split=split, shard=index, games=len(games), queries=len(sizes))))
    return dict(split=split, games=games, forced=forced, shard=shard, score_error=0.0)


def agreement(feature_dir: Path) -> dict[str, float]:
    """Student-vs-teacher agreement on the student's own states, per split (from metadata)."""
    manifest = json.loads((feature_dir / "manifest.json").read_text())
    out: dict[str, float] = {}
    for split, info in manifest["splits"].items():
        same = total = 0
        for shard in info["shards"]:
            for q in json.loads((feature_dir / shard["path"]).with_suffix(".json").read_text()):
                if len(q["options"]) > 1:
                    total += 1
                    same += q["target"] == q["teacher_target"]
        out[split] = same / total if total else float("nan")
    return out
