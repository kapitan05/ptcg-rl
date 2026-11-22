"""Package a Python agent in the playground sample_submission layout (main.py, deck.csv, cg/)."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "external/playground/data/sample_submission/sample_submission/sample_submission"
DECK_LOADER = 'file_path = Path(__file__).with_name("deck.csv")\n'
ROBUST_DECK_LOADER = '''def _deck_path() -> Path:
    """Locate deck.csv next to main.py, in cwd, or in the Kaggle agent directory."""
    candidates = [Path("deck.csv"), Path("/kaggle_simulations/agent/deck.csv")]
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).with_name("deck.csv"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("deck.csv not found")


file_path = _deck_path()
'''


def build(agent: str, output: Path, cg: Path) -> Path:
    """Copy the agent's main.py and deck.csv plus the cg package, then write a tar.gz."""
    source = ROOT / "src/agents" / agent
    main = (source / "main.py").read_text()
    if main.count(DECK_LOADER) != 1:
        raise ValueError("Unexpected deck loader in main.py; update build_submission.py")
    if not (cg / "api.py").is_file():
        raise FileNotFoundError(f"Not a cg package: {cg}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "main.py").write_text(main.replace(DECK_LOADER, ROBUST_DECK_LOADER))
    deck = (source / "deck.csv").read_text().split()
    if len(deck) != 60:
        raise ValueError("deck.csv must contain 60 card IDs")
    (output / "deck.csv").write_text("\n".join(deck) + "\n")
    shutil.copytree(cg, output / "cg", ignore=shutil.ignore_patterns("__pycache__"))
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(output.rglob("*")):
            tar.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    return archive


TRANSFORMER_CODE = (
    "gbdt/features.py",
    "gbdt/rich_features.py",
    "gbdt/schema.json",
    "transformer/features.py",
    "transformer/contract.py",
    "transformer/numpy_model.py",
)


def build_transformer(checkpoint: Path, output: Path, cg: Path) -> Path:
    """Package the Lucario Transformer: NumPy main.py, model.npz, feature code, deck, cg."""
    from agents.mega_lucario.transformer.numpy_model import export_npz

    source = ROOT / "src/agents/mega_lucario"
    if not (cg / "api.py").is_file():
        raise FileNotFoundError(f"Not a cg package: {cg}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copy2(source / "transformer/submission_main.py", output / "main.py")
    export_npz(checkpoint, output / "model.npz")
    for rel in TRANSFORMER_CODE:
        target = output / "agents/mega_lucario" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, target)
    deck = (source / "deck.csv").read_text().split()
    if len(deck) != 60:
        raise ValueError("deck.csv must contain 60 card IDs")
    (output / "deck.csv").write_text("\n".join(deck) + "\n")
    shutil.copytree(cg, output / "cg", ignore=shutil.ignore_patterns("__pycache__"))
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(output.rglob("*")):
            tar.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    return archive


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="mega_lucario", help="Directory under src/agents")
    p.add_argument("--output", type=Path, default=ROOT / "submissions/mega_lucario_rule")
    p.add_argument("--cg", type=Path, default=SAMPLE / "cg")
    p.add_argument(
        "--transformer-checkpoint",
        type=Path,
        help="Package the Lucario Transformer (NumPy inference) from this .pt instead",
    )
    a = p.parse_args()
    if a.transformer_checkpoint:
        sys.path.insert(0, str(ROOT / "src"))
        print(build_transformer(a.transformer_checkpoint, a.output, a.cg))
    else:
        print(build(a.agent, a.output, a.cg))


if __name__ == "__main__":
    main()
