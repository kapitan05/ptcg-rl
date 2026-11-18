"""Package a Python agent in the playground sample_submission layout (main.py, deck.csv, cg/)."""

from __future__ import annotations

import argparse
import shutil
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


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="mega_lucario", help="Directory under src/agents")
    p.add_argument("--output", type=Path, default=ROOT / "submissions/mega_lucario_rule")
    p.add_argument("--cg", type=Path, default=SAMPLE / "cg")
    a = p.parse_args()
    print(build(a.agent, a.output, a.cg))


if __name__ == "__main__":
    main()
