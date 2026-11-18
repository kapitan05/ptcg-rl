# Contributing (team workflow)

## Branches and PRs
- `main` is protected: changes go through a PR with 1 review and green CI.
- Branch names: `<owner>/<topic>` (e.g. `maksim/shared-layer`, `B/history-tokens`).
- Keep PRs small. Pipeline-logic changes need a test. Contract changes (features, architecture) need re-verified parity and a new `CONTRACT_ID`.

## One experiment = one config + one README
1. Copy an example from `configs/exp/<owner>/` and change **one** thing versus its baseline.
2. Copy `experiments/TEMPLATE.md` to `experiments/<owner>/<NN_name>/README.md` and fill in the hypothesis, baseline and metric **before** running.
3. Run it (`ptcg run … --submit` on Eden), then fill in the results: all seeds, CIs, plots, and a conclusion.
4. Commit the config + README + small result tables/figures. **Never commit** `runs/`, `outputs/`, `models/` or replays; artifacts go to the team HF Hub repo, with their sha256 in the README.

## Code style
- Python 3.12, typed signatures, `X | None`, pydantic for configs, polars/numpy for data.
- `uv run ruff format . && uv run ruff check .` and `uv run pytest tests/` must pass.
- Add dependencies only with `uv add` (never `pip install`).

## Secrets
- No tokens in code, configs, logs or the cluster filesystem (Eden data counts as public). Use environment variables or `chmod 600` files you place yourself.
