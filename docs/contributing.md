# Contributing (team workflow)

## Branches and PRs
- `main` is protected (GitHub branch protection):
  - every change goes through a PR; **1 approving review** from someone other than the author
  - CI job `test` (ruff, mypy, pytest, contract tests) must pass, and the branch must be up to date with `main`
  - a new push dismisses earlier approvals; all review conversations must be resolved
  - squash merge only (linear history); the branch is deleted after merge; no force pushes or deletion of `main`
  - repository admins can bypass in an emergency; don't use it for normal work
- Secret scanning with push protection is on: a push containing a recognised key is rejected. Keep keys in `.env` / `~/.netrc`, never in the repo or on Eden.
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
