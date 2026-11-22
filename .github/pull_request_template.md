## What and why
<!-- One or two sentences. Link the experiment README or issue. -->

## Type
- [ ] Code / infrastructure
- [ ] New experiment (config in `configs/exp/`, README from `experiments/TEMPLATE.md`, hypothesis written before results)
- [ ] Results / docs only

## Checks
- [ ] `uv run ruff check src tests && uv run ruff format --check src/ptcg_exp tests`
- [ ] `uv run mypy src/ptcg_exp --ignore-missing-imports --follow-imports=silent`
- [ ] `uv run pytest tests/`
- [ ] Contract change (features, tokens, architecture)? Parity re-verified and `CONTRACT_ID`/`SCHEMA_ID` changed
- [ ] Results claimed? ≥3 seeds, every seed reported, CIs, run IDs listed
- [ ] No secrets, personal paths, models, replays or data files added
