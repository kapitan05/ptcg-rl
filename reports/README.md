# Results

Everything here is generated from the cluster runs and committed; nothing needs Eden to view it.

| Where | What |
|---|---|
| [`overview.png`](overview.png) | every arm's pooled win rate (95% CI) vs. its start, the rule bots and the held-out GBDTs |
| `<A\|C>/<experiment>/winrates.png` | one experiment: win rate of its final model vs. each opponent, per seed and pooled |
| `C/<experiment>/learning_curve.png` | rating above the Lucario rule bot by PPO cycle, per seed |
| [`health_matrix.png`](health_matrix.png) | status of every PPO cycle (91): pass / warn / fail, labelled with the check that didn't pass |
| `C/<experiment>/health.png` | PPO health per cycle (KL, held-out KL, clip fraction, entropy, value calibration, teacher agreement) |
| [`A/01_gbdt_data_scale/data_scale.png`](A/01_gbdt_data_scale/data_scale.png) | A/01 data-scale curve |
| [`results/`](results) | tidy CSVs behind the figures: `pairs.csv` (head-to-head), `ratings.csv` (BT ratings), `cycles.csv` (PPO health) |
| [`runs/<run_id>/`](runs) | raw per-run records: manifest, metrics, eval summary, per-cycle health/canary/training JSON |

Interpretation, hypotheses and decisions are in each experiment's README: [`experiments/`](../experiments).

## Experiments
| ID | Question | Figures | Write-up |
|---|---|---|---|
| A/01 | GBDT imitation vs. teacher data | [data scale](A/01_gbdt_data_scale/data_scale.png) | [README](../experiments/A/01_gbdt_data_scale/README.md) |
| A/02 | Distilled Transformer vs. GBDT; one PPO cycle | [win rates](A/02_distill_ppo_smoke/winrates.png) | [README](../experiments/A/02_distill_ppo_smoke/README.md) |
| A/04 | DAgger vs. equal-budget BC | [win rates](A/04_dagger_r1/winrates.png) | [README](../experiments/A/04_dagger_r1/README.md) |
| C/01 | PPO baseline, 12 cycles × 5,000 games | [win rates](C/01_ppo_baseline/winrates.png) · [learning curve](C/01_ppo_baseline/learning_curve.png) · [health](C/01_ppo_baseline/health.png) | [README](../experiments/C/01_ppo_baseline/README.md) |
| C/02 | Critic weight 0.25 → 1.0 | [win rates](C/02_critic_value_weight/winrates.png) | [README](../experiments/C/02_critic_value_weight/README.md) |
| C/03 | +6 cycles, default pool (control) | [win rates](C/03_ppo_continue/winrates.png) · [learning curve](C/03_ppo_continue/learning_curve.png) · [health](C/03_ppo_continue/health.png) | [README](../experiments/C/03_ppo_continue/README.md) |
| C/04 | +6 cycles, snapshot league | [win rates](C/04_ppo_league/winrates.png) · [learning curve](C/04_ppo_league/learning_curve.png) · [health](C/04_ppo_league/health.png) | [README](../experiments/C/04_ppo_league/README.md) |
| C/05 | league + PFSP | [win rates](C/05_ppo_league_pfsp/winrates.png) · [learning curve](C/05_ppo_league_pfsp/learning_curve.png) · [health](C/05_ppo_league_pfsp/health.png) | [README](../experiments/C/05_ppo_league_pfsp/README.md) |
| C/06 | joint evaluation, promotion (C/06b: 2,000 games per pair) | [C/06b win rates](C/06_sprint3_eval/2k_games/winrates.png) · [C/06 (200 games)](C/06_sprint3_eval/winrates.png) | [README](../experiments/C/06_sprint3_eval/README.md) |

## Refresh
```bash
# 1. pull small run files from Eden (command in docs/setup.md "Results and tracking")
PYTHONPATH=src uv run python -m ptcg_exp.export_results   # reports/runs → reports/results/*.csv
PYTHONPATH=src uv run python -m ptcg_exp.figures          # → the PNGs above
```

Other views: the W&B project `ptcg-rl` (per-cycle curves, tables) and the results dashboard artifact, both fed from the same files.
