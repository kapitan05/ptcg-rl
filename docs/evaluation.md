# Evaluation protocol (frozen: rating_v1)

All comparisons in the project use **this exact protocol**. Changing it means a new protocol name (`rating_v2`) and re-evaluating every arm.

## Pool and games
- **Reference pool:** the 4 official rule bots (`lucario_rule`, `dragapult_rule`, `iono_rule`, `abomasnow_rule`) plus every candidate model under comparison.
- **Games:** `--games-per-pair 200` (100 per seat assignment). Both seat assignments use the same game seeds, so the comparison is paired.
- **Evaluation seed:** `ptcg_exp.seeds.eval_seed("rating_v1")`, drawn from a range disjoint from every training seed (asserted).
- **Errors:** any agent or engine error invalidates the evaluation.

## Metrics
| Metric | Source | Reported as |
|---|---|---|
| Bayesian Bradley–Terry rating | `src/tools/rating.py` (prior σ = 600, pair cap 500) | μ ± 1.96σ, LCB = μ − 2σ |
| Head-to-head win rate | `ptcg_exp.evaluate.pair_table` (draw = ½) | rate + 95% Wilson CI, per opponent, plus seat-0 rate |
| Imitation agreement | GBDT `query_top1` / Transformer top-1 on held-out **games** | mean over seeds |
| Time per decision | `src/tools/local_match.py` | max and p99 |
| Kaggle ladder | submissions | confirmation only, never for model selection |

## Seeds and statistics
- **≥3 training seeds per arm** (`seeds: [0, 1, 2]`). Every seed is reported, never only the best one.
- All seeds derive from the master seed via `ptcg_exp.seeds.derive(master, stage, name)` and are stored in each run's manifest.
- Each arm differs from its baseline in **one** variable, with equal games, updates and compute.
- **Promotion rule** (`ptcg_exp.evaluate.promote`): candidate LCB ≥ champion μ, and no opponent whose win rate drops beyond its CI.

## Correctness gates (a failed gate means the result doesn't count)
1. Engine and `cg` hashes are verified (`fetch_cg_engine.py`, `fetch_official_assets.py`).
2. C++ rule bots ≡ Python rule bots (`validate_cpp_agents.py`).
3. Python features ≡ C++ features, exactly (extraction).
4. GBDT C++ scores ≡ LightGBM within 1e-10 (`gbdt/scripts/audit.py`).
5. Transformer C++ ≡ PyTorch within 2e-5 (`transformer/scripts/validate.py`).
6. PPO behavior audit within 2e-5, and every new checkpoint is re-exported and validated.
7. Contract IDs are consistent across checkpoint, rollout and binary.
8. No leakage: train/validation/test split by game (hashes disjoint); evaluation seeds disjoint from training.

## PPO health
See `src/ptcg_exp/health.py`. Every cycle writes `health.json` with pass/warn/fail per check:
- **Data:** win rate per opponent and seat, game-length shift, behavior audit, temperature, seeds.
- **Per update:** approx KL, clip fraction, ratio p99, gradient spikes, finite metrics, update ratio.
- **Post-update:** held-out KL band, explained variance before and after, value ECE, entropy drop, C++ export parity.
- **Canary drift:** illegal picks, teacher agreement.

Thresholds live in the config (`health:`). **Any FAIL blocks promotion.**
