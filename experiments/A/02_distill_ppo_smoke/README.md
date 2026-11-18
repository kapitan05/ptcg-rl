# A/02: distil the 1,600-game GBDT into the Transformer, then one PPO cycle with full health checks

**Question:** pipeline (owner A). Does the distilled Transformer match its GBDT teacher? Does one real-data PPO cycle pass every health check?
**Hypothesis (pre-registered in the config):** the Transformer matches the teacher (top-1 and arena); the PPO cycle passes all checks.
**Teacher:** `games_1600_s0` from A/01. It was picked by test top-1 before the A/01 rating finished; it is the lowest-rated of the three 1,600-game seeds, but within their CIs.
**Config:** `configs/exp/A/02_distill_ppo_smoke.yaml`, which uses 5,000 teacher games (1,250 per opponent), 5 epochs on an RTX 6000, and a PPO cycle of 1,000 games with 16 updates.
**Seeds:** 0, 1, 2 (same teacher; distillation data, init and PPO seeds derived from the master seed).
- seed 0: run `A-distill_ppo_smoke-s0-1b8102e5`, jobs 1943748 (prefetch) and 1943742 (GPU, 39 min)
- seeds 1 and 2: prefetch jobs 1944315/1944316 (26 min each), then GPU jobs 1944319/1944320 (37 and 33 min)

## Summary over 3 seeds
| Seed | Teacher top-1 (epoch) | Value MSE | C++ max error | PPO health | Value ECE | Transformer BT | PPO BT | Transformer vs. GBDT | PPO vs. Transformer |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.955 (3) | 0.662 | 2.9e-6 | 19/19 pass | 0.114 | 1005.0 | 995.1 | 0.425 [0.36, 0.49] | 0.510 [0.44, 0.58] |
| 1 | 0.952 (3) | 0.699 | 2.2e-6 | 19/19 pass | 0.115 | 1002.0 | 996.0 | 0.440 [0.37, 0.51] | 0.530 [0.46, 0.60] |
| 2 | 0.957 (4) | 0.753 | 4.4e-6 | 18/19, **warn** | **0.157** | 1013.2 | 999.4 | 0.480 [0.41, 0.55] | 0.507 [0.44, 0.58] |
| **mean ± sd / pooled** | **0.955 ± 0.003** | 0.705 ± 0.046 | ≤ 4.4e-6, 0 mismatches | promotion allowed for all 3 | | **1006.7 ± 5.8** | 996.8 ± 2.3 | **0.448 [0.41, 0.49]** (600 games) | 0.515 [0.48, 0.55] (600 games) |

- **GBDT teacher rating in the same rosters:** 1025.6 ± 2.7.
- **Rule bot `lucario_rule`:** 1047.8 ± 4.5.
- **Pooled Transformer vs. GBDT:** 269 of 600 wins. The upper CI bound of 0.49 is below 0.5, so the Transformer is weaker on all 3 seeds and pooled.
- **Seed 2's warning:** value ECE 0.157 is just over the 0.15 warning threshold. It is a warning, not a failure. This is the same weak or overfitting critic, now caught by the check on one seed.

## Seed 0 in detail

### Distillation (validation: 16,259 decisions)
| Epoch | Teacher top-1 | KL | Value MSE |
|---|---|---|---|
| 1 | 0.935 | 0.044 | 0.694 |
| 3 (selected) | **0.955** | 0.024 | **0.662** |
| 5 | 0.955 | 0.023 | 0.714 |

**C++ export gate** (`validate.py`, tolerance 2e-5):
- max error 2.9e-6
- probe max error 4.3e-6 (1,000 validation queries)
- 0 action mismatches, 0 agent errors, 0 engine errors

### PPO cycle health (`ppo/cycle-000/health.json`): **19 / 19 pass**, promotion allowed (including `seeds_recorded`)
| Check | Value |
|---|---|
| behavior audit (logp / value / entropy) | 4.1e-6 / 1.5e-6 / 1.8e-6 |
| approx KL per update (mean / max) | 0.0008 / 0.0023 (fail above 0.016) |
| held-out KL | 0.0029 (band 0.001–0.016) |
| clip fraction (max) | 0.031 |
| ratio p99 | 1.24 |
| gradient spike (max / median) | 1.25× |
| update ratio ‖Δθ‖/‖θ‖ | 3.1e-4 |
| explained variance before / after | 0.31 / 0.31 |
| value ECE | **0.114** (pipecheck had 0.37 after 1 epoch) |
| entropy change | +2% |
| C++ export parity, illegal canary picks | pass, 0.0 |

### Arena (`rating_v1`: 7 agents, 200 games per pair, 4,200 games, 0 errors)
| Agent | BT rating (95% CI) |
|---|---|
| lucario_rule (the rule-bot teacher) | 1051.4 [1033, 1069] |
| dragapult_rule | 1035.1 [1017, 1053] |
| **GBDT 1600 s0 (the distillation teacher)** | **1024.6 [1006, 1043]** |
| **Transformer (distilled)** | **1005.0 [987, 1023]** |
| **PPO cycle 1** | **995.1 [977, 1013]** |
| abomasnow_rule | 976.2 [960, 992] |
| iono_rule | 912.7 [897, 929] |

**Head-to-head (Wilson 95% CI):**
- Transformer vs. GBDT teacher: **0.425 [0.36, 0.49]**. The Transformer is significantly weaker.
- PPO vs. Transformer: 0.51 [0.44, 0.58]. No measurable change.

BT ratings are relative to each roster, so they can't be compared directly with A/01. The GBDT's 1024.6 here versus 1010.9 there reflects different opponents.

## Conclusion
- **Hypothesis 1 is rejected (3 seeds).** At 95.5% per-decision agreement, the Transformer wins only 44.8% [41, 49] of games against its teacher: the A/01 effect again, one level down. The pipeline compounds an imitation loss at each step (rule bot → GBDT, −22 BT; GBDT → Transformer, −19 BT; 3-seed means). This is the argument for training the student on its own states (DAgger, sub-question A2) and for PPO starting from a stronger initialisation (C).
- **Hypothesis 2 holds.** On seeds 0 and 1, all 19 health checks pass on real data. Seed 2 has one warning (ECE) and still passes the gate. The value-head warning from the 1-epoch pipecheck is gone. The checks separate an untrained critic (ECE 0.37) from a trained one (0.11).
- **The critic is weak**: explained variance is 0.31 and validation value MSE bottoms out at epoch 3 (0.66), then rises. It is overfitting. Selecting the checkpoint on value loss, or giving the value head its own early stopping, is a cheap C ablation.
- **The PPO cycle is toy-sized** (1,000 games, 16 updates). "No change" (pooled 0.515 [0.48, 0.55]) is the expected result; it is not evidence about PPO.

## Correctness and reproducibility
- Gates passed: `transformer_validate_2e-5`, `ppo_health`. Eval refuses to run when any gate fails.
- A CPU determinism check (`jobs/determinism_check.sbatch`, job 1944314) reran the smoke pipeline in a new run dir. The GBDT model sha256 was identical and all 17 replay files were identical.
- The seed 0 manifest records git `c7bfa7f` with `dirty: true`, because code was rsynced with uncommitted changes. Seeds 1 and 2 ran on a clean commit, `73ab33d`, and record `code_sha256 c06c94e8…`, a content hash of `src/` and `configs/`.

## Limitations
- Each seed is evaluated in its own roster, so BT ratings are compared within a seed; the pooled head-to-head numbers are the robust comparison.
- The teacher choice was made before its rating existed (see above).
- 200 games per pair resolves about ±7 pp in win rate, so effects smaller than that are invisible.

Reports: `reports/A/02_distill_ppo_smoke/` (eval summary, health, validation, training history, manifest).
