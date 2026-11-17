# Project plan (checkpoint, 23–27 Nov 2026)

## Objective
**Research question:** under a fixed compute budget, which stages of an imitation → RL pipeline actually improve playing strength over a rule-based baseline in the Pokémon TCG, and why?

**Setting:** the Kaggle *PTCG AI Battle (playground)* with its own engine (R2 card pool), playing the Mega Lucario ex deck.

**Pipeline:** the 24th-place design, forked and adapted to the playground engine:
1. rule bot (teacher) → replays
2. GBDT LambdaRank imitation
3. distillation into a 980k-parameter Transformer (pointer policy + 51-bin value)
4. PPO self-play against an opponent pool
5. C++ inference for both the arena and the submission

## Sub-questions and owners
| Owner | Question | Baseline | Primary metric |
|---|---|---|---|
| A: infra/data | (A1) Does GBDT imitation scale with teacher data (120/400/1,600 games)? (A2) Does DAgger close the gap between per-decision accuracy and win rate? | rule bot; behavior cloning at equal labeled data | BT rating ± CI; win rate vs. the teacher (Wilson CI) |
| B: modeling | Do history tokens from `logs[]` and an auxiliary hidden-card belief head improve distillation and strength? | distilled Transformer (current tokens) | validation KL, top-1; BT rating |
| C: RL/eval | Which PPO choices matter: KL penalty toward the teacher, 2–4 epochs + KL early stop, value clip on/off, PFSP opponent weights? | repo-default PPO (`ppo/config.json`) | BT rating after N cycles at equal games; health metrics |

## Methodology (applies to everyone; details in `docs/evaluation.md`)
- **One change per arm.** Equal games, updates and compute; ≥3 seeds; every seed reported.
- **Fixed evaluation:** protocol `rating_v1`, 4 rule bots + candidates, 200 games per pair, both seats, evaluation seeds disjoint from training seeds.
- **Uncertainty:** Bayesian BT intervals, Wilson CIs per opponent, paired comparisons.
- **Correctness gates:** C++ ≡ Python parity for rule bots, features, GBDT (1e-10), Transformer and PPO (2e-5). No split leakage. A failed gate means no result.
- **PPO health:** per-cycle pass/warn/fail checks; a FAIL blocks promotion.
- **Reproducibility:** every run's manifest records git SHA, config hash, seed tree, environment and artifact hashes.

## Status at the checkpoint
- Playground engine integrated with full parity: 10,189 decisions, 0 mismatches.
- Baseline submitted to Kaggle: rule bot, ladder 600 → 538.9.
- Eden cluster pipeline running.
- A1 data-scale experiment: see `experiments/A/01_gbdt_data_scale/README.md`.
- Shared experiment layer, health checks, CI and docs merged via PR #1.

## Timeline (2-week sprints)
| Sprint | Dates | Goal | A | B | C |
|---|---|---|---|---|---|
| 1 | 9–22 Nov | checkpoint | data-scale GBDT; shared layer | distillation from the best GBDT | Transformer submission; PPO smoke |
| 2 | 23 Nov–6 Dec | core experiments I | DAgger round 1 vs. BC | history tokens (new contract, parity) | PPO baseline, 3 seeds |
| 3 | 7–20 Dec | core experiments II | DAgger rounds 2–3; gap analysis | belief head; size ablation (optional) | PPO ablation arms, 3 seeds each |
| 4 | 21 Dec–3 Jan | integration | combine the winners; final agent, 3 seeds | | ladder submissions |
| 5 | 4–18 Jan | report + reproducibility | fresh-clone reproduction | report | final tables, failure analysis |
| 6 | 19–25 Jan | presentation | | | |

## Risks
- **Covariate shift** (imitators lose whole games despite ~90% per-decision accuracy). This is the reason for A2 (DAgger) and for PPO.
- **Contract changes** (B) require C++ and Python changes together. Mitigation: a branch until parity passes; never mix contracts within one comparison.
- **Shared cluster:** only 2 nodes have GPUs (2× RTX 6000 each) and jobs are limited to 1 day. Mitigation: CPU-heavy stages on stud-1; resumable stages.
- **Kaggle runtime** (torch availability, time budget). Mitigation: a numpy fallback for inference; measure total time per match.
