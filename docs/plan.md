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

---

# Revision 18 Nov 2026: plan after A/01 and A/02

## Findings that set the priorities
1. **No learned agent beats the rule bot yet** (3-seed means): rule bot 1048, GBDT 1026, Transformer 1007, toy PPO 997 BT. Each imitation step loses about 20 BT, even at 92–96% per-decision agreement. Pooled, the Transformer vs. GBDT win rate is 0.448 [0.41, 0.49].
2. **The critic is weak:** explained variance 0.31, validation value loss rises after epoch 3, and value ECE warned on 1 of 3 seeds. PPO advantages depend on it.
3. **PPO is untested at real size.** This is the main open question for implementation and methodology (22 pts).
4. **Disk is a hard limit.** One full PPO cycle writes ~43 GB raw; the Eden quota is 200 GB. Rollouts are now archived per cycle (`keep_rollouts: false`): raw JSON compresses ~150×, and regenerable arrays are dropped with their sha256 kept.
5. **The opponent pool is narrow, and training and evaluation share it.** See the next section.

## Opponent pool: why it matters, and why it wasn't discussed earlier
**Current state:**
- **PPO trains against** (`ppo/config.json`): the 4 rule bots (mega_lucario, dragapult, iono, mega_abomasnow), the GBDT, and `self` (the live policy), with equal weights.
- **Evaluation** (`rating_v1`) uses the same 4 rule bots, plus candidates.
- On Kaggle, opponents are other teams' agents playing the **4 official decks**. They are mostly stronger than and different from the sample rule bots.

**Why it wasn't raised:** everything so far was imitation (A/01, A/02). For imitation, opponents only shape which states the teacher visits, and the repo's pool was the obvious fixed baseline. The pool becomes the main lever only once PPO trains at real size, so this revision raises it now, before C/01 results come in.

**Risks of the current pool:**
- **Training = evaluation (no held-out opponents).** PPO can learn to exploit 4 deterministic scripts, which looks like progress in `rating_v1` but doesn't transfer. The leaderboard (538.9 today) is the only out-of-distribution check, and it's slow and noisy.
- **Self-play without history:** playing only the current self can cycle (A beats B beats C beats A) and forget old counters. The 24th-place solution is literally a *population-based* ecosystem.
- **Mirror bias:** the learner is always Lucario. Other decks are played only by rule bots, which are weak and predictable.

**What we add (owner C, with A for the learned opponents):**
| Step | What | Code needed | When |
|---|---|---|---|
| P1 | **Held-out evaluation pool:** learned agents of the other 3 decks (GBDT per deck via the same pipeline; ~1 CPU-hour each), never used in training. Protocol `rating_v2` = `rating_v1` + held-out agents, reported separately. | config + the existing scripts (every deck has `gbdt/`, `transformer/`, `ppo/`) | Sprint 2 |
| P2 | **Snapshot league:** each cycle's checkpoint joins a pool of frozen opponents (`opponent_models` already accepts fixed `.bin` files); sample the last k plus older snapshots | ~40 lines in `run.py` (pass `--opponent-config` per cycle) | Sprint 2–3 |
| P3 | **PFSP weights:** weight opponents by how often the learner loses to them (`opponent_weights` exists); the C ablation from the original plan | ~30 lines | Sprint 3 |
| P4 | **Cross-deck learned opponents in training** (other decks' Transformers once P1 exists) | config | Sprint 3 |

**Success criterion:** C/01 vs. P2/P3 arms at equal games. Report both `rating_v1` (the training pool) and the held-out pool. An arm that improves only on the training pool counts as exploitation, not strength.

## Sprint 2 (18 Nov – 6 Dec): concrete work and Eden jobs
| # | Work | Owner | Status / Eden |
|---|---|---|---|
| 1 | **C/01 PPO baseline:** 12 cycles × 5,000 games × 128 updates, 3 seeds, starting from each A/02 seed's Transformer; rated at cycles 2, 5, 8 and 11 against the start and the GBDT | C | submitted: job 1944835 (after the C/00 pipecheck 1944834, after the disk cleanup 1944831) |
| 2 | **C/02 critic ablation:** value_weight 0.25 → 1.0, paired on the A/02 features and training seeds | C | submitted: job 1944836 |
| 3 | **Per-cycle PPO stage:** a health gate after every cycle, rollout archiving, resume at cycle granularity, learning-curve evaluation | shared | done (`run.py`), tested in C/00 |
| 4 | **DAgger round 1:** the student plays, the GBDT labels the student's states, retrain on teacher + student states (same label budget as BC) | A | to implement: a `dagger` stage (collect with `mega_lucario_transformer` → extract with the GBDT as labeler → train) |
| 5 | **P1 held-out pool:** GBDTs for dragapult, iono and abomasnow, plus `rating_v2` | A/C | to implement (configs) |
| 6 | **History tokens + parity** | B | to implement (new contract) |
| 7 | **Kaggle:** submit only a promoted champion; measure decision time | all | after C/01 |

**Decision rules:**
- If C/01 beats its start (pooled LCB > 0.5), the Sprint 3 ablations (P2, P3, KL-to-teacher) start from C/01.
- If not, the C/01 health logs decide the next step: a weak critic points to C/02, KL that's too small points to the learning rate or epochs.
- If DAgger closes the GBDT → Transformer gap, the Transformer gets re-distilled before further PPO.
