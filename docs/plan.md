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
| 1 | **C/01 PPO baseline:** 12 cycles × 5,000 games × 128 updates, 3 seeds, starting from each A/02 seed's Transformer; rated at cycles 2, 5, 8 and 11 against the start and the GBDT | C | **done**: s0/s1 beat start 0.625 [0.58, 0.67] and GBDT 0.625; tie the rule bots; s2 stopped by its health gate → rerun C/01b with KL early stop (`experiments/C/01_ppo_baseline`) |
| 2 | **C/02 critic ablation:** value_weight 0.25 → 1.0, paired on the A/02 features and training seeds | C | **done, rejected**: value MSE unchanged, top-1 −1.1 pp, BT −11 (`experiments/C/02_critic_value_weight`) |
| 3 | **Per-cycle PPO stage:** a health gate after every cycle, rollout archiving, resume at cycle granularity, learning-curve evaluation | shared | done (`run.py`), tested in C/00 |
| 4 | **DAgger round 1 (A/03–A/05):** the student plays 2,500 games vs. the rule bots; the GBDT relabels its states (`transformer/relabel.py`, self-test: relabel ≡ extract bit for bit on GBDT games, 100% agreement); merged with A/02's features; retrained. Control arm: the GBDT plays the same 2,500 seeded games (equal label budget). Both arms rated in one roster per seed. | A | **done, no gain**: DAgger vs. BC 0.505 [0.47, 0.55], vs. A/02 0.452; agreement on student states 94.2% (`experiments/A/04_dagger_r1`) |
| 5 | **P1 held-out pool:** GBDTs for dragapult, iono and abomasnow (400 games each, audit gate); the roster infers the deck from `outputs/<deck>/` | A/C | **done**: 3 GBDTs, audits exact; in all Sprint 2/3 eval rosters |
| 6 | **History tokens + parity** | B | to implement (new contract) |
| 7 | **Kaggle:** submit only a promoted champion; measure decision time | all | after C/01 |

**Decision rules:**
- If C/01 beats its start (pooled LCB > 0.5), the Sprint 3 ablations (P2, P3, KL-to-teacher) start from C/01.
- If not, the C/01 health logs decide the next step: a weak critic points to C/02, KL that's too small points to the learning rate or epochs.
- If DAgger closes the GBDT → Transformer gap, the Transformer gets re-distilled before further PPO.

## Sprint 3 (21 Nov – 6 Dec): opponent pool and submission
Decision rules applied: C/01 beat its start (LCB 0.58 > 0.5), so the pool ablations start from C/01's final policies. DAgger didn't help, so no re-distillation and no round 2. C/02 rejected, so value_weight stays 0.25.

| # | Work | Owner | Status / Eden |
|---|---|---|---|
| 1 | **C/01b:** seed 2 rerun with `kl_early_stop: 2.0` (deviation, reported) | C | **done** (1948509): 12 cycles no fail; BT 1020 → 1056; vs. start 0.562, rule bots 0.52–0.53 |
| 2 | **C/03 control:** continue each C/01 final for 6 cycles, default pool | C | **done** (1948510): 18/18 pass; vs. C/01 final 0.532 [0.49, 0.57]; abomasnow rule 0.582 [0.54, 0.62] (`experiments/C/03_ppo_continue`) |
| 3 | **C/04 snapshot league (P2):** + frozen snapshots (3 newest + 2 older, from C/01 c2/c5/c8, the start and own cycles) | C | implemented (`League`, `league_roster`); queued 1948511 |
| 4 | **C/05 league + PFSP (P3):** weight × max(0.2, 1 − win rate) from the previous cycle | C | **done** (1948512): 18/18 pass; ≈ C/03 and C/04 (`experiments/C/05_ppo_league_pfsp`) |
| 5 | **C/06 joint evaluation:** C/01 final, C/03, C/04, C/05, A/02, GBDT and held-out pool, one roster per seed | C | **done** (1948513; rerun C/06b 1984220 at 2,000 games per pair after finding shared-deal noise): all arms beat C/01 final by 4–5 pp and the Lucario rule bot at 0.56–0.57; arms tie each other (pool effect not supported). **Champion: C/05 s0**, Kaggle gate passed (`experiments/C/06_sprint3_eval`) |
| 6 | **Kaggle submission:** NumPy inference (`transformer/numpy_model.py`, no torch), `build_submission.py --transformer-checkpoint`, gate `submission_check.py` | all | **built and gated** for C/01 s0 final: scores ≡ C++ within 2.8e-6, 513/513 actions identical, cg games vs. rule bots 28-11-1, p99 10 ms/decision, 5.8 MB. Uploading to Kaggle is the user's step; re-gate the C/06 champion |
| 7 | **History features (B/01)** | B | designed (`experiments/B/01_history`): 20 free Global-token slots, parity-safe log fields only, as a second arena agent (v3 contract). Built after the PPO chain frees the build and disk |

All Sprint 3 PPO arms use the same start, seeds, 5,000 games per cycle and early stop, so they're paired. The jobs run one at a time (array throttle 1, chained with `afterany`), because one full cycle needs about 43 GB of the 65 GB free.

**P4 (cross-deck learned opponents in training) is dropped for now:** the only learned other-deck agents are the held-out GBDTs, and training against them would destroy the held-out pool.

**Success criteria (pre-registered):** C/04 vs. C/03, and C/05 vs. C/04, paired by seed. Report rating_v1 and the held-out pool separately. Promote the arm whose pooled head-to-head LCB against C/01 final is > 0.5 and which doesn't lose on the held-out pool.
