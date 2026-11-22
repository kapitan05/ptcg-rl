# C/04: snapshot league (P2), paired against C/03

**Question:** ppo_ablation (owner C). Does adding frozen past checkpoints to the opponent pool beat plain continuation at an equal game budget?
**Config:** `configs/exp/C/04_ppo_league.yaml` (hash `a00bacd2`). Same start, seeds, 6 cycles × 5,000 games × 128 updates and `kl_early_stop: 2.0` as C/03. Opponents: the 6 defaults + snapshots (3 most recent + 2 older, from C/01 cycles 2/5/8, the start, and own cycles), snapshot weight 1.0.
**Jobs:** 1948511 (array 0–2, throttle 1), ~2.8 h per seed. League code tested first in `03p_league_pipecheck` (1948508).

## Result: 18/18 cycles pass
| Pooled, 600 games (Wilson 95%) | C/04 league | C/03 control |
|---|---|---|
| vs. C/01 final | **0.547 [0.51, 0.59]** | 0.532 [0.49, 0.57] |
| vs. start Transformer | 0.613 | 0.630 |
| vs. GBDT | 0.632 | 0.630 |
| vs. Lucario rule bot | 0.536 [0.50, 0.58] | 0.535 [0.50, 0.58] |
| vs. Dragapult rule bot | 0.512 [0.47, 0.55] | 0.503 |
| vs. Abomasnow rule bot | 0.593 [0.55, 0.63] | 0.582 |
| vs. Iono rule bot | 0.757 | 0.753 |
| held-out Dragapult / Abomasnow / Iono GBDT | 0.663 / 0.563 / 0.803 | 0.666 / 0.567 / 0.800 |

Per seed vs. C/01 final: 0.525 / 0.545 / 0.570.

**Reading:**
- The league's pooled LCB vs. C/01 final (0.507) clears 0.5; C/03's (0.492) doesn't. But the arms differ by only +1.5 pp, well inside each other's intervals. The pre-registered comparison is the paired C/06 head-to-head (C/04 vs. C/03 in one roster).
- No change on the held-out pool or the rule bots: the league neither exploits the training pool more nor generalizes better so far.
- The Dragapult rule bot remains a tie for every arm.
