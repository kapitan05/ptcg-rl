# A/01: GBDT imitation vs. amount of teacher data

**Question:** data/imitation (owner A). Does imitation quality scale with teacher data?
**Hypothesis (written 2026-11-17, before the 400/1,600 results):** per-decision top-1 agreement and the head-to-head win rate vs. the rule-bot teacher both rise from 120 → 400 → 1,600 games. Our earlier local result (120 games: 41% vs. the teacher) is data-limited.
**Baseline:** the rule-bot teacher (`lucario_rule`), and the 120-game GBDT.
**Change:** only the number of teacher games, using nested subsets balanced across opponents and seats (`ptcg_exp.subsets`, first k games per opponent×seat dir).
**Fixed:** features (`SCHEMA_ID`), `standard` profile, 16 threads, val/test = 25% / 10% of **games**.
**Seeds:** 0, 1, 2 → `derive(seed, "gbdt", "split"|"train")` (recorded in `run.json`).
**Metric:** BT rating ± CI and win rate vs. the teacher with a Wilson CI (docs/evaluation.md, `rating_v1`). Secondary: test top-1 per bank, peak RAM.
**Compute:** Eden `student` partition, 16 CPUs per task. Jobs `1943714` (120), `1943715` (400), `1943716` (1,600).
**Run script:** `jobs/gbdt_scale.sbatch` (equivalent config: `configs/exp/A/01_gbdt_data_scale.yaml`).

## Results

### Per-decision agreement (query top-1, held-out games)
| Games | Seed | main val | main test | general val | general test | train peak RAM | train wall |
|---|---|---|---|---|---|---|---|
| 120 | 0 | 0.901 | 0.883 | 0.916 | 0.934 | 11 GB | 0:42 |
| 120 | 1 | 0.898 | 0.878 | 0.924 | 0.900 | 11 GB | 0:42 |
| 120 | 2 | 0.882 | 0.874 | 0.911 | 0.915 | 11 GB | 1:08 |
| 400 | 0 | 0.907 | 0.901 | 0.953 | 0.953 | 37 GB | 1:51 |
| 400 | 1 | 0.907 | 0.904 | 0.952 | 0.936 | 37 GB | 1:21 |
| 400 | 2 | 0.909 | 0.904 | 0.947 | 0.947 | 37 GB | 1:17 |
| 1600 | 0 | 0.924 | 0.921 | 0.967 | 0.967 | 148 GB | 5:01 |
| 1600 | 1 | 0.921 | 0.924 | 0.967 | 0.964 | 149 GB | 5:31 |
| 1600 | 2 | 0.920 | 0.924 | 0.965 | 0.965 | 149 GB | 4:05 |

**Mean test top-1:** main 0.878 → 0.903 → 0.923; general 0.916 → 0.945 → 0.965 (120 → 400 → 1,600 games). The seed spread at each size (≤ 0.03) is far smaller than the gaps between sizes.
**Memory:** peak RAM ≈ 0.09 GB per game (LightGBM, 16 threads), roughly linear. It explains why 256+ games ran out of memory on the 8 GB Mac. (On the Mac, 120 games peaked lower with fewer threads; part of the peak is per-thread LightGBM buffers.)

### Playing strength
_Pending: rating job over 4 rule bots + 9 GBDTs._

**Gates:** feature parity Python ≡ C++ (exact, enforced in extraction): passed for all 120-game runs. GBDT audit 1e-10: pending.
**Conclusion:** _pending_
