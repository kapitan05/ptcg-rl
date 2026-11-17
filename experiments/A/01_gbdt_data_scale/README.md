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
| Games | Seed | main val | main test | general val | general test | train peak RAM |
|---|---|---|---|---|---|---|
| 120 | 0 | 0.901 | 0.883 | 0.916 | 0.934 | 11.9 GB |
| 120 | 1 | 0.898 | 0.878 | 0.924 | 0.900 | 11.9 GB |
| 120 | 2 | 0.882 | 0.874 | 0.911 | 0.915 | 11.9 GB |
| 400 | | | | | | |
| 1600 | | | | | | |

### Playing strength
_Pending: rating job over 4 rule bots + 9 GBDTs._

**Gates:** feature parity Python ≡ C++ (exact, enforced in extraction): passed for all 120-game runs. GBDT audit 1e-10: pending.
**Conclusion:** _pending_
