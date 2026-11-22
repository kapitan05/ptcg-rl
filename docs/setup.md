# Setup

## Local (macOS or Linux)
```bash
uv sync --locked --dev
source macos.env            # macOS only: GCC 16 + a single OpenMP runtime
uv run python src/tools/fetch_cg_engine.py --distribution playground
uv run python src/tools/build_cpp_engine.py --distribution playground
uv run pytest tests/ -q
```
- macOS needs `brew install gcc libomp` (Apple clang lacks the C++20 features the engine uses).
- An 8 GB machine can run the smoke config and small GBDTs (≤ ~120 games). Bigger jobs go to Eden.

## Eden cluster (MINI PW)
1. Get an account, then SSH through the faculty jump host. Add to `~/.ssh/config`:
   ```
   Host mini
       HostName ssh.mini.pw.edu.pl
       User <mini_login>
   Host eden
       HostName eden.mini.pw.edu.pl
       User <eden_login>
       ProxyJump mini
   ```
   Then run `ssh-copy-id mini && ssh-copy-id eden`.
2. Set up the repo:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   git clone <repo> ~/ptcg-rl && cd ~/ptcg-rl
   UV_HTTP_TIMEOUT=600 ~/.local/bin/uv sync --locked --dev
   ```
3. Build and test **on a compute node**. The login node's CPU can't import numpy 2.5.
   ```bash
   srun -A <account> -p student -c 16 --mem=32G -t 60 .venv/bin/python src/tools/build_cpp_engine.py --distribution playground
   ```
4. Run experiments with `PYTHONPATH=src .venv/bin/python -m ptcg_exp.run <config> --submit`.
   - Set the account and partition with `PTCG_SLURM_ACCOUNT` / `PTCG_SLURM_PARTITION`.
   - Don't use `sbatch --export=...`: it gets jobs held with "user env retrieval failed".

Resources on the `student` partition: 3 nodes × 48 CPUs / ~756 GB RAM; stud-2 and stud-3 each have 2× RTX 6000; 1-day job limit.

## Results and tracking
- **Saved results:** `reports/runs/<run_id>/` holds the small per-run files from the cluster's `runs/`: manifest, metrics, eval summary/roster/ratings, and per PPO cycle health, canary, training and validation JSON. Machine-specific path prefixes are stripped. Models, replays and rollouts stay on the cluster (hashes are in the manifests).
- **Refresh from Eden** (from the repo root):
  ```bash
  rsync -a --prune-empty-dirs --exclude 'C01_s*' --exclude '*/rollout/' --exclude '*/replays/' \
    --exclude 'card_metadata.json' --exclude '*/features*/' --include '*/' \
    --include '*.json' \
    --include 'metrics.jsonl' --include 'ratings.csv' --include 'ppo_rosters/*.json' \
    --include 'wandb/**' --exclude '*' eden:ptcg-rl/runs/ reports/runs/
  ```
  Then strip machine path prefixes: `grep -rlE "/(mnt/evafs|home2)/" reports/runs | xargs sed -i "" -E "s#/(mnt/evafs/faculty/home|home2/faculty)/[^/]+/ptcg-rl/##g"` (macOS `sed`). `wandb/` is git-ignored; it is only there to be synced.
- **W&B:** jobs rendered by `ptcg_exp.slurm` log offline by default (`PTCG_WANDB=1`, `WANDB_MODE=offline`; set `PTCG_WANDB=0` to disable) into `runs/<id>/wandb/`. No API key is stored on the cluster. After pulling, run on a machine where you ran `uv run wandb login`:
  ```bash
  PYTHONPATH=src uv run python -m ptcg_exp.wandb_backfill --sync-offline reports/runs   # cluster offline runs
  PYTHONPATH=src uv run python -m ptcg_exp.wandb_backfill                               # runs without W&B logs
  ```
  Logged: config, per-cycle PPO metrics (`ppo_cycle/*`: training means, held-out KL, every numeric health check, canary), gates, BT ratings and pairwise win rates. Live and backfilled logging share one W&B run id per run ID.
