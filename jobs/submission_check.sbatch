#!/bin/bash
# Build the Transformer Kaggle submission from a run's checkpoint and gate it
# (NumPy ≡ C++ scores, identical actions, card table, cg games, decision time).
# Usage: sbatch jobs/submission_check.sbatch <run_dir_with_ppo_cycle> <name>
#SBATCH -A 22-stud-2627-z
#SBATCH -p student
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -J submission_check
#SBATCH -o /home2/faculty/mrazantsau/ptcg-rl/logs/%x-%j.out
set -euo pipefail
cd ~/ptcg-rl
export PYTHONPATH=src
CYCLE=$1; NAME=$2
OUT=submissions/$NAME
.venv/bin/python src/tools/build_submission.py --transformer-checkpoint $CYCLE/models/model.pt --output $OUT
sha256sum $CYCLE/models/model.pt $OUT/model.npz $OUT.tar.gz | tee $OUT.sha256
.venv/bin/python src/agents/mega_lucario/transformer/scripts/submission_check.py \
  --submission $OUT --replay-dir $CYCLE/validation/replays --games 5 --output $OUT.check.json
