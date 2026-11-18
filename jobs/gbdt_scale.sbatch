#!/bin/bash
# GBDT data-scale experiment (A/01). One array task = one training seed for one data size.
# Submit: sbatch --array=0-2 --mem=<M> jobs/gbdt_scale.sbatch <120|400|1600>
#SBATCH -A 22-stud-2627-z
#SBATCH -p student
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -J gbdt_scale
#SBATCH -o /home2/faculty/mrazantsau/ptcg-rl/logs/%x-%A_%a.out
set -euo pipefail
cd ~/ptcg-rl
export PYTHONPATH=src
PY=.venv/bin/python
SIZE=${1:?data size}
SEED=${SLURM_ARRAY_TASK_ID}
SPLIT_SEED=$($PY -c "from ptcg_exp.seeds import derive; print(derive($SEED, 'gbdt', 'split'))")
TRAIN_SEED=$($PY -c "from ptcg_exp.seeds import derive; print(derive($SEED, 'gbdt', 'train'))")
RUN=outputs/mega_lucario/scale/runs/games_${SIZE}_s${SEED}
mkdir -p "$RUN"
cat > "$RUN/run.json" <<JSON
{"experiment": "A/01_gbdt_data_scale", "size": $SIZE, "seed": $SEED,
 "split_seed": $SPLIT_SEED, "train_seed": $TRAIN_SEED,
 "git_sha": "$(git rev-parse HEAD)", "git_dirty": $([ -z "$(git status --porcelain -- src)" ] && echo false || echo true),
 "slurm_job": "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}", "host": "$(hostname)",
 "subset_sha256": "$(sha256sum outputs/mega_lucario/scale/games_${SIZE}/subset.json | cut -d' ' -f1)",
 "uv_lock_sha256": "$(sha256sum uv.lock | cut -d' ' -f1)", "start": "$(date -Is)"}
JSON
echo "size=$SIZE seed=$SEED split=$SPLIT_SEED train=$TRAIN_SEED"
/usr/bin/time -v $PY src/agents/mega_lucario/gbdt/scripts/extract.py \
  --replay-dir outputs/mega_lucario/scale/games_${SIZE} --output-dir "$RUN/features" \
  --seed "$SPLIT_SEED" 2> "$RUN/extract.time"
/usr/bin/time -v $PY src/agents/mega_lucario/gbdt/scripts/train.py \
  --feature-dir "$RUN/features" --output-dir "$RUN/model" \
  --threads 16 --seed "$TRAIN_SEED" --profile standard 2> "$RUN/train.time" | tee "$RUN/train.log"
grep -h "Maximum resident\|Elapsed (wall" "$RUN"/*.time
echo "done $(date -Is)"
