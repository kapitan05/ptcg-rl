#!/bin/bash
# A/01 follow-up: GBDT audit gate (1e-10) per size, then rating_v1 over 4 rule bots + 9 GBDTs, then report.
# Submit with --dependency=afterok:<train jobs>.
#SBATCH -A 22-stud-2627-z
#SBATCH -p student
#SBATCH -c 40
#SBATCH --mem=160G
#SBATCH -t 23:00:00
#SBATCH -J gbdt_scale_eval
#SBATCH -o /home2/faculty/mrazantsau/ptcg-rl/logs/%x-%j.out
set -euo pipefail
cd ~/ptcg-rl
export PYTHONPATH=src
PY=.venv/bin/python
RUNS=outputs/mega_lucario/scale/runs
OUT=outputs/mega_lucario/scale/eval
mkdir -p "$OUT"
echo "start $(date -Is) on $(hostname)"
# Gate 4: C++ GBDT scores == LightGBM within 1e-10, on GBDT-played games (one seed per size).
for SIZE in 120 400 1600; do
  R=$RUNS/games_${SIZE}_s0
  rm -rf "$R/audit_replays"
  $PY src/agents/mega_lucario/scripts/collect_replays.py --output-dir "$R/audit_replays" --games 1 \
    --teacher mega_lucario_gbdt --model "$R/model/model.gbdt" --workers 8 --seed 1234567
  $PY src/agents/mega_lucario/gbdt/scripts/audit.py --model "$R/model/model.gbdt" \
    --replay-dir "$R/audit_replays" --feature-dir "$R/features" --output "$R/audit.json"
  echo "audit ok: $SIZE"
done
ARGS=()
for d in $RUNS/games_*_s*; do n=$(basename $d); ARGS+=(--model "gbdt_${n#games_}=$d/model/model.gbdt"); done
$PY -m ptcg_exp.evaluate "${ARGS[@]}" --output-dir "$OUT" --games-per-pair 200 --workers 40
$PY -m ptcg_exp.report data-scale --runs $RUNS --eval "$OUT/summary.json" --out reports/A/01_gbdt_data_scale
echo "done $(date -Is)"
