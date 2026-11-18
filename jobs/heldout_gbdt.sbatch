#!/bin/bash
# P1 held-out opponents: a GBDT for each non-Lucario deck (400 rule-teacher games, 50 per
# opponent/seat), C++ ≡ Python audit gate, raw replays archived. Never used in any training.
# usage: sbatch --array=0-2 jobs/heldout_gbdt.sbatch      (0 dragapult, 1 iono, 2 mega_abomasnow)
#SBATCH -A 22-stud-2627-z
#SBATCH -p student
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -J heldout_gbdt
#SBATCH -o /home2/faculty/mrazantsau/ptcg-rl/logs/%x-%A_%a.out
set -euo pipefail
cd ~/ptcg-rl
export PYTHONPATH=src OMP_NUM_THREADS=32
DECKS=(dragapult iono mega_abomasnow)
D=${DECKS[${SLURM_ARRAY_TASK_ID:-0}]}
SEED=$((30260910 + ${SLURM_ARRAY_TASK_ID:-0} * 100000))   # disjoint from defaults and eval range
O=outputs/$D/heldout; mkdir -p $O
P=.venv/bin/python; S=src/agents/$D
[ -f $O/replays/manifest.json ] || $P $S/scripts/collect_replays.py --output-dir $O/replays --games 50 --seed $SEED --workers 32 --resume
[ -f $O/features/manifest.json ] || $P $S/gbdt/scripts/extract.py --replay-dir $O/replays --output-dir $O/features --seed $((SEED + 1))
[ -f $O/model/model.gbdt ] || /usr/bin/time -v $P $S/gbdt/scripts/train.py --feature-dir $O/features --output-dir $O/model --threads 32 --seed $((SEED + 2)) 2> $O/train_time.txt | tail -n 3
rm -rf $O/audit_replays
$P $S/scripts/collect_replays.py --output-dir $O/audit_replays --games 1 --teacher ${D}_gbdt --model $O/model/model.gbdt --seed $((SEED + 3)) --workers 8
$P $S/gbdt/scripts/audit.py --model $O/model/model.gbdt --replay-dir $O/audit_replays --output $O/audit.json --feature-dir $O/features --probe build/${D}_gbdt_probe
[ -d $O/replays ] && tar -C $O -cf - replays | zstd -q -T32 -9 -f -o $O/replays.tar.zst && zstd -q -t $O/replays.tar.zst && rm -rf $O/replays
echo "HELDOUT $D done: $(sha256sum $O/model/model.gbdt)"
