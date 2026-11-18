#!/bin/bash
# Correctness test for DAgger relabeling: on GBDT-played replays, relabel.py (Python teacher
# scoring) must reproduce extract.py (logged C++ scores) exactly, with 100% agreement.
#SBATCH -A 22-stud-2627-z
#SBATCH -p student
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J relabel_selftest
#SBATCH -o /home2/faculty/mrazantsau/ptcg-rl/logs/%x-%j.out
set -euo pipefail
cd ~/ptcg-rl
export PYTHONPATH=src
R=$(ls -d runs/shared-smoke-s0-*); T=$(mktemp -d -p runs relabel_selftest.XXXX)
M=$R/gbdt/model/model.gbdt
.venv/bin/python src/agents/mega_lucario/transformer/scripts/extract.py --replay-dir $R/gbdt/audit_replays --model $M --output-dir $T/extract --workers 4 --validation-fraction 0.25
PTCG_RELABEL_PLAYER=mega_lucario_gbdt .venv/bin/python src/agents/mega_lucario/transformer/scripts/relabel.py --replay-dir $R/gbdt/audit_replays --model $M --output-dir $T/relabel --workers 4 --validation-fraction 0.25
.venv/bin/python jobs/relabel_selftest.py $T/extract $T/relabel
rm -rf $T
