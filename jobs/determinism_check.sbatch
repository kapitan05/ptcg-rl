#!/bin/bash
# CPU determinism check: rerun the smoke pipeline in a fresh run dir; artifacts must be bit-identical.
#SBATCH -A 22-stud-2627-z
#SBATCH -p student
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -J determinism_check
#SBATCH -o /home2/faculty/mrazantsau/ptcg-rl/logs/%x-%j.out
set -euo pipefail
cd ~/ptcg-rl
export PYTHONPATH=src OMP_NUM_THREADS=4
.venv/bin/python -m ptcg_exp.run configs/base/smoke_repeat.yaml --seed 0 --stages data,gbdt
a=$(ls -d runs/shared-smoke-s0-*); b=$(ls -d runs/shared-smoke_repeat-s0-*)
status=0
for rel in gbdt/model/model.gbdt; do
  ha=$(sha256sum "$a/$rel" | cut -d' ' -f1); hb=$(sha256sum "$b/$rel" | cut -d' ' -f1)
  echo "$rel $ha $hb"; [ "$ha" = "$hb" ] || status=1
done
# replays: compare sorted per-file hashes (paths relative to each run)
(cd "$a/data" && find replays -type f | sort | xargs sha256sum) > /tmp/det_a.txt
(cd "$b/data" && find replays -type f | sort | xargs sha256sum) > /tmp/det_b.txt
if diff -q /tmp/det_a.txt /tmp/det_b.txt >/dev/null; then echo "replays identical ($(wc -l < /tmp/det_a.txt) files)"; else echo "replays DIFFER"; diff /tmp/det_a.txt /tmp/det_b.txt | head; status=1; fi
echo "DETERMINISM $([ $status = 0 ] && echo PASS || echo FAIL)"
exit $status
