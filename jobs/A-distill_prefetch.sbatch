#!/bin/bash
# CPU-only prefetch of A/02 distillation teacher games; usage: sbatch jobs/A-distill_prefetch.sbatch <seed>
#SBATCH -A 22-stud-2627-z
#SBATCH -p student
#SBATCH -c 46
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -J A-distill_prefetch
#SBATCH -o /home2/faculty/mrazantsau/ptcg-rl/logs/%x-%j.out
set -euo pipefail
cd ~/ptcg-rl
export PYTHONPATH=src
.venv/bin/python -m ptcg_exp.run configs/exp/A/02_distill_ppo_smoke.yaml --seed "${1:-0}" --stages prefetch
