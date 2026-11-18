#!/bin/bash
# Compress raw replay dirs (regenerable from seeds; game hashes are in the manifests) into
# <dir>.tar.zst. A dir is removed only after the archive passes `zstd -t` and its file count matches.
# usage: sbatch jobs/compress_replays.sbatch <dir> [<dir> ...]   (smallest first to gain headroom)
#SBATCH -A 22-stud-2627-z
#SBATCH -p student
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J compress_replays
#SBATCH -o /home2/faculty/mrazantsau/ptcg-rl/logs/%x-%j.out
set -euo pipefail
for d in "$@"; do
  d=${d%/}
  [ -d "$d" ] || { echo "skip (missing): $d"; continue; }
  n=$(find "$d" -type f | wc -l)
  before=$(du -sh "$d" | cut -f1)
  tar -C "$(dirname "$d")" -cf - "$(basename "$d")" | zstd -q -T32 -9 -o "$d.tar.zst"
  zstd -q -t "$d.tar.zst"
  m=$(zstd -dc "$d.tar.zst" | tar -tf - | grep -vc '/$')
  [ "$n" = "$m" ] || { echo "COUNT MISMATCH $d: $n files vs $m in archive"; exit 1; }
  sha256sum "$d.tar.zst" > "$d.tar.zst.sha256"
  rm -rf "$d"
  echo "ok $d: $n files, $before -> $(du -sh "$d.tar.zst" | cut -f1)"
done
df -h ~ | tail -1
