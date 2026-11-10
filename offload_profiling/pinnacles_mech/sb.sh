#!/bin/bash
# sbatch wrapper for run_mech.sh on the Pinnacles `test` partition (1 h max, 2x L40S).
# usage: sbatch sb.sh <model> <cfg> <arms> [KEY=VALUE ...]
# Extra KEY=VALUE arguments are exported for run_mech.sh (TAG, CHUNK_MB, NSYS, STEPS, NCCL_*...);
# positional export is used because the submitting environment is not reliably propagated.
#SBATCH -p test
#SBATCH -A project_ucm_dli35
#SBATCH --gres=gpu:l40s:2
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o /data/hanmeng/chunkflow/results/slurm-%j.out
m="$1"; c="$2"; a="$3"; shift 3
for kv in "$@"; do
    case "$kv" in *=*) export "$kv" ;; *) echo "sb.sh: ignoring non KEY=VALUE arg '$kv'" ;; esac
done
echo "sb.sh: model=$m cfg=$c arms=$a extra=[$*] -> TAG=${TAG:-} CHUNK_MB=${CHUNK_MB:-} NSYS=${NSYS:-}"
bash /data/hanmeng/chunkflow/run_mech.sh "$m" "$c" "$a"
