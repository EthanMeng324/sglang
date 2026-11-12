#!/bin/bash
# One-GPU unit test of the chunk-wise layerwise offload manager on a synthetic block stack:
# compares bound GPU weights with the originals and the hooked forward with the plain forward.
#SBATCH -p test
#SBATCH -A project_ucm_dli35
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:15:00
#SBATCH -o /data/hanmeng/chunkflow/results/slurm-%j.out
WS=/data/hanmeng/chunkflow
export CUDA_HOME=/opt/spack-v1.1.1/opt/spack/linux-cascadelake/nvhpc-25.7-kfehydowyjvjmze7hn7po2jkarxfml77/Linux_x86_64/25.7/cuda/12.9
export PATH="$WS/bin:$CUDA_HOME/bin:$PATH" CC=gcc CXX=g++
source "$WS/sglang/.venv/bin/activate"
cd "$WS"
echo "== unit test on $(hostname) $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u "$WS/offload_unit_test.py" "$@" 2>&1 | grep -v -E "UserWarning|warn_once|lru_cache"
echo "== DONE unit test $(date)"
