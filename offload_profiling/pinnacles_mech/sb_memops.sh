#!/bin/bash
# Feasibility test of the stream-memops prefetch gate on one L40S.
#SBATCH -p test
#SBATCH -A project_ucm_dli35
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:12:00
#SBATCH -o /data/hanmeng/chunkflow/results/slurm-%j.out
WS=/data/hanmeng/chunkflow
source "$WS/sglang/.venv/bin/activate"
export CUDA_HOME=/opt/spack-v1.1.1/opt/spack/linux-cascadelake/nvhpc-25.7-kfehydowyjvjmze7hn7po2jkarxfml77/Linux_x86_64/25.7/cuda/12.9
echo "== node $(hostname) $(date)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
python3 "$WS/memops_test.py" 2>&1
echo "== DONE memops $(date)"
