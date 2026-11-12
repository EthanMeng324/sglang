#!/bin/bash
# One node, ONE L40S (no sequence parallelism, no collectives): isolates the offload manager from
# communication. Wrapper around sb_mn.sh (NNODES=1, GPUS_PER_NODE=1 -> --num-gpus 1 --ulysses-degree 1).
# usage: sbatch sb_1g.sh <model> <cfg> <arms> [KEY=VALUE ...]      e.g. sbatch sb_1g.sh wan 41 no,launch TAG=g1
#SBATCH -p test
#SBATCH -A project_ucm_dli35
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gnode017
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 01:00:00
#SBATCH -o /data/hanmeng/chunkflow/results/slurm-%j.out
export GPUS_PER_NODE=1
exec bash /data/hanmeng/chunkflow/sb_mn.sh "$@"
