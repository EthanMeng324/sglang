#!/bin/bash
# Two nodes, ONE L40S per node (SP=2 across InfiniBand): the same parallel degree as the single-node cells,
# with the all-to-all crossing the NIC instead of the intra-node PCIe path. Wrapper around sb_mn.sh.
# usage: sbatch sb_mn1.sh <model> <cfg> <arms> [KEY=VALUE ...]      e.g. sbatch sb_mn1.sh wan 41 no,old TAG=ib1
#SBATCH -p test
#SBATCH -A project_ucm_dli35
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gnode017
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 01:00:00
#SBATCH -o /data/hanmeng/chunkflow/results/slurm-%j.out
export GPUS_PER_NODE=1
exec bash /data/hanmeng/chunkflow/sb_mn.sh "$@"
