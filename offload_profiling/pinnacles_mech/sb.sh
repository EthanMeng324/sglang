#!/bin/bash
# sbatch wrapper for run_mech.sh on the Pinnacles `test` partition (1 h max, 2x L40S).
# usage: sbatch [--export=ALL,TAG=x] sb.sh <model> <cfg> <arms>
#SBATCH -p test
#SBATCH -A project_ucm_dli35
#SBATCH --gres=gpu:l40s:2
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o /data/hanmeng/chunkflow/results/slurm-%j.out
bash /data/hanmeng/chunkflow/run_mech.sh "$1" "$2" "$3"
