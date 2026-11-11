#!/bin/bash
#SBATCH --job-name=wan_h100x2
#SBATCH --account=torch_pr_1002_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/yx2432/rebuttal_mn/results/h100-%j.out
PLAT=h100 bash /scratch/yx2432/rebuttal_mn/exp_driver.sh
