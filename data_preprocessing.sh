#!/bin/bash

#SBATCH --job-name=brats_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00

#SBATCH --output=slurm_logs/preprocess_%j.out
#SBATCH --error=slurm_logs/preprocess_%j.err

echo "Starting preprocessing..."

source ~/.bashrc

conda activate BraTS

python data_preprocessing.py