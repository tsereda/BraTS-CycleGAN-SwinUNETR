#!/bin/bash

#SBATCH --job-name=brats_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G # Adjust memory as needed
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 # Request 1 GPU

#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "Starting training..."

source ~/.bashrc

conda activate BraTS

nvidia-smi
python gpucheck.py
python segmentation/train.py