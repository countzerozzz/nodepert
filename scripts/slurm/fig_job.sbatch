#!/bin/bash
#SBATCH --job-name=nodepert
#SBATCH --output=logs/ym_%A.out
#SBATCH --time=0-1:00
#SBATCH --nodes=1
#SBATCH --mem=5G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# srun -u python figcode/crash_dynamics.py
srun -u python figcode/width_linear.py