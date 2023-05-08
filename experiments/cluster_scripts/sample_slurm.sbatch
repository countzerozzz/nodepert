#!/bin/bash
#SBATCH --job-name=nodepert
#SBATCH --output=logs/ym_%A.out
#SBATCH --time=12:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --partition=gpu_rtx
#SBATCH --gres=gpu:1

nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
srun -u $1

echo "Job finished."
exit