#!/bin/bash
#SBATCH --job-name=nodepert
#SBATCH --output=logs/slurm_%A.out
#SBATCH --time=12:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --partition=gpu_rtx
#SBATCH --gres=gpu:1

nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
srun python nodepert/main.py -network fc -dataset mnist -log_expdata True \
                              -n_hl 2 -hl_size 500 -lr 5e-3 -batchsize 100 \
                              -num_epochs 10 -update_rule np

echo "Job finished."
exit
