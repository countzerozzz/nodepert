#!/bin/bash
#SBATCH --job-name=nodepert
#SBATCH --output=logs/ym_%A.out
#SBATCH --time=0-18:00
#SBATCH --nodes=1
#SBATCH --mem=7G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

if [ $exp == 'trial' ]
then
    srun -u python nptest.py
    # srun -u python npconvtest.py

elif [ $exp == 'tmp-tf1' ]
then
    srun -u python tmp_tf/convTF.py -log_expdata True -num_epochs 400 -optimizer 'sgd' -final_actfunc 'sigmoid' -loss_func 'mse' -dropout False -jobid ${SLURM_ARRAY_TASK_ID}
    srun -u python tmp_tf/convTF.py -log_expdata True -num_epochs 400 -optimizer 'sgd' -final_actfunc 'softmax' -loss_func 'mse' -dropout False -jobid ${SLURM_ARRAY_TASK_ID}
    srun -u python tmp_tf/convTF.py -log_expdata True -num_epochs 400 -optimizer 'sgd' -final_actfunc 'softmax' -loss_func 'mse' -dropout True -jobid ${SLURM_ARRAY_TASK_ID}
    srun -u python tmp_tf/convTF.py -log_expdata True -num_epochs 400 -optimizer 'sgd-momentum' -final_actfunc 'softmax' -loss_func 'mse' -dropout True -jobid ${SLURM_ARRAY_TASK_ID}

elif [ $exp == 'tmp-tf2' ]
then
    srun -u python tmp_tf/convTF.py -log_expdata True -num_epochs 400 -optimizer 'sgd' -final_actfunc 'sigmoid' -loss_func 'cross-entropy' -dropout False -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python tmp_tf/convTF.py -log_expdata True -num_epochs 400 -optimizer 'sgd' -final_actfunc 'softmax' -loss_func 'cross-entropy' -dropout False -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python tmp_tf/convTF.py -log_expdata True -num_epochs 400 -optimizer 'sgd' -final_actfunc 'softmax' -loss_func 'cross-entropy' -dropout True -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python tmp_tf/convTF.py -log_expdata True -num_epochs 400 -optimizer 'sgd-momentum' -final_actfunc 'softmax' -loss_func 'cross-entropy' -dropout True -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'conv-base' ]
then
    # srun -u python experiments/conv/conv_base.py -log_expdata False -num_epochs 5 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/conv/conv_base.py -log_expdata False -num_epochs 10 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'conv-large' ]
then
    srun -u python experiments/conv/All-CNN-A.py -log_expdata True -num_epochs 10 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'crash-dynamics' ]
then
    srun -u python experiments/crash_dynamics.py -log_expdata True -n_hl 3 -hl_size 500 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'weight-decay' ]
then
    srun -u python experiments/weight_decay/weight_decay.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 50 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}   
    srun -u python experiments/weight_decay/weight_decay.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 50 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'adam-update' ]
then
    srun -u python experiments/adam_update.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 50 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/adam_update.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 50 -update_rule np-adam -jobid ${SLURM_ARRAY_TASK_ID}  

    srun -u python experiments/adam_update.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 50 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/adam_update.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 50 -update_rule sgd-adam -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'vary-lr' ]
then
    srun -u python experiments/vary_lr.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/vary_lr.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'vary-batchsize' ]
then
    srun -u python experiments/supplimentary/vary_batchsize.py -log_expdata True -n_hl 2 -batchsize 1 -num_epochs 1 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/supplimentary/vary_batchsize.py -log_expdata True -n_hl 2 -batchsize 10 -num_epochs 10 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/supplimentary/vary_batchsize.py -log_expdata True -n_hl 2 -batchsize 100 -num_epochs 100 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/supplimentary/vary_batchsize.py -log_expdata True -n_hl 2 -batchsize 1000 -num_epochs 1000 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    

    srun -u python experiments/supplimentary/vary_batchsize.py -log_expdata True -n_hl 2 -batchsize 1 -num_epochs 1 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/supplimentary/vary_batchsize.py -log_expdata True -n_hl 2 -batchsize 10 -num_epochs 10 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/supplimentary/vary_batchsize.py -log_expdata True -n_hl 2 -batchsize 100 -num_epochs 100 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/supplimentary/vary_batchsize.py -log_expdata True -n_hl 2 -batchsize 1000 -num_epochs 1000 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'linesearch' ]
then
    srun -u python experiments/linesearch.py -log_expdata True -n_hl 2 -hl_size 500 -lr 0.1 -num_epochs 100 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'scale-dataset' ]
then
    # srun -u python experiments/scaling/dataset.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 5000 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/dataset.py -log_expdata True -n_hl 3 -hl_size 2000 -num_epochs 5000 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'scale-width1' ]
then
    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 100 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 500 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 1000 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'scale-width2' ]
then
    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 5000 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 10000 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'scale-width3' ]
then

    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 100 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 500 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 1000 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'scale-width4' ]
then
    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 5000 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/width.py -log_expdata True -n_hl 1 -hl_size 10000 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    


elif [ $exp == 'scale-depth1' ]
then
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 4 -hl_size 500 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 6 -hl_size 500 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    
elif [ $exp == 'scale-depth2' ]
then
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 8 -hl_size 500 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 10 -hl_size 500 -num_epochs 200 -update_rule np -jobid ${SLURM_ARRAY_TASK_ID}    

elif [ $exp == 'scale-depth3' ]
then
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 2 -hl_size 500 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 4 -hl_size 500 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 6 -hl_size 500 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    
elif [ $exp == 'scale-depth4' ]
then
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 8 -hl_size 500 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    
    srun -u python experiments/scaling/depth.py -log_expdata True -n_hl 10 -hl_size 500 -num_epochs 200 -update_rule sgd -jobid ${SLURM_ARRAY_TASK_ID}    

fi

