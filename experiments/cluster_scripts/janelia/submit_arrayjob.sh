if [ $1 == 'fc' ]
then
    for update_rule in sgd, np
    do
        for wd in 0 1e-6 3e-6 1e-5 3e-5 1e-4
        do
            bsub -n 12 -J "$update_rule[1-9]" -o scripts/output.out -q gpu_rtx -gpu "num=1" "python experiments/conv/conv_base.py -log_expdata True -batchsize 100 -num_epochs 1000 -update_rule $update_rule -wd $wd -jobid \$LSB_JOBINDEX"
        done
    done

if [ $1 == 'conv-base' ]
then
    for update_rule in np
    do
        for wd in 0 1e-6 3e-6 1e-5 3e-5 1e-4
        do
            bsub -n 12 -J "$update_rule[1-9]" -o scripts/output.out -q gpu_tesla -gpu "num=1" "python experiments/conv/conv_base.py -log_expdata True -batchsize 100 -num_epochs 1000 -update_rule $update_rule -wd $wd -jobid \$LSB_JOBINDEX"
        done
    done

elif [ $1 == 'all-cnn-net' ]
then 
    for update_rule in np
    do
        for wd in 0 1e-6 3e-6 1e-5 3e-5 1e-4
        do
            bsub -n 5 -J "$update_rule[1-12]" -o scripts/output.out -q gpu_tesla -gpu "num=1" "python experiments/conv/all_cnn_net.py -log_expdata True -batchsize 100 -num_epochs 2000 -update_rule $update_rule -wd $wd -jobid \$LSB_JOBINDEX"
        done
    done

fi