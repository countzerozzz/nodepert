if [ $1 == 'conv-base' ]
then
    for update_rule in np
    do
        for wd in 1e-2 1e-3 1e-4 1e-5
        do
            bsub -n 8 -J "$update_rule[1-2]" -o output.out -q gpu_tesla -gpu "num=1" "python experiments/conv/conv_base.py -log_expdata True -batchsize 100 -num_epochs 50 -update_rule $update_rule -wd $wd -jobid \$LSB_JOBINDEX"
        done
    done

elif [ $1 == 'all-cnn-net' ]
then 
    for update_rule in np
    do
        for wd in 1e-2 1e-3 1e-4 1e-5
        do
            bsub -n 8 -J "$update_rule[1-2]" -o output.out -q gpu_tesla -gpu "num=1" "python experiments/conv/all_cnn_net.py -log_expdata True -batchsize 100 -num_epochs 50 -update_rule $update_rule -wd $wd -jobid \$LSB_JOBINDEX"
        done
    done

fi