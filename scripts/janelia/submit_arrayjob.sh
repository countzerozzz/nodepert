if [ $1 == 'fc-test' ]
then
    bsub -n 4 -J "npexps[1-3]" -q gpu_a100 -gpu "num=1" 'python fc_test.py -log_expdata True -batchsize 100 -num_epochs 3 -update_rule sgd -jobid $LSB_JOBINDEX'

elif [ $1 == 'conv-test' ]
then
    bsub -n 4 -J “npexps” -q gpu_a100 -gpu "num=1" 'python small_conv_test.py -log_expdata True -batchsize 100 -num_epochs 10 -update_rule sgd'

elif [ $1 == 'conv-base' ]
then  
    bsub -n 12 -J "npexps[1-5]" -o output.out -q gpu_tesla -gpu "num=1" 'python experiments/conv/conv_base.py -log_expdata True -batchsize 100 -num_epochs 300 -update_rule np -jobid $LSB_JOBINDEX'

elif [ $1 == 'all-cnn-net' ]
then
    bsub -n 12 -J "npexps[1-5]" -o output.out -q gpu_tesla -gpu "num=1" 'python experiments/conv/all_cnn_net.py -log_expdata True -batchsize 100 -num_epochs 300 -update_rule np -jobid $LSB_JOBINDEX'

fi

exit