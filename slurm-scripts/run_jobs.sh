#!/bin/sh

python experiments/conv/conv_mnist.py -log_expdata True -n_hl 3 -hl_size 500 -num_epochs 200 -update_rule np -jobid 0
python experiments/conv/conv_mnist.py -log_expdata True -n_hl 3 -hl_size 500 -num_epochs 200 -update_rule np -jobid 1
