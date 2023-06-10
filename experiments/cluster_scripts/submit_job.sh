#!/bin/sh

# replace bsub with the scheduling command for the particular cluster
bsub -n $1 -J "npexps[1-${2}]" -o experiments/cluster_scripts/output.out \
                 -q $3 -gpu "num=1" "${4} -jobid \$LSB_JOBINDEX"
