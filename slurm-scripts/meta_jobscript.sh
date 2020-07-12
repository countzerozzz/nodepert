#!/bin/sh

if [ $1 == 'trial' ]
then
    sbatch --array=5 --export=ALL,exp='trial' submit_job.sbatch

elif [ $1 == 'crash-dynamics' ]
then
    sbatch --array=5,15,25,35,45 --export=ALL,exp='crash-dynamics' submit_job.sbatch

elif [ $1 == 'weight-decay' ]
then
    sbatch --array=0-24 --export=ALL,exp='weight-decay' submit_job.sbatch

elif [ $1 == 'adam-update' ]
then
    sbatch --array=0-24 --export=ALL,exp='adam-update' submit_job.sbatch

elif [ $1 == 'vary-lr' ]
then
    sbatch --array=0-11 --export=ALL,exp='vary-lr' submit_job.sbatch

elif [ $1 == 'vary-batchsize' ]
then
    sbatch --array=0-24 --export=ALL,exp='vary-batchsize' submit_job.sbatch

elif [ $1 == 'scale-dataset' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-dataset' submit_job.sbatch

elif [ $1 == 'scale-width-lin' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-width-lin' submit_job.sbatch

elif [ $1 == 'scale-depth-lin' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-depth-lin' submit_job.sbatch

elif [ $1 == 'scale-width' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-width' submit_job.sbatch

elif [ $1 == 'scale-depth' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-depth' submit_job.sbatch

fi