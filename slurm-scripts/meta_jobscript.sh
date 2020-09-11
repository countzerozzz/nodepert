#!/bin/sh

if [ $1 == 'trial' ]
then
    sbatch --array=0 --export=ALL,exp='trial' slurm-scripts/submit_job.sbatch

elif [ $1 == 'linesearch' ]
then
    sbatch --array=0-5 --export=ALL,exp='linesearch' slurm-scripts/submit_job.sbatch

elif [ $1 == 'crash-dynamics' ]
then
    sbatch --array=13 --export=ALL,exp='crash-dynamics' slurm-scripts/submit_job.sbatch

elif [ $1 == 'weight-decay' ]
then
    sbatch --array=0-24 --export=ALL,exp='weight-decay' slurm-scripts/submit_job.sbatch

elif [ $1 == 'adam-update' ]
then
    sbatch --array=0-24 --export=ALL,exp='adam-update' slurm-scripts/submit_job.sbatch

elif [ $1 == 'vary-lr' ]
then
    sbatch --array=0-4 --export=ALL,exp='vary-lr' slurm-scripts/submit_job.sbatch

elif [ $1 == 'vary-batchsize' ]
then
    sbatch --array=0-24 --export=ALL,exp='vary-batchsize' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-dataset' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-dataset' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-width-lin' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-width-lin' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-depth-lin' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-depth-lin' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-width' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-width' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-depth' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-depth' slurm-scripts/submit_job.sbatch

fi