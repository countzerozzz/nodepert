#!/bin/sh

if [ $1 == 'trial' ]
then
    sbatch --array=0 --export=ALL,exp='trial' slurm-scripts/submit_job.sbatch

elif [ $1 == 'tmp-tf' ]
then
    sbatch --array=0 --export=ALL,exp='tmp-tf' slurm-scripts/submit_job.sbatch

elif [ $1 == 'conv-base' ]
then
    sbatch --array=0 --export=ALL,exp='conv-base' slurm-scripts/submit_job.sbatch

elif [ $1 == 'conv-large' ]
then
    sbatch --array=0 --export=ALL,exp='conv-large' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-dataset' ]
then
    sbatch --array=0-3 --export=ALL,exp='scale-dataset' slurm-scripts/submit_job.sbatch

elif [ $1 == 'linesearch' ]
then
    sbatch --array=0-4 --export=ALL,exp='linesearch' slurm-scripts/submit_job.sbatch

elif [ $1 == 'crash-dynamics' ]
then
    # sbatch --array=13 --export=ALL,exp='crash-dynamics' slurm-scripts/submit_job.sbatch
    sbatch --array=0-19 --export=ALL,exp='crash-dynamics' slurm-scripts/submit_job.sbatch

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

elif [ $1 == 'scale-width-lin' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-width-lin' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-depth-lin' ]
then
    sbatch --array=0-24 --export=ALL,exp='scale-depth-lin' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-width1' ]
then
    sbatch --array=0-49 --export=ALL,exp='scale-width1' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-width2' ]
then
    sbatch --array=0-49 --export=ALL,exp='scale-width2' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-width3' ]
then
    sbatch --array=0-49 --export=ALL,exp='scale-width3' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-width4' ]
then
    sbatch --array=0-49 --export=ALL,exp='scale-width4' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-depth1' ]
then
    sbatch --array=0-49 --export=ALL,exp='scale-depth1' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-depth2' ]
then
    sbatch --array=0-49 --export=ALL,exp='scale-depth2' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-depth3' ]
then
    sbatch --array=0-49 --export=ALL,exp='scale-depth3' slurm-scripts/submit_job.sbatch

elif [ $1 == 'scale-depth4' ]
then
    sbatch --array=0-49 --export=ALL,exp='scale-depth4' slurm-scripts/submit_job.sbatch

fi