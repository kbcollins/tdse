#!/bin/bash
#SBATCH --partition=gpu           # select which resource to run your job on
#SBATCH --gpus=1                  # select how many gpus to use (each node as 2)
#SBATCH --job-name=inverse        # create a short name for the job
#SBATCH --output='out_%x'         # stdout file (%A: job ID, a%: job array index, x%: job name)
#SBATCH --time=01:00:00           # set total run time limit for job


######### HELPFUL ENVIRONMENT VARIABLES #########

# $SLURM_JOB_NAME - short name of job set by --job-name


################## BASH SCRIPT ##################

# define Bash variables
date="07-13-2022"
filelist="numts-20_dt-.010"
#filelist=( $(ls ./"$date") ) # bash array of files in directory ./"$date"

# run pyscript
#for i in {0..0}; do  # only do the first potential
for i in {0..6}; do
  for thisfile in "${filelist[@]}"; do
    mkdir -p ./"$date"/"$thisfile"/v"$i"/results-"$SLURM_JOB_NAME"
    python tdse-wave-function-chebyshev-script-"$SLURM_JOB_NAME"-kbc.py ./"$date"/"$thisfile"/v"$i"
  done
done