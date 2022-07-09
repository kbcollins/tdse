#!/bin/bash
#SBATCH --partition=gpu           # select which resource to run your job on
#SBATCH --gpus=1                  # select how many gpus to use (each node as 2)
#SBATCH --job-name=inverse        # create a short name for the job
#SBATCH --output='out_%x'         # stdout file (%A: job ID, a%: job array index, x%: job name)
#SBATCH --time=03:00:00           # set total run time limit for job


################## BASH SCRIPT ##################

#for i in {0..6}
#do
##	echo $i
#	python tdse-amp-squared-main-script-inverse-kbc.py ./v$i
#done


# Define Bash variables
date="07-08-2022"
numtsargs=(20 50 80 110 140)
dtargs=("1e-2" "1e-3" "1e-4" "1e-5" "1e-6")