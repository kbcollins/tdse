#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-forward
#SBATCH --time=02:00:00

#	echo $i
mkdir -p timepotential
python tdse-amp-squared-time-potential-forward-script-kbc.py ./timepotential
