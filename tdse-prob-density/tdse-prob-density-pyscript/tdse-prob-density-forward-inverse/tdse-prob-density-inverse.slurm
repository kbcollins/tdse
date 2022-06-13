#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-forward
#SBATCH --time=00:05:00

python tdse-prob-density-inverse-kbc.py ./v0

#for i in {0..6}
#do
##	echo $i
#	python tdse-prob-density-inverse-kbc.py ./v$i
#done
