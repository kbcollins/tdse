#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-inverse
#SBATCH --time=00:30:00

#python tdse-amp-squared-inverse-kbc.py ./v0

for i in {0..6}
do
#	echo $i
	python tdse-prob-density-inverse-kbc.py ./v$i
done
