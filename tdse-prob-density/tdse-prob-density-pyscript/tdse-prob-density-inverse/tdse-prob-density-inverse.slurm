#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-forward
#SBATCH --time=00:05:00

for i in {0..6}
do
#	echo $i
	mkdir -p v$i
	python tdse-prob-density-forward-kbc.py $i
done
