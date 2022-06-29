#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-forward
#SBATCH --time=00:05:00

#mkdir -p v0
#python tdse-amp-squared-main-script-forward-kbc.py 0

for i in {0..6}
do
#	echo $i
	mkdir -p v$i
	python tdse-amp-squared-main-script-forward-kbc.py $i
done
