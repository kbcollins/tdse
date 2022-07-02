#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-results
#SBATCH --time=00:35:00

for i in {0..6}
do
#	echo $i
	python tdse-amp-squared-main-script-results-kbc.py ./v$i
done
