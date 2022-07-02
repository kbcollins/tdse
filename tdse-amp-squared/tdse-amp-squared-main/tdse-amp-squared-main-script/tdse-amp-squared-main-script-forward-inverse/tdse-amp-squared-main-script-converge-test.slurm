#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-converge-test
#SBATCH --output='out-converge-test'
#SBATCH --time=02:00:00


for i in {0..6}
do
#	echo $i
	python tdse-amp-squared-main-script-converge-test-kbc.py ./v$i
done
