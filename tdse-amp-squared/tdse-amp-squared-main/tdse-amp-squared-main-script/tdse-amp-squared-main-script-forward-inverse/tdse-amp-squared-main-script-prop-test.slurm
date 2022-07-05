#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-prop-test
#SBATCH --output='out-prop-test'
#SBATCH --time=00:30:00


for i in {0..6}
do
#	echo $i
	python tdse-amp-squared-main-script-prop-test-kbc.py ./v$i
done
