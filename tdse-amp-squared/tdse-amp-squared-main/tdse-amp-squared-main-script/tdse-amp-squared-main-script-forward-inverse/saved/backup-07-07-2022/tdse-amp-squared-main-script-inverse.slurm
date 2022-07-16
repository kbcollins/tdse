#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-inverse
#SBATCH --output='out-inverse'
#SBATCH --time=00:35:00

#python tdse-wave-function-main-script-inverse-kbc.py ./v0

for i in {0..6}
do
#	echo $i
	python tdse-amp-squared-main-script-inverse-kbc.py ./v$i
done
