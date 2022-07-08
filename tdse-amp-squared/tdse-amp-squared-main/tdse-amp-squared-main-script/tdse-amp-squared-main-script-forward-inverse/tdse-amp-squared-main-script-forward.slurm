#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-forward
#SBATCH --output='out-forward'
#SBATCH --time=00:35:00

#mkdir -p v0
#python tdse-amp-squared-main-script-forward-kbc.py 0

for i in {0..6}
do
#	echo $i
	mkdir -p ./07-08-2022_numts-20_dt-1e-2/v$i
	python tdse-amp-squared-main-script-forward-kbc.py ./07-08-2022_numts-20_dt-1e-2 $i 20 1e-2
done
