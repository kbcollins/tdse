#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-forward
#SBATCH --time=02:00:00

#	echo $i
mkdir -p gausswavepacket
python tdse-amat-from-snapshots-script-kbc.py ./gausswavepacket
