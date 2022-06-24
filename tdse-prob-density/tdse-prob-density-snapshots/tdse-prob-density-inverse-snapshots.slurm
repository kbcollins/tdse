#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-inverse
#SBATCH --time=02:00:00

python tdse-prob-density-inverse-snapshots-script-kbc.py ./gausswavepacket
