#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=kbc-inverse
#SBATCH --time=02:00:00

#python tdse-amp-squared-snapshots-inverse-script-kbc.py ./gausswavepacket
python tdse-amp-squared-snapshots-inverse-script-kbc.py ./timepotential