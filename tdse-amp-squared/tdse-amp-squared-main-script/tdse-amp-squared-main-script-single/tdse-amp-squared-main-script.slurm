#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=tdse-amp-squared-main-script-kbc
#SBATCH --output=./tdse-amp-squared-main-script-kbc-output/tdse-amp-squared-main-script-kbc.out
#SBATCH --error=./tdse-amp-squared-main-script-kbc-output/tdse-amp-squared-main-script-kbc.err
#SBATCH --time=00:30:00

pwd
conda info --all
python tdse-prob-density-pyscript-kbc.py
