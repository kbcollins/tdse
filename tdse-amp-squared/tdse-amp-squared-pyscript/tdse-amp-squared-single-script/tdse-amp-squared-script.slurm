#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=tdse-amp-squared-pyscript-kbc
#SBATCH --output=./tdse-amp-squared-pyscript-kbc-output/tdse-amp-squared-pyscript-kbc.out
#SBATCH --error=./tdse-amp-squared-pyscript-kbc-output/tdse-amp-squared-pyscript-kbc.err
#SBATCH --time=00:30:00

pwd
conda info --all
python tdse-prob-density-pyscript-kbc.py
