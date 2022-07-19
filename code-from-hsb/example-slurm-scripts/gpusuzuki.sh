#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=suzuki
#SBATCH --output=./suzuki.out
#SBATCH --error=./suzuki.err
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:2

source activate myjax
python gpusuzuki.py


