#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=jax-gpu-test
#SBATCH --output=./jax-gpu-test.out
#SBATCH --error=./jax-gpu-test.err
#SBATCH --time=00:05:00

source activate work052022
python jax-test.py
