#!/bin/bash
#SBATCH -p test
#SBATCH --constraint=gpu
#SBATCH --job-name=jax-gpu-test
#SBATCH --gres=gpu:2
#SBATCH --output=./jax-gpu-test.out
#SBATCH --error=./jax-gpu-test.err

python jax-test.py


