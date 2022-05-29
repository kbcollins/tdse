#!/bin/bash
#SBATCH --partition test
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=jax_gpu_test
#SBATCH --output=./jax-gpu-test.out
#SBATCH --error=./jax-gpu-test.err

python jax-test.py


