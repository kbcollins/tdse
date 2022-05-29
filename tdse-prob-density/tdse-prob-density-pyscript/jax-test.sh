#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=jax-gpu-test
#SBATCH --output=./jax-gpu-test.out
#SBATCH --error=./jax-gpu-test.err
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:2

python jax-test.py
