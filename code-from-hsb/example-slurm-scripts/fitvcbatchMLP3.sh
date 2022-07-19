#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=fitvc
#SBATCH --output=./fitvc.out
#SBATCH --error=./fitvc.err
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:2

source activate myjax
python fitvcbatchMLP3.py


