#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --gres=gpu:40g
#SBATCH --mem=16GB

# Your commands here
echo "Running on GPU with 40GB VRAM, 16GB RAM"
nvidia-smi
python3 entry_point.py
