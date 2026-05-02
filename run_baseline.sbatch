#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH -G 1
#SBATCH -C HX00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

cd ~/scratch/cs7643-project
source .venv/bin/activate

python run_baseline.py
