#!/bin/bash
#SBATCH --job-name=dpo_b05
#SBATCH -G 1
#SBATCH -C HX00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/dpo_beta05_%j.out
#SBATCH --error=logs/dpo_beta05_%j.err

cd ~/scratch/cs4644-project
source .venv/bin/activate

python train_dpo.py \
    --sft_checkpoint checkpoints/sft_rank16 \
    --dpo_data dpo_pairs.jsonl \
    --beta 0.05 \
    --epochs 1 \
    --output_dir checkpoints/dpo_beta05
