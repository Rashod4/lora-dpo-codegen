#!/bin/bash
#SBATCH --job-name=dpo_v2_b010
#SBATCH -G 1
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/dpo_v2_beta010_%j.out
#SBATCH --error=logs/dpo_v2_beta010_%j.err

cd ~/scratch/cs4644-project
source .venv/bin/activate

python train_dpo.py \
    --sft_checkpoint checkpoints/sft_rank16 \
    --dpo_data dpo_pairs.jsonl \
    --beta 0.1 \
    --epochs 5 \
    --lr 1e-5 \
    --output_dir checkpoints/dpo_v2_beta010
