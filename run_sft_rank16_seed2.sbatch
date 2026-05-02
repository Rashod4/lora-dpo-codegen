#!/bin/bash
#SBATCH --job-name=sft_r16_s2
#SBATCH -G 1
#SBATCH -C HX00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sft_r16_s2_%j.out
#SBATCH --error=logs/sft_r16_s2_%j.err

cd ~/scratch/cs4644-project
source .venv/bin/activate

python train_sft.py \
    --lora_rank 16 \
    --lora_alpha 32 \
    --epochs 3 \
    --seed 2 \
    --output_dir checkpoints/sft_rank16_seed2
