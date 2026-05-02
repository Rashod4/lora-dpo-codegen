#!/bin/bash
#SBATCH --job-name=sft_r16_n200
#SBATCH -G 1
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sft_r16_n200_%j.out
#SBATCH --error=logs/sft_r16_n200_%j.err

cd ~/scratch/cs4644-project
source .venv/bin/activate

python train_sft.py \
    --lora_rank 16 \
    --lora_alpha 32 \
    --epochs 3 \
    --max_train_samples 200 \
    --output_dir checkpoints/sft_rank16_n200
