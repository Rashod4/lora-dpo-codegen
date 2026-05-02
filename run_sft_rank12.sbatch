#!/bin/bash
#SBATCH --job-name=sft_rank12
#SBATCH -G 1
#SBATCH -C HX00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sft_rank12_%j.out
#SBATCH --error=logs/sft_rank12_%j.err

cd ~/scratch/cs4644-project
source .venv/bin/activate

python train_sft.py \
    --lora_rank 12 \
    --lora_alpha 24 \
    --epochs 3 \
    --output_dir checkpoints/sft_rank12
