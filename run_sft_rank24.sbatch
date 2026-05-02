#!/bin/bash
#SBATCH --job-name=sft_rank24
#SBATCH -G 1
#SBATCH -C HX00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sft_rank24_%j.out
#SBATCH --error=logs/sft_rank24_%j.err

cd ~/scratch/cs4644-project
source .venv/bin/activate

python train_sft.py \
    --lora_rank 24 \
    --lora_alpha 48 \
    --epochs 3 \
    --output_dir checkpoints/sft_rank24
