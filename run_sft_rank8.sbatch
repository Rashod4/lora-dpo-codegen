#!/bin/bash
#SBATCH --job-name=sft_rank8
#SBATCH -G 1
#SBATCH -C HX00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sft_rank8_%j.out
#SBATCH --error=logs/sft_rank8_%j.err

cd ~/scratch/cs4644-project
source .venv/bin/activate

python train_sft.py \
    --lora_rank 8 \
    --lora_alpha 16 \
    --epochs 3 \
    --output_dir checkpoints/sft_rank8
