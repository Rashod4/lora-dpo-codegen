#!/bin/bash
#SBATCH --job-name=eval_dpo
#SBATCH -G 1
#SBATCH -C HX00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_dpo_%j.out
#SBATCH --error=logs/eval_dpo_%j.err

cd ~/scratch/cs7643-project
source .venv/bin/activate

if [ -z "$DPO_CHECKPOINT" ]; then
    echo "ERROR: set DPO_CHECKPOINT env var"
    exit 1
fi

python -u eval_checkpoint.py \
    --checkpoint "$DPO_CHECKPOINT" \
    --output "results_${RUN_NAME:-dpo_main}.jsonl"
