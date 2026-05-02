"""
DPO training on top of an SFT-LoRA checkpoint.
Reads preference pairs from JSONL ({prompt, chosen, rejected} as strings),
wraps them in chat template, and continues training the LoRA adapter
with DPO loss against the same model (adapter-disabled) as reference.
"""

import argparse
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def to_conversational(pair):
    """
    Convert raw-string {prompt, chosen, rejected} into conversational format.
    TRL applies the tokenizer's chat template automatically when it sees
    role/content dicts.
    """
    return {
        "prompt":   [{"role": "user",      "content": pair["prompt"]}],
        "chosen":   [{"role": "assistant", "content": pair["chosen"]}],
        "rejected": [{"role": "assistant", "content": pair["rejected"]}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", required=True,
                        help="LoRA adapter from SFT, e.g. checkpoints/sft_rank16")
    parser.add_argument("--dpo_data", required=True,
                        help="JSONL of preference pairs from Person B")
    parser.add_argument("--output_dir", default="checkpoints/dpo_main")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (smaller = stronger preference signal)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="DPO LR is much smaller than SFT (5e-7 to 5e-6 typical)")
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading base model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="bfloat16",
        device_map="cuda",
    )

    print(f"Loading SFT LoRA adapter from {args.sft_checkpoint}...")
    # is_trainable=True so DPO can continue updating the same LoRA weights.
    # ref_model=None tells TRL to use the SAME model with the adapter disabled
    # as the reference policy — clean PEFT-DPO trick, no second copy in memory.
    model = PeftModel.from_pretrained(
        base_model,
        args.sft_checkpoint,
        is_trainable=True,
    )
    model.print_trainable_parameters()

    print(f"Loading preference pairs from {args.dpo_data}...")
    pairs = load_pairs(args.dpo_data)
    if args.max_train_samples:
        pairs = pairs[: args.max_train_samples]
    print(f"Training on {len(pairs)} pairs")

    dataset = Dataset.from_list([to_conversational(p) for p in pairs])

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        max_length=1024,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
