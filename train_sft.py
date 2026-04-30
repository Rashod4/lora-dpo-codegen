"""
Supervised fine-tuning with LoRA on MBPP for code generation.
Uses Microsoft Phi-3.5-mini-instruct as the base model.
"""
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"


def format_mbpp_example(example):
    """Format MBPP into a prompt + completion text string."""
    # MBPP "sanitized" uses 'prompt' for the description, not 'text'
    description = example.get("prompt") or example.get("text") or ""
    prompt = f"# {description}\n"
    if example.get("test_list"):
        prompt += f"# Example: {example['test_list'][0]}\n"
    completion = example["code"]
    return {"text": prompt + completion}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_rank16")
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="If set, use only this many training examples (for smoke tests)",
    )
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="bfloat16",
        device_map="cuda",
        trust_remote_code=True,
    )

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "qkv_proj", "o_proj",
            "gate_up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading MBPP dataset...")
    dataset = load_dataset("mbpp", "sanitized")
    train_data = dataset["train"].map(
        format_mbpp_example,
        remove_columns=dataset["train"].column_names,
    )
    if args.max_train_samples:
        train_data = train_data.select(range(args.max_train_samples))
    print(f"Training on {len(train_data)} examples")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        args=sft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()


