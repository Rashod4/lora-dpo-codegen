"""
Run base Phi-3.5-mini-instruct on HumanEval to get our floor pass@1.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_humaneval import run_eval

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

print(f"Loading {MODEL_NAME}...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="bfloat16",
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()


def generate(prompt: str) -> str:
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    full = tok.decode(out[0], skip_special_tokens=True)
    prompt_len = len(tok.decode(inputs["input_ids"][0], skip_special_tokens=True))
    return full[prompt_len:]


print("Running HumanEval (~15 min on H200)...")
results = run_eval(generate, n_samples=1, output_path="results_baseline.jsonl")
print(f"\nBaseline results: {results}")
