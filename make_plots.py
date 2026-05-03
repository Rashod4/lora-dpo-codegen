"""
Build all figures for the report from results_*.jsonl files.
Run: python make_plots.py
Outputs to plots/.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("plots", exist_ok=True)


def load_passes(path):
    if not os.path.exists(path):
        return None
    results = [json.loads(line) for line in open(path)]
    return [r["passes"][0] for r in results]


def pass_at_1(passes):
    return sum(passes) / len(passes) if passes else None


def bootstrap_ci(passes, n_iter=10000, ci=0.95):
    arr = np.array(passes, dtype=float)
    samples = np.random.choice(arr, size=(n_iter, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return lo, hi


# All possible runs and their result file paths
runs = {
    "Base":             "results_base_v2.jsonl",
    "SFT rank=8":       "results_sft_rank8.jsonl",
    "SFT rank=12":      "results_sft_rank12.jsonl",
    "SFT rank=16":      "results_sft_rank16.jsonl",
    "SFT rank=24":      "results_sft_rank24.jsonl",
    "SFT rank=32":      "results_sft_rank32.jsonl",
    "SFT n=120":        "results_sft_rank16_n120.jsonl",
    "SFT n=200":        "results_sft_rank16_n200.jsonl",
    "SFT seed=1":       "results_sft_rank16_seed1.jsonl",
    "SFT seed=2":       "results_sft_rank16_seed2.jsonl",
    "SFT seed=3":       "results_sft_rank16_seed3.jsonl",
    "SFT+DPO":          "results_dpo_main.jsonl",
}

data = {}
for name, path in runs.items():
    passes = load_passes(path)
    if passes is not None:
        data[name] = {
            "pass_at_1": pass_at_1(passes),
            "ci": bootstrap_ci(passes),
            "n": len(passes),
        }

# === Figure 1: Headline bar chart of all available pass@1 numbers ===
fig, ax = plt.subplots(figsize=(11, 6))
names = list(data.keys())
values = [data[n]["pass_at_1"] for n in names]
errs_lo = [data[n]["pass_at_1"] - data[n]["ci"][0] for n in names]
errs_hi = [data[n]["ci"][1] - data[n]["pass_at_1"] for n in names]
ax.bar(range(len(names)), values, yerr=[errs_lo, errs_hi], capsize=5, color='#4a90d9')
if "Base" in data:
    ax.axhline(data["Base"]["pass_at_1"], color='gray', linestyle='--', alpha=0.6, label='Base')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_ylabel("HumanEval pass@1")
ax.set_title("HumanEval pass@1 across training conditions (95% bootstrap CI)")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("plots/fig1_headline.pdf", bbox_inches='tight')
plt.savefig("plots/fig1_headline.png", dpi=200, bbox_inches='tight')
print("Saved plots/fig1_headline.{pdf,png}")

# === Figure 2: LoRA rank ablation (H2) ===
ranks_data = {8: "SFT rank=8", 12: "SFT rank=12", 16: "SFT rank=16",
              24: "SFT rank=24", 32: "SFT rank=32"}
ranks = sorted([r for r, n in ranks_data.items() if n in data])
if len(ranks) >= 2:
    rank_values = [data[ranks_data[r]]["pass_at_1"] for r in ranks]
    rank_lo = [data[ranks_data[r]]["ci"][0] for r in ranks]
    rank_hi = [data[ranks_data[r]]["ci"][1] for r in ranks]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ranks, rank_values, 'o-', markersize=10, linewidth=2, color='#4a90d9', label='SFT')
    ax.fill_between(ranks, rank_lo, rank_hi, alpha=0.2, color='#4a90d9')
    if "Base" in data:
        ax.axhline(data["Base"]["pass_at_1"], color='gray', linestyle='--', label='Base')
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("HumanEval pass@1")
    ax.set_title("LoRA rank ablation (H2)")
    ax.set_xticks(ranks)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/fig2_rank_ablation.pdf", bbox_inches='tight')
    plt.savefig("plots/fig2_rank_ablation.png", dpi=200, bbox_inches='tight')
    print("Saved plots/fig2_rank_ablation.{pdf,png}")

# === Figure 3: Data-size ablation (H3 setup) ===
data_sizes = {120: "SFT n=120", 200: "SFT n=200", 374: "SFT rank=16"}
sizes = sorted([s for s, n in data_sizes.items() if n in data])
if len(sizes) >= 2:
    size_values = [data[data_sizes[s]]["pass_at_1"] for s in sizes]
    size_lo = [data[data_sizes[s]]["ci"][0] for s in sizes]
    size_hi = [data[data_sizes[s]]["ci"][1] for s in sizes]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, size_values, 'o-', markersize=10, linewidth=2, color='#d94a90', label='SFT')
    ax.fill_between(sizes, size_lo, size_hi, alpha=0.2, color='#d94a90')
    if "Base" in data:
        ax.axhline(data["Base"]["pass_at_1"], color='gray', linestyle='--', label='Base')
    ax.set_xlabel("Training set size (MBPP examples)")
    ax.set_ylabel("HumanEval pass@1")
    ax.set_title("SFT data-size ablation (H3 setup)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/fig3_data_size.pdf", bbox_inches='tight')
    plt.savefig("plots/fig3_data_size.png", dpi=200, bbox_inches='tight')
    print("Saved plots/fig3_data_size.{pdf,png}")

# === Print summary table ===
print()
print(f"{'Run':<22} {'pass@1':<8} {'95% CI':<22} {'n':<5}")
print("-" * 60)
for name, d in data.items():
    ci_str = f"[{d['ci'][0]:.4f}, {d['ci'][1]:.4f}]"
    print(f"{name:<22} {d['pass_at_1']:<8.4f} {ci_str:<22} {d['n']:<5}")
