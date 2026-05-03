"""
Build all figures for the report from results_*.jsonl files.
Run: python make_plots.py
Outputs to plots/ directory.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("plots", exist_ok=True)


def load_passes(path):
    """Return list of pass@1 booleans (one per HumanEval problem)."""
    if not os.path.exists(path):
        return None
    results = [json.loads(line) for line in open(path)]
    return [r["passes"][0] for r in results]


def pass_at_1(passes):
    return sum(passes) / len(passes) if passes else None


def bootstrap_ci(passes, n_iter=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(passes, dtype=float)
    samples = rng.choice(arr, size=(n_iter, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return lo, hi


# Load all results files
runs = {
    "Base":           "results_base_v2.jsonl",
    "SFT rank=8":     "results_sft_rank8.jsonl",
    "SFT rank=12":    "results_sft_rank12.jsonl",
    "SFT rank=16":    "results_sft_rank16.jsonl",
    "SFT rank=24":    "results_sft_rank24.jsonl",
    "SFT rank=32":    "results_sft_rank32.jsonl",
    "SFT n=120":      "results_sft_rank16_n120.jsonl",
    "SFT n=200":      "results_sft_rank16_n200.jsonl",
    "SFT seed=1":     "results_sft_rank16_seed1.jsonl",
    "SFT seed=2":     "results_sft_rank16_seed2.jsonl",
    "SFT seed=3":     "results_sft_rank16_seed3.jsonl",
    "DPO main":       "results_dpo_main.jsonl",
    "DPO β=0.05":     "results_dpo_v2_beta005.jsonl",
    "DPO β=0.10":     "results_dpo_v2_beta010.jsonl",
    "DPO β=0.50":     "results_dpo_v2_beta050.jsonl",
    "DPO aggressive": "results_dpo_aggressive.jsonl",
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

print(f"Loaded {len(data)} result files\n")

# === Figure 1: Headline bar chart of all conditions ===
print("Building Figure 1: Headline bar chart...")
fig, ax = plt.subplots(figsize=(13, 6))

# Color-code by stage: gray=base, blue=SFT variants, red=DPO variants
colors = []
for name in data:
    if "Base" in name:
        colors.append("#666666")
    elif "DPO" in name:
        colors.append("#d94a4a")
    else:
        colors.append("#4a90d9")

names = list(data.keys())
values = [data[n]["pass_at_1"] for n in names]
errs_lo = [data[n]["pass_at_1"] - data[n]["ci"][0] for n in names]
errs_hi = [data[n]["ci"][1] - data[n]["pass_at_1"] for n in names]

ax.bar(range(len(names)), values, yerr=[errs_lo, errs_hi], capsize=4,
       color=colors, edgecolor="black", linewidth=0.5)

# Reference lines
if "Base" in data:
    ax.axhline(data["Base"]["pass_at_1"], color="gray", linestyle="--",
               alpha=0.6, label=f"Base ({data['Base']['pass_at_1']:.3f})")
if "SFT rank=16" in data:
    ax.axhline(data["SFT rank=16"]["pass_at_1"], color="#4a90d9", linestyle=":",
               alpha=0.6, label=f"SFT rank=16 ({data['SFT rank=16']['pass_at_1']:.3f})")

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("HumanEval pass@1", fontsize=11)
ax.set_title("HumanEval pass@1 across all training conditions (95% bootstrap CI)",
             fontsize=12)
ax.legend(loc="lower left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(values) * 1.15)
plt.tight_layout()
plt.savefig("plots/fig1_headline.pdf", bbox_inches="tight")
plt.savefig("plots/fig1_headline.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved plots/fig1_headline.{pdf,png}")

# === Figure 2: LoRA rank ablation (H2) ===
print("Building Figure 2: LoRA rank ablation...")
ranks_data = {8: "SFT rank=8", 12: "SFT rank=12", 16: "SFT rank=16",
              24: "SFT rank=24", 32: "SFT rank=32"}
ranks = sorted([r for r, n in ranks_data.items() if n in data])
if len(ranks) >= 2:
    rank_values = [data[ranks_data[r]]["pass_at_1"] for r in ranks]
    rank_lo = [data[ranks_data[r]]["ci"][0] for r in ranks]
    rank_hi = [data[ranks_data[r]]["ci"][1] for r in ranks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, rank_values, "o-", markersize=10, linewidth=2,
            color="#4a90d9", label="SFT")
    ax.fill_between(ranks, rank_lo, rank_hi, alpha=0.2, color="#4a90d9")
    if "Base" in data:
        ax.axhline(data["Base"]["pass_at_1"], color="gray", linestyle="--",
                   label=f"Base ({data['Base']['pass_at_1']:.3f})")
    ax.set_xlabel("LoRA rank", fontsize=11)
    ax.set_ylabel("HumanEval pass@1", fontsize=11)
    ax.set_title("LoRA rank ablation (H2): all ranks regress vs base",
                 fontsize=12)
    ax.set_xticks(ranks)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/fig2_rank_ablation.pdf", bbox_inches="tight")
    plt.savefig("plots/fig2_rank_ablation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved plots/fig2_rank_ablation.{pdf,png}")

# === Figure 3: Data-size ablation ===
print("Building Figure 3: Data-size ablation...")
data_sizes = {120: "SFT n=120", 200: "SFT n=200", 374: "SFT rank=16"}
sizes = sorted([s for s, n in data_sizes.items() if n in data])
if len(sizes) >= 2:
    size_values = [data[data_sizes[s]]["pass_at_1"] for s in sizes]
    size_lo = [data[data_sizes[s]]["ci"][0] for s in sizes]
    size_hi = [data[data_sizes[s]]["ci"][1] for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, size_values, "o-", markersize=10, linewidth=2,
            color="#d94a90", label="SFT")
    ax.fill_between(sizes, size_lo, size_hi, alpha=0.2, color="#d94a90")
    if "Base" in data:
        ax.axhline(data["Base"]["pass_at_1"], color="gray", linestyle="--",
                   label=f"Base ({data['Base']['pass_at_1']:.3f})")
    ax.set_xlabel("Training set size (MBPP examples)", fontsize=11)
    ax.set_ylabel("HumanEval pass@1", fontsize=11)
    ax.set_title("SFT data-size ablation (rank=16)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/fig3_data_size.pdf", bbox_inches="tight")
    plt.savefig("plots/fig3_data_size.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved plots/fig3_data_size.{pdf,png}")

# === Figure 4: DPO beta sweep (the headline DPO result) ===
print("Building Figure 4: DPO beta sweep...")
beta_data = {0.05: "DPO β=0.05", 0.10: "DPO β=0.10", 0.50: "DPO β=0.50"}
betas = sorted([b for b, n in beta_data.items() if n in data])
if len(betas) >= 2:
    beta_values = [data[beta_data[b]]["pass_at_1"] for b in betas]
    beta_lo = [data[beta_data[b]]["ci"][0] for b in betas]
    beta_hi = [data[beta_data[b]]["ci"][1] for b in betas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(betas, beta_values, "o-", markersize=12, linewidth=2,
            color="#d94a4a", label="DPO checkpoints")
    ax.fill_between(betas, beta_lo, beta_hi, alpha=0.2, color="#d94a4a")

    # Reference: base and SFT
    if "Base" in data:
        ax.axhline(data["Base"]["pass_at_1"], color="gray", linestyle="--",
                   label=f"Base ({data['Base']['pass_at_1']:.3f})")
    if "SFT rank=16" in data:
        ax.axhline(data["SFT rank=16"]["pass_at_1"], color="#4a90d9", linestyle=":",
                   label=f"SFT rank=16 ({data['SFT rank=16']['pass_at_1']:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("DPO β (log scale)", fontsize=11)
    ax.set_ylabel("HumanEval pass@1", fontsize=11)
    ax.set_title("DPO beta sweep: smaller β (more drift) → more recovery",
                 fontsize=12)
    ax.set_xticks(betas)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/fig4_dpo_beta.pdf", bbox_inches="tight")
    plt.savefig("plots/fig4_dpo_beta.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved plots/fig4_dpo_beta.{pdf,png}")

# === Figure 5: Per-problem confusion (sft_rank16 vs base) ===
print("Building Figure 5: Per-problem confusion matrix...")
if "Base" in data and "SFT rank=16" in data:
    base_passes = load_passes("results_base_v2.jsonl")
    sft_passes = load_passes("results_sft_rank16.jsonl")

    both_pass = sum(1 for b, s in zip(base_passes, sft_passes) if b and s)
    both_fail = sum(1 for b, s in zip(base_passes, sft_passes) if not b and not s)
    base_only = sum(1 for b, s in zip(base_passes, sft_passes) if b and not s)
    sft_only = sum(1 for b, s in zip(base_passes, sft_passes) if not b and s)

    matrix = np.array([[both_pass, base_only], [sft_only, both_fail]])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    labels = [["Both pass\n", "Base passed\nSFT failed"],
              ["Base failed\nSFT passed", "Both fail\n"]]

    for i in range(2):
        for j in range(2):
            cell_color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, f"{labels[i][j]}{matrix[i,j]}",
                    ha="center", va="center", fontsize=12, color=cell_color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["SFT passed", "SFT failed"], fontsize=10)
    ax.set_yticklabels(["Base passed", "Base failed"], fontsize=10)
    ax.set_title(f"Per-problem outcomes: SFT rank=16 vs Base\n"
                 f"Net regression: {sft_only - base_only} problems "
                 f"({(sft_only - base_only) / 164 * 100:+.1f} pp)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/fig5_confusion.pdf", bbox_inches="tight")
    plt.savefig("plots/fig5_confusion.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved plots/fig5_confusion.{pdf,png}")

# === Print summary table ===
print()
print("=" * 75)
print(f"{'Run':<22} {'pass@1':<8} {'95% CI':<22} {'n':<5}")
print("-" * 75)
for name, d in data.items():
    ci_str = f"[{d['ci'][0]:.4f}, {d['ci'][1]:.4f}]"
    print(f"{name:<22} {d['pass_at_1']:<8.4f} {ci_str:<22} {d['n']:<5}")
print("=" * 75)
print(f"\nAll plots saved to plots/ directory.")
