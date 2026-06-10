#!/usr/bin/env python3
"""Generate all evaluation figures for Chapter 7."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).resolve().parent.parent.parent / "Proiect_de_diplomă" / "pics"
OUT.mkdir(parents=True, exist_ok=True)

GREY  = "#555555"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

GENS       = ["ProjectedGAN", "SD v1.5 LoRA", "SD3 LoRA"]
GENS_SHORT = ["ProjGAN",      "SD LoRA",      "SD3 LoRA"]

# ── Data ────────────────────────────────────────────────────────────────────
auc = {
    "SVD baseline":   [0.413, 0.572, 0.775],
    "Concat VAE":     [0.473, 0.524, 0.874],
    "Gated VAE":      [0.412, 0.572, 0.778],
    "Masked VAE":     [0.462, 0.529, 0.858],
    "ViT (supervised)": [0.961, 0.897, 0.954],
}

colors = {
    "SVD baseline":     "#7EB6D9",
    "Concat VAE":       "#2E86C1",
    "Gated VAE":        "#F0A500",
    "Masked VAE":       "#27AE60",
    "ViT (supervised)": "#C0392B",
}

# ── Figure 1: Ablation bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
models = list(auc.keys())
x = np.arange(len(GENS))
n = len(models)
w = 0.14
offsets = np.linspace(-(n-1)/2 * w, (n-1)/2 * w, n)

for i, (model, vals) in enumerate(auc.items()):
    bars = ax.bar(x + offsets[i], vals, w, label=model,
                  color=colors[model], edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=GREY)

ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Random (AUC=0.5)")
ax.set_xticks(x)
ax.set_xticklabels(GENS, fontsize=11)
ax.set_ylabel("AUC (ROC)", fontsize=11)
ax.set_ylim(0.35, 1.05)
ax.set_title("Ablation: AUC per model and generator", fontsize=12)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
plt.tight_layout()
fig.savefig(OUT / "eval_ablation_auc.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved eval_ablation_auc.png")

# ── Figure 2: Per-channel AUC on SD3 and SD LoRA ────────────────────────────
channels = ["SVD", "DCT", "R", "G", "B"]
step3_sd3  = [0.840, 0.860, 0.784, 0.815, 0.800]
step5_sd3  = [0.799, 0.855, 0.756, 0.765, 0.756]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

x = np.arange(len(channels))
w = 0.35
for ax, (s3, s5, title, ylim) in zip(
    axes,
    [
        (step3_sd3, step5_sd3, "SD3 LoRA — per-channel AUC", (0.70, 0.92)),
        ([0.530, 0.470, 0.605, 0.612, 0.616],
         [0.530, 0.465, 0.590, 0.600, 0.608],
         "SD v1.5 LoRA — per-channel AUC", (0.40, 0.70)),
    ]
):
    b3 = ax.bar(x - w/2, s3, w, label="Concat VAE (Step 3)", color="#2E86C1", edgecolor="white")
    b5 = ax.bar(x + w/2, s5, w, label="Masked VAE (Step 5)", color="#27AE60", edgecolor="white")
    for bars in [b3, b5]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color=GREY)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.9)

plt.tight_layout()
fig.savefig(OUT / "eval_perchannel_auc.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved eval_perchannel_auc.png")

# ── Figure 3: Robustness ─────────────────────────────────────────────────────
conditions = ["Clean", "JPEG Q=50", "JPEG Q=30", "JPEG Q=10",
              "Blur σ=1", "Blur σ=2", "Blur σ=3"]

rob = {
    "sd3": {
        "Step 3": [0.874, 0.891, 0.897, 0.899, 0.870, 0.867, 0.867],
        "Step 5": [0.858, 0.877, 0.885, 0.892, 0.873, 0.870, 0.866],
    },
    "sdlora": {
        "Step 3": [0.524, 0.481, 0.489, 0.502, 0.471, 0.460, 0.453],
        "Step 5": [0.529, 0.490, 0.504, 0.531, 0.489, 0.479, 0.471],
    },
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
titles = {"sd3": "Robustness — SD3 LoRA", "sdlora": "Robustness — SD v1.5 LoRA"}
ylims  = {"sd3": (0.83, 0.92), "sdlora": (0.42, 0.56)}

for ax, (key, data) in zip(axes, rob.items()):
    xs = range(len(conditions))
    ax.plot(xs, data["Step 3"], "o-", color="#2E86C1", linewidth=2,
            markersize=6, label="Concat VAE (Step 3)")
    ax.plot(xs, data["Step 5"], "s--", color="#27AE60", linewidth=2,
            markersize=6, label="Masked VAE (Step 5)")
    ax.axvline(3.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(1.75, ylims[key][0] + 0.004, "← JPEG compression", ha="center",
            fontsize=8, color="gray")
    ax.text(5.25, ylims[key][0] + 0.004, "Gaussian blur →", ha="center",
            fontsize=8, color="gray")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(*ylims[key])
    ax.set_title(titles[key], fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)

plt.tight_layout()
fig.savefig(OUT / "eval_robustness.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved eval_robustness.png")

print("All plots saved to", OUT)
