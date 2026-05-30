#!/usr/bin/env python3
"""
Step 7 — Explainability plots for the diploma thesis.

Produces three figures saved under checkpoints/_ablation/plots/:

  fig1_per_channel_auc.{png,pdf}
      Bar chart: per-channel ROC-AUC for SVD / DCT / R / G / B modalities,
      one subplot per generator. Built from existing results.json files —
      no inference required.

  fig2_mse_histograms.{png,pdf}
      Overlapping density histograms: real vs fake reconstruction MSE for each
      generator. Shows how well the total anomaly score separates the two classes.
      Requires loading the concat_vae checkpoint and running inference on the test
      set. Raw per-sample scores are cached to
        checkpoints/_ablation/raw_scores_concat.npz
      so subsequent runs skip inference.

  fig3_heatmaps.{png,pdf}
      Qualitative per-image reconstruction error maps. For each generator, shows
      one real example and one fake example side-by-side with their total MSE
      heatmap, DCT-channel error, and SVD-channel error.

Usage (cluster, first run):
  python scripts/plot_step7_explainability.py

Usage (cluster, skip inference after first run):
  python scripts/plot_step7_explainability.py --skip_inference

Flags:
  --checkpoints_dir   default: <project_root>/checkpoints
  --data_dir          default: <project_root>/data
  --out_dir           default: checkpoints/_ablation/plots
  --ckpt              override concat_vae/best.pth path
  --latent_dim        default: 256
  --batch             default: 32
  --num_workers       default: 4
  --skip_inference    skip Figures 2 and 3 (only bar chart)
  --n_heatmap         examples per class per generator for heatmaps (default: 1)
  --heatmap_seed      random seed for picking heatmap examples (default: 7)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # no display needed on cluster
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from detection.concat_vae.dataset import EvalDataset
from detection.concat_vae.model import ConcatUNetVAE
from detection.concat_vae.transform import ConcatTransform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENERATORS = ["projected_gan_512", "sd_lora", "sd3_lora"]
GENERATOR_LABELS = {
    "projected_gan_512": "ProjectedGAN",
    "sd_lora":           "SD LoRA",
    "sd3_lora":          "SD3 LoRA",
}
CHANNEL_NAMES = ["SVD", "DCT", "R", "G", "B"]
CHANNEL_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f", "#76b7b2"]

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


# ---------------------------------------------------------------------------
# Helpers — load existing JSON results
# ---------------------------------------------------------------------------
def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _results_to_per_channel(results_list):
    """Return {generator: {channel_lower: auc}} from a results.json list."""
    if not results_list:
        return {}
    return {r["generator"]: r.get("per_channel_auc", {}) for r in results_list}


def _results_to_overall_auc(results_list):
    """Return {generator: roc_auc} from a results.json list."""
    if not results_list:
        return {}
    return {r["generator"]: r["roc_auc"] for r in results_list}


# ---------------------------------------------------------------------------
# Helpers — inference
# ---------------------------------------------------------------------------
def _collect_scores(model, loader, device):
    """Single forward pass; return (errors [N], errors_pc [N,5], labels [N])."""
    model.eval()
    all_e, all_pc, all_lbl = [], [], []
    with torch.inference_mode():
        for x_in, x_tgt, lbl in tqdm(loader, leave=False):
            x_in, x_tgt = x_in.to(device), x_tgt.to(device)
            recon, _, _ = model(x_in)
            err_map = F.mse_loss(recon, x_tgt, reduction="none")  # [B,5,H,W]
            err_pc  = err_map.mean(dim=[2, 3])                     # [B,5]
            err     = err_pc.mean(dim=1)                           # [B]
            all_e.append(err.cpu().numpy())
            all_pc.append(err_pc.cpu().numpy())
            all_lbl.append(lbl.numpy())
    return (np.concatenate(all_e),
            np.concatenate(all_pc, axis=0),
            np.concatenate(all_lbl))


def load_or_compute_scores(model, data_dir, cache_path: Path, device, batch_size, num_workers):
    """Load raw per-sample scores from cache or run inference and save."""
    if cache_path.exists():
        print(f"  Loading cached raw scores from {cache_path}")
        data = np.load(cache_path)
        return {
            gen: {
                "errors":    data[f"{gen}_errors"],
                "errors_pc": data[f"{gen}_errors_pc"],
                "labels":    data[f"{gen}_labels"],
            }
            for gen in GENERATORS
        }

    print("  Running inference to collect raw scores ...")
    scores = {}
    transform = ConcatTransform()
    for gen in GENERATORS:
        test_ds = EvalDataset(data_dir, gen, subset="test", transform=transform, seed=42)
        loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        errors, errors_pc, labels = _collect_scores(model, loader, device)
        scores[gen] = {"errors": errors, "errors_pc": errors_pc, "labels": labels}
        auc = roc_auc_score(labels, errors)
        print(f"    {gen}: AUC={auc:.4f}  (n={len(errors)})")

    flat = {}
    for gen, d in scores.items():
        flat[f"{gen}_errors"]    = d["errors"]
        flat[f"{gen}_errors_pc"] = d["errors_pc"]
        flat[f"{gen}_labels"]    = d["labels"]
    np.savez(cache_path, **flat)
    print(f"  Saved raw scores to {cache_path}")
    return scores


# ---------------------------------------------------------------------------
# Helpers — heatmap examples
# ---------------------------------------------------------------------------
def collect_heatmap_examples(model, data_dir, generator, device, n_per_class=1, seed=7):
    """
    Return a list of dicts (n_per_class real + n_per_class fake), each with:
      label         : 0 (real) or 1 (fake)
      image_rgb     : np.ndarray (H, W, 3) in [0, 1]
      error_total   : np.ndarray (H, W) — mean MSE across 5 channels
      error_pc      : np.ndarray (5, H, W) — per-channel spatial MSE
    """
    transform = ConcatTransform()
    test_ds = EvalDataset(data_dir, generator, subset="test", transform=transform, seed=42)

    rng = np.random.default_rng(seed)
    real_idx = [i for i, (_, lbl) in enumerate(test_ds.samples) if lbl == 0]
    fake_idx = [i for i, (_, lbl) in enumerate(test_ds.samples) if lbl == 1]
    chosen = (list(rng.choice(real_idx, min(n_per_class, len(real_idx)), replace=False)) +
              list(rng.choice(fake_idx, min(n_per_class, len(fake_idx)), replace=False)))

    examples = []
    model.eval()
    with torch.inference_mode():
        for idx in chosen:
            x_in, x_tgt, label = test_ds[idx]
            x_in  = x_in.unsqueeze(0).to(device)
            x_tgt = x_tgt.unsqueeze(0).to(device)
            recon, _, _ = model(x_in)
            err_map = F.mse_loss(recon, x_tgt, reduction="none")[0].cpu().numpy()  # [5,H,W]
            # Recover RGB from target channels 2–4 ([-1,1] → [0,1])
            rgb = np.clip((x_tgt[0, 2:5].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
            examples.append({
                "label":       int(label),
                "image_rgb":   rgb,
                "error_total": err_map.mean(axis=0),
                "error_pc":    err_map,
            })

    # sort: real before fake for consistent row order
    examples.sort(key=lambda e: e["label"])
    return examples


# ---------------------------------------------------------------------------
# Figure 1 — per-channel AUC bar chart
# ---------------------------------------------------------------------------
def plot_per_channel_auc(step3_pc, overall_auc3, out_dir: Path):
    """
    One subplot per generator. 5 bars (SVD/DCT/R/G/B) per subplot.
    Dashed line at 0.5 (random). Dotted line at the combined AUC for that generator.
    """
    fig, axes = plt.subplots(1, len(GENERATORS), figsize=(13, 4.2), sharey=True)

    x = np.arange(len(CHANNEL_NAMES))
    bar_w = 0.55

    for ax, gen in zip(axes, GENERATORS):
        pc = step3_pc.get(gen, {})
        vals = [pc.get(ch.lower(), 0.0) for ch in CHANNEL_NAMES]

        bars = ax.bar(x, vals, width=bar_w, color=CHANNEL_COLORS,
                      edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1.0,
                   zorder=2, label="Random (0.5)")

        overall = overall_auc3.get(gen)
        if overall is not None:
            ax.axhline(overall, color="#222222", linestyle=":", linewidth=1.4,
                       zorder=2, label=f"Combined AUC = {overall:.3f}")

        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

        ax.set_title(GENERATOR_LABELS.get(gen, gen), pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(CHANNEL_NAMES)
        ax.set_ylim(0.30, 1.02)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

        if gen == GENERATORS[0]:
            ax.set_ylabel("ROC-AUC")
        if gen == GENERATORS[-1]:
            ax.legend(loc="lower right", framealpha=0.85)

    fig.suptitle("Per-channel ROC-AUC — Concat Multimodal VAE (Step 3)", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_per_channel_auc")


# ---------------------------------------------------------------------------
# Figure 2 — MSE score histograms
# ---------------------------------------------------------------------------
def plot_mse_histograms(scores: dict, out_dir: Path):
    """
    One subplot per generator. Real (blue) vs fake (orange) density histograms
    of the per-sample mean reconstruction MSE. AUC shown as text annotation.
    """
    fig, axes = plt.subplots(1, len(GENERATORS), figsize=(13, 4.2))

    for ax, gen in zip(axes, GENERATORS):
        d = scores.get(gen)
        if d is None:
            ax.set_visible(False)
            continue

        errors, labels = d["errors"], d["labels"]
        real_e = errors[labels == 0]
        fake_e = errors[labels == 1]
        cap = np.percentile(errors, 99)
        bins = np.linspace(0.0, cap, 60)

        ax.hist(real_e[real_e <= cap], bins=bins, density=True, alpha=0.65,
                color="#4e79a7", label=f"Real  (n={len(real_e)})")
        ax.hist(fake_e[fake_e <= cap], bins=bins, density=True, alpha=0.65,
                color="#f28e2b", label=f"Fake  (n={len(fake_e)})")

        auc = roc_auc_score(labels, errors)
        ax.text(0.97, 0.97, f"AUC = {auc:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(GENERATOR_LABELS.get(gen, gen), pad=6)
        ax.set_xlabel("Mean reconstruction MSE")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", framealpha=0.85)
        if gen == GENERATORS[0]:
            ax.set_ylabel("Density")

    fig.suptitle("Reconstruction MSE Distributions: Real vs Fake — Step 3", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_mse_histograms")


# ---------------------------------------------------------------------------
# Figure 3 — reconstruction heatmaps
# ---------------------------------------------------------------------------
def plot_heatmaps(all_examples: dict, out_dir: Path):
    """
    Grid: rows = (real, fake) per generator (6 rows total).
    Cols: Original RGB | Total MSE | DCT channel MSE | SVD channel MSE.
    """
    col_titles = ["Original", "Total MSE", "DCT error", "SVD error"]
    n_cols = len(col_titles)

    rows = []
    for gen in GENERATORS:
        for ex in all_examples.get(gen, []):
            rows.append((gen, ex))

    if not rows:
        print("  No heatmap examples collected — skipping fig3")
        return

    n_rows = len(rows)
    row_h  = 3.2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.6, row_h * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, (gen, ex) in enumerate(rows):
        label_str = "Real" if ex["label"] == 0 else "Fake"
        row_title = f"{GENERATOR_LABELS.get(gen, gen)} — {label_str}"

        rgb       = ex["image_rgb"]          # (H,W,3) [0,1]
        err_total = ex["error_total"]        # (H,W)
        err_dct   = ex["error_pc"][1]        # ch 1 = DCT
        err_svd   = ex["error_pc"][0]        # ch 0 = SVD

        # Column 0: original image
        axes[ri, 0].imshow(rgb)
        axes[ri, 0].axis("off")
        # Row label on left margin
        axes[ri, 0].text(-0.04, 0.5, row_title, transform=axes[ri, 0].transAxes,
                         rotation=90, ha="right", va="center", fontsize=9)

        # Columns 1–3: error heatmaps with individual colourbars
        for ci, err in enumerate([err_total, err_dct, err_svd], start=1):
            vmax = np.percentile(err, 98)
            im = axes[ri, ci].imshow(err, cmap="hot", vmin=0, vmax=max(vmax, 1e-8))
            axes[ri, ci].axis("off")
            fig.colorbar(im, ax=axes[ri, ci], fraction=0.046, pad=0.04, shrink=0.85)

    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=11, pad=4)

    fig.suptitle("Per-image Reconstruction Error Heatmaps — Concat VAE (Step 3)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_heatmaps")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _save(fig, out_dir: Path, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  Saved {stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoints_dir", default=str(PROJECT_DIR / "checkpoints"))
    ap.add_argument("--data_dir",        default=str(PROJECT_DIR / "data"))
    ap.add_argument("--out_dir",         default=None)
    ap.add_argument("--ckpt",            default=None, help="Override concat_vae/best.pth")
    ap.add_argument("--latent_dim",      type=int,   default=256)
    ap.add_argument("--batch",           type=int,   default=32)
    ap.add_argument("--num_workers",     type=int,   default=4)
    ap.add_argument("--skip_inference",  action="store_true",
                    help="Plot only Figure 1 (bar chart); skip inference-based figures")
    ap.add_argument("--n_heatmap",       type=int,   default=1,
                    help="Real+fake examples per generator for fig3 (default: 1)")
    ap.add_argument("--heatmap_seed",    type=int,   default=7,
                    help="RNG seed for picking heatmap examples (default: 7)")
    args = ap.parse_args()

    ck      = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (ck / "_ablation" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 7 — Explainability plots ===")
    print(f"Output directory: {out_dir}\n")

    # ---- Load existing results.json ----------------------------------------
    step3 = _load_json(ck / "concat_vae"  / "results.json")
    step3_pc  = _results_to_per_channel(step3)
    overall3  = _results_to_overall_auc(step3)

    if not step3_pc:
        print("WARNING: concat_vae/results.json not found or empty — "
              "Figure 1 will be empty.")

    # ---- Figure 1 (no inference needed) ------------------------------------
    print("Plotting Figure 1: per-channel AUC bar chart ...")
    plot_per_channel_auc(step3_pc, overall3, out_dir)

    if args.skip_inference:
        print("\n--skip_inference set; Figures 2 and 3 skipped.")
        print(f"Done. Plots in {out_dir}")
        return

    # ---- Load concat_vae model --------------------------------------------
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt) if args.ckpt else (ck / "concat_vae" / "best.pth")
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}. "
              f"Use --ckpt to specify the path.")
        sys.exit(1)

    print(f"Loading concat_vae checkpoint from {ckpt_path} ...")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = ConcatUNetVAE(latent_dim=args.latent_dim, in_channels=5, out_channels=5).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded (device={device})\n")

    # ---- Figure 2 — raw score histograms ----------------------------------
    print("Collecting raw per-sample scores for Figure 2 ...")
    cache_path = ck / "_ablation" / "raw_scores_concat.npz"
    scores = load_or_compute_scores(
        model, args.data_dir, cache_path, device, args.batch, args.num_workers
    )
    print("Plotting Figure 2: MSE histograms ...")
    plot_mse_histograms(scores, out_dir)

    # ---- Figure 3 — heatmaps ----------------------------------------------
    print(f"\nCollecting heatmap examples "
          f"({args.n_heatmap} real + {args.n_heatmap} fake per generator, seed={args.heatmap_seed}) ...")
    all_examples = {}
    for gen in GENERATORS:
        all_examples[gen] = collect_heatmap_examples(
            model, args.data_dir, gen, device,
            n_per_class=args.n_heatmap, seed=args.heatmap_seed
        )
    print("Plotting Figure 3: reconstruction heatmaps ...")
    plot_heatmaps(all_examples, out_dir)

    print(f"\n=== Done. All plots saved to {out_dir} ===")
    print("  fig1_per_channel_auc.png/pdf")
    print("  fig2_mse_histograms.png/pdf")
    print("  fig3_heatmaps.png/pdf")


if __name__ == "__main__":
    main()
