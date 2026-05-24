#!/usr/bin/env python3
"""
Quick reconstruction sanity check for the FF++ SVD U-Net VAE.

For a handful of real and fake test-set images, saves a grid showing:
  input (i_low) | target (i_gray) | model reconstruction | abs-error heatmap

Also prints per-sample MSE so you can see directly whether real and fake
distributions overlap or separate. This is the diagnostic that confirms the
code/model are functioning and that the structural ceiling is the limit —
not a bug.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ..svd_unet_vae.model import SVDUNetVAE
from .dataset import (
    FFppEvalDataset, MANIPULATIONS, _default_transform,
    parse_source_id,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def to_display(tensor_1xHxW: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] grayscale tensor to a [0, 1] numpy array for matplotlib."""
    return ((tensor_1xHxW.squeeze(0).cpu().numpy() + 1.0) / 2.0).clip(0, 1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data/ffpp"))
    parser.add_argument("--ckpt", default=str(SCRIPT_DIR / "../../checkpoints/ffpp_svd_vae/best.pth"))
    parser.add_argument("--output", default=str(SCRIPT_DIR / "../../checkpoints/ffpp_svd_vae/recon_visualization.png"))
    parser.add_argument("--manipulation", default="Deepfakes",
                        help="Which fake category to compare against real")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of real and fake samples to visualise (each)")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = SVDUNetVAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build eval dataset on the test split; we only need a few samples
    ds = FFppEvalDataset(
        args.data_dir, args.manipulation, subset="test",
        transform=_default_transform(), seed=args.seed,
    )

    # Pull the first n_samples real and n_samples fake samples
    real_samples, fake_samples = [], []
    for path, label in ds.samples:
        if label == 0 and len(real_samples) < args.n_samples:
            real_samples.append(path)
        elif label == 1 and len(fake_samples) < args.n_samples:
            fake_samples.append(path)
        if len(real_samples) >= args.n_samples and len(fake_samples) >= args.n_samples:
            break

    print(f"Collected {len(real_samples)} real and {len(fake_samples)} fake samples")

    transform = _default_transform()

    rows = []   # each row is (label_str, i_low, i_gray, recon, mse_value)
    for src_paths, label_name in [(real_samples, "real"), (fake_samples, "fake")]:
        for path in src_paths:
            img = Image.open(path).convert("RGB")
            i_low, i_gray = transform(img)
            i_low = i_low.unsqueeze(0).to(device)
            i_gray = i_gray.unsqueeze(0).to(device)
            with torch.inference_mode():
                recon, _, _ = model(i_low)
            mse = F.mse_loss(recon, i_gray).item()
            rows.append((label_name, path.stem, i_low[0], i_gray[0], recon[0], mse))

    # Print per-sample MSE side by side so you can see overlap directly
    print("\nPer-sample MSE (recon vs i_gray):")
    real_mses = [m for label, *_, m in rows if label == "real"]
    fake_mses = [m for label, *_, m in rows if label == "fake"]
    for label, name, _, _, _, m in rows:
        print(f"  {label:>4}  {name:<24}  MSE={m:.5f}")
    print(f"\n  real mean: {np.mean(real_mses):.5f} ± {np.std(real_mses):.5f}")
    print(f"  fake mean: {np.mean(fake_mses):.5f} ± {np.std(fake_mses):.5f}")
    print(f"  gap (fake - real): {np.mean(fake_mses) - np.mean(real_mses):.5f}")

    # Build matplotlib grid: rows = samples, cols = (input | target | recon | error)
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Input (I_low)", "Target (I_gray)", "Reconstruction", "|error|"]
    for ax_row, (label, name, i_low, i_gray, recon, mse) in zip(axes, rows):
        i_low_img = to_display(i_low)
        i_gray_img = to_display(i_gray)
        recon_img = to_display(recon)
        err_img = np.abs(i_gray_img - recon_img)

        ax_row[0].imshow(i_low_img, cmap="gray", vmin=0, vmax=1)
        ax_row[1].imshow(i_gray_img, cmap="gray", vmin=0, vmax=1)
        ax_row[2].imshow(recon_img, cmap="gray", vmin=0, vmax=1)
        ax_row[3].imshow(err_img, cmap="hot", vmin=0, vmax=0.3)

        ax_row[0].set_ylabel(f"{label}\n{name}\nMSE={mse:.4f}", rotation=0,
                             labelpad=70, va="center", fontsize=9)
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved visualisation to {output_path}")


if __name__ == "__main__":
    main()
