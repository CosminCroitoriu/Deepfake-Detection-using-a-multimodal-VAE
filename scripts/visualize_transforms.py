#!/usr/bin/env python3
"""
Visualise the SVD and DCT modalities extracted from a single dataset image.

Outputs four panels:
  1. Original RGB
  2. SVD low-rank approximation  (I_low — what the model receives as channel 0)
  3. SVD residual                (I_gray − I_low, scaled ×3 for visibility)
  4. Block DCT coefficient map   (log-magnitude, normalised to [0, 1])

Usage:
    python scripts/visualize_transforms.py path/to/image.png
    python scripts/visualize_transforms.py path/to/image.png --out result.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.concat_vae.transform import (
    block_dct,
    DCT_BLOCK_SIZE,
    ENERGY_THRESHOLD,
    TARGET_SIZE,
)


def compute_svd_lowrank(gray_tensor: torch.Tensor, energy_thresh: float):
    """Return (i_low H×W float32 in [0,1], rank k) from a 1×H×W tensor."""
    mat = gray_tensor.squeeze(0)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    cumulative = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    k = int((cumulative >= energy_thresh).nonzero(as_tuple=False)[0].item()) + 1
    i_low = ((U[:, :k] * S[:k]) @ Vh[:k, :]).clamp(0.0, 1.0)
    return i_low, k


def main():
    parser = argparse.ArgumentParser(description="Visualise SVD and DCT transforms")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--out", default=None, help="Save figure to this path instead of displaying")
    parser.add_argument(
        "--energy", type=float, default=ENERGY_THRESHOLD,
        help=f"SVD energy threshold (default: {ENERGY_THRESHOLD})"
    )
    args = parser.parse_args()

    # ── Load and resize ──────────────────────────────────────────────────────
    img = Image.open(args.image).convert("RGB").resize(
        (TARGET_SIZE, TARGET_SIZE), Image.LANCZOS
    )
    rgb_t = TF.to_tensor(img)                                         # 3×H×W [0,1]
    gray_t = TF.rgb_to_grayscale(rgb_t, num_output_channels=1)        # 1×H×W [0,1]

    # ── SVD ──────────────────────────────────────────────────────────────────
    i_low, k = compute_svd_lowrank(gray_t, args.energy)                # H×W [0,1]
    i_gray = gray_t.squeeze(0)                                         # H×W [0,1]
    residual = (i_gray - i_low).numpy()

    # ── DCT ──────────────────────────────────────────────────────────────────
    dct_raw = block_dct(i_gray.numpy(), DCT_BLOCK_SIZE)
    dct_log = np.log1p(np.abs(dct_raw))
    dct_vis = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-8)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    axes[0].imshow(np.array(img))
    axes[0].set_title("Original (RGB)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(i_low.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"SVD low-rank  $I_{{\\mathrm{{low}}}}$\n"
        f"rank $k = {k}$ / {TARGET_SIZE}  ({int(args.energy * 100)}% energy)",
        fontsize=11,
    )
    axes[1].axis("off")

    # Centre residual at 0.5 and scale ×3 so small differences are visible
    axes[2].imshow(np.clip(residual * 3.0 + 0.5, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        "$I_{\\mathrm{gray}} - I_{\\mathrm{low}}$  (residual)\nscaled ×3, centred at 0.5",
        fontsize=11,
    )
    axes[2].axis("off")

    im = axes[3].imshow(dct_vis, cmap="inferno", vmin=0, vmax=1)
    n_blocks = TARGET_SIZE // DCT_BLOCK_SIZE
    axes[3].set_title(
        f"Block DCT  (log-magnitude)\n"
        f"{DCT_BLOCK_SIZE}×{DCT_BLOCK_SIZE} px blocks  →  {n_blocks}×{n_blocks} grid",
        fontsize=11,
    )
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(Path(args.image).name, fontsize=10)
    plt.tight_layout()

    out = Path(args.out) if args.out else None
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")

        stem = out.parent / out.stem
        svd_path = Path(f"{stem}_svd.png")
        dct_path = Path(f"{stem}_dct.png")
        plt.imsave(svd_path, i_low.numpy(), cmap="gray", vmin=0, vmax=1)
        plt.imsave(dct_path, dct_vis, cmap="inferno", vmin=0, vmax=1)
        print(f"Saved → {svd_path}")
        print(f"Saved → {dct_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
