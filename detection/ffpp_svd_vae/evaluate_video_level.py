#!/usr/bin/env python3
"""
Video-level AUC evaluation for the FF++ SVD U-Net VAE.

The per-frame MSE has variance that can wash out a weak per-frame signal.
By grouping the 32 frames of each video and averaging their reconstruction
errors before computing AUC, noise is reduced by √32 ≈ 5.7×. If there is
*any* consistent per-frame signal, video-level AUC will be visibly higher.

Sweeps across all saved epoch_NNN.pth checkpoints and reports per-epoch
video-level AUC per manipulation.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..svd_unet_vae.model import SVDUNetVAE
from .dataset import FFppEvalDataset, MANIPULATIONS, parse_source_id, _default_transform

SCRIPT_DIR = Path(__file__).resolve().parent


def _video_key(path: Path, is_fake: bool) -> str:
    """
    A unique video identifier.

    Real frame '035_0030.png'         → video key 'real/035'
    Fake  frame '035_036_0030.png'    → video key 'fake/035_036'

    Real and fake share source 035 but the eval task is "is this real or fake?"
    so we keep them distinct.
    """
    stem = path.stem
    parts = stem.split("_")
    if is_fake:
        # Fake frames: source_target_frameidx
        vid_id = "_".join(parts[:2])
        return f"fake/{vid_id}"
    else:
        # Real frames: source_frameidx
        return f"real/{parts[0]}"


def collect_video_scores(model, dataset, device, batch_size, num_workers):
    """
    Run the model over every sample in `dataset`, group MSE by video key,
    average per video. Returns (errors_per_video, labels_per_video) arrays.
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    model.eval()

    # Accumulate per-video sums + counts
    sums = defaultdict(float)
    counts = defaultdict(int)
    labels_map = {}  # video_key -> 0/1

    # Iterate in dataset order, which we need so we can recover paths in the same order
    idx_to_sample = dataset.samples  # list of (Path, label)
    cursor = 0
    with torch.inference_mode():
        for i_low, i_gray, labels in tqdm(loader, leave=False):
            i_low = i_low.to(device)
            i_gray = i_gray.to(device)
            recon, _, _ = model(i_low)
            err = F.mse_loss(recon, i_gray, reduction="none").mean(dim=[1, 2, 3]).cpu().numpy()
            labels_np = labels.numpy()
            bsz = len(labels_np)

            for i in range(bsz):
                path, _ = idx_to_sample[cursor + i]
                is_fake = bool(labels_np[i])
                key = _video_key(path, is_fake)
                sums[key] += float(err[i])
                counts[key] += 1
                labels_map[key] = int(labels_np[i])
            cursor += bsz

    # Average per video
    video_keys = sorted(sums.keys())
    errors = np.array([sums[k] / counts[k] for k in video_keys])
    labels = np.array([labels_map[k] for k in video_keys])
    return errors, labels, video_keys


def auc_for(model, data_dir, manipulation, device, batch_size, num_workers):
    transform = _default_transform()
    # Use the FULL eval set (not just the test subset) so each video has many
    # frames contributing to its average — gives the cleanest video-level result
    ds = FFppEvalDataset(data_dir, manipulation, subset="test", transform=transform, seed=42)
    errs, labels, video_keys = collect_video_scores(model, ds, device, batch_size, num_workers)
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    auc = float(roc_auc_score(labels, errs))
    return auc, n_real, n_fake


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data/ffpp"))
    parser.add_argument("--ckpt_dir", default=str(SCRIPT_DIR / "../../checkpoints/ffpp_svd_vae"))
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--manipulations", nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pth"))
    if not checkpoints:
        print(f"No epoch_*.pth files found in {ckpt_dir}")
        return

    data_root = Path(args.data_dir)
    if args.manipulations:
        manipulations = args.manipulations
    else:
        manipulations = [m for m in MANIPULATIONS if (data_root / m).is_dir()]

    print(f"Video-level sweep over {len(checkpoints)} checkpoints, {len(manipulations)} manipulations")
    print(f"Manipulations: {manipulations}\n")

    model = SVDUNetVAE(latent_dim=args.latent_dim).to(device)
    rows = []

    header = f"{'Epoch':>6}  " + "  ".join(f"{m[:12]:>12}" for m in manipulations) + f"  {'Avg':>6}"
    print(header)
    print("-" * len(header))

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt["epoch"] + 1

        aucs = []
        sizes = []
        for m in manipulations:
            auc, n_real, n_fake = auc_for(model, args.data_dir, m, device,
                                          args.batch, args.num_workers)
            aucs.append(auc)
            sizes.append((n_real, n_fake))
        avg = sum(aucs) / len(aucs)
        rows.append({
            "epoch": epoch,
            "aucs": dict(zip(manipulations, aucs)),
            "video_counts": dict(zip(manipulations, sizes)),
            "avg": avg,
        })
        print(f"{epoch:>6}  " + "  ".join(f"{a:>12.4f}" for a in aucs) + f"  {avg:>6.4f}")

    # Print video counts (helpful sanity check)
    print()
    print("Video counts (real, fake) per manipulation:")
    for m, (nr, nf) in zip(manipulations, sizes):
        print(f"  {m:<16} real={nr}  fake={nf}")

    with open(ckpt_dir / "video_level_sweep_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nResults saved to {ckpt_dir / 'video_level_sweep_results.json'}")

    best = max(rows, key=lambda r: r["avg"])
    print(f"\nBest epoch by avg video-level AUC: {best['epoch']} (avg={best['avg']:.4f})")


if __name__ == "__main__":
    main()
