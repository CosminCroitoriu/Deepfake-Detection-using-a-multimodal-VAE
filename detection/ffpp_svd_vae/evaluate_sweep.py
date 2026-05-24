#!/usr/bin/env python3
"""
Evaluate the FF++ SVD U-Net VAE across all saved per-epoch checkpoints.

Useful when the val-loss minimum doesn't coincide with the best AUC — common
in unsupervised anomaly detection where the model can over-train into a
generic-image reconstructor that loses real-vs-fake discrimination.

Output: a per-epoch table of AUCs for each manipulation method.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..svd_unet_vae.model import SVDUNetVAE
from .dataset import FFppEvalDataset, MANIPULATIONS, _default_transform

SCRIPT_DIR = Path(__file__).resolve().parent


def collect_scores(model, loader, device):
    model.eval()
    all_errors, all_labels = [], []
    with torch.inference_mode():
        for i_low, i_gray, labels in tqdm(loader, leave=False):
            i_low = i_low.to(device)
            i_gray = i_gray.to(device)
            recon, _, _ = model(i_low)
            err = F.mse_loss(recon, i_gray, reduction="none").mean(dim=[1, 2, 3])
            all_errors.append(err.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_errors), np.concatenate(all_labels)


def auc_for(model, data_dir, manipulation, device, batch_size, num_workers):
    transform = _default_transform()
    ds = FFppEvalDataset(data_dir, manipulation, subset="test", transform=transform, seed=42)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    errs, labels = collect_scores(model, loader, device)
    return float(roc_auc_score(labels, errs))


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
        print("Make sure training was run with --save_every_epoch")
        return

    data_root = Path(args.data_dir)
    if args.manipulations:
        manipulations = args.manipulations
    else:
        manipulations = [m for m in MANIPULATIONS if (data_root / m).is_dir()]

    print(f"Sweep over {len(checkpoints)} checkpoints, {len(manipulations)} manipulations")
    print(f"Manipulations: {manipulations}\n")

    rows = []
    model = SVDUNetVAE(latent_dim=args.latent_dim).to(device)

    header = f"{'Epoch':>6}  " + "  ".join(f"{m[:12]:>12}" for m in manipulations) + f"  {'Avg':>6}"
    print(header)
    print("-" * len(header))

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt["epoch"] + 1

        aucs = []
        for m in manipulations:
            aucs.append(auc_for(model, args.data_dir, m, device, args.batch, args.num_workers))
        avg = sum(aucs) / len(aucs)
        rows.append({"epoch": epoch, "aucs": dict(zip(manipulations, aucs)), "avg": avg})

        print(f"{epoch:>6}  " + "  ".join(f"{a:>12.4f}" for a in aucs) + f"  {avg:>6.4f}")

    # save
    with open(ckpt_dir / "sweep_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nResults saved to {ckpt_dir / 'sweep_results.json'}")

    # print best epoch by average AUC
    best = max(rows, key=lambda r: r["avg"])
    print(f"\nBest epoch by avg AUC: {best['epoch']} (avg={best['avg']:.4f})")


if __name__ == "__main__":
    main()
