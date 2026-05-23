#!/usr/bin/env python3
"""
Evaluate the FF++-trained SVD U-Net VAE against each manipulation method.
Anomaly score: MSE(model(i_low), i_gray) — matches the Step 2 bug-fix.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


def youden_threshold(errors, labels):
    thresholds = np.unique(errors)
    best_j, best_t = -1.0, thresholds[0]
    for t in thresholds:
        preds = (errors >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def evaluate_manipulation(model, data_dir, manipulation, device, batch_size, num_workers):
    transform = _default_transform()
    seed = 42

    thresh_ds = FFppEvalDataset(data_dir, manipulation, subset="thresh",
                                transform=transform, seed=seed)
    test_ds = FFppEvalDataset(data_dir, manipulation, subset="test",
                              transform=transform, seed=seed)

    thresh_loader = DataLoader(thresh_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    print(f"  [{manipulation}] threshold split: {len(thresh_ds)}, test split: {len(test_ds)}")

    thresh_errors, thresh_labels = collect_scores(model, thresh_loader, device)
    threshold = youden_threshold(thresh_errors, thresh_labels)

    test_errors, test_labels = collect_scores(model, test_loader, device)
    preds = (test_errors >= threshold).astype(int)

    return {
        "manipulation": manipulation,
        "threshold": threshold,
        "n_thresh": len(thresh_ds),
        "n_test": len(test_ds),
        "roc_auc": float(roc_auc_score(test_labels, test_errors)),
        "accuracy": float(accuracy_score(test_labels, preds)),
        "precision": float(precision_score(test_labels, preds, zero_division=0)),
        "recall": float(recall_score(test_labels, preds, zero_division=0)),
        "f1": float(f1_score(test_labels, preds, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data/ffpp"))
    parser.add_argument("--ckpt", default=str(SCRIPT_DIR / "../../checkpoints/ffpp_svd_vae/best.pth"))
    parser.add_argument("--results_dir", default=str(SCRIPT_DIR / "../../checkpoints/ffpp_svd_vae"))
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--manipulations", nargs="+", default=None,
                        help="Which manipulations to evaluate (default: auto-detect)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = SVDUNetVAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data_root = Path(args.data_dir)
    if args.manipulations:
        manipulations = args.manipulations
    else:
        manipulations = [m for m in MANIPULATIONS if (data_root / m).is_dir()]
    print(f"Manipulations to evaluate: {manipulations}\n")

    all_results = []
    for m in manipulations:
        print(f"Evaluating {m} ...")
        res = evaluate_manipulation(model, args.data_dir, m, device,
                                    args.batch, args.num_workers)
        all_results.append(res)
        print(
            f"  AUC={res['roc_auc']:.4f}  Acc={res['accuracy']:.4f}  "
            f"F1={res['f1']:.4f}  threshold={res['threshold']:.6f}"
        )

    results_path = Path(args.results_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n{:<20} {:>8} {:>8} {:>9} {:>8} {:>8}".format(
        "Manipulation", "AUC", "Accuracy", "Precision", "Recall", "F1"
    ))
    print("-" * 60)
    for r in all_results:
        print("{:<20} {:>8.4f} {:>8.4f} {:>9.4f} {:>8.4f} {:>8.4f}".format(
            r["manipulation"], r["roc_auc"], r["accuracy"],
            r["precision"], r["recall"], r["f1"]
        ))


if __name__ == "__main__":
    main()
