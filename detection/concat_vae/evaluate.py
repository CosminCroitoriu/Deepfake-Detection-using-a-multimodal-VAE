#!/usr/bin/env python3
"""
Evaluate the Concatenation Multimodal VAE against each generator separately.

Anomaly score = mean MSE across all 5 reconstructed modalities.
Per-channel errors are also logged so you can see which modality drives
detection for each generator (e.g. DCT carries GAN signal, SVD carries
diffusion signal).
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

from .dataset import EvalDataset, GENERATOR_DIRS
from .model import ConcatUNetVAE
from .transform import ConcatTransform

SCRIPT_DIR = Path(__file__).resolve().parent
CHANNEL_NAMES = ["svd", "dct", "r", "g", "b"]


def collect_scores(model, loader, device):
    """Return (errors, errors_per_channel, labels) for an EvalDataset loader.

    Single forward pass per batch — per-channel and total errors derived from
    the same reconstruction.
    """
    model.eval()
    all_errors, all_pc, all_labels = [], [], []
    with torch.inference_mode():
        for input_5ch, target_5ch, labels in tqdm(loader, leave=False):
            input_5ch = input_5ch.to(device)
            target_5ch = target_5ch.to(device)
            recon, _, _ = model(input_5ch)
            err_map = F.mse_loss(recon, target_5ch, reduction="none")  # [B, 5, H, W]
            err_pc = err_map.mean(dim=[2, 3])                          # [B, 5]
            err = err_pc.mean(dim=1)                                   # [B]
            all_errors.append(err.cpu().numpy())
            all_pc.append(err_pc.cpu().numpy())
            all_labels.append(labels.numpy())
    return (
        np.concatenate(all_errors),
        np.concatenate(all_pc, axis=0),
        np.concatenate(all_labels),
    )


def youden_threshold(errors: np.ndarray, labels: np.ndarray) -> float:
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


def evaluate_generator(model, data_dir, generator, device, batch_size, num_workers):
    transform = ConcatTransform()
    seed = 42

    thresh_ds = EvalDataset(data_dir, generator, subset="thresh",
                            transform=transform, seed=seed)
    test_ds = EvalDataset(data_dir, generator, subset="test",
                          transform=transform, seed=seed)

    thresh_loader = DataLoader(thresh_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    print(f"  [{generator}] threshold split: {len(thresh_ds)}, test split: {len(test_ds)}")

    thresh_errors, _, thresh_labels = collect_scores(model, thresh_loader, device)
    threshold = youden_threshold(thresh_errors, thresh_labels)

    test_errors, test_errors_pc, test_labels = collect_scores(model, test_loader, device)
    preds = (test_errors >= threshold).astype(int)

    # per-channel AUC (which modality discriminates best?)
    per_channel_auc = {
        name: float(roc_auc_score(test_labels, test_errors_pc[:, i]))
        for i, name in enumerate(CHANNEL_NAMES)
    }

    return {
        "generator": generator,
        "threshold": threshold,
        "n_thresh": len(thresh_ds),
        "n_test": len(test_ds),
        "roc_auc": float(roc_auc_score(test_labels, test_errors)),
        "accuracy": float(accuracy_score(test_labels, preds)),
        "precision": float(precision_score(test_labels, preds, zero_division=0)),
        "recall": float(recall_score(test_labels, preds, zero_division=0)),
        "f1": float(f1_score(test_labels, preds, zero_division=0)),
        "per_channel_auc": per_channel_auc,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--ckpt", default=str(SCRIPT_DIR / "../../checkpoints/concat_vae/best.pth"))
    parser.add_argument("--results_dir", default=str(SCRIPT_DIR / "../../checkpoints/concat_vae"))
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--generators", nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = ConcatUNetVAE(latent_dim=args.latent_dim, in_channels=5, out_channels=5).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    fake_root = Path(args.data_dir) / "fake"
    if args.generators:
        generators = args.generators
    else:
        generators = [
            d.name for d in sorted(fake_root.iterdir())
            if d.is_dir() and d.name in GENERATOR_DIRS
        ]
    print(f"Generators to evaluate: {generators}\n")

    all_results = []
    for gen in generators:
        print(f"Evaluating {gen} ...")
        res = evaluate_generator(model, args.data_dir, gen, device,
                                 args.batch, args.num_workers)
        all_results.append(res)
        print(
            f"  AUC={res['roc_auc']:.4f}  Acc={res['accuracy']:.4f}  "
            f"F1={res['f1']:.4f}  threshold={res['threshold']:.6f}"
        )
        print(
            f"  per-channel AUC: "
            + "  ".join(f"{k}={v:.3f}" for k, v in res["per_channel_auc"].items())
        )

    results_path = Path(args.results_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n{:<20} {:>8} {:>8} {:>9} {:>8} {:>8}".format(
        "Generator", "AUC", "Accuracy", "Precision", "Recall", "F1"
    ))
    print("-" * 60)
    for r in all_results:
        print("{:<20} {:>8.4f} {:>8.4f} {:>9.4f} {:>8.4f} {:>8.4f}".format(
            r["generator"], r["roc_auc"], r["accuracy"],
            r["precision"], r["recall"], r["f1"]
        ))


if __name__ == "__main__":
    main()
