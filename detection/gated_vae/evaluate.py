#!/usr/bin/env python3
"""
Evaluate the Gated Multimodal VAE (Step 4) per generator.

For each generator we report:
  - overall AUC using the gate-weighted anomaly score
  - per-modality AUC using each modality's raw MSE alone (no gating)
  - mean gate weight (alpha) per modality, split by real vs fake samples,
    so the explainability analysis in Step 7 has the data it needs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..concat_vae.transform import ConcatTransform
from .dataset import EvalDataset, GENERATOR_DIRS
from .model import GatedMultimodalVAE

SCRIPT_DIR = Path(__file__).resolve().parent
MODALITY_NAMES = ["svd", "dct", "rgb"]


def collect_scores(model, loader, device):
    """Return arrays: gated_scores, per_modality_mse [N×3], alpha [N×3], labels."""
    model.eval()
    all_score, all_per_mod, all_alpha, all_labels = [], [], [], []
    with torch.inference_mode():
        for input_5ch, target_5ch, labels in tqdm(loader, leave=False):
            input_5ch = input_5ch.to(device)
            target_5ch = target_5ch.to(device)
            score, per_mod, alpha = model.reconstruction_error(
                input_5ch, target_5ch, return_alpha=True
            )
            all_score.append(score.cpu().numpy())
            all_per_mod.append(per_mod.cpu().numpy())
            all_alpha.append(alpha.cpu().numpy())
            all_labels.append(labels.numpy())
    return (
        np.concatenate(all_score),
        np.concatenate(all_per_mod, axis=0),
        np.concatenate(all_alpha, axis=0),
        np.concatenate(all_labels),
    )


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

    th_score, _, _, th_labels = collect_scores(model, thresh_loader, device)
    threshold = youden_threshold(th_score, th_labels)

    score, per_mod, alpha, labels = collect_scores(model, test_loader, device)
    preds = (score >= threshold).astype(int)

    per_modality_auc = {
        name: float(roc_auc_score(labels, per_mod[:, i]))
        for i, name in enumerate(MODALITY_NAMES)
    }

    mean_alpha_real = alpha[labels == 0].mean(axis=0).tolist() if (labels == 0).any() else [0.0] * 3
    mean_alpha_fake = alpha[labels == 1].mean(axis=0).tolist() if (labels == 1).any() else [0.0] * 3

    return {
        "generator": generator,
        "threshold": threshold,
        "n_thresh": len(thresh_ds),
        "n_test": len(test_ds),
        "roc_auc": float(roc_auc_score(labels, score)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "per_modality_auc": per_modality_auc,
        "mean_alpha_real": dict(zip(MODALITY_NAMES, mean_alpha_real)),
        "mean_alpha_fake": dict(zip(MODALITY_NAMES, mean_alpha_fake)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--ckpt", default=str(SCRIPT_DIR / "../../checkpoints/gated_vae/best.pth"))
    parser.add_argument("--results_dir", default=str(SCRIPT_DIR / "../../checkpoints/gated_vae"))
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--generators", nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GatedMultimodalVAE(latent_dim=args.latent_dim).to(device)
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
            f"  AUC={res['roc_auc']:.4f}  Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}\n"
            f"  per-modality AUC: " +
            "  ".join(f"{k}={v:.3f}" for k, v in res["per_modality_auc"].items()) + "\n"
            f"  α (real): " +
            "  ".join(f"{k}={v:.3f}" for k, v in res["mean_alpha_real"].items()) + "\n"
            f"  α (fake): " +
            "  ".join(f"{k}={v:.3f}" for k, v in res["mean_alpha_fake"].items())
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
