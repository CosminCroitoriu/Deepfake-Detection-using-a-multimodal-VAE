#!/usr/bin/env python3
"""
Evaluate a pretrained CNN classifier per generator.

Anomaly score: sigmoid(logit) — probability the model assigns to class 1 (fake).
Protocol matches the ViT classifier and VAE evaluators exactly:
  1. Calibrate a Youden-optimal threshold on the "thresh" split.
  2. Report AUC, Accuracy, Precision, Recall, F1 on the held-out "test" split.
Results are saved to results.json in the checkpoint directory.
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

from .dataset import ClassifierEvalDataset, GENERATOR_DIRS, eval_transform
from .model import PretrainedCNNClassifier

SCRIPT_DIR = Path(__file__).resolve().parent


def collect_scores(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.inference_mode():
        for imgs, labels in tqdm(loader, leave=False):
            probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def youden_threshold(probs, labels):
    thresholds = np.unique(probs)
    best_j, best_t = -1.0, thresholds[0]
    for t in thresholds:
        preds = (probs >= t).astype(int)
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
    thresh_ds = ClassifierEvalDataset(
        data_dir, generator, subset="thresh", seed=42, transform=eval_transform
    )
    test_ds = ClassifierEvalDataset(
        data_dir, generator, subset="test", seed=42, transform=eval_transform
    )

    thresh_loader = DataLoader(thresh_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size, num_workers=num_workers)

    print(f"  [{generator}] threshold split: {len(thresh_ds)}, test split: {len(test_ds)}")

    thresh_probs, thresh_labels = collect_scores(model, thresh_loader, device)
    threshold = youden_threshold(thresh_probs, thresh_labels)

    test_probs, test_labels = collect_scores(model, test_loader, device)
    preds = (test_probs >= threshold).astype(int)

    return {
        "generator":  generator,
        "threshold":  threshold,
        "n_thresh":   len(thresh_ds),
        "n_test":     len(test_ds),
        "roc_auc":    float(roc_auc_score(test_labels, test_probs)),
        "accuracy":   float(accuracy_score(test_labels, preds)),
        "precision":  float(precision_score(test_labels, preds, zero_division=0)),
        "recall":     float(recall_score(test_labels, preds, zero_division=0)),
        "f1":         float(f1_score(test_labels, preds, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arch",         default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--data_dir",     default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--ckpt",         default=None,
                        help="Path to best.pth (default: checkpoints/{arch}/best.pth)")
    parser.add_argument("--results_dir",  default=None,
                        help="Directory for results.json (default: checkpoints/{arch})")
    parser.add_argument("--batch",        type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--generators",   nargs="+", default=None)
    args = parser.parse_args()

    ckpt_path   = Path(args.ckpt or (SCRIPT_DIR / f"../../checkpoints/{args.arch}/best.pth"))
    results_dir = Path(args.results_dir or ckpt_path.parent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.arch} from {ckpt_path} ...")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = PretrainedCNNClassifier(arch=args.arch, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    fake_root  = Path(args.data_dir) / "fake"
    generators = args.generators or [
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
        print(f"  AUC={res['roc_auc']:.4f}  Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")

    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n{:<25} {:>8} {:>8} {:>9} {:>8} {:>8}".format(
        "Generator", "AUC", "Accuracy", "Precision", "Recall", "F1"
    ))
    print("-" * 65)
    for r in all_results:
        print("{:<25} {:>8.4f} {:>8.4f} {:>9.4f} {:>8.4f} {:>8.4f}".format(
            r["generator"], r["roc_auc"], r["accuracy"],
            r["precision"], r["recall"], r["f1"]
        ))

    if len(all_results) > 1:
        avg_auc = np.mean([r["roc_auc"] for r in all_results])
        print(f"\nAverage AUC across {len(all_results)} generators: {avg_auc:.4f}")


if __name__ == "__main__":
    main()
