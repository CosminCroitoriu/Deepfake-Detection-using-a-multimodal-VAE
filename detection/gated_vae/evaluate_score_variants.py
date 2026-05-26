#!/usr/bin/env python3
"""
Evaluate the Gated Multimodal VAE under MULTIPLE anomaly-score formulas.

The model's per-modality MSEs and gate weights α are computed once per sample;
different scoring strategies are computed in post and ranked side-by-side.

Strategies compared:

  1. alpha     : α_svd·MSE_svd + α_dct·MSE_dct + α_rgb·MSE_rgb       (current/baseline)
  2. inv_alpha : (1−α_svd)·MSE_svd + (1−α_dct)·MSE_dct + (1−α_rgb)·MSE_rgb
                 (Option B — gate identifies reliable modalities,
                 anomalies are in *unreliable* ones)
  3. mean      : (MSE_svd + MSE_dct + MSE_rgb) / 3
                 (gate-ignored diagnostic — closest analog to Step 3 Concat VAE)
  4. max       : max(MSE_svd, MSE_dct, MSE_rgb)
                 (worst-reconstructed modality wins; intuitive for anomaly)
  5. entropy   : (log 3 − H(α))      (Option C — gate's deviation from
                                       training-time uniform is the signal)
  6. mean_plus_entropy : mean + 0.1·(log 3 − H(α))  (hybrid)

Also reports per-modality AUC, mean α for real and for fake, and the per-sample
α-entropy mean for real vs fake — useful for diagnosing whether the gate
genuinely reacts to fakes.
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

from ..concat_vae.transform import ConcatTransform
from .dataset import EvalDataset, GENERATOR_DIRS
from .model import GatedMultimodalVAE

SCRIPT_DIR = Path(__file__).resolve().parent
MODALITY_NAMES = ["svd", "dct", "rgb"]


def collect_raw(model, loader, device):
    """Single forward pass through the dataset; return raw arrays:
       per_mod_mse[N×3]  alpha[N×3]  labels[N]"""
    model.eval()
    pm, al, lb = [], [], []
    with torch.inference_mode():
        for input_5ch, target_5ch, labels in tqdm(loader, leave=False):
            input_5ch = input_5ch.to(device)
            target_5ch = target_5ch.to(device)
            recon_svd, recon_dct, recon_rgb, _, _, alpha = model(input_5ch)
            tgt_svd, tgt_dct, tgt_rgb = (
                target_5ch[:, 0:1], target_5ch[:, 1:2], target_5ch[:, 2:5],
            )
            mse_svd = F.mse_loss(recon_svd, tgt_svd, reduction="none").mean(dim=[1, 2, 3])
            mse_dct = F.mse_loss(recon_dct, tgt_dct, reduction="none").mean(dim=[1, 2, 3])
            mse_rgb = F.mse_loss(recon_rgb, tgt_rgb, reduction="none").mean(dim=[1, 2, 3])
            per_mod = torch.stack([mse_svd, mse_dct, mse_rgb], dim=1)  # B×3
            pm.append(per_mod.cpu().numpy())
            al.append(alpha.cpu().numpy())
            lb.append(labels.numpy())
    return np.concatenate(pm, 0), np.concatenate(al, 0), np.concatenate(lb)


def score_strategies(per_mod, alpha):
    """Return dict[strategy_name -> per-sample score array]."""
    log3 = np.log(3.0)
    H = -(alpha * np.log(alpha + 1e-9)).sum(axis=1)            # per-sample entropy

    scores = {
        "alpha":              (alpha * per_mod).sum(axis=1),
        "inv_alpha":          ((1.0 - alpha) * per_mod).sum(axis=1),
        "mean":               per_mod.mean(axis=1),
        "max":                per_mod.max(axis=1),
        "entropy":            log3 - H,
        "mean_plus_entropy":  per_mod.mean(axis=1) + 0.1 * (log3 - H),
    }
    return scores, H


def evaluate_generator(model, data_dir, generator, device, batch_size, num_workers):
    transform = ConcatTransform()
    test_ds = EvalDataset(data_dir, generator, subset="test", transform=transform, seed=42)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    print(f"  [{generator}] test split: {len(test_ds)}")
    per_mod, alpha, labels = collect_raw(model, test_loader, device)

    # Per-strategy AUC
    scores, H = score_strategies(per_mod, alpha)
    strategy_aucs = {k: float(roc_auc_score(labels, v)) for k, v in scores.items()}

    # Per-modality AUC (raw, no gating)
    per_mod_aucs = {
        name: float(roc_auc_score(labels, per_mod[:, i]))
        for i, name in enumerate(MODALITY_NAMES)
    }

    # α stats split by class
    mean_alpha_real = alpha[labels == 0].mean(axis=0).tolist() if (labels == 0).any() else [0.0]*3
    mean_alpha_fake = alpha[labels == 1].mean(axis=0).tolist() if (labels == 1).any() else [0.0]*3
    mean_H_real = float(H[labels == 0].mean()) if (labels == 0).any() else 0.0
    mean_H_fake = float(H[labels == 1].mean()) if (labels == 1).any() else 0.0

    return {
        "generator": generator,
        "n_test": len(test_ds),
        "strategy_aucs": strategy_aucs,
        "per_modality_aucs": per_mod_aucs,
        "mean_alpha_real": dict(zip(MODALITY_NAMES, mean_alpha_real)),
        "mean_alpha_fake": dict(zip(MODALITY_NAMES, mean_alpha_fake)),
        "mean_alpha_entropy_real": mean_H_real,
        "mean_alpha_entropy_fake": mean_H_fake,
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

        print(f"  per-modality AUC: " + "  ".join(f"{k}={v:.3f}" for k, v in res["per_modality_aucs"].items()))
        print(f"  α real entropy={res['mean_alpha_entropy_real']:.3f}    α fake entropy={res['mean_alpha_entropy_fake']:.3f}")
        print(f"  α (real): " + "  ".join(f"{k}={v:.3f}" for k, v in res["mean_alpha_real"].items()))
        print(f"  α (fake): " + "  ".join(f"{k}={v:.3f}" for k, v in res["mean_alpha_fake"].items()))
        print(f"  Strategy AUCs:")
        for strat, auc in res["strategy_aucs"].items():
            print(f"    {strat:<22} = {auc:.4f}")
        print()

    results_path = Path(args.results_dir) / "score_variants_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Side-by-side summary table
    strats = list(all_results[0]["strategy_aucs"].keys())
    header = f"{'Generator':<22}" + "".join(f"{s:>20}" for s in strats)
    print("\n" + header)
    print("-" * len(header))
    for r in all_results:
        line = f"{r['generator']:<22}"
        for s in strats:
            line += f"{r['strategy_aucs'][s]:>20.4f}"
        print(line)


if __name__ == "__main__":
    main()
