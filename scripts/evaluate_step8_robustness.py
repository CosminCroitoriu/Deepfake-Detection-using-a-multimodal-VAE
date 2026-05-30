#!/usr/bin/env python3
"""
Step 8 — Robustness evaluation under image degradation.

Tests both the Concat VAE (Step 3) and the Masked Concat VAE (Step 5) under
six post-processing conditions applied at inference time:

  JPEG compression : quality = 50, 30, 10
  Gaussian blur    : radius  =  1,  2,  3

Perturbations are applied to the PIL image BEFORE the ConcatTransform, simulating
real-world images that have been re-compressed or blurred before the detector sees
them. Both real and fake images are perturbed equally — the evaluation measures how
much the model's discriminative ability degrades when inputs are degraded.

Clean AUC is loaded from existing results.json files (no re-run needed).

Outputs are saved to checkpoints/_ablation/:
  robustness_results.json     — full AUC numbers for every condition
  robustness_table.md         — markdown table per generator (Step 3 / Step 5 / Δ)
  robustness_table.tex        — LaTeX version of the same table

Usage (cluster):
  python scripts/evaluate_step8_robustness.py

Flags:
  --checkpoints_dir   default: <project_root>/checkpoints
  --data_dir          default: <project_root>/data
  --out_dir           default: checkpoints/_ablation
  --ckpt_concat       override concat_vae/best.pth
  --ckpt_masked       override masked_vae/best.pth
  --latent_dim        default: 256
  --mask_prob         default: 0.3
  --batch             default: 32
  --num_workers       default: 4
"""
import argparse
import io
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
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
from detection.masked_vae.model import MaskedConcatVAE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENERATORS = ["projected_gan_512", "sd_lora", "sd3_lora"]
GENERATOR_LABELS = {
    "projected_gan_512": "ProjectedGAN",
    "sd_lora":           "SD LoRA",
    "sd3_lora":          "SD3 LoRA",
}

# Perturbation definitions — (name, function)
PERTURBATIONS = [
    ("jpeg_q50",   lambda img: _jpeg(img, 50)),
    ("jpeg_q30",   lambda img: _jpeg(img, 30)),
    ("jpeg_q10",   lambda img: _jpeg(img, 10)),
    ("blur_s1",    lambda img: _blur(img, 1)),
    ("blur_s2",    lambda img: _blur(img, 2)),
    ("blur_s3",    lambda img: _blur(img, 3)),
]
PERTURBATION_LABELS = {
    "clean":      "Clean",
    "jpeg_q50":   "JPEG Q=50",
    "jpeg_q30":   "JPEG Q=30",
    "jpeg_q10":   "JPEG Q=10",
    "blur_s1":    "Blur σ=1",
    "blur_s2":    "Blur σ=2",
    "blur_s3":    "Blur σ=3",
}
ALL_CONDITIONS = ["clean"] + [name for name, _ in PERTURBATIONS]


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------
def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class PerturbedEvalDataset(Dataset):
    """
    Wraps the samples list from an EvalDataset and applies a PIL perturbation
    to each image before the ConcatTransform.
    """

    def __init__(self, base_dataset: EvalDataset, perturb_fn: Optional[Callable]):
        self.samples = base_dataset.samples
        self.transform = base_dataset.transform
        self.perturb_fn = perturb_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.perturb_fn is not None:
            img = self.perturb_fn(img)
        input_5ch, target_5ch = self.transform(img)
        return input_5ch, target_5ch, label


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def collect_scores(model, loader, device):
    """Single forward pass; return (errors [N], labels [N])."""
    model.eval()
    all_e, all_lbl = [], []
    with torch.inference_mode():
        for x_in, x_tgt, lbl in tqdm(loader, leave=False):
            x_in, x_tgt = x_in.to(device), x_tgt.to(device)
            recon, _, _ = model(x_in)
            err = F.mse_loss(recon, x_tgt, reduction="none").mean(dim=[1, 2, 3])
            all_e.append(err.cpu().numpy())
            all_lbl.append(lbl.numpy())
    return np.concatenate(all_e), np.concatenate(all_lbl)


def evaluate_under_perturbation(model, data_dir, generator, perturb_fn, device,
                                 batch_size, num_workers):
    transform = ConcatTransform()
    base_ds = EvalDataset(data_dir, generator, subset="test", transform=transform, seed=42)
    ds = PerturbedEvalDataset(base_ds, perturb_fn)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    errors, labels = collect_scores(model, loader, device)
    return float(roc_auc_score(labels, errors))


# ---------------------------------------------------------------------------
# Load clean AUC from existing results.json
# ---------------------------------------------------------------------------
def load_clean_auc(results_path: Path):
    """Return {generator: auc} from a results.json file, or {} if not found."""
    if not results_path.exists():
        return {}
    data = json.loads(results_path.read_text())
    return {r["generator"]: r["roc_auc"] for r in data}


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------
def _fmt(v):
    return f"{v:.3f}" if v is not None else "—"


def _delta(step3, step5):
    if step3 is None or step5 is None:
        return "—"
    d = step5 - step3
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def build_markdown_tables(results: dict):
    """
    Build one markdown table per generator.
    results[gen][model][condition] = auc
    """
    lines = []
    for gen in GENERATORS:
        lines.append(f"### {GENERATOR_LABELS.get(gen, gen)}\n")
        lines.append("| Condition | Step 3 (Concat) | Step 5 (Masked) | Δ (5−3) |")
        lines.append("|---|---|---|---|")
        for cond in ALL_CONDITIONS:
            s3 = results.get(gen, {}).get("step3", {}).get(cond)
            s5 = results.get(gen, {}).get("step5", {}).get(cond)
            lines.append(f"| {PERTURBATION_LABELS[cond]} | {_fmt(s3)} | {_fmt(s5)} | {_delta(s3, s5)} |")
        lines.append("")
    return "\n".join(lines)


def build_latex_table(results: dict):
    """Single combined LaTeX table: rows = conditions, col groups = generators."""
    # One row per condition, 3×3 columns (step3/step5/delta per generator)
    gen_headers = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{GENERATOR_LABELS[g]}}}" for g in GENERATORS
    )
    sub_headers = " & ".join([r"S3 & S5 & $\Delta$"] * len(GENERATORS))

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Robustness evaluation: ROC-AUC of Step 3 (Concat VAE) and Step 5 (Masked VAE) "
        r"under JPEG compression and Gaussian blur. $\Delta = \text{Step~5} - \text{Step~3}$; "
        r"positive means the masked model degrades less.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{l" + "ccc" * len(GENERATORS) + "}",
        r"\toprule",
        r"Condition & " + gen_headers + r" \\",
        r" & " + sub_headers + r" \\",
        r"\midrule",
    ]

    for cond in ALL_CONDITIONS:
        cells = [PERTURBATION_LABELS[cond]]
        for gen in GENERATORS:
            s3 = results.get(gen, {}).get("step3", {}).get(cond)
            s5 = results.get(gen, {}).get("step5", {}).get(cond)
            d = _delta(s3, s5)
            # Bold delta if Step 5 >= Step 3
            if s3 is not None and s5 is not None and s5 >= s3:
                d = r"\textbf{" + d + "}"
            cells += [_fmt(s3), _fmt(s5), d]
        lines.append(" & ".join(cells) + r" \\")
        if cond == "clean":
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


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
    ap.add_argument("--ckpt_concat",     default=None)
    ap.add_argument("--ckpt_masked",     default=None)
    ap.add_argument("--latent_dim",      type=int,   default=256)
    ap.add_argument("--mask_prob",       type=float, default=0.3)
    ap.add_argument("--batch",           type=int,   default=32)
    ap.add_argument("--num_workers",     type=int,   default=4)
    args = ap.parse_args()

    ck      = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (ck / "_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Step 8 — Robustness evaluation ===")
    print(f"Device: {device}")
    print(f"Output: {out_dir}\n")

    # ---- Load clean AUC from existing results.json ------------------------
    clean_concat = load_clean_auc(ck / "concat_vae" / "results.json")
    clean_masked = load_clean_auc(ck / "masked_vae" / "results.json")
    print("Clean AUC loaded:")
    for gen in GENERATORS:
        s3 = clean_concat.get(gen)
        s5 = clean_masked.get(gen)
        print(f"  {gen}: Step3={_fmt(s3)}  Step5={_fmt(s5)}")
    print()

    # ---- Load models -------------------------------------------------------
    concat_path = Path(args.ckpt_concat) if args.ckpt_concat else (ck / "concat_vae" / "best.pth")
    masked_path = Path(args.ckpt_masked) if args.ckpt_masked else (ck / "masked_vae" / "best.pth")

    print(f"Loading Step 3 (Concat VAE) from {concat_path} ...")
    ckpt3  = torch.load(concat_path, map_location=device)
    model3 = ConcatUNetVAE(latent_dim=args.latent_dim, in_channels=5, out_channels=5).to(device)
    model3.load_state_dict(ckpt3["model"])
    model3.eval()

    print(f"Loading Step 5 (Masked VAE) from {masked_path} ...")
    ckpt5  = torch.load(masked_path, map_location=device)
    model5 = MaskedConcatVAE(latent_dim=args.latent_dim, mask_prob=args.mask_prob).to(device)
    model5.load_state_dict(ckpt5["model"])
    model5.eval()
    print()

    # ---- Sweep perturbations ----------------------------------------------
    # results[gen][model_key][condition] = auc
    results = {gen: {"step3": {}, "step5": {}} for gen in GENERATORS}

    # Seed clean AUC values
    for gen in GENERATORS:
        if gen in clean_concat:
            results[gen]["step3"]["clean"] = clean_concat[gen]
        if gen in clean_masked:
            results[gen]["step5"]["clean"] = clean_masked[gen]

    for cond_name, perturb_fn in PERTURBATIONS:
        print(f"--- {PERTURBATION_LABELS[cond_name]} ---")
        for gen in GENERATORS:
            auc3 = evaluate_under_perturbation(
                model3, args.data_dir, gen, perturb_fn, device, args.batch, args.num_workers
            )
            auc5 = evaluate_under_perturbation(
                model5, args.data_dir, gen, perturb_fn, device, args.batch, args.num_workers
            )
            results[gen]["step3"][cond_name] = auc3
            results[gen]["step5"][cond_name] = auc5
            delta = auc5 - auc3
            sign = "+" if delta >= 0 else ""
            print(f"  {gen:<22}  Step3={auc3:.4f}  Step5={auc5:.4f}  Δ={sign}{delta:.4f}")
        print()

    # ---- Save JSON --------------------------------------------------------
    json_path = out_dir / "robustness_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {json_path}\n")

    # ---- Build and save tables --------------------------------------------
    md_text = f"# Step 8 — Robustness results (per-sample AUC)\n\n{build_markdown_tables(results)}"
    md_path = out_dir / "robustness_table.md"
    md_path.write_text(md_text)
    print(f"Saved {md_path}")

    tex_text = build_latex_table(results)
    tex_path = out_dir / "robustness_table.tex"
    tex_path.write_text(tex_text)
    print(f"Saved {tex_path}")

    # ---- Console summary for SD3 (the headline generator) -----------------
    print("\n=== SD3 LoRA — robustness summary ===")
    print(f"{'Condition':<14}  {'Step 3':>8}  {'Step 5':>8}  {'Δ':>8}")
    print("-" * 44)
    for cond in ALL_CONDITIONS:
        s3 = results["sd3_lora"]["step3"].get(cond)
        s5 = results["sd3_lora"]["step5"].get(cond)
        print(f"{PERTURBATION_LABELS[cond]:<14}  {_fmt(s3):>8}  {_fmt(s5):>8}  {_delta(s3, s5):>8}")


if __name__ == "__main__":
    main()
