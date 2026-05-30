#!/usr/bin/env python3
"""
Build the main thesis ablation table from all step results.

Reads results.json files produced by each step's evaluation script and emits:
  - A clean markdown table (for the wiki)
  - A LaTeX table (for the thesis)
  - A console-printable summary (for sanity-checking)

Per-step result file locations (all under checkpoints/):
  Step 2 (SVD)         : checkpoints/svd_unet_vae/results.json
  Step 3 (Concat)      : checkpoints/concat_vae/results.json
  Step 4 (Gated)       : checkpoints/gated_vae/score_variants_results.json
                         (multi-strategy result; we report α-weighted + best fusion)
  Step 5 (Masked)      : checkpoints/masked_vae/results.json
  FF++ same-data       : checkpoints/ffpp_svd_vae/results.json
  FF++ video sweep     : checkpoints/ffpp_svd_vae/sweep_results.json

Outputs go to checkpoints/_ablation/.

Usage:
  python scripts/build_ablation_table.py
"""
import argparse
import json
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Order in which generators are reported in Table B columns.
GENERATORS = ["projected_gan_512", "sd_lora", "sd3_lora"]
GENERATOR_HEADERS = {
    "projected_gan_512": "ProjGAN",
    "sd_lora":           "SD LoRA",
    "sd3_lora":          "SD3 LoRA",
}


# -- Loaders --------------------------------------------------------------------

def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def step_auc_map(results_list, generators=GENERATORS):
    """results.json files store a list of dicts keyed by generator. Return {gen: auc}."""
    if results_list is None:
        return {g: None for g in generators}
    by_gen = {r["generator"]: r for r in results_list}
    return {g: by_gen[g]["roc_auc"] if g in by_gen else None for g in generators}


def step_per_channel_map(results_list, generators=GENERATORS, key_options=("per_channel_auc", "per_modality_aucs")):
    """For models that report per-channel/per-modality AUC, return {gen: {ch: auc}}."""
    if results_list is None:
        return {g: None for g in generators}
    by_gen = {r["generator"]: r for r in results_list}
    out = {}
    for g in generators:
        if g not in by_gen:
            out[g] = None
            continue
        entry = by_gen[g]
        per_chan = None
        for key in key_options:
            if key in entry:
                per_chan = entry[key]
                break
        out[g] = per_chan
    return out


def gated_strategy_aucs(variant_list, generators=GENERATORS):
    """Step 4's score_variants_results.json: list of dicts, each with .strategy_aucs."""
    if variant_list is None:
        return {}
    by_gen = {r["generator"]: r for r in variant_list}
    strategies = list(next(iter(by_gen.values()))["strategy_aucs"].keys())
    out = {}
    for strat in strategies:
        out[strat] = {g: by_gen[g]["strategy_aucs"][strat] if g in by_gen else None for g in generators}
    return out


# -- Formatters -----------------------------------------------------------------

def fmt_auc(v):
    return f"{v:.3f}" if v is not None else "—"


def fmt_avg(d):
    vals = [v for v in d.values() if v is not None]
    return fmt_auc(mean(vals)) if vals else "—"


def format_row(name, auc_map, bold_best=False):
    cells = [fmt_auc(auc_map[g]) for g in GENERATORS]
    avg = fmt_avg(auc_map)
    if bold_best:
        # Bold the best (max) cell in this row
        vals = [auc_map[g] for g in GENERATORS]
        best_i = max(range(len(vals)), key=lambda i: vals[i] if vals[i] is not None else -1)
        cells[best_i] = f"**{cells[best_i]}**"
    return f"| {name} | " + " | ".join(cells) + f" | {avg} |"


# -- Outputs --------------------------------------------------------------------

def markdown_main_table(rows):
    header = "| Variant | " + " | ".join(GENERATOR_HEADERS[g] for g in GENERATORS) + " | Avg |"
    sep = "|" + "|".join(["---"] * (len(GENERATORS) + 2)) + "|"
    return "\n".join([header, sep] + rows)


def latex_main_table(rows_data):
    """rows_data: list of (name, auc_map) pairs."""
    n_cols = len(GENERATORS) + 2
    align = "l" + "c" * (n_cols - 1)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation across model variants on the CrisisNLP detection task. "
        r"ROC-AUC per generator family, computed at the per-sample level with video-source-disjoint test splits.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        r"Variant & " + " & ".join(GENERATOR_HEADERS[g] for g in GENERATORS) + r" & Avg \\",
        r"\midrule",
    ]
    for name, auc_map in rows_data:
        cells = [fmt_auc(auc_map[g]) for g in GENERATORS]
        vals = [auc_map[g] for g in GENERATORS]
        best_i = max(range(len(vals)), key=lambda i: vals[i] if vals[i] is not None else -1)
        cells[best_i] = r"\textbf{" + cells[best_i] + "}"
        avg = fmt_avg(auc_map)
        lines.append(f"{name} & " + " & ".join(cells) + f" & {avg} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints_dir", default=str(PROJECT_DIR / "checkpoints"))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    ck = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (ck / "_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load every step's results --------------------------------------------
    step2  = _load_json(ck / "svd_unet_vae"  / "results.json")
    step3  = _load_json(ck / "concat_vae"    / "results.json")
    step4v = _load_json(ck / "gated_vae"     / "score_variants_results.json")
    step5  = _load_json(ck / "masked_vae"    / "results.json")
    ffpp   = _load_json(ck / "ffpp_svd_vae"  / "results.json")
    ffpp_v = _load_json(ck / "ffpp_svd_vae"  / "video_level_sweep_results.json")

    step2_auc = step_auc_map(step2)
    step3_auc = step_auc_map(step3)
    step5_auc = step_auc_map(step5)
    step4_strategies = gated_strategy_aucs(step4v)
    step4_alpha = step4_strategies.get("alpha", {g: None for g in GENERATORS})
    step4_mean  = step4_strategies.get("mean",  {g: None for g in GENERATORS})

    # -- Per-channel breakdown for Step 3 (used in explainability table) -------
    step3_per_channel = step_per_channel_map(step3)

    # -- Build the main Table B -----------------------------------------------
    rows_data = [
        ("Step 2 — SVD-only baseline",           step2_auc),
        ("Step 3 — Concat Multimodal VAE",       step3_auc),
        ("Step 4 — Gated VAE (α-weighted)",      step4_alpha),
        ("Step 4 — Gated VAE (mean fusion)",     step4_mean),
        ("Step 5 — Masked Multimodal VAE",       step5_auc),
    ]
    md_rows = [format_row(name, auc_map, bold_best=True) for name, auc_map in rows_data]
    main_md = markdown_main_table(md_rows)
    main_tex = latex_main_table(rows_data)

    # -- Per-channel breakdown table for Step 3 --------------------------------
    explainability_rows = []
    for g in GENERATORS:
        per_chan = step3_per_channel.get(g)
        if per_chan is None:
            continue
        cells = [fmt_auc(per_chan.get(ch)) for ch in ["svd", "dct", "r", "g", "b"]]
        explainability_rows.append(f"| {GENERATOR_HEADERS[g]} | " + " | ".join(cells) + " |")
    explainability_md = "\n".join([
        "| Generator | SVD | DCT | R | G | B |",
        "|---|---|---|---|---|---|",
        *explainability_rows,
    ])

    # -- FF++ cross-check table -------------------------------------------------
    ffpp_auc = step_auc_map(ffpp, generators=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]) if ffpp else None
    ffpp_md = "(no FF++ same-dataset results.json found)"
    if ffpp_auc:
        ffpp_md = "| Manipulation | Frame-level AUC |\n|---|---|\n"
        for m in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            v = ffpp_auc.get(m)
            ffpp_md += f"| {m} | {fmt_auc(v)} |\n"

    # -- Sarkar reproduction row (hand-entered; not from JSON) -----------------
    sarkar_md = (
        "| Source | AUC | Setup |\n"
        "|---|---|---|\n"
        "| Our pipeline (rigorous) | ~0.50 | Per-sample, video-level splits, grayscale SVD |\n"
        "| **Sarkar's unmodified code** | **0.467** | Their bugs + leaky frame-level splits, RGB per-channel SVD 80% |\n"
    )

    # -- Compose the full markdown report --------------------------------------
    full_md = f"""# Ablation results (CrisisNLP, per-sample AUC)

## Table B — Main result

{main_md}

Best cell per row in **bold**. Step 4 reports two rows: as-designed α-weighted
scoring (which suffers the self-cancellation failure mode) and the best alternative
fusion (mean), which equals the per-modality average.

## Step 3 — Per-channel AUC breakdown (explainability data)

{explainability_md}

DCT is the single best modality for SD3 LoRA detection in our Concat VAE.

## FF++ Validation Track — frame-level AUC (Sarkar reproduction)

{ffpp_md}

## Two-path FF++ reproduction summary

{sarkar_md}

Both paths independently confirm Sarkar et al.'s published 0.881 average AUC is
not reproducible from the public artifacts.
"""

    # -- Write outputs ---------------------------------------------------------
    (out_dir / "table_b.md").write_text(full_md)
    (out_dir / "table_b.tex").write_text(main_tex)

    # -- Console summary -------------------------------------------------------
    print(full_md)
    print(f"\n--- Wrote outputs to {out_dir} ---")
    print(f"  table_b.md   ({(out_dir / 'table_b.md').stat().st_size} bytes)")
    print(f"  table_b.tex  ({(out_dir / 'table_b.tex').stat().st_size} bytes)")


if __name__ == "__main__":
    main()
