#!/usr/bin/env python3
"""
demo.py — Demo detecție imagini generate cu VAE multimodal.

Rulează toate cele 4 modele VAE pe o imagine aleasă și afișează
scorul de anomalie și decizia (REAL / GENERAT) pentru fiecare model.

Pragurile Youden sunt citite automat din checkpoints/<model>/results.json
(generat de evaluate.py pe cluster). Dacă fișierul lipsește, se folosesc
valori de rezervă hardcodate.

Utilizare:
    python demo.py <imagine> [--label real|fake] [--generator sd3_lora|sd_lora|projected_gan_512]

Exemple:
    python demo.py pics/seed0266.png --label fake --generator sd3_lora
    python demo.py pics/hurricane_real.png --label real

NOTĂ: Descarcă checkpointurile și results.json de pe cluster înainte de rulare:
    scp -r <user>@<cluster>:~/Deepfake-Detection-using-a-multimodal-VAE/checkpoints ./
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from detection.concat_vae.transform import ConcatTransform
from detection.svd_unet_vae.svd_transform import SVDTransform
from detection.concat_vae.model import ConcatUNetVAE
from detection.svd_unet_vae.model import SVDUNetVAE
from detection.gated_vae.model import GatedMultimodalVAE
from detection.masked_vae.model import MaskedConcatVAE

# ---------------------------------------------------------------------------
# Configurație
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    "SVD Baseline": PROJECT_DIR / "checkpoints/svd_unet_vae/best.pth",
    "Concat VAE":   PROJECT_DIR / "checkpoints/concat_vae/best.pth",
    "Gated VAE":    PROJECT_DIR / "checkpoints/gated_vae/best.pth",
    "Masked VAE":   PROJECT_DIR / "checkpoints/masked_vae/best.pth",
}

# results.json salvat de evaluate.py — conține pragurile Youden per generator
RESULTS_JSON = {
    "SVD Baseline": PROJECT_DIR / "checkpoints/svd_unet_vae/results.json",
    "Concat VAE":   PROJECT_DIR / "checkpoints/concat_vae/results.json",
    "Gated VAE":    PROJECT_DIR / "checkpoints/gated_vae/results.json",
    "Masked VAE":   PROJECT_DIR / "checkpoints/masked_vae/results.json",
}

# Valori de rezervă dacă results.json lipsește.
# Concat VAE: calculat din _ablation/raw_scores_concat.npz pe SD3 LoRA.
# Celelalte: aproximate — rulează evaluate.py pe cluster pentru valori exacte.
FALLBACK_THRESHOLDS = {
    "SVD Baseline": 0.000099,
    "Concat VAE":   0.000385,
    "Gated VAE":    0.000385,
    "Masked VAE":   0.000385,
}

# AUC pe SD3 LoRA — afișat ca referință de calitate a fiecărui model
AUC_SD3 = {
    "SVD Baseline": 0.775,
    "Concat VAE":   0.874,
    "Gated VAE":    0.778,
    "Masked VAE":   0.858,
}

LATENT_DIM = 256
MODELS_ORDER = ["SVD Baseline", "Concat VAE", "Gated VAE", "Masked VAE"]

# Interval de afișare pentru bara vizuală (estimat din distribuțiile reale)
SCORE_DISPLAY_LO = 0.000080
SCORE_DISPLAY_HI = 0.000700

# ---------------------------------------------------------------------------
# Praguri — citite din results.json, cu fallback la valorile hardcodate
# ---------------------------------------------------------------------------

def load_threshold(name: str, generator: str) -> tuple[float, bool]:
    """
    Returnează (threshold, from_file).
    Caută în results.json intrarea pentru generatorul specificat.
    Dacă fișierul sau intrarea lipsesc, returnează valoarea de rezervă.
    """
    results_path = RESULTS_JSON[name]
    if results_path.exists():
        with open(results_path) as f:
            entries = json.load(f)
        for entry in entries:
            if entry.get("generator") == generator:
                return float(entry["threshold"]), True
    return FALLBACK_THRESHOLDS[name], False


# ---------------------------------------------------------------------------
# Încărcare modele
# ---------------------------------------------------------------------------

def load_model(name: str, device: torch.device):
    ckpt_path = CHECKPOINTS[name]
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if name == "SVD Baseline":
        model = SVDUNetVAE(latent_dim=LATENT_DIM)
    elif name == "Concat VAE":
        model = ConcatUNetVAE(latent_dim=LATENT_DIM, in_channels=5, out_channels=5)
    elif name == "Gated VAE":
        model = GatedMultimodalVAE(latent_dim=LATENT_DIM)
    elif name == "Masked VAE":
        model = MaskedConcatVAE(latent_dim=LATENT_DIM, in_channels=5, out_channels=5)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device)

# ---------------------------------------------------------------------------
# Calcul scor de anomalie
# ---------------------------------------------------------------------------

def compute_score(model, name: str, input_t: torch.Tensor,
                  target_t: torch.Tensor, device: torch.device) -> float:
    inp = input_t.unsqueeze(0).to(device)
    tgt = target_t.unsqueeze(0).to(device)

    with torch.inference_mode():
        if name == "Gated VAE":
            # Scor ponderat cu gate-ul per modalitate
            score, _ = model.reconstruction_error(inp, tgt)
            return score.item()
        else:
            recon, _, _ = model(inp)
            return F.mse_loss(recon, tgt).item()

# ---------------------------------------------------------------------------
# Afișaj
# ---------------------------------------------------------------------------

def score_bar(score: float, threshold: float, width: int = 32) -> str:
    """
    Bară vizuală: scor față de intervalul de referință.
    '|' marchează pragul; '█' arată nivelul scorului.
    """
    lo, hi = SCORE_DISPLAY_LO, SCORE_DISPLAY_HI

    def to_pos(v):
        return max(0, min(width - 1, int((v - lo) / (hi - lo) * width)))

    pos_score = to_pos(score)
    pos_thresh = to_pos(threshold)

    bar = ["░"] * width
    for i in range(pos_score):
        bar[i] = "█"
    if 0 <= pos_thresh < width:
        bar[pos_thresh] = "│"

    return "".join(bar)


def print_result(name: str, score: float, threshold: float,
                 decision: str, label: str | None):
    correct_marker = ""
    if label is not None:
        expected = "GENERAT" if label == "fake" else "REAL"
        correct_marker = "  ✓" if decision == expected else "  ✗"

    color_start = "\033[91m" if decision == "GENERAT" else "\033[92m"
    color_end   = "\033[0m"

    print(f"  {name:<16}  scor={score:.6f}  prag={threshold:.6f}  "
          f"{color_start}{decision:<8}{color_end}{correct_marker}  "
          f"(AUC SD3={AUC_SD3[name]:.3f})")

    bar = score_bar(score, threshold)
    label_lo = "REAL"
    label_hi = "GENERAT"
    print(f"  {'':16}  {label_lo} [{bar}] {label_hi}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Calea către imaginea de evaluat")
    parser.add_argument(
        "--label", choices=["real", "fake"],
        help="Eticheta adevărată a imaginii (opțional, pentru verificare)",
    )
    parser.add_argument(
        "--generator",
        default="sd3_lora",
        choices=["sd3_lora", "sd_lora", "projected_gan_512"],
        help="Generatorul față de care se calibrează pragul (implicit: sd3_lora)",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"\n[EROARE] Imaginea nu există: {img_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    concat_transform = ConcatTransform()
    svd_transform    = SVDTransform()

    img = Image.open(img_path).convert("RGB")

    print()
    print("╔" + "═" * 66 + "╗")
    print(f"║  Imagine   : {img_path.name:<51} ║")
    if args.label:
        truth = ("REALĂ" if args.label == "real" else "GENERATĂ (FAKE)")
        print(f"║  Etichetă  : {truth:<51} ║")
    print(f"║  Generator : {args.generator:<51} ║")
    print(f"║  Device    : {str(device):<51} ║")
    print("╚" + "═" * 66 + "╝")
    print()

    results = []

    for name in MODELS_ORDER:
        print(f"  [{name}] Încarc checkpoint ...", end=" ", flush=True)
        model = load_model(name, device)

        if model is None:
            missing = CHECKPOINTS[name].relative_to(PROJECT_DIR)
            print(f"LIPSĂ  ({missing})")
            results.append((name, None, None, None, None))
            print()
            continue

        print("OK  |  Calculez scor ...", end=" ", flush=True)

        if name == "SVD Baseline":
            input_t, target_t = svd_transform(img)
        else:
            input_t, target_t = concat_transform(img)

        score = compute_score(model, name, input_t, target_t, device)
        threshold, from_file = load_threshold(name, args.generator)
        source = "results.json" if from_file else "fallback"
        decision = "GENERAT" if score >= threshold else "REAL"
        results.append((name, score, threshold, decision, source))

        print("gata")

    print()
    print("─" * 68)
    print(f"  {'Model':<16}  {'Scor':>10}  {'Prag':>10}  {'Decizie':<10}  AUC(SD3)")
    print("─" * 68)

    for name, score, threshold, decision, source in results:
        if score is None:
            print(f"  {name:<16}  {'—':>10}  {'—':>10}  {'N/A':<10}  {AUC_SD3[name]:.3f}")
            print()
            continue
        print_result(name, score, threshold, decision, args.label)
        prag_sursa = "  (prag din results.json)" if source == "results.json" else "  (prag aproximat — rulează evaluate.py)"
        print(f"  {'':16}{prag_sursa}")
        print()

    print("─" * 68)

    available = [(n, s, t, d) for n, s, t, d, _ in results if s is not None]
    if args.label and available:
        expected = "GENERAT" if args.label == "fake" else "REAL"
        correct  = sum(1 for _, _, _, d in available if d == expected)
        total    = len(available)
        print(f"\n  Modele corecte: {correct}/{total}")

    print()


if __name__ == "__main__":
    main()
