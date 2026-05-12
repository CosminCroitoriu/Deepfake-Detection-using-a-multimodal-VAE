#!/usr/bin/env python3
"""
Fine-tune ProjectedGAN on CrisisNLP disaster images (all classes mixed).

ProjectedGAN discriminates on features from a pretrained EfficientNet/CSP backbone,
making it more data-efficient than StyleGAN2 on heterogeneous, unaligned datasets
like CrisisNLP social media photos.

Run preprocess.py first.

Usage:
  python train_projected_gan.py
  python train_projected_gan.py --kimg 3000 --gpus 2
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]
REPO_URL = "https://github.com/autonomousvision/projected_gan.git"

# Paths are anchored to the script's location so the script works regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent


def clone_repo(repo_dir: Path):
    if repo_dir.exists():
        print(f"Repo already at {repo_dir}")
        return
    print(f"Cloning ProjectedGAN into {repo_dir} ...")
    subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)


def apply_patches(repo_dir: Path):
    """Apply compatibility patches for Python 3.12 / PyTorch 2.x / timm 0.6+."""
    # Patch 1: misc.py — PyTorch ≥1.11 removed the dataset arg from Sampler.__init__
    misc_path = repo_dir / "torch_utils" / "misc.py"
    text = misc_path.read_text()
    if "super().__init__(dataset)" in text:
        misc_path.write_text(text.replace("super().__init__(dataset)", "super().__init__()"))
        print("  Patched torch_utils/misc.py")

    # Patch 2: projector.py — timm ≥0.6 dropped act1 as standalone attribute on EfficientNet
    proj_path = repo_dir / "pg_modules" / "projector.py"
    text = proj_path.read_text()
    if "model.act1" in text and "hasattr(model, 'act1')" not in text:
        text = text.replace(
            "act1 = model.act1",
            "act1 = model.act1 if hasattr(model, 'act1') else torch.nn.SiLU()",
        )
        proj_path.write_text(text)
        print("  Patched pg_modules/projector.py")

    # Patch 3: train.py — PyTorch ≥2.0 rejects mixed int/float betas in Adam
    train_path = repo_dir / "train.py"
    text = train_path.read_text()
    patched = text.replace("betas=[0,", "betas=[0.0,")
    if patched != text:
        train_path.write_text(patched)
        print("  Patched train.py (betas)")


def merge_classes(data_root: Path, merged_dir: Path, classes: list) -> int:
    merged_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cls in classes:
        cls_dir = data_root / cls
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} not found, skipping")
            continue
        for img in cls_dir.glob("*.png"):
            dst = merged_dir / f"{cls}_{img.name}"
            if not dst.exists():
                shutil.copy2(img, dst)
                n += 1
    return n


def make_dataset_zip(src_dir: Path, zip_path: Path, repo_dir: Path):
    if zip_path.exists():
        print(f"Dataset zip exists: {zip_path.name}")
        return
    print(f"Creating dataset zip ...")
    subprocess.run(
        [
            sys.executable, str(repo_dir / "dataset_tool.py"),
            "--source", str(src_dir),
            "--dest", str(zip_path),
        ],
        check=True,
    )


def train(
    dataset_zip: Path,
    out_dir: Path,
    repo_dir: Path,
    gpus: int,
    batch: int,
    kimg: int,
    resume: str | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(repo_dir / "train.py"),
        "--outdir", str(out_dir),
        "--data", str(dataset_zip),
        "--cfg", "fastgan",
        "--gpus", str(gpus),
        "--batch", str(batch),
        "--kimg", str(kimg),
        "--snap", "50",
        "--mirror", "1",
    ]
    if resume:
        cmd += ["--resume", resume]
    print("\nLaunching ProjectedGAN training ...")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--repo_dir", default=str(SCRIPT_DIR / "projected_gan"))
    parser.add_argument("--checkpoints_dir", default=None,
                        help="Override checkpoint directory (default: checkpoints/projected_gan_<res>)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--kimg", type=int, default=5000)
    parser.add_argument("--img_res", type=int, default=512, choices=[256, 512],
                        help="Training resolution (default: 512)")
    parser.add_argument("--classes", nargs="+", default=DISASTER_CLASSES,
                        choices=DISASTER_CLASSES)
    parser.add_argument("--resume", default=None,
                        help="Path to a .pkl snapshot to resume from")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    data_root = Path(args.data_dir) / f"real_{args.img_res}"
    default_ckpt = str(SCRIPT_DIR / f"../../checkpoints/projected_gan_{args.img_res}")
    ckpt_root = Path(args.checkpoints_dir or default_ckpt)

    clone_repo(repo_dir)
    apply_patches(repo_dir)

    merged_dir = ckpt_root / "merged_dataset"
    print("Merging class folders ...")
    n = merge_classes(data_root, merged_dir, args.classes)
    print(f"  {n} total images in {merged_dir}")

    zip_path = ckpt_root / "crisis_all.zip"
    make_dataset_zip(merged_dir, zip_path, repo_dir)

    out_dir = ckpt_root / "training_runs"

    # Auto-resume from latest snapshot if no explicit resume given
    resume = args.resume
    if resume is None:
        snaps = sorted(out_dir.rglob("network-snapshot-*.pkl"))
        if snaps:
            resume = str(snaps[-1])
            print(f"Auto-resuming from {resume}")

    train(zip_path, out_dir, repo_dir, args.gpus, args.batch, args.kimg, resume)

    print(f"\nProjectedGAN training complete. Checkpoints in {out_dir}")
    print("Next: run generate_projected_gan.py")


if __name__ == "__main__":
    main()
