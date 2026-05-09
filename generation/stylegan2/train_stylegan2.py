#!/usr/bin/env python3
"""
Fine-tune StyleGAN2-ADA on CrisisNLP disaster images.
Trains one model per disaster class to avoid mode collapse from scene diversity.
ADA (Adaptive Discriminator Augmentation) is designed for datasets under 100K images.

Run preprocess.py first.

Usage:
  python train_stylegan2.py
  python train_stylegan2.py --classes flood fire --kimg 3000 --gpus 2
"""
import argparse
import subprocess
import sys
from pathlib import Path

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]
REPO_URL = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"

SCRIPT_DIR = Path(__file__).resolve().parent


def clone_repo(repo_dir: Path):
    if repo_dir.exists():
        print(f"Repo already at {repo_dir}")
        return
    print(f"Cloning StyleGAN2-ADA into {repo_dir} ...")
    subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)


def apply_patches(repo_dir: Path):
    # Patch 1: misc.py — PyTorch ≥1.11 removed the dataset arg from Sampler.__init__
    misc_path = repo_dir / "torch_utils" / "misc.py"
    text = misc_path.read_text()
    if "super().__init__(dataset)" in text:
        misc_path.write_text(text.replace("super().__init__(dataset)", "super().__init__()"))
        print("  Patched torch_utils/misc.py")

    # Patch 2: train.py — PyTorch ≥2.0 rejects mixed int/float betas in Adam
    train_path = repo_dir / "train.py"
    text = train_path.read_text()
    patched = text.replace("betas=[0,", "betas=[0.0,")
    if patched != text:
        train_path.write_text(patched)
        print("  Patched train.py (betas)")

    # Patch 4: training/loss.py — R1 uses create_graph=True (needs 2nd-order grid_sample grad,
    # absent in PyTorch 2.x); set False so R1 value is reported but gradient is detached
    loss_path = repo_dir / "training" / "loss.py"
    text = loss_path.read_text()
    patched = text.replace("create_graph=True", "create_graph=False")
    if patched != text:
        loss_path.write_text(patched)
        print("  Patched training/loss.py (R1 create_graph)")

    # Patch 3: grid_sample_gradfix.py — PyTorch >=2.x dropped grid_sampler_2d_backward;
    # replace entire file with a minimal stub that always uses native F.grid_sample
    grid_path = repo_dir / "torch_utils" / "ops" / "grid_sample_gradfix.py"
    if grid_path.exists() and "_GridSample2dForward" in grid_path.read_text():
        grid_path.write_text(
            "# Patched: PyTorch >=2.x dropped grid_sampler_2d_backward\n"
            "import torch\n\n"
            "enabled = False\n\n"
            "def grid_sample(input, grid, *, mode='bilinear', padding_mode='zeros', align_corners=False):\n"
            "    return torch.nn.functional.grid_sample(\n"
            "        input=input, grid=grid, mode=mode,\n"
            "        padding_mode=padding_mode, align_corners=align_corners\n"
            "    )\n"
        )
        print("  Patched torch_utils/ops/grid_sample_gradfix.py")


def make_dataset_zip(src_dir: Path, zip_path: Path, repo_dir: Path):
    if zip_path.exists():
        print(f"  Dataset zip exists: {zip_path.name}")
        return
    print(f"  Creating dataset zip from {src_dir} ...")
    subprocess.run(
        [
            sys.executable, str(repo_dir / "dataset_tool.py"),
            "--source", str(src_dir),
            "--dest", str(zip_path),
        ],
        check=True,
    )


def latest_snapshot(out_dir: Path) -> str | None:
    """Return path to latest network-snapshot-*.pkl in the run directory, or None."""
    run_dirs = sorted(out_dir.glob("*-stylegan2ada-*"))
    for run_dir in reversed(run_dirs):
        snaps = sorted(run_dir.glob("network-snapshot-*.pkl"))
        if snaps:
            return str(snaps[-1])
    return None


def train_class(
    cls: str,
    dataset_zip: Path,
    out_dir: Path,
    repo_dir: Path,
    gpus: int,
    batch: int,
    kimg: int,
    resume: str | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume from latest snapshot if no explicit resume given
    if resume is None:
        resume = latest_snapshot(out_dir)
        if resume:
            print(f"  Auto-resuming from {resume}")

    cmd = [
        sys.executable, str(repo_dir / "train.py"),
        "--outdir", str(out_dir),
        "--data", str(dataset_zip),
        "--cfg", "paper512",
        "--gpus", str(gpus),
        "--batch", str(batch),
        "--gamma", "65",
        "--aug", "ada",
        "--target", "0.6",
        "--kimg", str(kimg),
        "--snap", "50",
        "--mirror", "1",
    ]
    if resume:
        cmd += ["--resume", resume]
    print(f"\n[{cls}] Launching training ...")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--repo_dir", default=str(SCRIPT_DIR / "stylegan2-ada-pytorch"))
    parser.add_argument("--checkpoints_dir", default=str(SCRIPT_DIR / "../../checkpoints/stylegan2"))
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32,
                        help="Total batch size across all GPUs")
    parser.add_argument("--kimg", type=int, default=5000,
                        help="Training length in thousands of real images")
    parser.add_argument("--classes", nargs="+", default=DISASTER_CLASSES,
                        choices=DISASTER_CLASSES)
    parser.add_argument("--resume", default=None,
                        help="Explicit .pkl checkpoint to resume from (overrides auto-resume)")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    data_root = Path(args.data_dir) / "real_512"
    ckpt_root = Path(args.checkpoints_dir)

    clone_repo(repo_dir)
    apply_patches(repo_dir)

    for cls in args.classes:
        cls_dir = data_root / cls
        if not cls_dir.exists():
            print(f"WARNING: {cls_dir} not found, skipping {cls}")
            continue

        n = len(list(cls_dir.glob("*.png")))
        print(f"\n=== {cls} ({n} images) ===")
        if n < 200:
            print(f"  WARNING: only {n} images — quality may be poor")

        zip_path = ckpt_root / "datasets" / f"{cls}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        make_dataset_zip(cls_dir, zip_path, repo_dir)

        out_dir = ckpt_root / cls
        train_class(cls, zip_path, out_dir, repo_dir,
                    args.gpus, args.batch, args.kimg, args.resume)

    print(f"\nAll done. Checkpoints in {ckpt_root}/")
    print("Next: run generate_stylegan2.py")


if __name__ == "__main__":
    main()
