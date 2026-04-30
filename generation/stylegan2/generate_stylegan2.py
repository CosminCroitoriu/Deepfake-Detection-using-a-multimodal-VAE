#!/usr/bin/env python3
"""
Generate fake disaster images from trained StyleGAN2-ADA checkpoints.
Uses the latest checkpoint in each class's training directory.

Input:  ../checkpoints/stylegan2/<class>/network-snapshot-*.pkl
Output: ../data/fake/stylegan2/<class>/  (PNG images)
"""
import argparse
import subprocess
import sys
from pathlib import Path

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    pkls = sorted(ckpt_dir.glob("network-snapshot-*.pkl"))
    return pkls[-1] if pkls else None


def generate(
    checkpoint: Path,
    out_dir: Path,
    n_images: int,
    seed_offset: int,
    truncation: float,
    repo_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = f"{seed_offset}-{seed_offset + n_images - 1}"
    cmd = [
        sys.executable, str(repo_dir / "generate.py"),
        "--outdir", str(out_dir),
        "--network", str(checkpoint),
        "--seeds", seeds,
        "--trunc", str(truncation),
    ]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--checkpoints_dir", default="../checkpoints/stylegan2")
    parser.add_argument("--repo_dir", default="./stylegan2-ada-pytorch")
    parser.add_argument("--n_images", type=int, default=500,
                        help="Images to generate per class (default: 500)")
    parser.add_argument("--truncation", type=float, default=0.7,
                        help="Truncation psi: lower = more quality / less diversity")
    parser.add_argument("--classes", nargs="+", default=DISASTER_CLASSES,
                        choices=DISASTER_CLASSES)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    ckpt_root = Path(args.checkpoints_dir)
    out_root = Path(args.data_dir) / "fake" / "stylegan2"

    for cls in args.classes:
        ckpt_dir = ckpt_root / cls
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is None:
            print(f"WARNING: no checkpoint for '{cls}' in {ckpt_dir}, skipping")
            print("  Run train_stylegan2.py first.")
            continue

        print(f"\n[{cls}] checkpoint: {ckpt.name}")
        out_dir = out_root / cls
        generate(ckpt, out_dir, args.n_images, seed_offset=0,
                 truncation=args.truncation, repo_dir=repo_dir)
        print(f"  Saved {args.n_images} images -> {out_dir}")

    print("\nStyleGAN2-ADA generation complete.")


if __name__ == "__main__":
    main()
