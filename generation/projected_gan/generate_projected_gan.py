#!/usr/bin/env python3
"""
Generate fake disaster images from a trained ProjectedGAN checkpoint.

ProjectedGAN is trained unconditionally on mixed classes, so images are
distributed evenly across class folders by seed range.

Input:  ../checkpoints/projected_gan/training_runs/
Output: ../data/fake/projected_gan/<class>/
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]


def latest_checkpoint(run_root: Path) -> Path | None:
    pkls = sorted(run_root.rglob("network-snapshot-*.pkl"))
    return pkls[-1] if pkls else None


def generate_pool(
    checkpoint: Path,
    tmp_dir: Path,
    n_total: int,
    truncation: float,
    repo_dir: Path,
):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seeds = f"0-{n_total - 1}"
    cmd = [
        sys.executable, str(repo_dir / "generate.py"),
        "--outdir", str(tmp_dir),
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
    parser.add_argument("--checkpoints_dir", default="../checkpoints/projected_gan")
    parser.add_argument("--repo_dir", default="./projected_gan")
    parser.add_argument("--n_images", type=int, default=500,
                        help="Images to generate per class (default: 500)")
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--classes", nargs="+", default=DISASTER_CLASSES,
                        choices=DISASTER_CLASSES)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    run_root = Path(args.checkpoints_dir) / "training_runs"
    checkpoint = latest_checkpoint(run_root)

    if checkpoint is None:
        print(f"ERROR: no checkpoint found under {run_root}")
        print("Run train_projected_gan.py first.")
        return 1

    print(f"Checkpoint: {checkpoint}")

    n_classes = len(args.classes)
    n_total = args.n_images * n_classes
    out_root = Path(args.data_dir) / "fake" / "projected_gan"
    tmp_dir = out_root / "_pool"

    print(f"Generating {n_total} images ({args.n_images} per class) ...")
    generate_pool(checkpoint, tmp_dir, n_total, args.truncation, repo_dir)

    # Distribute generated images across class folders by seed order
    all_imgs = sorted(tmp_dir.glob("*.png"))
    per_cls = len(all_imgs) // n_classes
    for i, cls in enumerate(args.classes):
        cls_dir = out_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for img in all_imgs[i * per_cls : (i + 1) * per_cls]:
            img.rename(cls_dir / img.name)
        print(f"  {cls}: {per_cls} images -> {cls_dir}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nProjectedGAN generation complete.")


if __name__ == "__main__":
    main()
