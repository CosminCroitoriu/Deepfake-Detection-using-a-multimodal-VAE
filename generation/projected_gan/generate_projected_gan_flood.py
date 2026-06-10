#!/usr/bin/env python3
"""
Generate flood fakes from the flood-only ProjectedGAN checkpoint.

Thin wrapper around generate_projected_gan.py pointed at the flood checkpoint
dir (checkpoints/projected_gan_flood_512) and writing to a separate output
folder (data/fake/projected_gan_flood/flood). Leaves the all-classes generation
output untouched.

Usage:
  python generate_projected_gan_flood.py
  python generate_projected_gan_flood.py --n_images 1000 --truncation 0.7
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "generate_projected_gan.py"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--checkpoints_dir",
                        default=str(SCRIPT_DIR / "../../checkpoints/projected_gan_flood_512"))
    parser.add_argument("--repo_dir", default=str(SCRIPT_DIR / "projected_gan"))
    parser.add_argument("--output_dir",
                        default=str(SCRIPT_DIR / "../../data/fake/projected_gan_flood"))
    parser.add_argument("--n_images", type=int, default=1000,
                        help="Flood images to generate (default: 1000)")
    parser.add_argument("--truncation", type=float, default=0.7)
    args = parser.parse_args()

    cmd = [
        sys.executable, str(BASE_SCRIPT),
        "--data_dir", args.data_dir,
        "--checkpoints_dir", args.checkpoints_dir,
        "--repo_dir", args.repo_dir,
        "--output_dir", args.output_dir,
        "--n_images", str(args.n_images),
        "--truncation", str(args.truncation),
        "--classes", "flood",
    ]
    print("Flood-only ProjectedGAN generation")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
