#!/usr/bin/env python3
"""
Flood-only ProjectedGAN run (single-disaster ablation).

Thin wrapper around train_projected_gan.py. It launches the same training
pipeline but restricted to the flood class and writes to a separate checkpoint
directory, so the existing all-classes (10k) run is left untouched.

Motivation: the all-classes generators were trained across every disaster type
(and, for the LoRAs, the extra non-disaster images), which spreads model
capacity thin and yields generic samples. This run tests whether a single
coherent visual mode (flood, the largest class at ~3.2k images) produces
higher-fidelity fakes.

Mirror augmentation stays on (inherited from the base script) to buy back some
effective data, since a single class sees fewer images than the mixed set.

Usage:
  python train_projected_gan_flood.py
  python train_projected_gan_flood.py --kimg 5000 --gpus 1 --batch 32
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "train_projected_gan.py"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--checkpoints_dir",
                        default=str(SCRIPT_DIR / "../../checkpoints/projected_gan_flood_512"))
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--kimg", type=int, default=5000)
    parser.add_argument("--img_res", type=int, default=512, choices=[256, 512])
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cmd = [
        sys.executable, str(BASE_SCRIPT),
        "--data_dir", args.data_dir,
        "--checkpoints_dir", args.checkpoints_dir,
        "--gpus", str(args.gpus),
        "--batch", str(args.batch),
        "--kimg", str(args.kimg),
        "--img_res", str(args.img_res),
        "--classes", "flood",
    ]
    if args.resume:
        cmd += ["--resume", args.resume]

    print("Flood-only ProjectedGAN run")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
