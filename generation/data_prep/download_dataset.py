#!/usr/bin/env python3
"""
Organize the CrisisNLP ASONAM20 crisis image dataset into the expected
data/ directory structure.

1. Download from: https://crisisnlp.qcri.org/crisis-image-datasets-asonam20
2. Run:
     python download_dataset.py --zip_path /path/to/crisisnlp.zip
   or, if already extracted:
     python download_dataset.py --raw_dir /path/to/extracted_folder/

Output:
  ../data/real/earthquake/
  ../data/real/fire/
  ../data/real/flood/
  ../data/real/hurricane/
  ../data/real/landslide/
  ../data/real/not_disaster/
  ../data/real/other_disaster/
"""
import argparse
import shutil
import zipfile
from pathlib import Path

ALL_CLASSES = [
    "earthquake", "fire", "flood", "hurricane", "landslide",
    "not_disaster", "other_disaster",
]

# CrisisNLP zip may use spaces or underscores in folder names
CLASS_VARIANTS = {cls: [cls, cls.replace("_", " ")] for cls in ALL_CLASSES}


def find_class_dirs(root: Path) -> dict:
    found = {cls: [] for cls in ALL_CLASSES}
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        name = p.name.lower().replace(" ", "_")
        if name in found:
            found[name].append(p)
    return found


def organize(src_root: Path, out_real: Path):
    class_dirs = find_class_dirs(src_root)
    counts = {}
    for cls, dirs in class_dirs.items():
        if not dirs:
            print(f"  WARNING: no folder found for '{cls}'")
            counts[cls] = 0
            continue
        dst = out_real / cls
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for d in dirs:
            split_tag = d.parent.name  # train / dev / test
            for img in d.iterdir():
                if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    dst_file = dst / f"{split_tag}_{img.name}"
                    if not dst_file.exists():
                        shutil.copy2(img, dst_file)
                        n += 1
        counts[cls] = n
        print(f"  {cls}: {n} images")

    total = sum(counts.values())
    print(f"  Total: {total} images")
    if total == 0:
        print(
            "\nERROR: No images found. Verify the zip contains:\n"
            "  .../Task1_disaster_types_multiclass/{train,dev,test}/<class>/"
        )
    return total


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data",
                        help="Base data directory (default: ../data)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--zip_path", help="Path to downloaded CrisisNLP zip archive")
    group.add_argument("--raw_dir", help="Path to already-extracted dataset directory")
    args = parser.parse_args()

    out_real = Path(args.data_dir) / "real"

    if args.zip_path:
        zip_path = Path(args.zip_path)
        print(f"Extracting {zip_path} ...")
        tmp = Path(args.data_dir) / "_extract_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        print("Organizing ...")
        organize(tmp, out_real)
        shutil.rmtree(tmp)

    elif args.raw_dir:
        print(f"Organizing from {args.raw_dir} ...")
        organize(Path(args.raw_dir), out_real)

    else:
        print("CrisisNLP ASONAM20 — Download Instructions")
        print("=" * 45)
        print()
        print("1. Visit: https://crisisnlp.qcri.org/crisis-image-datasets-asonam20")
        print("2. Fill out the form and download the zip archive.")
        print("3. Run one of:")
        print()
        print("   python download_dataset.py --zip_path /path/to/crisisnlp.zip")
        print("   python download_dataset.py --raw_dir /path/to/extracted_folder/")
        print()
        print("Expected output:")
        for cls in ALL_CLASSES:
            print(f"   ../data/real/{cls}/")
        return

    print(f"\nDone. Next step: run preprocess.py to resize images to 256x256.")


if __name__ == "__main__":
    main()
