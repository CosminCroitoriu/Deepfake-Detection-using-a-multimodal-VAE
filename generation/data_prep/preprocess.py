#!/usr/bin/env python3
"""
Preprocess CrisisNLP real images: center-crop to square, resize to target size,
save as PNG.

Run prepare_dataset.py first.

Input:  ../data/real/<class>/   → ../data/real_512/<class>/
        ../data/real_extra/     → ../data/real_extra_512/   (--include_extra)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

from PIL import Image
from tqdm import tqdm

DISASTER_CLASSES = [
    "earthquake", "fire", "flood", "hurricane", "landslide",
]


def center_crop_resize(img: Image.Image, size: int = 512) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.LANCZOS)


def process_one(src: Path, dst: Path, size: int) -> bool:
    try:
        img = Image.open(src).convert("RGB")
        img = center_crop_resize(img, size)
        img.save(dst, "PNG", optimize=True)
        return True
    except Exception as e:
        print(f"  SKIP {src.name}: {e}")
        return False


def process_dir(src_dir: Path, dst_dir: Path, size: int, workers: int, label: str) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (p, dst_dir / (p.stem + ".png"))
        for p in src_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not tasks:
        return 0

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, src, dst, size): None for src, dst in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=label, leave=False):
            if fut.result():
                ok += 1
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--include_extra", action="store_true",
        help="Also process real_extra/ -> real_extra_512/ (Tasks 2 & 3 images for VAE training)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    total_ok = 0

    # Task 1: per-class disaster images
    src_root = data_dir / "real"
    dst_root = data_dir / f"real_{args.size}"
    print(f"Processing disaster class images ...")
    for cls in DISASTER_CLASSES:
        src_cls = src_root / cls
        if not src_cls.exists():
            print(f"  WARNING: {src_cls} not found, skipping")
            continue
        n = process_dir(src_cls, dst_root / cls, args.size, args.workers, cls)
        total_ok += n
        print(f"  {cls}: {n}")

    # Tasks 2 & 3: extra images for VAE training (unlabeled)
    if args.include_extra:
        src_extra = data_dir / "real_extra"
        dst_extra = data_dir / f"real_extra_{args.size}"
        if src_extra.exists():
            print(f"\nProcessing extra images (Tasks 2 & 3) ...")
            n = process_dir(src_extra, dst_extra, args.size, args.workers, "real_extra")
            total_ok += n
            print(f"  real_extra: {n}")
        else:
            print(f"\nWARNING: {src_extra} not found — run prepare_dataset.py first")

    print(f"\nTotal: {total_ok} images processed → {dst_root}")


if __name__ == "__main__":
    main()
