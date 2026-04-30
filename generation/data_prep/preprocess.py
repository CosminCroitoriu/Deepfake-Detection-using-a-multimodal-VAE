#!/usr/bin/env python3
"""
Preprocess CrisisNLP real images: center-crop to square, resize to 256x256,
save as PNG.

Input:  ../data/real/<class>/
Output: ../data/real_256/<class>/
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ALL_CLASSES = [
    "earthquake", "fire", "flood", "hurricane", "landslide",
    "not_disaster", "other_disaster",
]


def center_crop_resize(img: Image.Image, size: int = 256) -> Image.Image:
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src_root = Path(args.data_dir) / "real"
    dst_root = Path(args.data_dir) / "real_256"
    dst_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cls in ALL_CLASSES:
        src_cls = src_root / cls
        if not src_cls.exists():
            print(f"WARNING: {src_cls} not found, skipping")
            continue
        dst_cls = dst_root / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        for img_path in src_cls.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                dst_path = dst_cls / (img_path.stem + ".png")
                tasks.append((img_path, dst_path, cls))

    print(f"Processing {len(tasks)} images with {args.workers} workers ...")
    ok = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, src, dst, args.size): (src, dst, cls)
            for src, dst, cls in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            src, dst, cls = futures[fut]
            if fut.result():
                ok += 1

    print(f"\nSaved {ok}/{len(tasks)} images to {dst_root}")
    print("\nCounts per class:")
    for cls in ALL_CLASSES:
        n = len(list((dst_root / cls).glob("*.png"))) if (dst_root / cls).exists() else 0
        if n:
            print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()
