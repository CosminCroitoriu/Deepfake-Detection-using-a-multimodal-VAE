#!/usr/bin/env python3
"""
Prepare CrisisNLP real images from the crisis_vision_benchmarks TSV structure.

Reads all four task TSVs, applies filtering rules, and copies images into the
data/ directory structure expected by preprocess.py.

Filtering rules:
  Task 1 (disaster_types): keep earthquake, fire, flood, hurricane, landslide.
                            Drop not_disaster and other_disaster.
  Task 4 (informative):    build an exclusion set of not_informative image paths
                            and apply it across all tasks.
  Tasks 2, 3:              include all images (all are disaster scenes by definition).

Output:
  data/real/<class>/                    Task 1 images per disaster class
  data/real_extra/                      Tasks 2/3 images not already in Task 1
  data/splits/task1_{train,dev,test}.txt  Split manifests (relative to data/)

Usage:
  python prepare_dataset.py --dataset_dir /path/to/crisis_vision_benchmarks
  python prepare_dataset.py --dataset_dir /path/to/crisis_vision_benchmarks --data_dir /path/to/data
"""
import argparse
import csv
import shutil
from pathlib import Path

KEEP_CLASSES = frozenset(["earthquake", "fire", "flood", "hurricane", "landslide"])

SPLITS = ["train", "dev", "test"]

TASK_TSV = {
    "disaster_types": "tasks/disaster_types/consolidated/consolidated_disaster_types_{split}_final.tsv",
    "informative":    "tasks/informative/consolidated/consolidated_info_{split}_final.tsv",
    "humanitarian":   "tasks/humanitarian/consolidated/consolidated_hum_{split}_final.tsv",
    "damage_severity":"tasks/damage_severity/consolidated/consolidated_damage_{split}_final.tsv",
}


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_not_informative_set(dataset_dir: Path) -> set[str]:
    """Return the set of image_paths labeled not_informative in Task 4 across all splits."""
    excluded = set()
    for split in SPLITS:
        tsv = dataset_dir / TASK_TSV["informative"].format(split=split)
        if not tsv.exists():
            print(f"  WARNING: {tsv} not found")
            continue
        for row in read_tsv(tsv):
            if row["class_label"] == "not_informative":
                excluded.add(row["image_path"])
    return excluded


def copy_image(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Path to crisis_vision_benchmarks/ directory",
    )
    parser.add_argument(
        "--data_dir", default="../data",
        help="Output base directory (default: ../data)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    real_dir = data_dir / "real"
    extra_dir = data_dir / "real_extra"
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build the not-informative exclusion set from Task 4
    # ------------------------------------------------------------------
    print("Building not-informative exclusion set (Task 4) ...")
    not_informative = build_not_informative_set(dataset_dir)
    print(f"  {len(not_informative):,} image paths will be excluded")

    # ------------------------------------------------------------------
    # Task 1 — Disaster Types
    # Copy to real/<class>/ and write per-split manifests.
    # ------------------------------------------------------------------
    print("\nProcessing Task 1 (disaster_types) ...")
    class_counts: dict[str, int] = {cls: 0 for cls in KEEP_CLASSES}
    copied_img_rels: set[str] = set()  # image_path values already copied
    split_manifests: dict[str, list[str]] = {s: [] for s in SPLITS}

    for split in SPLITS:
        tsv = dataset_dir / TASK_TSV["disaster_types"].format(split=split)
        if not tsv.exists():
            print(f"  WARNING: {tsv} not found, skipping {split}")
            continue

        for row in read_tsv(tsv):
            label = row["class_label"]
            img_rel = row["image_path"]

            if label not in KEEP_CLASSES:
                continue
            if img_rel in not_informative:
                continue

            src = dataset_dir / img_rel
            fname = Path(img_rel).name
            dst = real_dir / label / fname

            if copy_image(src, dst):
                class_counts[label] += 1
                split_manifests[split].append(f"real/{label}/{fname}")
                copied_img_rels.add(img_rel)

    for split, paths in split_manifests.items():
        manifest = splits_dir / f"task1_{split}.txt"
        manifest.write_text("\n".join(paths) + ("\n" if paths else ""))

    print("  Split sizes:")
    for split, paths in split_manifests.items():
        print(f"    {split}: {len(paths)}")
    print("  Class counts (all splits combined):")
    for cls in sorted(class_counts):
        print(f"    {cls}: {class_counts[cls]}")

    # ------------------------------------------------------------------
    # Tasks 2, 3, 4 — Extra real images for VAE training (unlabeled)
    # Copy to real_extra/ — no class subdirs, VAE trains without labels.
    # Images already present in Task 1 are skipped.
    # Task 4: only keep rows labeled informative (the not_informative rows
    # are already excluded via the exclusion set, but this makes it explicit).
    # ------------------------------------------------------------------
    print("\nProcessing Tasks 2, 3, 4 (extra images for VAE training) ...")
    seen_fnames: set[str] = set()
    extra_count = 0

    for task in ["humanitarian", "damage_severity", "informative"]:
        for split in SPLITS:
            tsv = dataset_dir / TASK_TSV[task].format(split=split)
            if not tsv.exists():
                continue

            for row in read_tsv(tsv):
                img_rel = row["image_path"]

                if task == "informative" and row["class_label"] != "informative":
                    continue
                if img_rel in not_informative:
                    continue
                if img_rel in copied_img_rels:
                    continue  # already in Task 1 or earlier task

                src = dataset_dir / img_rel
                fname = Path(img_rel).name
                # Resolve filename collisions across source datasets
                if fname in seen_fnames:
                    fname = f"{Path(img_rel).parent.name}_{fname}"
                dst = extra_dir / fname

                if copy_image(src, dst):
                    extra_count += 1
                    copied_img_rels.add(img_rel)
                    seen_fnames.add(fname)

    print(f"  {extra_count:,} extra images -> {extra_dir}")
    print(f"\nDone. Next step: run preprocess.py to resize images to 256x256.")
    print(f"  python preprocess.py --data_dir {data_dir} --include_extra")


if __name__ == "__main__":
    main()
