"""
Flood-only dataset classes for the SVD U-Net VAE.

Identical split logic to detection.svd_unet_vae.dataset, with two changes for
the single-disaster experiment:

  1. Only the "flood" class is used (no other disaster types).
  2. The training split does NOT pull in real_extra_512. The whole point of
     this run is to train on one coherent mode, so the extra mixed images are
     deliberately left out.

Fakes are read from the flood-only generator folders produced by the
generate_*_flood.py scripts (data/fake/sd_lora_flood, data/fake/sd3_lora_flood).
"""
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # allow large ICC profiles
from torch.utils.data import Dataset

from .svd_transform import SVDTransform

DISASTER_CLASSES = ["flood"]
GENERATOR_DIRS = ["sd_lora_flood", "sd3_lora_flood"]


def _collect_images(root: Path, classes: List[str]) -> List[Path]:
    paths = []
    for cls in classes:
        d = root / cls
        if d.is_dir():
            paths.extend(sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")))
    return paths


class RealTrainDataset(Dataset):
    """Real flood images, training split — returns (I_low, I_gray) pairs."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train" | "val" | "test"
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        seed: int = 42,
        transform: SVDTransform = None,
    ):
        data_root = Path(data_dir)

        # Flood images only — deterministic train/val/test split
        flood_paths = _collect_images(data_root / "real_512", DISASTER_CLASSES)
        rng = random.Random(seed)
        rng.shuffle(flood_paths)
        n = len(flood_paths)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if split == "train":
            self.paths = flood_paths[:n_train]
        elif split == "val":
            self.paths = flood_paths[n_train: n_train + n_val]
        else:
            self.paths = flood_paths[n_train + n_val:]

        self.transform = transform or SVDTransform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class EvalDataset(Dataset):
    """
    Mixed real-test + fake flood images for binary evaluation.

    label 0 -> real
    label 1 -> fake (from a single flood generator)
    """

    def __init__(
        self,
        data_dir: str,
        generator: str,
        subset: str = "test",   # "thresh" | "test"
        fake_thresh_frac: float = 0.20,
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        seed: int = 42,
        transform: SVDTransform = None,
    ):
        data_root = Path(data_dir)

        # real flood test split (same split as training, test portion only)
        real_root = data_root / "real_512"
        all_real = _collect_images(real_root, DISASTER_CLASSES)
        rng = random.Random(seed)
        rng.shuffle(all_real)
        n = len(all_real)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        real_test = all_real[n_train + n_val:]

        # fake flood images for the requested generator
        fake_root = data_root / "fake" / generator
        all_fake = _collect_images(fake_root, DISASTER_CLASSES)
        rng2 = random.Random(seed + 1)
        rng2.shuffle(all_fake)

        # balance: cap fakes to the number of real test images, then split both 20/80
        n_eval = min(len(all_fake), len(real_test))
        real_eval = real_test[:n_eval]
        fake_eval = all_fake[:n_eval]

        n_thresh = int(n_eval * fake_thresh_frac)
        if subset == "thresh":
            real_split = real_eval[:n_thresh]
            fake_split = fake_eval[:n_thresh]
        else:
            real_split = real_eval[n_thresh:]
            fake_split = fake_eval[n_thresh:]

        # build combined list of (path, label)
        self.samples: List[Tuple[Path, int]] = (
            [(p, 0) for p in real_split] + [(p, 1) for p in fake_split]
        )
        rng3 = random.Random(seed + 2)
        rng3.shuffle(self.samples)

        self.transform = transform or SVDTransform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        i_low, i_gray = self.transform(img)
        return i_low, i_gray, label
