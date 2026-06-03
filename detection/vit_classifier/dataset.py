"""
Dataset classes for the supervised ViT classifier.

Split strategy
--------------
Real images   : 80/10/10 train/val/test from real_512  (same as VAE models)
Fake images   : 80/20 train/eval split applied *per generator* so each
                generator contributes ~500 evaluation images (2 500 × 0.20).

Training data : real_train + 80% of all generators pooled together.
Eval data     : per-generator 20% held-out set, then split into
                20% threshold-calibration + 80% test (matching VAE protocol).
"""
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset
from torchvision import transforms

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]
# Generators that actually have images on disk (projected_gan_512 is empty)
GENERATOR_DIRS = ["sd_lora", "sd3_lora", "projected_gan", "projected_gan_512_1"]

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _collect_images(root: Path, classes: List[str]) -> List[Path]:
    paths = []
    for cls in classes:
        d = root / cls
        if d.is_dir():
            paths.extend(sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")))
    return paths


def _fake_splits(data_root: Path, generator: str, fake_eval_frac: float, seed: int):
    """Return (train_paths, eval_paths) for a single generator."""
    all_fake = _collect_images(data_root / "fake" / generator, DISASTER_CLASSES)
    rng = random.Random(seed + 10)
    rng.shuffle(all_fake)
    n_eval = int(len(all_fake) * fake_eval_frac)
    return all_fake[n_eval:], all_fake[:n_eval]   # train, eval


class ClassifierTrainDataset(Dataset):
    """
    Balanced real/fake dataset for supervised training.

    Both classes are capped to the smaller class size per split to keep
    the training loss unbiased.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        fake_eval_frac: float = 0.20,
        seed: int = 42,
        transform=None,
    ):
        data_root = Path(data_dir)

        # ── Real images ────────────────────────────────────────────────────
        all_real = _collect_images(data_root / "real_512", DISASTER_CLASSES)
        rng = random.Random(seed)
        rng.shuffle(all_real)
        n = len(all_real)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        if split == "train":
            real_paths = all_real[:n_train]
            extra = data_root / "real_extra_512"
            if extra.exists():
                real_paths = real_paths + sorted(extra.glob("*.png")) + sorted(extra.glob("*.jpg"))
        elif split == "val":
            real_paths = all_real[n_train: n_train + n_val]
        else:
            real_paths = all_real[n_train + n_val:]

        # ── Fake images: pool all generators, take training portion ───────
        fake_paths: List[Path] = []
        for gen in GENERATOR_DIRS:
            gen_root = data_root / "fake" / gen
            if not gen_root.exists():
                continue
            gen_train, gen_eval = _fake_splits(data_root, gen, fake_eval_frac, seed)
            gen_all = gen_train  # eval split is held out
            rng2 = random.Random(seed + 3)
            rng2.shuffle(gen_all)
            n_g = len(gen_all)
            n_g_train = int(n_g * (train_frac / (train_frac + val_frac)))
            if split == "train":
                fake_paths.extend(gen_all[:n_g_train])
            else:
                fake_paths.extend(gen_all[n_g_train:])

        # ── Balance and shuffle ────────────────────────────────────────────
        rng3 = random.Random(seed + 4)
        n_min = min(len(real_paths), len(fake_paths))
        real_sampled = rng3.sample(real_paths, n_min)
        fake_sampled = rng3.sample(fake_paths, n_min)

        self.samples: List[Tuple[Path, int]] = (
            [(p, 0) for p in real_sampled] +
            [(p, 1) for p in fake_sampled]
        )
        rng3.shuffle(self.samples)
        self.transform = transform if transform is not None else train_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class ClassifierEvalDataset(Dataset):
    """
    Per-generator evaluation dataset matching the VAE EvalDataset protocol.

    Fake images come from the held-out 20% that was *not* used during training.
    Real images come from the same held-out real_test as the VAE models.

    subset="thresh" — 20% of eval images for Youden threshold calibration
    subset="test"   — remaining 80% for reporting metrics
    """

    def __init__(
        self,
        data_dir: str,
        generator: str,
        subset: str = "test",
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        fake_eval_frac: float = 0.20,
        fake_thresh_frac: float = 0.20,
        seed: int = 42,
        transform=None,
    ):
        data_root = Path(data_dir)

        # Real test images — same held-out split as VAE models
        all_real = _collect_images(data_root / "real_512", DISASTER_CLASSES)
        rng = random.Random(seed)
        rng.shuffle(all_real)
        n = len(all_real)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        real_test = all_real[n_train + n_val:]

        # Fake eval images: the 20% held out from classifier training
        _, fake_eval = _fake_splits(data_root, generator, fake_eval_frac, seed)

        n_eval = min(len(fake_eval), len(real_test))
        real_eval = real_test[:n_eval]
        fake_eval  = fake_eval[:n_eval]

        n_thresh = int(n_eval * fake_thresh_frac)
        if subset == "thresh":
            real_split = real_eval[:n_thresh]
            fake_split = fake_eval[:n_thresh]
        else:
            real_split = real_eval[n_thresh:]
            fake_split = fake_eval[n_thresh:]

        self.samples: List[Tuple[Path, int]] = (
            [(p, 0) for p in real_split] +
            [(p, 1) for p in fake_split]
        )
        rng2 = random.Random(seed + 2)
        rng2.shuffle(self.samples)
        self.transform = transform if transform is not None else eval_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
