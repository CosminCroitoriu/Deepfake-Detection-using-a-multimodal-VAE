"""
Dataset classes for the Gated Multimodal VAE.

Reuses the 5-channel ConcatTransform from detection.concat_vae — the model
slices the 5-channel input into per-modality tensors internally.
"""
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

from torch.utils.data import Dataset

from ..concat_vae.transform import ConcatTransform

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]
GENERATOR_DIRS = ["sd_lora", "sd3_lora", "projected_gan_512", "stylegan2", "controlnet"]


def _collect_images(root: Path, classes: List[str]) -> List[Path]:
    paths = []
    for cls in classes:
        d = root / cls
        if d.is_dir():
            paths.extend(sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")))
    return paths


class RealTrainDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        seed: int = 42,
        transform: ConcatTransform = None,
    ):
        data_root = Path(data_dir)

        task1_paths = _collect_images(data_root / "real_512", DISASTER_CLASSES)
        rng = random.Random(seed)
        rng.shuffle(task1_paths)
        n = len(task1_paths)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if split == "train":
            self.paths = task1_paths[:n_train]
            extra_root = data_root / "real_extra_512"
            if extra_root.exists():
                extra = sorted(extra_root.glob("*.png")) + sorted(extra_root.glob("*.jpg"))
                self.paths = self.paths + extra
        elif split == "val":
            self.paths = task1_paths[n_train: n_train + n_val]
        else:
            self.paths = task1_paths[n_train + n_val:]

        self.transform = transform or ConcatTransform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class EvalDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        generator: str,
        subset: str = "test",
        fake_thresh_frac: float = 0.20,
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        seed: int = 42,
        transform: ConcatTransform = None,
    ):
        data_root = Path(data_dir)

        real_root = data_root / "real_512"
        all_real = _collect_images(real_root, DISASTER_CLASSES)
        rng = random.Random(seed)
        rng.shuffle(all_real)
        n = len(all_real)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        real_test = all_real[n_train + n_val:]

        fake_root = data_root / "fake" / generator
        all_fake = _collect_images(fake_root, DISASTER_CLASSES)
        rng2 = random.Random(seed + 1)
        rng2.shuffle(all_fake)

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

        self.samples: List[Tuple[Path, int]] = (
            [(p, 0) for p in real_split] + [(p, 1) for p in fake_split]
        )
        rng3 = random.Random(seed + 2)
        rng3.shuffle(self.samples)

        self.transform = transform or ConcatTransform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        input_5ch, target_5ch = self.transform(img)
        return input_5ch, target_5ch, label
