"""
FF++ dataset wrappers for the SVD U-Net VAE.

Mirrors the structure of detection.svd_unet_vae.dataset but adapted to
FaceForensics++:
  - flat per-manipulation directories (no per-class subdirs)
  - SVDTransform pinned at 256×256 (matches the Sarkar paper)
"""
import random
from pathlib import Path
from typing import List

from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

from torch.utils.data import Dataset

from ..svd_unet_vae.svd_transform import SVDTransform

FFPP_TARGET_SIZE = 256
MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _list_pngs(d: Path) -> List[Path]:
    return sorted(d.glob("*.png"))


def _default_transform() -> SVDTransform:
    return SVDTransform(target_size=FFPP_TARGET_SIZE)


class FFppRealDataset(Dataset):
    """Real face frames from FF++ `original/`. Returns (i_low, i_gray)."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        seed: int = 42,
        transform: SVDTransform = None,
    ):
        all_paths = _list_pngs(Path(data_dir) / "original")
        rng = random.Random(seed)
        rng.shuffle(all_paths)
        n = len(all_paths)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if split == "train":
            self.paths = all_paths[:n_train]
        elif split == "val":
            self.paths = all_paths[n_train: n_train + n_val]
        else:
            self.paths = all_paths[n_train + n_val:]

        self.transform = transform or _default_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class FFppEvalDataset(Dataset):
    """Real test split + fakes from one manipulation. Returns (i_low, i_gray, label)."""

    def __init__(
        self,
        data_dir: str,
        manipulation: str,
        subset: str = "test",
        fake_thresh_frac: float = 0.20,
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        seed: int = 42,
        transform: SVDTransform = None,
    ):
        data_root = Path(data_dir)
        all_real = _list_pngs(data_root / "original")
        rng = random.Random(seed)
        rng.shuffle(all_real)
        n = len(all_real)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        real_test = all_real[n_train + n_val:]

        all_fake = _list_pngs(data_root / manipulation)
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

        self.samples = (
            [(p, 0) for p in real_split] + [(p, 1) for p in fake_split]
        )
        rng3 = random.Random(seed + 2)
        rng3.shuffle(self.samples)

        self.transform = transform or _default_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        i_low, i_gray = self.transform(img)
        return i_low, i_gray, label
