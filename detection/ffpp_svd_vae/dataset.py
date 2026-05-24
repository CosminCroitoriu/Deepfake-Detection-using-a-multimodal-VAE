"""
FF++ dataset wrappers for the SVD U-Net VAE.

Video-level splits by source identity (no leakage):
  - sources   0-719 → train (~720 videos × 32 frames = ~23k frames)
  - sources 720-859 → val   (~140 videos × 32 frames = ~4.5k frames)
  - sources 860-999 → test  (~140 videos × 32 frames = ~4.5k frames)

This matches what FF++'s official splits achieve. The previous random
frame-level split caused identity leakage: same videos appeared in both
train and test, and fakes were derived from those same source videos —
giving the model effectively zero distribution shift between train and
test → AUC = 0.5.

Filenames carry the source ID as the first underscore-delimited token:
  original/<source>_<frame>.png            e.g. 035_0030.png
  Deepfakes/<source>_<target>_<frame>.png  e.g. 035_036_0030.png
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

# Video-level splits — by source identity (matches FF++ official protocol)
SPLIT_BOUNDARIES = {
    "train": (0, 720),
    "val": (720, 860),
    "test": (860, 1000),
}


def _list_pngs(d: Path) -> List[Path]:
    return sorted(d.glob("*.png"))


def _default_transform() -> SVDTransform:
    return SVDTransform(target_size=FFPP_TARGET_SIZE)


def parse_source_id(path: Path) -> str:
    """First underscore-delimited token of the stem is always the source video ID.

    Real    : 035_0030.png    -> '035'
    Fake    : 035_036_0030.png -> '035'
    """
    return path.stem.split("_")[0]


def video_id_in_split(video_id_str: str, split: str) -> bool:
    """Check whether a video ID (as zero-padded string) belongs to a given split."""
    try:
        n = int(video_id_str)
    except ValueError:
        return False
    lo, hi = SPLIT_BOUNDARIES[split]
    return lo <= n < hi


def _filter_paths_by_split(paths: List[Path], split: str) -> List[Path]:
    """Keep only frames whose source video ID falls in this split's range."""
    return [p for p in paths if video_id_in_split(parse_source_id(p), split)]


class FFppRealDataset(Dataset):
    """Real face frames from FF++ `original/`. Returns (i_low, i_gray).

    Frames are filtered by source video ID: all frames from the same video
    go to the same split. No identity leakage between train/val/test.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seed: int = 42,
        transform: SVDTransform = None,
        # train_frac / val_frac kept for back-compat; no longer used
        train_frac: float = None,
        val_frac: float = None,
    ):
        all_paths = _list_pngs(Path(data_dir) / "original")
        self.paths = _filter_paths_by_split(all_paths, split)
        # deterministic shuffle within the split (so DataLoader sees varied batches even
        # without shuffle=True, though shuffle=True is used in train_loader anyway)
        rng = random.Random(seed)
        rng.shuffle(self.paths)
        self.transform = transform or _default_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class FFppEvalDataset(Dataset):
    """Real test split + fakes from one manipulation. Returns (i_low, i_gray, label).

    Both real and fake frames are filtered by source video ID into the same
    split. A fake video like Deepfakes/035_036 is included only if source 035
    is in this split, matching the official FF++ protocol.
    """

    def __init__(
        self,
        data_dir: str,
        manipulation: str,
        subset: str = "test",          # "thresh" | "test"
        eval_split: str = "test",      # which FF++ split to evaluate on: "val" or "test"
        fake_thresh_frac: float = 0.20,
        seed: int = 42,
        transform: SVDTransform = None,
        # train_frac / val_frac kept for back-compat
        train_frac: float = None,
        val_frac: float = None,
    ):
        data_root = Path(data_dir)

        # Real frames in the requested FF++ split (e.g. sources 860-999 for test)
        all_real = _list_pngs(data_root / "original")
        real_in_split = _filter_paths_by_split(all_real, eval_split)
        rng = random.Random(seed)
        rng.shuffle(real_in_split)

        # Fake frames whose SOURCE video is in the same split (no leakage)
        all_fake = _list_pngs(data_root / manipulation)
        fake_in_split = _filter_paths_by_split(all_fake, eval_split)
        rng2 = random.Random(seed + 1)
        rng2.shuffle(fake_in_split)

        # Balance class sizes
        n_eval = min(len(fake_in_split), len(real_in_split))
        real_eval = real_in_split[:n_eval]
        fake_eval = fake_in_split[:n_eval]

        # Within the eval set, peel off `fake_thresh_frac` for Youden's-Index thresholding
        n_thresh = int(n_eval * fake_thresh_frac)
        if subset == "thresh":
            real_split_paths = real_eval[:n_thresh]
            fake_split_paths = fake_eval[:n_thresh]
        else:
            real_split_paths = real_eval[n_thresh:]
            fake_split_paths = fake_eval[n_thresh:]

        self.samples = (
            [(p, 0) for p in real_split_paths] + [(p, 1) for p in fake_split_paths]
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
