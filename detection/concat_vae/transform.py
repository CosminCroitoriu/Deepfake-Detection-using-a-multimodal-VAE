"""
Multimodal transform: returns (input_5ch, target_5ch) per image.

Modalities:
  ch 0 — SVD low-rank greyscale  (90% energy threshold)
  ch 1 — block DCT of greyscale  (16×16 blocks, ortho-normalised)
  ch 2 — R channel
  ch 3 — G channel
  ch 4 — B channel

input_5ch  : [I_low,    DCT(I), R, G, B]   (model sees this)
target_5ch : [I_gray,   DCT(I), R, G, B]   (model reconstructs this)

Note the asymmetry on channel 0: input is the SVD-compressed image, target is the
full greyscale. This preserves Sarkar's "uncompress" task for the SVD stream.
Channels 1–4 are identity (input == target), giving each modality its own
anomaly-detection objective via the VAE bottleneck.

All channels are normalised to [-1, 1].
"""
import numpy as np
import scipy.fft
import torch
import torchvision.transforms.functional as TF
from PIL import Image

TARGET_SIZE = 512
ENERGY_THRESHOLD = 0.90
DCT_BLOCK_SIZE = 16          # 512 / 16 = 32 blocks per side, divides cleanly
# With ortho-normalised DCT on [0,1] greyscale, the DC coefficient of a
# 16×16 block scales as ~mean(block) * block_size = up to 16. AC components
# are typically much smaller (|AC| < 4 for natural images). Scale by 16 so DC
# lands in [0, 1] without clipping and AC in roughly [-0.25, 0.25].
DCT_SCALE = 16.0


def svd_lowrank(gray_tensor: torch.Tensor, energy_thresh: float = ENERGY_THRESHOLD) -> torch.Tensor:
    """Low-rank approximation of a single-channel CHW tensor (C=1)."""
    mat = gray_tensor.squeeze(0)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    total_energy = (S ** 2).sum()
    cumulative = torch.cumsum(S ** 2, dim=0)
    k = int((cumulative / total_energy >= energy_thresh).nonzero(as_tuple=False)[0].item()) + 1
    low = (U[:, :k] * S[:k]) @ Vh[:k, :]
    return low.clamp(0.0, 1.0).unsqueeze(0)


def block_dct(gray_np: np.ndarray, block_size: int = DCT_BLOCK_SIZE) -> np.ndarray:
    """
    Vectorised block DCT-II with ortho normalisation.
    Input  : H×W float32 in [0, 1]
    Output : H×W float32, raw DCT coefficients per block.
    """
    H, W = gray_np.shape
    n_h, n_w = H // block_size, W // block_size
    # split into blocks: (n_h, block, n_w, block) -> (n_h*n_w, block, block)
    blocks = (
        gray_np.reshape(n_h, block_size, n_w, block_size)
               .transpose(0, 2, 1, 3)
               .reshape(-1, block_size, block_size)
    )
    dct_blocks = scipy.fft.dctn(blocks, axes=(-2, -1), norm="ortho")
    # reassemble
    return (
        dct_blocks.reshape(n_h, n_w, block_size, block_size)
                  .transpose(0, 2, 1, 3)
                  .reshape(H, W)
                  .astype(np.float32)
    )


class ConcatTransform:
    """PIL → (input_5ch, target_5ch) float32 tensors, both 5×512×512 in [-1, 1]."""

    def __init__(
        self,
        target_size: int = TARGET_SIZE,
        energy_thresh: float = ENERGY_THRESHOLD,
        block_size: int = DCT_BLOCK_SIZE,
        dct_scale: float = DCT_SCALE,
    ):
        self.target_size = target_size
        self.energy_thresh = energy_thresh
        self.block_size = block_size
        self.dct_scale = dct_scale

    def __call__(self, img: Image.Image):
        img = img.convert("RGB").resize(
            (self.target_size, self.target_size), Image.LANCZOS
        )
        rgb_t = TF.to_tensor(img)                              # 3×H×W in [0, 1]
        gray_t = TF.rgb_to_grayscale(rgb_t, num_output_channels=1)  # 1×H×W
        i_low = svd_lowrank(gray_t, self.energy_thresh)        # 1×H×W

        # DCT in numpy land then back to tensor
        gray_np = gray_t.squeeze(0).numpy()
        dct_np = block_dct(gray_np, self.block_size)
        dct_t = torch.from_numpy(dct_np).unsqueeze(0)          # 1×H×W
        # Normalise DCT roughly into [-1, 1]
        dct_t = (dct_t / self.dct_scale).clamp(-1.0, 1.0)

        # Bring [0, 1] channels into [-1, 1]
        i_low = i_low * 2.0 - 1.0
        i_gray = gray_t * 2.0 - 1.0
        rgb_norm = rgb_t * 2.0 - 1.0

        input_5ch = torch.cat([i_low, dct_t, rgb_norm], dim=0)      # 5×H×W
        target_5ch = torch.cat([i_gray, dct_t, rgb_norm], dim=0)    # 5×H×W
        return input_5ch, target_5ch
