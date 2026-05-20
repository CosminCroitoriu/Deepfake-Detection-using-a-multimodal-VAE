"""
SVD-based image transform for deepfake detection preprocessing.

Converts a colour image to a low-rank greyscale approximation by keeping
enough singular values to capture ≥ ENERGY_THRESHOLD of the total spectral
energy.  Both the low-rank approximation (model input) and the original
greyscale image (reconstruction target) are returned at TARGET_SIZE×TARGET_SIZE.
"""
import torch
import torchvision.transforms.functional as TF
from PIL import Image

TARGET_SIZE = 512
ENERGY_THRESHOLD = 0.90


def svd_lowrank(gray_tensor: torch.Tensor, energy_thresh: float = ENERGY_THRESHOLD) -> torch.Tensor:
    """Return low-rank approximation of a single-channel CHW tensor (C=1)."""
    mat = gray_tensor.squeeze(0)  # H×W
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    total_energy = (S ** 2).sum()
    cumulative = torch.cumsum(S ** 2, dim=0)
    k = int((cumulative / total_energy >= energy_thresh).nonzero(as_tuple=False)[0].item()) + 1
    low = (U[:, :k] * S[:k]) @ Vh[:k, :]
    low = low.clamp(0.0, 1.0).unsqueeze(0)
    return low


class SVDTransform:
    """
    Callable that processes a PIL Image into (I_low, I_gray).

    I_low  — SVD low-rank approximation, normalised to [-1, 1]  (model input)
    I_gray — original greyscale image,   normalised to [-1, 1]  (reconstruction target)
    Both are 1×TARGET_SIZE×TARGET_SIZE float32 tensors.
    """

    def __init__(
        self,
        target_size: int = TARGET_SIZE,
        energy_thresh: float = ENERGY_THRESHOLD,
    ):
        self.target_size = target_size
        self.energy_thresh = energy_thresh

    def __call__(self, img: Image.Image):
        img = img.convert("RGB").resize(
            (self.target_size, self.target_size), Image.LANCZOS
        )
        gray = TF.to_grayscale(img, num_output_channels=1)
        gray_t = TF.to_tensor(gray)  # [0, 1]
        low_t = svd_lowrank(gray_t, self.energy_thresh)
        # normalise both to [-1, 1]
        i_low = low_t * 2.0 - 1.0
        i_gray = gray_t * 2.0 - 1.0
        return i_low, i_gray
