"""
Masked Concatenation Multimodal VAE (Step 5).

Architectural base: Step 3's ConcatUNetVAE (5-channel input/output, single
shared U-Net encoder + decoder). This is our strongest detector — we build
masking on top of it rather than on Step 4's gated architecture because
Step 4's identified failure modes (gate suppresses signal at inference,
separate encoders learn weaker features) would confound the masking signal.

Masking mechanic: during training, the 5-channel input is partitioned into
three modality groups — SVD (channel 0), DCT (channel 1), RGB (channels 2-4).
Each group is independently zeroed with probability `mask_prob` per sample.
The model has to reconstruct the full unmasked target from whatever
modalities remain.

At inference, no masking is applied — the model sees the full input but has
learned (during training) to handle modality dropout. The Step 8 robustness
evaluation will test whether this generalises to JPEG/blur degradation.
"""
import torch

from ..concat_vae.model import ConcatUNetVAE, vae_loss  # re-use the loss

# Channel groupings for the 5-channel [I_low, DCT, R, G, B] input
MODALITY_CHANNEL_GROUPS = [
    [0],          # SVD low-rank
    [1],          # DCT block coefficients
    [2, 3, 4],    # RGB
]


class MaskedConcatVAE(ConcatUNetVAE):
    """ConcatUNetVAE + training-time modality channel dropout."""

    def __init__(
        self,
        latent_dim: int = 256,
        in_channels: int = 5,
        out_channels: int = 5,
        mask_prob: float = 0.3,
    ):
        super().__init__(latent_dim=latent_dim, in_channels=in_channels, out_channels=out_channels)
        self.mask_prob = mask_prob

    def forward(self, x_5ch):
        # Only mask during training; eval gets the full input
        if self.training and self.mask_prob > 0:
            x_5ch = self._apply_modality_mask(x_5ch)
        return super().forward(x_5ch)

    def _apply_modality_mask(self, x_5ch):
        """
        Per-sample, per-modality-group Bernoulli masking. A masked group's
        channels are zeroed out; this is `0` in [-1, 1] normalised space, which
        the model can learn to interpret as "no information from this modality."
        Per-sample variation ensures every modality is seen plain in some
        batches and missing in others.
        """
        B = x_5ch.size(0)
        x_masked = x_5ch.clone()
        for channels in MODALITY_CHANNEL_GROUPS:
            # Sample which samples to mask for THIS modality group
            mask = (torch.rand(B, device=x_5ch.device) < self.mask_prob)   # B
            # Convert to keep multiplier: 0 if masked, 1 if kept
            keep = (~mask).float().view(B, 1, 1, 1)                        # B×1×1×1
            for ch in channels:
                x_masked[:, ch:ch + 1] = x_masked[:, ch:ch + 1] * keep
        return x_masked


__all__ = ["MaskedConcatVAE", "vae_loss"]
