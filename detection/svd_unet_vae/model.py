"""
SVD U-Net VAE for deepfake detection.

Architecture (Sarkar et al., 2025 adaptation for 256×256 greyscale):
  Encoder: 6 convolutional blocks (stride-2), channels [32,64,128,256,512,512]
  Bottleneck: convolutional VAE (mu + logvar via 1×1 Conv2d, 4×4 spatial)
  Decoder: 6 transposed-conv blocks with skip connections (concat)
  Input:  1×512×512   (SVD low-rank approximation, normalised to [-1,1])
  Output: 1×512×512   (reconstructed greyscale, normalised to [-1,1])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SVDUNetVAE(nn.Module):
    def __init__(self, latent_dim: int = 256, in_channels: int = 1):
        super().__init__()
        enc_chs = [32, 64, 128, 256, 512, 512]

        # Encoder (each block halves spatial dims via stride-2 first conv)
        self.enc = nn.ModuleList()
        ch_in = in_channels
        for ch_out in enc_chs:
            self.enc.append(ConvBlock(ch_in, ch_out, stride=2))
            ch_in = ch_out

        # VAE bottleneck: 512 ch at 4×4
        self.mu_conv = nn.Conv2d(512, latent_dim, 1)
        self.logvar_conv = nn.Conv2d(512, latent_dim, 1)
        self.decode_proj = nn.ConvTranspose2d(latent_dim, 512, 1)

        # Decoder
        # skip channels come from encoder outputs (reversed, skip the last)
        skip_chs = list(reversed(enc_chs[:-1]))   # [512, 256, 128, 64, 32]
        dec_in_chs = [512, 512, 256, 128, 64, 32]
        dec_out_chs = [512, 256, 128, 64, 32, 32]

        self.dec = nn.ModuleList()
        for i, (in_c, out_c) in enumerate(zip(dec_in_chs, dec_out_chs)):
            skip_c = skip_chs[i] if i < len(skip_chs) else 0
            self.dec.append(UpBlock(in_c, skip_c, out_c))

        self.head = nn.Conv2d(32, 1, 1)

    # ------------------------------------------------------------------
    def encode(self, x):
        skips = []
        h = x
        for block in self.enc:
            h = block(h)
            skips.append(h)
        mu = self.mu_conv(h)
        logvar = self.logvar_conv(h)
        return mu, logvar, skips

    def reparameterise(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z, skips):
        h = self.decode_proj(z)
        skip_list = list(reversed(skips[:-1]))  # skip the innermost (same as z)
        for i, block in enumerate(self.dec):
            skip = skip_list[i] if i < len(skip_list) else None
            h = block(h, skip)
        return torch.tanh(self.head(h))

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z, skips)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction (no gradient)."""
        with torch.inference_mode():
            recon, _, _ = self.forward(x)
        err = F.mse_loss(recon, x, reduction="none")
        return err.mean(dim=[1, 2, 3])


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 0.6,
    beta_l1: float = 0.6,
) -> torch.Tensor:
    mse = F.mse_loss(recon, target)
    l1 = F.l1_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta_kl * kl + beta_l1 * l1
