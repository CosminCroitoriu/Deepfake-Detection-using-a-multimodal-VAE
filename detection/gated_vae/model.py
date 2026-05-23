"""
Gated Multimodal U-Net VAE (Step 4).

Three independent U-Net encoders (SVD, DCT, RGB) each produce a per-modality
(mu, logvar) at the same 8×8×latent_dim bottleneck. A small MLP gate takes
the global-average-pooled mus of all three streams and outputs a 3-dim softmax
(alpha_svd, alpha_dct, alpha_rgb) — per-sample weights for the three modalities.

The shared latent is the alpha-weighted combination of the three mus and logvars.
Three independent decoders take z (plus their own encoder's skip connections)
and reconstruct their respective targets:
  Decoder_SVD  : z + skips_svd → I_gray (1 ch)        — Sarkar's uncompress task
  Decoder_DCT  : z + skips_dct → DCT(I) (1 ch)        — identity recon
  Decoder_RGB  : z + skips_rgb → RGB(I) (3 ch)        — identity recon

The same alpha that fuses the latent also weights the per-modality MSE at
inference time:
  anomaly_score = α_svd · MSE_svd + α_dct · MSE_dct + α_rgb · MSE_rgb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Building blocks (same as SVD U-Net VAE)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Per-modality encoder and decoder (one each per modality)
# ----------------------------------------------------------------------------
class ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int = 256):
        super().__init__()
        enc_chs = [32, 64, 128, 256, 512, 512]
        self.enc = nn.ModuleList()
        ch_in = in_channels
        for ch_out in enc_chs:
            self.enc.append(ConvBlock(ch_in, ch_out, stride=2))
            ch_in = ch_out
        self.mu_conv = nn.Conv2d(512, latent_dim, 1)
        self.logvar_conv = nn.Conv2d(512, latent_dim, 1)

    def forward(self, x):
        skips = []
        h = x
        for block in self.enc:
            h = block(h)
            skips.append(h)
        mu = self.mu_conv(h)
        logvar = self.logvar_conv(h)
        return mu, logvar, skips


class ModalityDecoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int = 256):
        super().__init__()
        self.decode_proj = nn.ConvTranspose2d(latent_dim, 512, 1)
        skip_chs = list(reversed([32, 64, 128, 256, 512]))
        dec_in_chs = [512, 512, 256, 128, 64, 32]
        dec_out_chs = [512, 256, 128, 64, 32, 32]
        self.dec = nn.ModuleList()
        for i, (in_c, out_c) in enumerate(zip(dec_in_chs, dec_out_chs)):
            skip_c = skip_chs[i] if i < len(skip_chs) else 0
            self.dec.append(UpBlock(in_c, skip_c, out_c))
        self.head = nn.Conv2d(32, out_channels, 1)

    def forward(self, z, skips):
        h = self.decode_proj(z)
        skip_list = list(reversed(skips[:-1]))
        for i, block in enumerate(self.dec):
            skip = skip_list[i] if i < len(skip_list) else None
            h = block(h, skip)
        return torch.tanh(self.head(h))


# ----------------------------------------------------------------------------
# Gate MLP — produces 3-dim softmax weights from per-modality mu features
# ----------------------------------------------------------------------------
class GateMLP(nn.Module):
    """
    Takes globally-average-pooled mus of three modalities (each B×latent_dim),
    concatenates them, and outputs 3 softmax weights (alpha_svd, alpha_dct, alpha_rgb).
    """
    def __init__(self, latent_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * latent_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, mu_svd, mu_dct, mu_rgb):
        # Global average pool over spatial dimensions: B×C×H×W → B×C
        g_svd = mu_svd.mean(dim=[2, 3])
        g_dct = mu_dct.mean(dim=[2, 3])
        g_rgb = mu_rgb.mean(dim=[2, 3])
        h = torch.cat([g_svd, g_dct, g_rgb], dim=1)
        logits = self.mlp(h)
        alpha = torch.softmax(logits, dim=1)  # B × 3
        return alpha


# ----------------------------------------------------------------------------
# Full Gated Multimodal VAE
# ----------------------------------------------------------------------------
class GatedMultimodalVAE(nn.Module):
    def __init__(self, latent_dim: int = 256, gate_hidden: int = 128):
        super().__init__()
        # Three encoders
        self.enc_svd = ModalityEncoder(in_channels=1, latent_dim=latent_dim)
        self.enc_dct = ModalityEncoder(in_channels=1, latent_dim=latent_dim)
        self.enc_rgb = ModalityEncoder(in_channels=3, latent_dim=latent_dim)
        # Gate
        self.gate = GateMLP(latent_dim=latent_dim, hidden=gate_hidden)
        # Three decoders
        self.dec_svd = ModalityDecoder(out_channels=1, latent_dim=latent_dim)
        self.dec_dct = ModalityDecoder(out_channels=1, latent_dim=latent_dim)
        self.dec_rgb = ModalityDecoder(out_channels=3, latent_dim=latent_dim)

    # ----- internal helpers ---------------------------------------------------
    @staticmethod
    def _split(x_5ch):
        """Slice the 5-channel concat input into per-modality tensors."""
        return x_5ch[:, 0:1], x_5ch[:, 1:2], x_5ch[:, 2:5]

    @staticmethod
    def _reparameterise(mu, logvar, training):
        if training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    @staticmethod
    def _fuse(alpha, mu_svd, mu_dct, mu_rgb, logvar_svd, logvar_dct, logvar_rgb):
        # alpha is B×3 ; broadcast to B×1×1×1 for each component
        a_svd = alpha[:, 0].view(-1, 1, 1, 1)
        a_dct = alpha[:, 1].view(-1, 1, 1, 1)
        a_rgb = alpha[:, 2].view(-1, 1, 1, 1)
        mu = a_svd * mu_svd + a_dct * mu_dct + a_rgb * mu_rgb
        logvar = a_svd * logvar_svd + a_dct * logvar_dct + a_rgb * logvar_rgb
        return mu, logvar

    # ----- forward ------------------------------------------------------------
    def forward(self, x_5ch):
        x_svd, x_dct, x_rgb = self._split(x_5ch)

        mu_svd, lv_svd, skips_svd = self.enc_svd(x_svd)
        mu_dct, lv_dct, skips_dct = self.enc_dct(x_dct)
        mu_rgb, lv_rgb, skips_rgb = self.enc_rgb(x_rgb)

        alpha = self.gate(mu_svd, mu_dct, mu_rgb)               # B × 3
        mu, logvar = self._fuse(alpha, mu_svd, mu_dct, mu_rgb,
                                lv_svd, lv_dct, lv_rgb)
        z = self._reparameterise(mu, logvar, self.training)

        recon_svd = self.dec_svd(z, skips_svd)                  # B × 1 × H × W
        recon_dct = self.dec_dct(z, skips_dct)                  # B × 1 × H × W
        recon_rgb = self.dec_rgb(z, skips_rgb)                  # B × 3 × H × W
        return recon_svd, recon_dct, recon_rgb, mu, logvar, alpha

    # ----- anomaly score ------------------------------------------------------
    def reconstruction_error(self, x_5ch, target_5ch, return_alpha: bool = False):
        """
        Per-sample anomaly score: alpha-weighted sum of per-modality MSEs.

        Returns:
            score    : B  (total anomaly score per sample)
            per_mod  : B × 3  (raw MSE per modality, no alpha weighting; for diagnostics)
            alpha    : B × 3  (gate weights; only if return_alpha=True)
        """
        with torch.inference_mode():
            recon_svd, recon_dct, recon_rgb, _, _, alpha = self.forward(x_5ch)

        tgt_svd, tgt_dct, tgt_rgb = self._split(target_5ch)
        mse_svd = F.mse_loss(recon_svd, tgt_svd, reduction="none").mean(dim=[1, 2, 3])  # B
        mse_dct = F.mse_loss(recon_dct, tgt_dct, reduction="none").mean(dim=[1, 2, 3])  # B
        mse_rgb = F.mse_loss(recon_rgb, tgt_rgb, reduction="none").mean(dim=[1, 2, 3])  # B
        per_mod = torch.stack([mse_svd, mse_dct, mse_rgb], dim=1)                       # B × 3

        score = (alpha * per_mod).sum(dim=1)                                            # B
        if return_alpha:
            return score, per_mod, alpha
        return score, per_mod


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def gated_vae_loss(
    recon_svd, recon_dct, recon_rgb,
    target_5ch,
    mu, logvar,
    beta_kl: float = 0.6,
    beta_l1: float = 0.6,
):
    """
    L = L_recon_svd + L_recon_dct + L_recon_rgb + beta_kl·KL + beta_l1·L1_total

    Note: per the implementation plan, all three modalities contribute equally
    to the reconstruction loss. The gate's job is to weight them at the latent
    fusion (and at inference scoring), not in the loss.
    """
    tgt_svd, tgt_dct, tgt_rgb = (
        target_5ch[:, 0:1],
        target_5ch[:, 1:2],
        target_5ch[:, 2:5],
    )

    mse_svd = F.mse_loss(recon_svd, tgt_svd)
    mse_dct = F.mse_loss(recon_dct, tgt_dct)
    mse_rgb = F.mse_loss(recon_rgb, tgt_rgb)
    mse_total = mse_svd + mse_dct + mse_rgb

    l1_svd = F.l1_loss(recon_svd, tgt_svd)
    l1_dct = F.l1_loss(recon_dct, tgt_dct)
    l1_rgb = F.l1_loss(recon_rgb, tgt_rgb)
    l1_total = l1_svd + l1_dct + l1_rgb

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return mse_total + beta_kl * kl + beta_l1 * l1_total
