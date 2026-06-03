"""
Simple Vision Transformer binary classifier (real=0, fake=1).

Architecture: ViT-S-style — patch tokenisation → CLS token → transformer
encoder → classification head. Trained end-to-end with cross-entropy on
labelled real/fake pairs.

This is a supervised comparison baseline for the unsupervised VAE models.
The key difference: the VAE models never see fake images during training;
this model sees labelled fake images from all four generators.

Default config (embed_dim=256, depth=6, heads=8, patch=16, img=224):
  ~5.3 M parameters, which is deliberately lighter than the VAE models
  (~15-30 M) to keep training fast.
"""
import torch
import torch.nn as nn


class _TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTClassifier(nn.Module):
    """
    Vision Transformer binary classifier (real vs. fake natural-hazard images).

    Parameters
    ----------
    img_size    : input spatial resolution after resize (square)
    patch_size  : patch side length (must divide img_size evenly)
    in_channels : RGB = 3
    embed_dim   : token dimensionality
    depth       : number of transformer blocks
    num_heads   : attention heads (embed_dim must be divisible by num_heads)
    mlp_ratio   : MLP hidden-dim = embed_dim × mlp_ratio
    dropout     : attention + MLP dropout during training
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        num_patches = (img_size // patch_size) ** 2

        # Patchify + project to embed_dim in one Conv2d
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Binary head: logit for class 1 (fake). Use BCEWithLogitsLoss in trainer.
        self.head = nn.Linear(embed_dim, 1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit for class 1 (fake), shape [B]."""
        B = x.size(0)
        x = self.patch_embed(x)                      # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)             # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)               # [B, N+1, D]
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0]).squeeze(1)          # [B]

    def fake_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid of the logit — used as anomaly score during evaluation."""
        with torch.inference_mode():
            return torch.sigmoid(self.forward(x))
