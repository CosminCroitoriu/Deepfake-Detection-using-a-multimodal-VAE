"""
Pretrained CNN binary classifiers (real=0, fake=1) via timm.

Supported architectures:
  resnet50        — 23.5 M params, deep residual network
  efficientnet_b0 —  4.0 M params, compound-scaled mobile network

Both are loaded with ImageNet pretrained weights. The timm classification
head (originally 1000-class softmax) is replaced with a single linear unit
for binary BCEWithLogitsLoss. All layers including the backbone are trained
end-to-end — no layer freezing.

These are supervised comparison baselines for the unsupervised VAE models.
The key difference from the from-scratch ViT: pretrained weights give the
backbone a rich low-level feature vocabulary before any fake-image signal.
"""
import timm
import torch.nn as nn

SUPPORTED = ["resnet50", "efficientnet_b0"]


class PretrainedCNNClassifier(nn.Module):
    """
    Thin wrapper around a timm pretrained backbone with a binary head.

    timm's num_classes=1 replaces the original classifier with Linear(features, 1).
    The forward pass returns a raw logit of shape [B]; use BCEWithLogitsLoss
    at training and sigmoid() as the anomaly score at evaluation.
    """

    def __init__(self, arch: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if arch not in SUPPORTED:
            raise ValueError(f"arch must be one of {SUPPORTED}, got '{arch}'")
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=1)
        self.arch = arch

    def forward(self, x):
        return self.backbone(x).squeeze(1)   # [B, 1] → [B]
