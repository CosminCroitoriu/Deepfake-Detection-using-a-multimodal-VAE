"""
Dataset re-export for the pretrained CNN classifiers.

Data pipeline is identical to the ViT classifier — same splits, same
ImageNet normalization, same 224×224 resize. Importing directly avoids
duplicating the split logic.
"""
from ..vit_classifier.dataset import (
    DISASTER_CLASSES,
    GENERATOR_DIRS,
    ClassifierTrainDataset,
    ClassifierEvalDataset,
    train_transform,
    eval_transform,
)

__all__ = [
    "DISASTER_CLASSES",
    "GENERATOR_DIRS",
    "ClassifierTrainDataset",
    "ClassifierEvalDataset",
    "train_transform",
    "eval_transform",
]
