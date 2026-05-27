"""
Dataset classes for the Masked Concatenation VAE.

Identical to detection.concat_vae.dataset — re-exported here so the trainer
and evaluator have a self-contained module to import from. The data is
the same; only the model is different.
"""
from ..concat_vae.dataset import (
    DISASTER_CLASSES,
    GENERATOR_DIRS,
    RealTrainDataset,
    EvalDataset,
)

__all__ = ["DISASTER_CLASSES", "GENERATOR_DIRS", "RealTrainDataset", "EvalDataset"]
