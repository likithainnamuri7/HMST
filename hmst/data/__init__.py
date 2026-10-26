"""
data/__init__.py

Exports dataset classes and split helpers for each supported dataset.

UID2021 (internal name: uid)
  960 images, MOS scores in mos_UID.xlsx, scene-level 80/20 split.
"""
from .dataset import (
    UWIQADataset, SAUDDataset, UIDDataset,
    make_uwiqa_splits, make_saud_splits, make_uid_splits,
)
from .transforms import get_train_transforms, get_val_transforms

__all__ = [
    "UWIQADataset",
    "SAUDDataset",
    "UIDDataset",
    "make_uwiqa_splits",
    "make_saud_splits",
    "make_uid_splits",
    "get_train_transforms",
    "get_val_transforms",
]
