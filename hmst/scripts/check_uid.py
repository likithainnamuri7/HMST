"""
scripts/check_uid.py
--------------------
Phase-4 sanity checker for the UID2021 dataset integration.

Checks:
  1. Split sizes: 39 train-scenes / 9 val-scenes / 12 test-scenes -> 624 / 144 / 192 images.
  2. Pairwise scene isolation between train, val, and test.
  3. Three sample items: path, score in [0,1], tensor shape (3,224,224), no NaN.
  4. Score min/max after normalisation (expect ~0.0 and ~1.0).
  5. UWIQA split sizes (712/178) and SAUD split sizes (800/200), skipped if
     data is not present on this machine.
"""

import sys
import os
import math

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from data import (
    UIDDataset,
    make_uid_splits,
    get_val_transforms,
)
from scripts.train import make_absolute

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# ── Paths ──────────────────────────────────────────────────────────────────
IMAGE_DIR  = make_absolute("../UID2021/UID2021", PROJECT_ROOT)
SCORE_FILE = make_absolute("../mos_UID.xlsx",    PROJECT_ROOT)

print("=" * 60)
print("CHECK_UID — Phase 4 sanity checker")
print("=" * 60)
print(f"image_dir  : {IMAGE_DIR}")
print(f"score_file : {SCORE_FILE}")
print()

# ── 1. Split sizes & scene isolation ───────────────────────────────────────
print("[1] Building train/val/test splits ...")
train_idx, test_idx = make_uid_splits(IMAGE_DIR, SCORE_FILE, seed=42)

val_size  = max(16, (len(train_idx) // 5) // 16 * 16)   # whole scenes of 16 images
val_idx   = train_idx[-val_size:]
train_idx = train_idx[:-val_size]

EXPECTED_TRAIN = 624
EXPECTED_VAL   = 144
EXPECTED_TEST  = 192

print(f"    train indices: {len(train_idx)}  (expect {EXPECTED_TRAIN})")
print(f"    val   indices: {len(val_idx)}    (expect {EXPECTED_VAL})")
print(f"    test  indices: {len(test_idx)}   (expect {EXPECTED_TEST})")

assert len(train_idx) == EXPECTED_TRAIN, (
    f"FAIL: expected {EXPECTED_TRAIN} train images, got {len(train_idx)}"
)
assert len(val_idx) == EXPECTED_VAL, (
    f"FAIL: expected {EXPECTED_VAL} val images, got {len(val_idx)}"
)
assert len(test_idx) == EXPECTED_TEST, (
    f"FAIL: expected {EXPECTED_TEST} test images, got {len(test_idx)}"
)
print("    Split sizes: PASS")

# Pairwise scene isolation between train, val, and test
from data.dataset import _build_uid_samples
all_samples = _build_uid_samples(IMAGE_DIR, SCORE_FILE)

train_scenes = {all_samples[i][2] for i in train_idx}
val_scenes   = {all_samples[i][2] for i in val_idx}
test_scenes  = {all_samples[i][2] for i in test_idx}

overlap_tv = train_scenes & val_scenes
overlap_tt = train_scenes & test_scenes
overlap_vt = val_scenes & test_scenes

print(f"    Train scenes: {len(train_scenes)}  Val scenes: {len(val_scenes)}  Test scenes: {len(test_scenes)}")
assert not overlap_tv, f"FAIL: scene(s) appear in both train and val: {sorted(overlap_tv)}"
assert not overlap_tt, f"FAIL: scene(s) appear in both train and test: {sorted(overlap_tt)}"
assert not overlap_vt, f"FAIL: scene(s) appear in both val and test: {sorted(overlap_vt)}"
print("    Pairwise scene isolation: PASS")
print()

# ── 2. Three sample items ───────────────────────────────────────────────────
print("[2] Loading 3 sample items ...")
tf = get_val_transforms(224)
ds = UIDDataset(IMAGE_DIR, SCORE_FILE, train_idx, transform=tf)

for i in range(min(3, len(ds))):
    item = ds[i]
    path  = item["path"]
    score = item["score"]
    image = item["image"]

    print(f"    Sample {i}:")
    print(f"      path  : {os.path.basename(path)}")
    print(f"      score : {score.item():.6f}")
    print(f"      shape : {list(image.shape)}  (expect [3, 224, 224])")

    assert image.shape == (3, 224, 224), \
        f"FAIL: unexpected shape {image.shape}"
    assert not torch.isnan(image).any(), "FAIL: NaN in image tensor"
    assert not torch.isnan(score).any(), "FAIL: NaN in score tensor"
    assert 0.0 <= score.item() <= 1.0, \
        f"FAIL: score {score.item()} out of [0,1]"

print("    All 3 samples: PASS")
print()

# ── 3. Score min/max after normalisation ───────────────────────────────────
print("[3] Score range after normalisation ...")
all_train_ds = UIDDataset(IMAGE_DIR, SCORE_FILE, train_idx + val_idx + test_idx)
all_scores   = [s[1] for s in all_train_ds.samples]
s_min = min(all_scores)
s_max = max(all_scores)
print(f"    Normalised min: {s_min:.6f}  (expect ~0.0)")
print(f"    Normalised max: {s_max:.6f}  (expect ~1.0)")
assert s_min >= 0.0, f"FAIL: min score {s_min} < 0"
assert s_max <= 1.0 + 1e-5, f"FAIL: max score {s_max} > 1"
print("    Score range: PASS")
print()

# ── 4. UWIQA regression (skipped if data absent) ───────────────────────────
print("[4] UWIQA split size regression ...")
try:
    import yaml, openpyxl
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "full.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    uwiqa_cfg  = cfg["uwiqa"]
    uwiqa_dir  = make_absolute(uwiqa_cfg["image_dir"],  PROJECT_ROOT)
    uwiqa_sf   = make_absolute(uwiqa_cfg["score_file"], PROJECT_ROOT)
    if not os.path.exists(uwiqa_dir) or not os.path.exists(uwiqa_sf):
        raise FileNotFoundError
    from data import make_uwiqa_splits
    u_train, u_test = make_uwiqa_splits(uwiqa_dir, uwiqa_sf, seed=42)
    print(f"    UWIQA train: {len(u_train)}  test: {len(u_test)}")
    assert len(u_train) == 712 and len(u_test) == 178, \
        f"FAIL: UWIQA split changed! got {len(u_train)}/{len(u_test)}"
    print("    UWIQA split sizes: PASS (712/178)")
except FileNotFoundError:
    print("    UWIQA data not present on this machine — SKIPPED")
print()

# ── 5. SAUD regression (skipped if data absent) ────────────────────────────
print("[5] SAUD split size regression ...")
try:
    import yaml, scipy.io
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "full.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    saud_cfg = cfg["saud"]
    saud_mat = make_absolute(saud_cfg["bt_scores_mat"], PROJECT_ROOT)
    if not os.path.exists(saud_mat):
        raise FileNotFoundError
    from data import make_saud_splits
    s_train, s_test = make_saud_splits(saud_mat, seed=42)
    print(f"    SAUD  train: {len(s_train)}  test: {len(s_test)}")
    assert len(s_train) == 800 and len(s_test) == 200, \
        f"FAIL: SAUD split changed! got {len(s_train)}/{len(s_test)}"
    print("    SAUD split sizes: PASS (800/200)")
except FileNotFoundError:
    print("    SAUD data not present on this machine — SKIPPED")
print()

print("=" * 60)
print("ALL CHECK_UID CHECKS PASSED")
print("=" * 60)
