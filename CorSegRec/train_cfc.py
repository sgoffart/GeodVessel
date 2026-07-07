"""
Train CFC for centerline classification.

3 classes of training points (paper Section 4.2):
  - Positive (1): centerline voxels
  - Negative (0): non-centerline lumen voxels  (N_pos samples)
  - Negative (0): near-wall outside voxels     (N_pos samples)
  → ratio ~1:4 positive:negative total

Feature: multiscale patch (size 15 + size 7), normalized, concatenated.
"""

import os
import nrrd
import numpy as np
from scipy.ndimage import zoom, binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize
from models.cfc_model import CFC

# ---- CONFIG ----
DATA_DIR = (
    "./datasets/ASOCA/Dataset001_ASOCA"
)
SAVE_PATH = (
    "./"
    "./ASOCA/cfc_model.pkl"
)

N_POS_PER_CASE = 300      # centerline points per case
MAX_TOTAL = 20000         # global cap
NEAR_WALL_DIST = 7        # max distance outside lumen for hard negatives
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ---- FEATURE EXTRACTION ----
def extract_patch(vol, center, size):
    """Extract cubic patch, zero-pad at boundaries."""
    x, y, z = center
    r = size // 2

    patch = vol[
        max(0, x - r): x + r + 1,
        max(0, y - r): y + r + 1,
        max(0, z - r): z + r + 1,
    ]

    # zero-pad to exact size
    pad_width = [(0, size - s) for s in patch.shape]
    patch = np.pad(patch, pad_width, mode='constant')
    return patch.astype(np.float32)


def normalize_patch(patch):
    """Z-score normalize a single patch."""
    mu = patch.mean()
    sigma = patch.std()
    return (patch - mu) / (sigma + 1e-6)


def extract_features(vol, coord):
    """
    Multiscale feature vector used in both training and inference.
    Steps:
      1. Extract 7^3 and 15^3 patches
      2. Z-score normalize each independently
      3. Downsample 15^3 → 7^3 via zoom
      4. Flatten and concatenate → shape (2 * 7^3,) = (686,)
    """
    p_small = extract_patch(vol, coord, 7)
    p_large = extract_patch(vol, coord, 15)

    p_small = normalize_patch(p_small)
    p_large = normalize_patch(p_large)

    # downsample large to match small
    scale = 7.0 / 15.0
    p_large_ds = zoom(p_large, (scale, scale, scale), order=1)

    # ensure exact shape after zoom
    p_large_ds = p_large_ds[:7, :7, :7]

    feat = np.concatenate([
        p_small.flatten(),
        p_large_ds.flatten()
    ]).astype(np.float32)

    return feat  # shape (686,)


# ---- SKELETON EXTRACTION ----
def get_skeleton(label_mask):
    """
    Binary skeleton via skimage skeletonize_3d.
    Returns boolean array same shape as label_mask.
    """
    binary = (label_mask > 0).astype(np.uint8)
    skel = skeletonize(binary)
    return skel.astype(bool)


# ---- NEAR-WALL NEGATIVES ----
def get_near_wall_outside(label_mask, max_dist=7):
    """
    Points outside the lumen but within max_dist voxels of the wall.
    Used as hard negatives (paper Section 4.2).
    """
    outside = (label_mask == 0)
    dist = distance_transform_edt(outside)   # distance from each outside voxel to nearest lumen voxel
    near_wall = outside & (dist <= max_dist)
    return np.argwhere(near_wall)


# ---- MAIN ----
X_all, y_all = [], []

print("Scanning training cases...")
img_dir = os.path.join(DATA_DIR, "imagesTr")
lbl_dir = os.path.join(DATA_DIR, "labelsTr")

cases = sorted([
    f for f in os.listdir(img_dir)
    if f.endswith("_0000.nrrd")
])

print(f"Found {len(cases)} cases.\n")

for case_file in cases:
    case_id = case_file.replace("_0000.nrrd", "")

    vol_path = os.path.join(img_dir, case_file)
    lbl_path = os.path.join(lbl_dir, f"{case_id}.nrrd")

    if not os.path.exists(lbl_path):
        print(f"  ⚠️  Label missing for {case_id}, skipping.")
        continue

    vol, _ = nrrd.read(vol_path)
    lbl, _ = nrrd.read(lbl_path)

    vol = vol.astype(np.float32)

    # ---- get point sets ----
    print(f"  {case_id}: computing skeleton...", end=" ", flush=True)
    skel = get_skeleton(lbl)

    centerline_pts  = np.argwhere(skel)                          # positive
    lumen_pts       = np.argwhere((lbl > 0) & (~skel))          # inside lumen, not skeleton
    near_wall_pts   = get_near_wall_outside(lbl, NEAR_WALL_DIST) # outside, near wall

    n_pos = min(N_POS_PER_CASE, len(centerline_pts))
    # paper ratio: 1 positive : 2 lumen-neg : 2 near-wall-neg  → 1:4 total
    n_neg_each = min(2 * n_pos, len(lumen_pts), len(near_wall_pts))

    print(
        f"skel={len(centerline_pts)}, lumen={len(lumen_pts)}, "
        f"near_wall={len(near_wall_pts)} → "
        f"sampling pos={n_pos}, neg={n_neg_each}×2"
    )

    if n_pos == 0:
        print("  ⚠️  Empty skeleton, skipping.")
        continue

    # ---- sample ----
    pos_idx = np.random.choice(len(centerline_pts),   n_pos,      replace=False)
    neg_l   = np.random.choice(len(lumen_pts),        n_neg_each, replace=False)
    neg_w   = np.random.choice(len(near_wall_pts),    n_neg_each, replace=False)

    sample_groups = [
        (centerline_pts[pos_idx],  1),
        (lumen_pts[neg_l],         0),
        (near_wall_pts[neg_w],     0),
    ]

    for pts, label in sample_groups:
        for pt in pts:
            coord = tuple(pt)
            feat  = extract_features(vol, coord)
            X_all.append(feat)
            y_all.append(label)

print(f"\nRaw dataset size: {len(X_all)}")

# ---- convert ----
X = np.array(X_all, dtype=np.float32)
y = np.array(y_all, dtype=np.int32)

# ---- global cap ----
if len(X) > MAX_TOTAL:
    print(f"Subsampling to {MAX_TOTAL}...")
    idx = np.random.choice(len(X), MAX_TOTAL, replace=False)
    X = X[idx]
    y = y[idx]

pos_count = int((y == 1).sum())
neg_count = int((y == 0).sum())
print(f"Final: {len(X)} samples  (pos={pos_count}, neg={neg_count}, ratio=1:{neg_count/max(pos_count,1):.1f})\n")

# ---- train ----
print("Training CFC (this may take several minutes)...")
model = CFC()
model.fit(X, y)

model.save(SAVE_PATH)
print("✅ Done.")
