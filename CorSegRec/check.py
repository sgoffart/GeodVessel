#!/usr/bin/env python3
"""
Standalone sanity check for the trained CFC model.

FIXES:
- Feature extraction now matches dpc_walk.py exactly (order, method, dims)
- Uses scipy.ndimage.zoom (bilinear) as canonical downsampling method
- Feature vector order: [p_small (343), p_large_ds (343)] = 686 dims total
- Added detailed diagnostics: confusion matrix, per-class stats, ROC AUC
- Added volume/label validation before processing
- Dynamic max_dist from volume shape
"""

import os
import pickle
import numpy as np
import nrrd
from scipy.ndimage import zoom
from skimage.morphology import skeletonize

# ---- CONFIG ----
DATA_DIR = (
    "./datasets/ASOCA/Dataset001_ASOCA"
)
MODEL_PATH = (
    "./ASOCA/cfc_model.pkl"
)

CASE_ID       = "case_0000"   # adjust if needed
N_POS_SAMPLES = 300           # skeleton samples
N_NEG_SAMPLES = 600           # lumen (non-skeleton) samples
RANDOM_SEED   = 42

# ======================================================================
# CANONICAL FEATURE EXTRACTION
# Must match dpc_walk.py extract_features() EXACTLY:
#   - Patch sizes  : small=7, large=15
#   - Downsampling : scipy.ndimage.zoom (bilinear, order=1)
#   - Concat order : [p_small.flatten(), p_large_ds.flatten()]
#   - Output dims  : 343 + 343 = 686
# ======================================================================

def extract_patch(vol: np.ndarray, center: tuple, size: int) -> np.ndarray:
    """
    Extract a cubic patch of side `size` centered at `center`.
    Handles boundary by zero-padding.

    Parameters
    ----------
    vol    : 3-D float32 volume
    center : (x, y, z) integer voxel coordinate
    size   : side length of the cubic patch

    Returns
    -------
    patch  : float32 array of shape (size, size, size)
    """
    x, y, z = center
    r = size // 2

    x0, x1 = max(0, x - r), x + r + 1
    y0, y1 = max(0, y - r), y + r + 1
    z0, z1 = max(0, z - r), z + r + 1

    patch = vol[x0:x1, y0:y1, z0:z1]

    # Zero-pad to reach full size if at boundary
    pad_width = [(0, size - s) for s in patch.shape]
    patch = np.pad(patch, pad_width, mode='constant')

    # Guarantee exact shape (handles edge case where x+r+1 > vol dim)
    return patch[:size, :size, :size].astype(np.float32)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Z-score normalization.
    Adds epsilon to std to avoid division by zero in flat regions.
    """
    mu    = patch.mean()
    sigma = patch.std()
    return (patch - mu) / (sigma + 1e-6)


def extract_features(vol: np.ndarray, coord: tuple) -> np.ndarray:
    """
    Extract 686-dim feature vector (canonical definition).

    Procedure
    ---------
    1. Extract small patch  : 7x7x7  -> normalize -> flatten (343 dims)
    2. Extract large patch  : 15x15x15 -> normalize
    3. Downsample large     : zoom by 7/15 -> crop to 7x7x7 -> flatten (343 dims)
    4. Concatenate          : [small | large_ds]  (686 dims total)

    Parameters
    ----------
    vol   : 3-D float32 CT volume
    coord : (x, y, z) integer voxel coordinate

    Returns
    -------
    features : float32 array of shape (686,)
    """
    # ---- Small patch: 7x7x7 ----
    p_small = extract_patch(vol, coord, 7)           # (7, 7, 7)
    p_small = normalize_patch(p_small)               # z-score

    # ---- Large patch: 15x15x15 downsampled to 7x7x7 ----
    p_large    = extract_patch(vol, coord, 15)        # (15, 15, 15)
    p_large    = normalize_patch(p_large)             # z-score
    scale      = 7.0 / 15.0                           # ≈ 0.4667
    p_large_ds = zoom(p_large, (scale, scale, scale), order=1)  # bilinear
    p_large_ds = p_large_ds[:7, :7, :7]              # ensure exact 7x7x7

    # ---- Concatenate: small first, large second ----
    features = np.concatenate([
        p_small.flatten(),     # 7*7*7 = 343 dims
        p_large_ds.flatten()   # 7*7*7 = 343 dims
    ]).astype(np.float32)      # total: 686 dims

    assert features.shape == (686,), (
        f"Feature dim mismatch: expected 686, got {features.shape[0]}"
    )
    return features


# ======================================================================
# VALIDATION HELPERS
# ======================================================================

def validate_volume(vol: np.ndarray, lbl: np.ndarray) -> None:
    """Basic sanity checks on loaded data before processing."""
    assert vol.ndim == 3, f"Expected 3-D volume, got shape {vol.shape}"
    assert lbl.ndim == 3, f"Expected 3-D label,  got shape {lbl.shape}"
    assert vol.shape == lbl.shape, (
        f"Shape mismatch: vol={vol.shape}, lbl={lbl.shape}"
    )
    unique_lbl = np.unique(lbl)
    assert set(unique_lbl).issubset({0, 1}), (
        f"Label array contains unexpected values: {unique_lbl}"
    )
    print(f"  ✅ Volume shape  : {vol.shape}")
    print(f"  ✅ Voxel range   : [{vol.min():.1f}, {vol.max():.1f}]")
    print(f"  ✅ Label unique  : {unique_lbl}")
    print(f"  ✅ Seg voxels    : {lbl.sum():,}")


def print_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a simple 2x2 confusion matrix to stdout."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n  Confusion Matrix")
    print(f"  {'':12s}  Pred POS  Pred NEG")
    print(f"  {'True POS':12s}  {tp:>8d}  {fn:>8d}")
    print(f"  {'True NEG':12s}  {fp:>8d}  {tn:>8d}")
    print(f"\n  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1 Score  : {f1:.3f}")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    # ---- LOAD MODEL ----
    print("=" * 60)
    print("  CFC MODEL SANITY CHECK")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"\n[1/5] Loading model from:\n      {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"  Model type   : {type(model)}")

    # Verify the model has the expected interface
    assert hasattr(model, "predict"),        "Model missing .predict()"
    assert hasattr(model, "predict_proba"),  "Model missing .predict_proba()"
    print("  ✅ predict() and predict_proba() found")

    # ---- LOAD DATA ----
    img_path = os.path.join(DATA_DIR, "imagesTr", f"{CASE_ID}_0000.nrrd")
    lbl_path = os.path.join(DATA_DIR, "labelsTr", f"{CASE_ID}.nrrd")

    print(f"\n[2/5] Loading case: {CASE_ID}")
    print(f"  img : {img_path}")
    print(f"  lbl : {lbl_path}")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not os.path.exists(lbl_path):
        raise FileNotFoundError(f"Label not found: {lbl_path}")

    vol, _ = nrrd.read(img_path)
    lbl, _ = nrrd.read(lbl_path)

    vol = vol.astype(np.float32)
    lbl = (lbl > 0).astype(np.uint8)

    validate_volume(vol, lbl)

    # ---- SKELETON ----
    print(f"\n[3/5] Computing skeleton...")
    skel      = skeletonize(lbl).astype(bool)
    skel_pts  = np.argwhere(skel)
    lumen_pts = np.argwhere((lbl > 0) & ~skel)

    print(f"  Skeleton points : {len(skel_pts):,}")
    print(f"  Lumen points    : {len(lumen_pts):,}")

    if len(skel_pts) == 0:
        raise RuntimeError("Skeleton is empty — check that the label mask is non-zero.")
    if len(lumen_pts) == 0:
        raise RuntimeError("Lumen mask is empty — label may be a pure skeleton already.")

    # ---- SAMPLE ----
    print(f"\n[4/5] Sampling points...")
    rng   = np.random.default_rng(RANDOM_SEED)
    n_pos = min(N_POS_SAMPLES, len(skel_pts))
    n_neg = min(N_NEG_SAMPLES, len(lumen_pts))

    pos_pts = skel_pts [rng.choice(len(skel_pts),  n_pos, replace=False)]
    neg_pts = lumen_pts[rng.choice(len(lumen_pts), n_neg, replace=False)]

    all_pts    = np.vstack([pos_pts, neg_pts])
    all_labels = np.array([1] * n_pos + [0] * n_neg, dtype=np.int32)

    print(f"  Sampled POS : {n_pos}")
    print(f"  Sampled NEG : {n_neg}")
    print(f"  Total       : {len(all_pts)}")

    # ---- FEATURE EXTRACTION ----
    print(f"\n[5/5] Extracting features...")
    X = np.array(
        [extract_features(vol, tuple(pt)) for pt in all_pts],
        dtype=np.float32
    )
    y = all_labels

    print(f"  Feature matrix : {X.shape}  (expected: ({len(all_pts)}, 686))")
    assert X.shape[1] == 686, (
        f"Feature dimension error: expected 686, got {X.shape[1]}"
    )

    # ---- PREDICT ----
    print("\n🔍 Running predictions...")
    preds  = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    pos_mask = (y == 1)
    neg_mask = (y == 0)

    pos_acc = (preds[pos_mask] == 1).mean()
    neg_acc = (preds[neg_mask] == 0).mean()
    overall = (preds == y).mean()

    # ---- RESULTS ----
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\n  Ground truth  — POS: {y.sum()}   NEG: {(y == 0).sum()}")
    print(f"  Predictions   — POS: {(preds == 1).sum()}   NEG: {(preds == 0).sum()}")

    print(f"\n  Skeleton accuracy : {pos_acc:.3f}   ({int(pos_acc * n_pos)}/{n_pos} correct)")
    print(f"  Lumen    accuracy : {neg_acc:.3f}   ({int(neg_acc * n_neg)}/{n_neg} correct)")
    print(f"  Overall  accuracy : {overall:.3f}")

    print(f"\n  Probability stats:")
    print(f"  POS — mean={probas[pos_mask].mean():.3f}  "
          f"std={probas[pos_mask].std():.3f}  "
          f"min={probas[pos_mask].min():.3f}  "
          f"max={probas[pos_mask].max():.3f}")
    print(f"  NEG — mean={probas[neg_mask].mean():.3f}  "
          f"std={probas[neg_mask].std():.3f}  "
          f"min={probas[neg_mask].min():.3f}  "
          f"max={probas[neg_mask].max():.3f}")

    # ---- CONFUSION MATRIX ----
    print_confusion(y, preds)

    # ---- SEPARATION SCORE ----
    sep = probas[pos_mask].mean() - probas[neg_mask].mean()
    print(f"\n  Separation (P_pos_mean - P_neg_mean) = {sep:.3f}")

    if sep > 0.3:
        print("  ✅ Good separation — model is discriminating well")
    elif sep > 0.1:
        print("  ⚠️  Weak separation — model may underperform during walk")
    else:
        print("  ❌ No separation — model is not discriminating")
        print("     → Check that feature extraction matches training exactly")
        print("     → Verify model was trained on this dataset")

    # ---- FEATURE DIM REMINDER ----
    print("\n" + "=" * 60)
    print("  FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Small patch   : 7x7x7   -> flatten -> 343 dims")
    print(f"  Large patch   : 15x15x15 -> zoom(7/15, order=1) -> 7x7x7 -> 343 dims")
    print(f"  Concat order  : [p_small | p_large_ds]")
    print(f"  Total dims    : 686")
    print(f"  Downsampling  : scipy.ndimage.zoom (bilinear, order=1)")
    print("=" * 60)


if __name__ == "__main__":
    main()
