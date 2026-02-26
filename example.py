"""
Ultra-fast minimal example using geodesic_vessels with hyperparameters.py
No saving, no metrics, just function calls for a single patient.
"""

import numpy as np
import nrrd
from scipy.ndimage import label

from geodesic_vessels.extremities import Extremities3D
from geodesic_vessels.paths import GeodesicPaths3D
from geodesic_vessels.reconstructions import ReconstructVessels3D
from geodesic_vessels.hyperparameters import DEFAULT_PARAMS, optimize_dip_params

# -----------------------------
# CONFIG (edit paths)
# -----------------------------

BASE = "/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/ASOCA"
PATIENT_ID = "0002"
FOLD_ID = 0
RECONSTRUCTION_METHOD = "scale"
FORMULATION = "DIP"

IMG_PATH  = f"{BASE}/Dataset002_ASOCA_5fold/imagesTr/case_{PATIENT_ID}_0000.nrrd"
SEG_PATH  = f"{BASE}/nnUNet_results/Dataset002_ASOCA_5fold/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{FOLD_ID}/validation/case_{PATIENT_ID}.nrrd"
PROB_PATH = SEG_PATH.replace(".nrrd", ".npz")

# -----------------------------
# LOAD DATA
# -----------------------------
img, _ = nrrd.read(IMG_PATH)
seg, _ = nrrd.read(SEG_PATH)
prob = np.load(PROB_PATH)["probabilities"][1]

if prob.shape != img.shape:
    prob = np.transpose(prob, (2, 1, 0))

# -----------------------------
# HYPERPARAMETER OPTIMIZATION (optional)
# -----------------------------
# small n_samples/n_iterations for speed
best_params = optimize_dip_params(img, seg > 0, prob,
                                  weight_map=FORMULATION,
                                  n_samples=5,
                                  n_iterations=10,
                                  downsample=2)

params = DEFAULT_PARAMS  # or swap with best_params to use optimized values

# -----------------------------
# GEODESIC PIPELINE
# -----------------------------

seg_binary = (seg > 0).astype(np.uint8)
seg_label, _ = label(seg_binary, structure=np.ones((3,3,3)))

# Extremities
exts = Extremities3D(seg_label)
exts.build_graphs_for_components()
exts.compute_tangents_for_all_extremities()
exts.compute_radii_for_all_extremities_mask()

# Geodesic paths
paths = GeodesicPaths3D(exts, img, seg_label, prob)
paths.compute_all_paths(params)
paths.filter_duplicates()

# Reconstruct vessels
vessels = ReconstructVessels3D(paths)
vessels.reconstruct_all_paths(method=RECONSTRUCTION_METHOD)
seg_corrected = seg_binary.astype(bool) | (vessels.mask > 0)

# -----------------------------
# METRICS (commented out)
# -----------------------------
# from skimage.morphology import skeletonize
# from geodesic_vessels.metrics import compute_all_metrics
# skel_gt = skeletonize(gt).astype(np.uint8)
# skel_pred = skeletonize(seg_corrected).astype(np.uint8)
# metrics = compute_all_metrics(
#     {
#         "y_true": gt,
#         "y_pred": seg_corrected.astype(np.uint8),
#         "y_skel_true": skel_gt,
#         "y_skel_pred": skel_pred,
#         "y_coords_true": np.argwhere(gt),
#         "y_coords_pred": np.argwhere(seg_corrected),
#     },
#     ["dice", "precision", "recall", "cldice"]
# )

print("Geodesic pipeline completed for patient", PATIENT_ID)
print("seg_corrected shape:", seg_corrected.shape)