"""
Ultra-fast minimal example using geodesic_vessels with hyperparameters.py
No saving, no metrics, just function calls for a single patient.
"""

import numpy as np
import nrrd
from scipy.ndimage import label
# from skimage.morphology import skeletonize
# from geodesic_vessels.metrics import compute_all_metrics

from geodesic_vessels.extremities import Extremities3D
from geodesic_vessels.paths import GeodesicPaths3D
from geodesic_vessels.reconstructions import ReconstructVessels3D
from geodesic_vessels.hyperparameters import optimize_dip_params

# -----------------------------
# CONFIG (edit paths)
# -----------------------------

RECONSTRUCTION_METHOD = "scale"
FORMULATION = "DIP"

IMG_PATH  = "" # fill with the original image .nrrd file
SEG_PATH  = "" # fill with binary mask:  .nrrd file
PROB_PATH = "" # fill with binary mask:  .nrrd file

DEFAULT_PARAMS = {"alpha": 1,
                "beta": 0.01,
                "gamma": 1,
                "theta": 1,
                "lambda": 0.1,
                "deg": 75,
                "max_it": 25,
                "weight_map": "DIP", # or DIP+ / DI 
                "device": "cuda"}

# -----------------------------
# LOAD DATA
# -----------------------------
img, _ = nrrd.read(IMG_PATH)
seg, _ = nrrd.read(SEG_PATH)
prob = nrrd.load(PROB_PATH)

# -----------------------------
# HYPERPARAMETER OPTIMIZATION (optional)
# -----------------------------

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

