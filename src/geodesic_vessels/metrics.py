import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from scipy.ndimage import label
from skimage.metrics import hausdorff_distance
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from itertools import combinations

import torch
import torch.nn.functional as F


from typing import Callable,Any, TypeAlias

import networkx as nx

Data: TypeAlias = dict[str,Any]
MetricFn: TypeAlias = Callable[[Data],None]

# TODO : Probably move to a class of computed metrics 
# TODO : Optimize the computation (investigate to limite the calculation of label and skeletonize function (time-consuming functions))

# -------------------------------
# Mask utilities
# -------------------------------

def paths_to_mask(paths, shape):
    """
    Convert a list of path dictionaries into a binary mask.
    Each path dict has a 'path' key with a list of voxel coordinates.
    """
    mask = np.zeros(shape, dtype=bool)
    for res in paths:
        if "path" not in res:
            continue
        for z, y, x in res["path"]:
            mask[int(z), int(y), int(x)] = True
    return mask

# ------------------------------- 
# # Extract paths from components 
# # ------------------------------- 

def extract_paths_from_components(pred_mask, bridging_paths):
    """ Skeletonize each connected component and combine with bridging paths. """ 
    dim = int(pred_mask.ndim) # assuming squared matrix
    _structure = np.ones((3,) * dim, dtype=int) # 26-connectivity 
    labeled, num_features = label(pred_mask,structure =_structure)
    component_paths = [] 
    for i in range(1, num_features + 1): 
        print(f"Processing component {i}/{num_features}...") 
        coords = np.argwhere(labeled == i) 
        if len(coords) < 2: 
            continue 
        component_skel = skeletonize((labeled == i).astype(np.uint8)) 
        skel_coords = np.argwhere(component_skel) 
        if len(skel_coords) < 2: 
            continue 
        component_paths.append({ "start": tuple(skel_coords[0]), "path": [tuple(c) for c in skel_coords], "label": None }) 
    
    if bridging_paths==None:
        return component_paths
    else:
        return component_paths + bridging_paths
    

def extract_list_path_lengths(skel_arr:np.array)->list:
    """Path length for each of the individual skeleton"""
    dim = int(skel_arr.ndim) # assuming squared matrix
    _structure = np.ones((3,) * dim, dtype=int) # 26-connectivity 
    masks,num_skel = label(skel_arr,structure =_structure)
    lengths = []
    for _label in [0,num_skel]:
        length = np.count_nonzero(masks[masks == _label])
        lengths.append(length)
    return lengths

# -------------------------------
# Voxels level overlap metrics
# -------------------------------

def dice(data:dict[str,any])->float:
    y_pred = np.array(data['y_pred']).astype(bool)
    y_true = np.array(data["y_true"]).astype(bool)
    intersection = np.sum(y_pred & y_true)
    return (2.0 * intersection) / (np.sum(y_pred) + np.sum(y_true) + 1e-8)

def jaccard(data:dict[str,any])->float:
    y_pred = np.array(data['y_pred']).flatten()
    y_true = np.array(data["y_true"]).flatten()
    return jaccard_score(y_true,y_pred,zero_division=0)

def f1(data:dict[str,any])->float:
    y_pred = np.array(data['y_pred']).flatten()
    y_true = np.array(data["y_true"]).flatten()
    return f1_score(y_true,y_pred,zero_division=0)

def precision(data:dict[str,any])->float:
    y_pred = np.array(data['y_pred']).flatten()
    y_true = np.array(data["y_true"]).flatten()
    return precision_score(y_true,y_pred,zero_division=0)

def recall(data:dict[str,any])->float:
    y_pred = np.array(data['y_pred']).flatten()
    y_true = np.array(data["y_true"]).flatten()
    return recall_score(y_true,y_pred,zero_division=0)


def leakage(data: dict[str, any], eps=1e-8) -> float:
    """
    Compute leakage: fraction of predicted voxels outside ground-truth,
    normalized by the number of ground-truth voxels.
    """
    y_pred = np.array(data['y_pred']).astype(bool)
    y_true = np.array(data['y_true']).astype(bool)
    
    fp = np.logical_and(y_pred, np.logical_not(y_true)).sum()
    gt_volume = y_true.sum()
    
    if gt_volume == 0:
        return 0.0
    
    return float(fp / (gt_volume + eps))


# -------------------------------
# Distance based metrics
# -------------------------------

def assd(data:dict[str,any])->float:
    """
    Compute Average Symmetric Surface Distance (ASSD)
    pred_points, gt_points: Nx3 arrays of voxel coordinates
    """
    pred_points = data["y_coords_pred"]
    gt_points = data["y_coords_true"]
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)
    d_pred, _ = tree_pred.query(gt_points)
    d_gt, _ = tree_gt.query(pred_points)
    return (np.mean(d_pred) + np.mean(d_gt)) / 2.0

def hd(data: dict[str, any]) -> float:
    """
    Correct Hausdorff Distance for point clouds.
    """
    pred_points = np.asarray(data["y_coords_pred"])
    gt_points   = np.asarray(data["y_coords_true"])

    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf

    tree_pred = cKDTree(pred_points)
    tree_gt   = cKDTree(gt_points)

    # distances from GT → nearest pred
    d_gt, _ = tree_pred.query(gt_points, k=1)
    # distances from pred → nearest GT
    d_pred, _ = tree_gt.query(pred_points, k=1)

    return max(d_gt.max(), d_pred.max())

def hd95(data: dict[str, any]) -> float:
    pred_points = np.asarray(data["y_coords_pred"])
    gt_points   = np.asarray(data["y_coords_true"])

    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf

    tree_pred = cKDTree(pred_points)
    tree_gt   = cKDTree(gt_points)

    d_gt, _ = tree_pred.query(gt_points, k=1)
    d_pred, _ = tree_gt.query(pred_points, k=1)

    all_distances = np.concatenate([d_gt, d_pred])
    return np.percentile(all_distances, 95)

# -------------------------------
# Connectivity score
# -------------------------------

def ncgt(data:dict[str,any])->int:
    """
    Number of connected components in ground truth mask.
    """
    y_true = data["y_true"].astype(bool)
    
    dim = int(y_true.ndim) # assuming squared matrix
    _structure = np.ones((3,) * dim, dtype=int) # 26-connectivity 
    
    _,nb_true = label(y_true,structure=_structure)
    return nb_true

def ncpred(data:dict[str,any])->int:
    """
    Number of connected components in predicted mask.
    """
    y_pred = data["y_pred"].astype(bool)
    
    dim = int(y_pred.ndim) # assuming squared matrix
    _structure = np.ones((3,) * dim, dtype=int) # 26-connectivity 
    
    _,nb_pred = label(y_pred,structure=_structure)
    return nb_pred

def ccr(data:dict[str,any])->float:
    """
    Compute the Connected Component Ratio (CCR) between predicted and true masks.

    CCR = (# connected components in prediction) / (# connected components in ground truth)

    Parameters
    ----------
    data : dict
        Dictionary containing:
        - "y_true": ground truth mask (array-like)
        - "y_pred": predicted mask (array-like)

    Returns
    -------
    float
        The connected component ratio.
    """
    # laod array
    y_true = data["y_true"].astype(bool) #y_skel_true
    y_pred = data["y_pred"].astype(bool)
    
    # define dim
    dim = int(y_true.ndim) # assuming squared matrix
    _structure = np.ones((3,) * dim, dtype=int) # 26-connectivity 
    
    # labelling
    label_mask,nb_true = label(y_true,structure=_structure)
    label_mask_pred, nb_pred = label(y_pred,structure=_structure)
    
    # Avoid division by zero
    if nb_true == 0:
        raise ValueError("Ground-truth mask has 0 connected components; CCR is undefined.")
    
    return (nb_pred/nb_true)

def connectivity(data:dict[str,any])->float:
    """
    Ratio of correctly connected endpoints between predicted and GT masks.
    """
    if len(data["y_true"].shape) == 2: 
        _structure = np.ones((3, 3), dtype=int)  # 4-connectivity
    elif len(data["y_true"].shape) == 3:
        _structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
    pred_skel = data["y_skel_pred"]
    gt_skel = data["y_skel_true"]

    pred_cc, pred_n = label(pred_skel, structure=_structure)
    gt_cc, gt_n = label(gt_skel, structure=_structure)
    return min(pred_n, gt_n) / max(pred_n, gt_n) if max(pred_n, gt_n) > 0 else 1.0

def ccq_metrics(pred_mask, gt_mask, tolerance=3)->float:
    """
    Compute Correctness, Completeness, and Quality (CCQ).
    A predicted pixel is correct if it lies within <tolerance> pixels of GT.
    """
    pred_pts = np.argwhere(pred_mask)
    gt_pts = np.argwhere(gt_mask)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return {"correctness": 0, "completeness": 0, "quality": 0}

    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)

    # Distances from GT to nearest prediction and vice versa
    d_gt, _ = tree_pred.query(gt_pts)
    d_pred, _ = tree_gt.query(pred_pts)

    tp_gt = np.sum(d_gt <= tolerance)
    tp_pred = np.sum(d_pred <= tolerance)

    correctness = tp_pred / (len(pred_pts) + 1e-8)
    completeness = tp_gt / (len(gt_pts) + 1e-8)
    quality = (2 * correctness * completeness) / (correctness + completeness + 1e-8)

    return {"correctness": correctness, "completeness": completeness, "quality": quality}

def tp(data:dict[str,any],eps = 1e-8)->float:
    "topology prediction : fraction of skeleton predicted (y_skel_pred) that lies in ground truth mask (y_true)"
    if any(x not in data.keys() for x in ["y_skel_pred", "y_true"]):
        raise ValueError("'y_skel_pred' and/or 'y_true' missing")

    y_skel_pred = data["y_skel_pred"].astype(bool)
    y_true = data["y_true"].astype(bool)

    intersection = np.logical_and(y_skel_pred, y_true).sum()
    denom = y_skel_pred.sum()

    if denom == 0:
        return 0.0
    return float(intersection / (denom + eps))

def ts(data:dict[str,any],eps = 1e-8)->float:
    "topology sensitivity : fraction of ground truth mask (y_true) that lies in skeleton predicted (y_skel_pred)"
    if any(x not in data.keys() for x in ["y_skel_true", "y_pred"]):
        raise ValueError("'y_skel_true' and/or 'y_pred' missing")

    y_skel_true = data["y_skel_true"].astype(bool)
    y_pred = data["y_pred"].astype(bool)

    intersection = np.logical_and(y_skel_true, y_pred).sum()
    denom = y_skel_true.sum()

    if denom == 0:
        return 0.0
    return float(intersection / (denom + eps))

def cldice(data: dict[str, any], eps=1e-8) -> float:
    _tp = tp(data, eps)
    _ts = ts(data, eps)

    if (_tp + _ts) == 0:
        return 0.0

    return float(2 * _tp * _ts / (_tp + _ts + eps))
    
# -------------------------------
# Path level metrics
# -------------------------------

def por(data:dict[str,any],eps = 1e-8)->float:
    """compute Path Overlap Ration(POR)"""
    overlap = np.sum(data["y_skel_pred"] & data["y_true"])
    return (overlap / (np.sum(data["y_skel_pred"]) + eps))

def apl(data:dict[str,any],eps = 1e-8)->float:
    """compute Average Path Length (APL)"""
    # run paht leghts extractors
    path_lengths = extract_list_path_lengths(data["y_skel_pred"])
    apl = np.mean(path_lengths) if len(path_lengths) > 0 else 0.0
    return apl

def apls(data: dict[str, any], eps=1e-8) -> float:
    """
    Compute APLS in the same way as APL, using extract_list_path_lengths.
    """
    pred_lengths = extract_list_path_lengths(data["y_skel_pred"])
    gt_lengths   = extract_list_path_lengths(data["y_skel_true"])

    if len(pred_lengths) == 0 or len(gt_lengths) == 0:
        return 0.0

    apl_pred = np.mean(pred_lengths)
    apl_gt   = np.mean(gt_lengths)

    return 1.0 - abs(apl_pred - apl_gt) / (apl_gt + eps)

def tlts(data:dict[str,any], tolerance=0.15)->float:
    """
    Compute Too-Long-Too-Short (TLTS): fraction of predicted paths whose
    lengths deviate less than 15% from corresponding GT paths.
    """
    pred_paths = data["y_skel_pred"]
    gt_paths = data["y_skel_true"]
    
    if len(pred_paths) == 0 or len(gt_paths) == 0:
        return 0.0

    gt_lengths = [len(res["path"]) for res in gt_paths if "path" in res]
    pred_lengths = [len(res["path"]) for res in pred_paths if "path" in res]

    if len(gt_lengths) == 0 or len(pred_lengths) == 0:
        return 0.0

    min_len = min(len(gt_lengths), len(pred_lengths))
    matched_gt = gt_lengths[:min_len]
    matched_pred = pred_lengths[:min_len]

    deviations = np.abs(np.array(matched_pred) - np.array(matched_gt)) / (np.array(matched_gt) + 1e-8)
    return np.sum(deviations < tolerance) / len(deviations)

# -------------------------------
# Evaluation
# -------------------------------

metrics: dict[str,MetricFn]={
    "dice": dice,
    "cldice": cldice,
    "tp":tp,
    "ts":ts,
    "hd":hd,
    "hd95":hd95,
    "jaccard":jaccard,
    "f1":f1,
    "leakage":leakage,
    'precision':precision, 
    'recall':recall,
    "connectivity": connectivity,
    "assd":assd,
    "apls":apls,
    "tlts":tlts,
    "apl":apl,
    "por":por,
    "ccr":ccr,
    "ncgt":ncgt,
    "ncpred":ncpred
}



def compute_metric(data:dict,metric:str):
    metricFn = metrics.get(metric)
    if metricFn is None:
        raise ValueError(f"No metrics found : {metric}")
    return metricFn(data)

def compute_all_metrics(data:dict,metrics=None):
    """
    Evaluate reconstructed paths against original and ground truth masks.
    Returns a dictionary of metrics.
    """
    allowed_keys = {
        "y_true",
        "y_pred",
        "y_skel_true",
        "y_skel_pred",
        "y_coords_true",
        "y_coords_pred"
        }

    if metrics is None:
        raise ValueError("Metrics list is empty")

    # Check that all keys in `data` are allowed
    if not all(k in allowed_keys for k in data.keys()):
        raise ValueError("Content of data dict is out of the possible keys")

    results = {}
    for _metric in metrics:
        results[_metric] = float(compute_metric(data, _metric)) # convert to float and not np.float

    return results
