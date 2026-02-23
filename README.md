# geodesic-vessels

## Getting started

# 🩸 GeodesicVessels

**geodesic_vessels** is a Python package for **geodesic-based vessel reconnection and reconstruction**.
It aims to restore **vascular continuity** by reconnecting fragmented vessel segments using **geodesic paths**, while maintaining realistic topology and geometry.
The package is designed for use in medical image analysis, particularly angiographic, CT, and MRI vascular datasets.

---

## 🚀 Key Features

* **Geodesic Reconnection:**
  Automatically reconnects broken or missing vessel branches using geodesic shortest paths guided by probability or intensity maps.

* **Level-Set & Implicit Reconstruction:**
  Optionally integrates level-set–based reconstruction to rebuild smooth, tubular vessel geometries.

* **2D and 3D Support:**
  Compatible with both 2D slices and full 3D volumes.

* **Memory-Efficient Implementation:**
  Handles large medical images through efficient patch-wise operations.

* **Integrated Evaluation Tools:**
  Includes metrics such as Dice, CLDice, connectivity, APLS, CCQ, and TLTS to assess reconstruction quality.

---

## 📦 Installation

```bash
git clone https://github.com/.../geodesic-vessels.git
cd geodesic-vessels
pip install -e .
```

### Requirements

* Python ≥ 3.11

You can install dependencies manually:

```bash
pip install numpy scipy scikit-image networkx torch matplotlib
```


## ⚙️ Usage Example

```python
import numpy as np
from geodesic_vessels import geodesic_reconnect, evaluate_paths

# Load your binary vessel segmentation
vessel_mask = np.load("segmentation.npy")

# Perform geodesic-based reconnection
reconnected_mask, paths = geodesic_reconnect(vessel_mask, prob_map=None)

# Evaluate the reconstruction
metrics = evaluate_paths(paths, vessel_mask, ground_truth_mask)
print(metrics)
```

---

## 🧠 Methodology Overview

The **GeodesicVessels** pipeline consists of:

1. **Centerline Extraction:**
   Skeletonize the input segmentation to obtain disconnected vessel centerlines.

2. **Geodesic Path Finding:**
   Use a cost map derived from the inverse of intensity or probability to compute minimal geodesic connections between vessel endpoints.

3. **Tubular Reconstruction:**
   Reconstruct smooth vessel geometry along each geodesic path using implicit or spline-based modeling.

4. **Topology Preservation:**
   Merge reconstructed paths with the original mask while maintaining vessel connectivity.

---

## 📊 Metrics & Evaluation

| Metric           | Description                                         |
| :--------------- | :-------------------------------------------------- |
| **Dice**         | Overlap between predicted and ground truth vessels. |
| **CLDice**       | Connectivity-aware Dice metric.                     |
| **Connectivity** | Ratio of correctly connected components.            |
| **APLS**         | Average Path Length Similarity.                     |
| **CCQ**          | Correctness, Completeness, and Quality.             |
| **TLTS**         | Too-Long-Too-Short path ratio.                      |

---
