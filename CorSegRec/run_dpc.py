# run_dpc.py
import os
import nrrd
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import label

from models.cfc_model import CFC
from dpc.dpc_walk import DPC

# ---- CONFIG ----
BASE    = "./datasets/ASOCA"
CASE_ID = "case_0002"

# ---- PATHS ----
img_path = os.path.join(BASE, "Dataset001_ASOCA/imagesVal",f"{CASE_ID}_0000.nrrd")
seg_path = os.path.join(BASE,
    "nnUNet_results/Dataset001_ASOCA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation",
    f"{CASE_ID}.nrrd")

# ---- LOAD ----
print(f"Loading image: {img_path}")
vol, _ = nrrd.read(img_path)

print(f"Loading segmentation: {seg_path}")
seg, _ = nrrd.read(seg_path)

print(f"Volume shape : {vol.shape}")
print(f"Seg shape    : {seg.shape}")
print(f"Seg voxels   : {(seg > 0).sum()}")

# ---- LOAD MODEL ----
model = CFC()
model.load(os.path.join(BASE, "cfc_model.pkl"))

# ---- FIND DISCONNECTED COMPONENTS ----
def find_cross_component_endpoints(seg, min_component_size=100, min_distance=20.0):
    """
    Find two points from TWO DIFFERENT connected components.
    Picks the closest pair of endpoints across components
    (simulates a gap to reconnect).

    Returns
    -------
    start : tuple  — point in component A
    end   : tuple  — point in component B
    comp_a_id, comp_b_id : int
    """
    print("\n[CC] Running connected component analysis...")
    labeled, n_components = label(seg > 0)
    print(f"[CC] Found {n_components} connected components")

    # Get component sizes
    comp_sizes = {}
    for comp_id in range(1, n_components + 1):
        size = (labeled == comp_id).sum()
        comp_sizes[comp_id] = size

    # Filter small components (noise)
    valid_comps = [cid for cid, sz in comp_sizes.items() if sz >= min_component_size]
    print(f"[CC] Valid components (size >= {min_component_size}): {len(valid_comps)}")

    for cid in valid_comps:
        print(f"     Component {cid}: {comp_sizes[cid]} voxels")

    if len(valid_comps) < 2:
        raise ValueError(
            f"❌ Only {len(valid_comps)} valid component(s) found. "
            f"Need at least 2 to run DPC reconnection.\n"
            f"   Try reducing min_component_size (currently {min_component_size})."
        )

    # For each component, sample candidate endpoints
    # Strategy: use points on the "boundary" of each component (closest to other components)
    best_dist = np.inf
    best_start, best_end = None, None
    best_a, best_b = None, None

    # Sample points from each component
    comp_samples = {}
    for cid in valid_comps:
        pts = np.argwhere(labeled == cid)
        n_sample = min(200, len(pts))
        idx = np.random.choice(len(pts), n_sample, replace=False)
        comp_samples[cid] = pts[idx]

    # Find closest pair across different components
    print(f"\n[CC] Finding closest cross-component pair...")
    for i, cid_a in enumerate(valid_comps):
        for cid_b in valid_comps[i+1:]:
            pts_a = comp_samples[cid_a]
            pts_b = comp_samples[cid_b]

            # Vectorized distance matrix
            # pts_a: (Na, 3), pts_b: (Nb, 3)
            diff = pts_a[:, None, :] - pts_b[None, :, :]   # (Na, Nb, 3)
            dists = np.linalg.norm(diff, axis=-1)            # (Na, Nb)

            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            min_d = dists[min_idx]

            if min_d >= min_distance and min_d < best_dist:
                best_dist = min_d
                best_start = tuple(pts_a[min_idx[0]])
                best_end   = tuple(pts_b[min_idx[1]])
                best_a, best_b = cid_a, cid_b

    if best_start is None:
        # Fallback: just take any two components regardless of distance
        print("[CC] ⚠️  No pair found with min_distance constraint, using closest pair overall")
        cid_a, cid_b = valid_comps[0], valid_comps[1]
        pts_a, pts_b = comp_samples[cid_a], comp_samples[cid_b]
        diff  = pts_a[:, None, :] - pts_b[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        best_start = tuple(pts_a[min_idx[0]])
        best_end   = tuple(pts_b[min_idx[1]])
        best_dist  = dists[min_idx]
        best_a, best_b = cid_a, cid_b

    print(f"\n[CC] ✅ Selected:")
    print(f"     Component A (id={best_a}, size={comp_sizes[best_a]}): {best_start}")
    print(f"     Component B (id={best_b}, size={comp_sizes[best_b]}): {best_end}")
    print(f"     Gap distance: {best_dist:.2f} voxels")

    return best_start, best_end, best_a, best_b


# ---- VISUALIZE ----
def visualize_dpc_3d(seg, path, labeled, comp_a, comp_b, save_path):
    print("\nPreparing 3D visualization...")

    if len(path) == 0:
        print("❌ Empty path, skipping visualization")
        return

    path_arr = np.array(path)
    px, py, pz = path_arr[:, 0], path_arr[:, 1], path_arr[:, 2]

    traces = []

    # Component A (blue)
    pts_a = np.argwhere(labeled == comp_a)
    if len(pts_a) > 30000:
        pts_a = pts_a[np.random.choice(len(pts_a), 30000, replace=False)]
    traces.append(go.Scatter3d(
        x=pts_a[:,0], y=pts_a[:,1], z=pts_a[:,2],
        mode='markers',
        marker=dict(size=1, color='blue', opacity=0.2),
        name=f'Component A (id={comp_a})'
    ))

    # Component B (cyan)
    pts_b = np.argwhere(labeled == comp_b)
    if len(pts_b) > 30000:
        pts_b = pts_b[np.random.choice(len(pts_b), 30000, replace=False)]
    traces.append(go.Scatter3d(
        x=pts_b[:,0], y=pts_b[:,1], z=pts_b[:,2],
        mode='markers',
        marker=dict(size=1, color='cyan', opacity=0.2),
        name=f'Component B (id={comp_b})'
    ))

    # DPC path (red)
    traces.append(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='lines+markers',
        line=dict(color='red', width=5),
        marker=dict(size=3, color='red'),
        name='DPC Reconnection Path'
    ))

    # Start / End
    traces.append(go.Scatter3d(
        x=[px[0]], y=[py[0]], z=[pz[0]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='diamond'),
        name='Start (Comp A)'
    ))
    traces.append(go.Scatter3d(
        x=[px[-1]], y=[py[-1]], z=[pz[-1]],
        mode='markers',
        marker=dict(size=10, color='orange', symbol='diamond'),
        name='End (Comp B)'
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"DPC Reconnection: Component {comp_a} → Component {comp_b} ({len(path)} steps)",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
    )
    fig.write_html(save_path)
    print(f"✅ Saved to: {save_path}")


# =====================================================
# MAIN
# =====================================================

# ---- FIND CROSS-COMPONENT ENDPOINTS ----
start, end, comp_a, comp_b = find_cross_component_endpoints(
    seg,
    min_component_size=100,  # ignore tiny noise components
    min_distance=5.0         # minimum gap (voxels) between components
)

print(f"\nStart : {start}  seg={seg[start]}")
print(f"End   : {end}    seg={seg[end]}")
print(f"Distance: {np.linalg.norm(np.array(start) - np.array(end)):.2f} voxels")

# ---- CONNECTED COMPONENT MAP (for visualization) ----
from scipy.ndimage import label as scipy_label
labeled_vol, _ = scipy_label(seg > 0)

# ---- INIT DPC ----
dpc = DPC(
    vol, seg, model=model,
    omega=5.0,
    divergence_penalty=0.2
)

# ---- RUN ----
print("\n⏳ Running DPC pathfinding...")
path = dpc.walk(
    start, end,
    max_steps=5000,
    stay_in_seg=False,       # ← MUST BE FALSE: path crosses the gap (outside seg)
    convergence_threshold=3.0,
    max_stuck_steps=200
)

print(f"\n✅ Path length  : {len(path)}")
print(f"   Start        : {path[0]}")
print(f"   End reached  : {path[-1]}")
print(f"   Final dist   : {np.linalg.norm(np.array(path[-1]) - np.array(end)):.2f}")

# ---- VISUALIZE ----
out_html = os.path.join(BASE, f"dpc_result_{CASE_ID}.html")
visualize_dpc_3d(seg, path, labeled_vol, comp_a, comp_b, out_html)

print("\n✅ DONE!")
