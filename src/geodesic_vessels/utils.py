import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

import plotly.graph_objects as go
import plotly.io as pio

from scipy.ndimage import binary_dilation
from sklearn.mixture import GaussianMixture

import pyvista as pv
# os.environ["DISPLAY"] = ":1"  # Ensure PyVista uses the correct display for off-screen rendering
pv.global_theme.allow_empty_mesh = True
pv.OFF_SCREEN = True

def save_comparison(y_true, y_pred, y_pred_new,img, filename="comparison.png", cmap='gray'):
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()
    if torch.is_tensor(y_pred_new): y_pred_new = y_pred_new.cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(img, cmap=cmap)
    axes[0].set_title('image')
    axes[0].axis('off')

    axes[1].imshow(y_true, cmap=cmap)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(y_pred, cmap=cmap)
    axes[2].set_title('Predicted (Old)')
    axes[2].axis('off')

    axes[3].imshow(y_pred_new, cmap=cmap)
    axes[3].set_title('Predicted (New)')
    axes[3].axis('off')

    plt.tight_layout()

    # Save figure in same folder as the script
    out_path = os.path.join(os.getcwd(), filename)
    plt.savefig(out_path, dpi=400, bbox_inches='tight')

    print(f"Saved figure to: {out_path}")

    plt.close(fig)   # Close to avoid memory leaks in loops

def visualize_diff(seg_gt, seg_data, seg_data_new, img_data,filename="diff_visualization.png"):
    """
    Visualize voxel differences between 2D segmentations and ground truth.
    Colors:
        - Light Green = True Positive
        - Red         = False Positive
        - Orange      = False Negative
    """

    # Binarize (2D)
    gt  = (seg_gt > 0)
    old = (seg_data > 0)
    new = (seg_data_new > 0)

    # --- OLD vs GT ---
    old_tp =  old & gt
    old_fp =  old & ~gt
    old_fn = ~old & gt

    # --- NEW vs GT ---
    new_tp =  new & gt
    new_fp =  new & ~gt
    new_fn = ~new & gt

    def build_rgb(tp, fp, fn):
        # Create an empty RGB image
        rgb = np.zeros((tp.shape[0], tp.shape[1], 3), dtype=np.uint8)

        rgb[tp] = [144, 238, 144]   # light green
        rgb[fp] = [255, 0, 0]       # red
        rgb[fn] = [255, 165, 0]     # orange

        return rgb

    vis_old = build_rgb(old_tp, old_fp, old_fn)
    vis_new = build_rgb(new_tp, new_fp, new_fn)

    # ---- Visualization ---- #
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(img_data, cmap="gray")
    axs[0].set_title("Image")
    axs[0].axis("off")

    # axs[1].imshow(seg_gt, cmap="gray")
    # axs[1].set_title("Ground Truth")
    # axs[1].axis("off")

    axs[2].imshow(vis_old)
    axs[2].set_title("Predicted (Old)")
    axs[2].axis("off")

    axs[3].imshow(vis_new)
    axs[3].set_title("Predicted (New)")
    axs[3].axis("off")

    # Save figure in same folder as the script
    out_path = os.path.join(os.getcwd(), filename)
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    
    plt.show()
    print(f"Saved figure to: {out_path}")
    plt.close(fig)   # Close to avoid memory leaks in loops

def plotly_segmentation_results_3D(patient_id, 
                                   seg_gt_arr,
                                   seg_prob_arr,
                                   seg_arr, 
                                   seg_corr_arr,
                                   filedir = "./",
                                   show_plotly=False):
    
    diff_mask = seg_arr != seg_corr_arr
    # --------------------------
    # Prepare overlays for Plotly
    # --------------------------
    # convert to (x, y, z) scatter points
    def mask_to_xyz(mask):
        pts = np.argwhere(mask)
        # Plotly expects x,y,z
        return pts[:,2], pts[:,1], pts[:,0]

    x_diff, y_diff, z_diff = mask_to_xyz(diff_mask)
    x_gt,   y_gt,   z_gt   = mask_to_xyz(seg_gt_arr > 0)
    x_seg, y_seg, z_seg = mask_to_xyz(seg_arr>0)
    x_seg_prob, y_seg_prob, z_seg_prob = mask_to_xyz(seg_prob_arr>1e-5)


    # --------------------------
    # Create Plotly figure
    # --------------------------
    fig = go.Figure()

    # Corrections (red)
    fig.add_trace(go.Scatter3d(
        x=x_diff, y=y_diff, z=z_diff,
        mode='markers',
        marker=dict(size=2, color='darkred'),
        name="Corrections"
    ))

    # Ground truth (green)
    fig.add_trace(go.Scatter3d(
        x=x_gt, y=y_gt, z=z_gt,
        mode='markers',
        marker=dict(size=2, color='green'),
        name="GT"
    ))

    # segmentation
    fig.add_trace(go.Scatter3d(
        x=x_seg, y=y_seg, z=z_seg,
        mode='markers',
        marker=dict(size=2, color='red'),
        name="Seg"
    ))

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=f"Patient {patient_id}: Corrections & GT Overlap"
    )

    if show_plotly:
        fig.show()
    pio.write_html(fig, f"{filedir}/{patient_id}_compare_seg.html")

def render_volume_surface(vol, color=(1,1,1), opacity=1.0, window_size=(1200, 1200)):
    """
    Render 3D volume as isosurface mesh for proper lighting/shading.
    """
    # Debug info
    print(f"Volume: shape={vol.shape}, dtype={vol.dtype}, min={vol.min()}, max={vol.max()}, sum={vol.sum()}")
    
    # Check if volume is empty
    if vol.sum() == 0:
        print("Empty volume - returning white image")
        return np.ones((window_size[1], window_size[0], 3), dtype=np.uint8) * 255
    
    # Convert to binary if needed
    vol_binary = (vol > 0).astype(np.uint8)
    
    # Wrap volume
    grid = pv.ImageData(dimensions=vol_binary.shape)
    grid.point_data['values'] = vol_binary.flatten(order='F')
    
    # Extract surface mesh
    surf = grid.contour(isosurfaces=[0.5], scalars='values')
    
    print(f"Mesh: n_points={surf.n_points}, n_cells={surf.n_cells}")
    
    # Check if contour produced any geometry
    if surf.n_points == 0:
        print("No geometry produced - returning white image")
        return np.ones((window_size[1], window_size[0], 3), dtype=np.uint8) * 255

    # Offscreen plotter with HIGH RESOLUTION
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.background_color = "white"
    
    # Simple, reliable lighting
    p.enable_lightkit()
    
    # Add mesh - use RGB tuple normalized to 0-1
    p.add_mesh(
        surf,
        color=color,
        opacity=opacity,
        smooth_shading=False,
        lighting=True,
        ambient=0.2,
        diffuse=0.8,
        specular=0.3,
        specular_power=15
    )
    
    p.camera_position = 'xz'  # Front view
    p.reset_camera()
    
    # Screenshot
    img = p.screenshot(return_img=True)
    p.close()
    
    print(f"Image: shape={img.shape}, dtype={img.dtype}")
    
    return img

def render_combined_volumes(volumes_colors_opacities, window_size=(1200, 1200),p_cam = "iso"):
    """
    Render multiple volumes in the same scene with different colors.
    
    Args:
        volumes_colors_opacities: List of tuples (volume, color, opacity)
        window_size: Resolution of the render (width, height)
    """
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.background_color = "white"
    
    # Lighting
    p.enable_lightkit()
    
    has_geometry = False
    
    # Add all volumes to the same scene
    for vol, color, opacity in volumes_colors_opacities:
        if vol.sum() == 0:
            continue
            
        vol_binary = (vol > 0).astype(np.uint8)
        grid = pv.ImageData(dimensions=vol_binary.shape)
        grid.point_data['values'] = vol_binary.flatten(order='F')
        surf = grid.contour(isosurfaces=[0.5], scalars='values')
        
        if surf.n_points == 0:
            continue
            
        has_geometry = True
        p.add_mesh(
            surf,
            color=color,
            opacity=opacity,
            smooth_shading=True,
            lighting=True,
            ambient=0.2,
            diffuse=0.8,
            specular=0.3,
            specular_power=15
        )
    
    if not has_geometry:
        p.close()
        return np.ones((window_size[1], window_size[0], 3), dtype=np.uint8) * 255
    
    p.camera_position = p_cam  # Front view
    p.reset_camera()
    img = p.screenshot(return_img=True)
    p.close()
    
    return img

def _vol_to_surface(vol):
    """Convert a binary volume to a PyVista surface mesh."""
    vol_binary = (vol > 0).astype(np.uint8)
    grid = pv.ImageData(dimensions=vol_binary.shape)
    grid.point_data["values"] = vol_binary.flatten(order="F")
    return grid.contour(isosurfaces=[0.5], scalars="values")

# def _render_all_panels(panels, resolution=1200, camera_position="iso"):
#     """
#     Render N panels in ONE plotter using subplots so the camera
#     orientation is identical across all panels.

#     panels : list of list of (vol, color, opacity)
#     Returns: list of (H, W, 3) uint8 numpy arrays, one per panel.
#     """
#     n = len(panels)
#     p = pv.Plotter(
#         off_screen=True,
#         shape=(1, n),
#         window_size=(resolution * n, resolution),
#     )
#     p.background_color = "white"

#     # ── Add meshes to each subplot ────────────────────────────────────────
#     for col, vols_colors_opacities in enumerate(panels):
#         p.subplot(0, col)
#         p.enable_lightkit()
#         for vol, color, opacity in vols_colors_opacities:
#             if vol.sum() == 0:
#                 continue
#             surf = _vol_to_surface(vol)
#             if surf.n_points == 0:
#                 continue
#             p.add_mesh(
#                 surf,
#                 color=color,
#                 opacity=opacity,
#                 smooth_shading=True,
#                 lighting=True,
#                 ambient=0.2,
#                 diffuse=0.8,
#                 specular=0.3,
#                 specular_power=15,
#             )

#     # ── Set the same camera for every subplot ─────────────────────────────
#     p.subplot(0, 0)
#     p.camera_position = camera_position
#     p.reset_camera()
#     cam = p.camera  # capture camera state from subplot 0

#     for col in range(1, n):
#         p.subplot(0, col)
#         p.camera_position = camera_position
#         p.reset_camera()
#         p.camera.position    = cam.position
#         p.camera.focal_point = cam.focal_point
#         p.camera.up          = cam.up

#     # ── Screenshot → slice into individual panels ─────────────────────────
#     full = p.screenshot(return_img=True)   # (resolution, resolution*n, 3)
#     p.close()

#     w = resolution
#     return [full[:, col * w : (col + 1) * w, :] for col in range(n)]

def _render_all_panels(panels, resolution=1200, camera_position="iso"):
    """
    Render N panels as separate plotters with a shared camera.
    Avoids PyVista subplot borders.
    """
    # ── First pass: render panel 0 and capture its camera ────────────────
    def _render_single(vols_colors_opacities, cam=None):
        p = pv.Plotter(off_screen=True, window_size=(resolution, resolution))
        p.background_color = "white"
        p.enable_lightkit()
        for vol, color, opacity in vols_colors_opacities:
            if vol.sum() == 0:
                continue
            surf = _vol_to_surface(vol)
            if surf.n_points == 0:
                continue
            p.add_mesh(surf, color=color, opacity=opacity,
                       smooth_shading=True, lighting=True,
                       ambient=0.2, diffuse=0.8, specular=0.3, specular_power=15)
        if cam is None:
            p.camera_position = camera_position
            p.reset_camera()
        else:
            p.camera.position    = cam.position
            p.camera.focal_point = cam.focal_point
            p.camera.up          = cam.up
        img = p.screenshot(return_img=True)
        cam_out = p.camera
        p.close()
        return img, cam_out

    imgs = []
    cam = None
    for panel in panels:
        img, cam = _render_single(panel, cam=cam)
        imgs.append(img)

    return imgs

def plot_segmentation_3d_screenshot(patient_id, seg_gt_arr, seg_arr, seg_corr_arr,
                                    filedir="./", render_resolution=1200, p_cam="iso"):
    """
    3-panel 3D render with shared camera and per-panel titles.

    Panel 1 | Panel 2      | Panel 3
    GT      | Segmentation | Error map (TP / FP / FN)

    Error map colors:
      Blue   → TP : seg=1 & gt=1  (correct)
      Orange → FP : seg=1 & gt=0  (leakage / false positive)
      Purple → FN : seg=0 & gt=1  (missed / false negative)

    Args:
        render_resolution: Size of each panel in pixels.
                           1200 = high quality (default)
                           2400 = ultra high quality (recommended for publication)
                           3600 = maximum quality (may be slow)
    """
    # ── Voxel categories ──────────────────────────────────────────────────
    seg_gt_arr  = (seg_gt_arr  > 0).astype(np.uint8)
    seg_arr     = (seg_arr     > 0).astype(np.uint8)
    seg_corr_arr = (seg_corr_arr > 0).astype(np.uint8)

    added_voxels = seg_corr_arr & ~seg_arr  # voxels added by correction

    print(f"\nPatient {patient_id}")
    print(f"  GT={seg_gt_arr.sum()}  seg={seg_arr.sum()}  corrected={seg_corr_arr.sum()}")
    print(f"  Added by correction={added_voxels.sum()}")

    panels = [
        # Panel 1 – GT (green)
        [(seg_gt_arr, (0.2, 0.8, 0.2), 1.0)],
        # Panel 2 – original segmentation (red)
        [(seg_arr, (0.9, 0.1, 0.1), 1.0)],
        # Panel 3 – seg (red) + correction additions (blue)
        [
            (seg_arr,      (0.9, 0.1, 0.1), 1.0),  # original seg → red
            (added_voxels, (0.1, 0.4, 0.9), 1.0),  # added voxels → blue
        ],
    ]

    print("  Rendering panels (shared camera)...")
    panel_imgs = _render_all_panels(panels, resolution=render_resolution, camera_position=p_cam)

    # ── Compose figure with matplotlib ───────────────────────────────────
    panel_titles = ["Ground Truth", "Segmentation", "Error Map  (TP / FP / FN)"]

    # fig, axes = plt.subplots(
    #     1, 3,
    #     figsize=(render_resolution * 3 / 100, render_resolution / 100 + 1.2),
    # )
    fig, axes = plt.subplots(
        1, 3,
        figsize=(render_resolution * 3 / 100, render_resolution / 100 + 1.2),
        frameon=False,
    )
    fig.patch.set_visible(False)      # remove figure background

    for ax, img, title in zip(axes, panel_imgs, panel_titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_frame_on(False)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8, color="white")
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_elements = [
        Patch(facecolor=(0.2, 0.8, 0.2),  label="GT"),
        Patch(facecolor=(0.9, 0.1, 0.1),  label="Segmentation"),
        Patch(facecolor=(0.1, 0.4, 0.9),  label=f"Added by correction ({added_voxels.sum()} vox)"),
    ]
    # fig.legend(
    #     handles=legend_elements,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.02),
    #     ncol=5,
    #     fontsize=11,
    #     frameon=False,
    # )

    fig.suptitle(f"Patient: {patient_id}", fontsize=16, fontweight="bold", y=1.01)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    out_path = os.path.join(filedir, f"{patient_id}_3D_render.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# --------------------------------------------
# compute probability maps with GMM (Gaussian Mixture Models)
# ---------------------------------------------------------------------
# ============================================================
# CONTINUOUS GMM FOR ARTERY LUMEN (FIXED)
# ============================================================

import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.mixture import GaussianMixture

def compute_lumen_gmm_saturated(
    img,
    seg_mask,
    dilate_iter=2,
    bg_dilate_iter=6,
    artery_hu_window=(150, 350),
    body_threshold=-200,
    high_pct=99.5,
    max_samples=200_000
):
    """
    Produces a strong but saturated lumen probability map.
    Probabilities collapse to few values (near 0 or 1).
    """

    # -----------------------------
    # 1. Window + normalize image
    # -----------------------------
    # normalize the image to 0-1, within artery HU window 
    img_win = np.clip(img, artery_hu_window[0], artery_hu_window[1])
    img_win = (img_win - artery_hu_window[0]) / (artery_hu_window[1] - artery_hu_window[0]) # saturates values out of HU window

    # -----------------------------
    # 2. Vessel samples
    # -----------------------------
    vessel_support = binary_dilation(seg_mask, iterations=dilate_iter) # dilate the vessels to get more samples (larger area) 
    vessel_vals = img_win[vessel_support] # collect vessel values in the dilated vessels area
    vessel_vals = vessel_vals[vessel_vals <= np.percentile(vessel_vals, high_pct)] # remove extreme high-intensity artifacts

    # -----------------------------
    # 3. Background samples
    # -----------------------------
    body_mask = img > body_threshold # inside body
    bg_mask = body_mask & (~binary_dilation(seg_mask, iterations=bg_dilate_iter)) # exclude dilated vessels 
    bg_vals = img_win[bg_mask] # collect background values
    bg_vals = bg_vals[bg_vals <= np.percentile(bg_vals, high_pct)] # remove extreme high-intensity artifacts

    if vessel_vals.size < 200 or bg_vals.size < 200: # check samples size 
        raise RuntimeError("Not enough samples")

    # -----------------------------
    # 4. Balance + fit GMM
    # -----------------------------
    # subsample to balance classes btween vessel / background
    n = min(len(vessel_vals), len(bg_vals), max_samples)
    X = np.concatenate([
        vessel_vals[np.random.choice(len(vessel_vals), n, False)],
        bg_vals[np.random.choice(len(bg_vals), n, False)]
    ]).reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type="full", max_iter=200, random_state=0)
    gmm.fit(X)

    vessel_idx = np.argmax(gmm.means_.squeeze())

    # ----------------------------- 
    # 5. Posterior
    # -----------------------------
    prob = gmm.predict_proba(img_win.reshape(-1, 1))[:, vessel_idx] # vessel prob
    return prob.reshape(img.shape).astype(np.float32)

def compute_lumen_gmm_continuous(
    img,
    seg_mask,
    dilate_iter=2,
    bg_dilate_iter=6,
    artery_hu_window=(150, 350),
    body_threshold=-200,
    high_pct=99.5,
    max_samples=200_000,
):
    """
    Produces a smooth, continuous lumen probability map.
    """

    # -----------------------------
    # 1. Vessel samples (RAW HU)
    # -----------------------------
    vessel_support = binary_dilation(seg_mask, iterations=dilate_iter)
    vessel_vals = img[vessel_support]
    vessel_vals = vessel_vals[
        (vessel_vals > artery_hu_window[0]) &
        (vessel_vals < artery_hu_window[1])
    ]
    vessel_vals = vessel_vals[vessel_vals <= np.percentile(vessel_vals, high_pct)]

    # -----------------------------
    # 2. Background samples (inside body)
    # -----------------------------
    body_mask = img > body_threshold
    bg_mask = body_mask & (~binary_dilation(seg_mask, iterations=bg_dilate_iter))
    bg_vals = img[bg_mask]
    bg_vals = bg_vals[
        (bg_vals > 0) &
        (bg_vals < artery_hu_window[1])
    ]
    bg_vals = bg_vals[bg_vals <= np.percentile(bg_vals, high_pct)]

    if vessel_vals.size < 200 or bg_vals.size < 200:
        raise RuntimeError("Not enough samples")

    # -----------------------------
    # 3. Balance + fit GMM
    # -----------------------------
    n = min(len(vessel_vals), len(bg_vals), max_samples)
    X = np.concatenate([
        vessel_vals[np.random.choice(len(vessel_vals), n, False)],
        bg_vals[np.random.choice(len(bg_vals), n, False)]
    ]).reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        max_iter=300,
        random_state=0
    )
    gmm.fit(X)

    vessel_idx = np.argmax(gmm.means_.squeeze())

    # -----------------------------
    # 4. Posterior on RAW image
    # -----------------------------
    prob = gmm.predict_proba(img.reshape(-1, 1))[:, vessel_idx]
    prob = prob.reshape(img.shape)
    # kill air
    prob[img <= body_threshold] = 0.0
    
    return prob.reshape(img.shape).astype(np.float32)

# ============================================================
# CONTINUOUS LOCAL GMM (COMPONENT-BASED)
# ============================================================
def compute_lumen_gmm_continuous_component(
    img,
    seg_labels,
    comp_id,
    vessel_dilate=2,
    bg_dilate=12,
    artery_hu_window=(150, 350),
    body_threshold=-200,
    high_pct=99,
    max_samples=200_000,
):
    """
    Train GMM on ONE component (local context),
    apply posterior GLOBALLY to detect nearby vessels.
    """

    # -----------------------------
    # 1. Component mask (TRAINING)
    # -----------------------------
    comp_mask = (seg_labels == comp_id)

    if comp_mask.sum() < 50:
        raise RuntimeError(f"Component {comp_id} too small")

    vessel_support = binary_dilation(comp_mask, iterations=vessel_dilate)

    # -----------------------------
    # 2. Local background (TRAINING)
    # -----------------------------
    bg_roi = binary_dilation(comp_mask, iterations=bg_dilate)
    bg_mask = bg_roi & (~vessel_support) & (img > body_threshold)

    # -----------------------------
    # 3. Vessel samples
    # -----------------------------
    vessel_vals = img[vessel_support]
    vessel_vals = vessel_vals[
        (vessel_vals > artery_hu_window[0]) &
        (vessel_vals < artery_hu_window[1])
    ]
    vessel_vals = vessel_vals[
        vessel_vals <= np.percentile(vessel_vals, high_pct)
    ]

    # -----------------------------
    # 4. Background samples
    # -----------------------------
    bg_vals = img[bg_mask]
    bg_vals = bg_vals[
        (bg_vals > 0) &
        (bg_vals < artery_hu_window[1])
    ]
    bg_vals = bg_vals[
        bg_vals <= np.percentile(bg_vals, high_pct)
    ]

    if vessel_vals.size < 200 or bg_vals.size < 200:
        raise RuntimeError("Not enough samples")

    # -----------------------------
    # 5. Balance + fit GMM
    # -----------------------------
    n = min(len(vessel_vals), len(bg_vals), max_samples)

    X = np.concatenate([
        vessel_vals[np.random.choice(len(vessel_vals), n, False)],
        bg_vals[np.random.choice(len(bg_vals), n, False)]
    ]).reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        max_iter=200,
        random_state=0
    )
    gmm.fit(X)

    vessel_idx = np.argmax(gmm.means_.squeeze())

    # -----------------------------
    # 6. GLOBAL POSTERIOR
    # -----------------------------
    prob = np.zeros_like(img, dtype=np.float32)

    infer_mask = img > body_threshold
    vals = img[infer_mask].reshape(-1, 1)

    prob[infer_mask] = gmm.predict_proba(vals)[:, vessel_idx]

    return prob


# ---------------------------------------------------------------------
# Calibration of probability maps
# ---------------------------------------------------------------------

def calibrate_probability_map_monotonic_regression(prob_map, gt_map):
    """
    Calibrate a 3D probability map using monotonic (isotonic) regression.

    Parameters
    ----------
    prob_map : np.ndarray
        Predicted probability map (values in [0,1]).
    gt_map : np.ndarray
        Ground-truth binary map (0 or 1), same shape as prob_map.

    Returns
    -------
    calibrated_map : np.ndarray
        Calibrated probability map with preserved monotonicity.
    """
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    # Flatten predictions and ground truth
    prob_flat = prob_map.ravel()
    gt_flat = gt_map.ravel().astype(np.float32)

    # Fit isotonic regression: P(y=1 | p)
    iso_reg = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds='clip'
    )
    iso_reg.fit(prob_flat, gt_flat)

    # Apply calibration mapping
    calibrated_flat = iso_reg.transform(prob_flat)
    calibrated_map = calibrated_flat.reshape(prob_map.shape)

    return calibrated_map