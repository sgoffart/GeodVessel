import numpy as np
from scipy.ndimage import gaussian_filter
from  dataclasses import dataclass

from geodesic_vessels.extremities import *
from geodesic_vessels.paths import *

import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing
from skimage.morphology import remove_small_objects
from dataclasses import dataclass, field
from typing import List, Optional
from skimage.filters import threshold_otsu
import torch 

# -------------------------------
# Radius based methods
# -------------------------------

import numpy as np
from scipy.ndimage import gaussian_filter

def reconstruct_tubular_vessels_ultraslim(
    img_shape, path, radius_map=None, patch_factor=2.0, sigma=1.0
):
    """
    Memory-efficient, smooth vessel reconstruction along a given path.
    Compatible with 2D and 3D images. Produces a smooth tubular mask
    that represents the vessel structure.

    Parameters
    ----------
    img_shape : tuple
        Shape of the image (2D or 3D).
    path : list or ndarray
        Sequence of coordinates (y,x) for 2D or (z,y,x) for 3D.
    radius_map : list, optional
        Estimated radius at each path point. If None, uses default radius 4 voxels.
    patch_factor : float
        Scaling factor for the local patch around each path point (slightly larger than radius).
    sigma : float
        Gaussian smoothing parameter to produce smooth edges and remove voxel artifacts.

    Returns
    -------
    vessel_mask : ndarray (bool)
        Binary mask representing the reconstructed vessel.
    """
    dim = len(img_shape)
    assert dim in (2, 3), "Supports only 2D or 3D images."

    # Initialize the vessel mask
    vessel_mask = np.zeros(img_shape, dtype=bool)
    path = np.array(path, dtype=float)

    # Use default radius if none provided
    if radius_map is None:
        radius_map = [1.0] * len(path)
    radius_map = np.array(radius_map, dtype=float)

    # --- Compute tangents along the path ---
    # Approximate tangent at each path point by finite differences
    tangents = np.zeros_like(path)
    tangents[1:-1] = path[2:] - path[:-2]  # central difference for middle points
    tangents[0] = path[1] - path[0]        # forward difference for first point
    tangents[-1] = path[-1] - path[-2]     # backward difference for last point
    tangents /= np.linalg.norm(tangents, axis=1)[:, None] + 1e-8  # normalize

    # --- Iterate over each point in the path ---
    for voxel, r, tangent in zip(path, radius_map, tangents):
        # Define local patch size
        patch_radius = int(np.ceil(patch_factor * r))

        # --- Create coordinate grid for local patch ---
        if dim == 2:
            yc, xc = voxel
            y0, y1 = max(0, int(yc - patch_radius)), min(img_shape[0], int(yc + patch_radius + 1))
            x0, x1 = max(0, int(xc - patch_radius)), min(img_shape[1], int(xc + patch_radius + 1))
            Y, X = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
            coords = np.stack([Y - yc, X - xc], axis=-1)
        else:  # dim == 3
            zc, yc, xc = voxel
            z0, z1 = max(0, int(zc - patch_radius)), min(img_shape[0], int(zc + patch_radius + 1))
            y0, y1 = max(0, int(yc - patch_radius)), min(img_shape[1], int(yc + patch_radius + 1))
            x0, x1 = max(0, int(xc - patch_radius)), min(img_shape[2], int(xc + patch_radius + 1))
            Z, Y, X = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
            coords = np.stack([Z - zc, Y - yc, X - xc], axis=-1)

        # --- Implicit tubular modeling ---
        proj = np.sum(coords * tangent[None, ...], axis=-1)            # distance along tangent
        coords_perp = coords - proj[..., None] * tangent[None, ...]    # perpendicular component
        d_perp = np.linalg.norm(coords_perp, axis=-1)                 # perpendicular distance

        # Define radial and longitudinal limits (smooth tube)
        radial_limit = r
        longitudinal_limit = 1.2 * r  # slightly extend along tangent for smooth overlap

        # --- Soft-edged mask with smooth exponential falloff ---
        mask_local = (d_perp <= radial_limit) & (np.abs(proj) <= longitudinal_limit)
        falloff = np.exp(-((d_perp / radial_limit)**4 + (proj / longitudinal_limit)**4))
        mask_local = mask_local & (falloff > 0.1)  # keep smooth boundaries

        # --- Update the global vessel mask ---
        if dim == 2:
            vessel_mask[y0:y1, x0:x1] |= mask_local
        else:
            vessel_mask[z0:z1, y0:y1, x0:x1] |= mask_local

    # --- Apply Gaussian smoothing to remove voxel artifacts ---
    vessel_mask = gaussian_filter(vessel_mask.astype(float), sigma=sigma) > 0.5

    return vessel_mask

# -------------------------------
# Geodesic based methods
# -------------------------------

@dataclass
class ReconstructVessel3D:
    """Rebuild 3D vessels from paths."""

    path: GeodesicPath3D
    mask: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Convert torch tensors in the path to numpy-compatible integers."""
        if hasattr(self.path, "path"):
            cleaned_path = []
            for coord in self.path.path:
                if isinstance(coord, (list, tuple)):
                    cleaned_coord = tuple(
                        int(c.item()) if torch.is_tensor(c) else int(c)
                        for c in coord
                    )
                    cleaned_path.append(cleaned_coord)
                elif torch.is_tensor(coord):
                    cleaned_path.append(tuple(int(v.item()) for v in coord))
                else:
                    cleaned_path.append(tuple(coord))
            self.path.path = cleaned_path

    def reconstruct(self, method: str = "radius") -> np.ndarray:
        """
        High-level reconstruction interface.

        Parameters
        ----------
        method : str
            Reconstruction method to use. Options:
            - "geodesic": uses geodesic distance maps.
            - "radius": uses local radius estimation.

        Returns
        -------
        vessel_mask : np.ndarray
            Binary mask representing the reconstructed vessel.
        """
        if method == "geodesic":
            return self._reconstruct_from_geodesic_maps()
        elif method == "radius":
            return self._reconstruct_from_radius_estimation()
        elif method == "scale":
            return self._reconstruct_vessels_auto_scale()
        elif method == "centerline":
            return self._reconstruct_centerline_only()
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'geodesic' or 'radius'.")
    
    def compute_tangents_along_path(self, verbose=False):
            """
            Compute tangent vectors at each coordinate along the path.

            Tangent at point i is computed as the normalized vector between
            its neighboring coordinates (p_{i+1} - p_{i-1}).
            For endpoints, forward/backward differences are used.
            """
            _path = list(self.path.path)  # assuming self.path.path is iterable of coordinates
            n = len(_path)
            if n < 2:
                raise ValueError("Path too short to compute tangents")

            tangents = []
            coords = [np.array(c, dtype=float) for c in _path]

            for i in range(n):
                if i == 0:
                    # forward difference
                    t = coords[i + 1] - coords[i]
                elif i == n - 1:
                    # backward difference
                    t = coords[i] - coords[i - 1]
                else:
                    # central difference
                    t = coords[i + 1] - coords[i - 1]

                norm = np.linalg.norm(t)
                if norm > 0:
                    t /= norm
                else:
                    t = np.zeros_like(coords[i])

                tangents.append(t)
                if verbose:
                    print(f"Tangent at {tuple(map(int, coords[i]))}: {t}")

            return np.array(tangents)
    
    def _reconstruct_from_geodesic_maps(self, max_radius: int = 5, sigma: float = 0.8, relative_thresh: float = 0.02, eps : float = 0.1):
        """
        Reconstruct vessels by expanding around each centerline coordinate
        using local geodesic distance values.

        For each centerline voxel, a local region is added to the mask where
        geodesic distance <= (geodesic_value_at_center + relative_thresh).

        Parameters
        ----------
        max_radius : int
            Defines the cube region to consider around each path voxel (in voxels).
        sigma : float
            Gaussian smoothing strength.
        relative_thresh : float
            Geodesic distance offset to define local boundary around each path voxel.
        """
        # check the inputs 
        if len(self.path.path) == 0:
            raise ValueError("Path is empty!")
        if self.path.cost_map is None:
            raise ValueError("Missing geodesic distance map (cost_map)!")

        geod_map = self.path.cost_map
        shape = geod_map.shape
        vessel_mask = np.zeros(shape, dtype=bool)

        # Convert to integer indices from tensor to cpu-based integer (numpy compatibility)
        path_coords = []
        for coord in self.path.path:
            c = tuple(int(x.item()) if isinstance(x, torch.Tensor) else int(x) for x in coord)
            path_coords.append(c)
        
         # Compute tangents
        tangents = self.compute_tangents_along_path()

        # Iterate through centerline over all the centerline voxel (coord)
        for coord, tangent in zip(path_coords, tangents):
           
                z, y, x = coord 
                
                # create a box around the coord from the centerline with a radius of max_radius = 3 voxels : 7x7x7
                z0, z1 = max(0, z - max_radius), min(shape[0], z + max_radius + 1)
                y0, y1 = max(0, y - max_radius), min(shape[1], y + max_radius + 1)
                x0, x1 = max(0, x - max_radius), min(shape[2], x + max_radius + 1)

                local_block = geod_map[z0:z1, y0:y1, x0:x1] # corresponding to the box of interest and the geodesic values 
                center_val = geod_map[z, y, x] # centered around the value of interest

                # skip voxel if: 
                # - the centered voxel have inf value in cost_map (out of geodesic computation) 
                # - the centered voxel have zero value in cost_map (corresponding to exisitng segmentation)
                if np.isinf(center_val) or np.isnan(center_val):
                    continue  # skip invalid voxels 
                
                zz,yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
                rel = np.stack([zz - z,yy - y, xx - x], axis=-1)
                # --- Keep voxels close to the plane orthogonal to tangent ---
                dot = np.abs(np.tensordot(rel, tangent, axes=([-1], [0])))
                plane_mask = dot <= eps

                # Include voxels where geodesic distance is close to centerline
                # for each voxel from the centerline, I look around and want 
                # to apply a threshold corresponding to the center_val + relative threshold
                local_mask = local_block <= (center_val + relative_thresh) 
                vessel_mask[z0:z1, y0:y1, x0:x1] |= local_mask
            

        # Smooth mask for continuity
        vessel_mask = gaussian_filter(vessel_mask.astype(float), sigma=sigma) > 0.5

        filled_voxels = vessel_mask.sum()
        print(f"[INFO] Geodesic-based reconstruction done (voxels filled={filled_voxels}, "
            f"fraction={filled_voxels/np.prod(shape):.6f})")
        if filled_voxels == 0:
            print("[WARN] Empty vessel mask — check geodesic map or thresholds.")

        self.mask = vessel_mask
        return vessel_mask

    def _reconstruct_vessels_auto_scale(
        self,
        max_radius: int = 5,
        factor: float = 2.0,
        plane_eps: float = 0.5
    ):
        """
        Reconstruct vessels using an automatically scaled threshold
        based on centerline geodesic values.

        Parameters
        ----------
        max_radius : int
            Radius of local cube around each centerline voxel
        factor : float
            Multiplicative factor applied to centerline cost
        plane_eps : float
            Thickness of orthogonal plane constraint
        """
        cost_map = self.path.cost_map
        shape = cost_map.shape
        vessel_mask = np.zeros(shape, dtype=bool)

        centerline = self.path.path
        centerline = np.asarray(centerline, dtype=float)


        # ---- Compute tangents along the centerline ----
        tangents = np.zeros_like(centerline, dtype=float)
        if len(centerline) > 1:
            tangents[1:-1] = centerline[2:] - centerline[:-2]
            tangents[0] = centerline[1] - centerline[0]
            tangents[-1] = centerline[-1] - centerline[-2]

            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            valid = norms[:, 0] > 1e-6
            tangents[valid] /= norms[valid]

        # ---- Iterate along centerline ----
        for idx, coord in enumerate(centerline):
            z, y, x = coord.astype(int)

            if not np.isfinite(cost_map[z, y, x]):
                continue

            center_val = cost_map[z, y, x]
            thresh_val = center_val * factor

            # Local cube bounds
            z0, z1 = max(0, z - max_radius), min(shape[0], z + max_radius + 1)
            y0, y1 = max(0, y - max_radius), min(shape[1], y + max_radius + 1)
            x0, x1 = max(0, x - max_radius), min(shape[2], x + max_radius + 1)

            local_block = cost_map[z0:z1, y0:y1, x0:x1]

            # Orthogonal plane constraint
            zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
            rel = np.stack([zz - z, yy - y, xx - x], axis=-1)

            t = tangents[idx]
            if np.linalg.norm(t) > 0:
                plane_mask = np.abs(np.tensordot(rel, t, axes=([-1], [0]))) <= plane_eps
            else:
                plane_mask = np.ones(rel.shape[:-1], dtype=bool)

            vessel_mask[z0:z1, y0:y1, x0:x1] |= (
                (local_block <= thresh_val) & plane_mask
            )

        self.mask = vessel_mask
        return vessel_mask
    
    def _reconstruct_from_radius_estimation(self) -> np.ndarray:
        """
        Radius-based reconstruction in 3D using tubular patches around the centerline.
        """
        if not hasattr(self.path, "path") or len(self.path.path) == 0:
            raise ValueError("No centerline available in path for radius-based reconstruction.")

        radius_map = [2.0] * len(self.path.path)

        vessel_mask = reconstruct_tubular_vessels_ultraslim(
            img_shape=self.path.img.shape,
            path=self.path.path,
            radius_map=radius_map,
            sigma=0.6
        )

        print(f"[INFO] Radius-based 3D reconstruction done (vessel fraction={vessel_mask.mean():.4f})")
        self.mask = vessel_mask
        return vessel_mask

    def _reconstruct_centerline_only(self) -> np.ndarray:
        """
        Simple 3D reconstruction that marks only the centerline voxels as vessel.
        Works for torch tensors, numpy arrays, or tuples.
        """
        vessel_mask = np.zeros(self.path.cost_map.shape, dtype=bool)

        for coord in self.path.path:
            # convert each element to int safely
            z, y, x = [int(c.item()) if hasattr(c, 'item') else int(c) for c in coord]
            
            # bounds check
            if 0 <= z < vessel_mask.shape[0] and 0 <= y < vessel_mask.shape[1] and 0 <= x < vessel_mask.shape[2]:
                vessel_mask[z, y, x] = True

        self.mask = vessel_mask
        return vessel_mask

class ReconstructVessels3D:
    
    def __init__(self, _paths: GeodesicPaths3D):
        self.paths = _paths
        self.mask = field(default_factory=lambda: np.array([]))

    def reconstruct_all_paths(self, method:str):
        vessels = np.zeros_like(self.paths.img)
        for item in self.paths.paths:
            _tmp_vessel = ReconstructVessel3D(path = item,
                                               mask = item.cost_map)
            _tmp_vessel.reconstruct(method = method)
            vessels += _tmp_vessel.mask
        self.mask = vessels
    
    # vessels =ReconstructVessels3D(paths_objs_int)
    # vessels.reconstruct_all_paths(method = "radius")

    
# -------------------------------
# Example usage
# -------------------------------

def main(): 
    
    from PIL import Image
    from scipy.ndimage import label

    patient_id = '1' # 15 1
    comp_id =  10 #15 14 6
    ext_id = 1
    _device = 'cpu'

    # Paths to the image, segmentation, and probability map files
    # These paths should be updated to point to the actual files
    # in your environment.

    image_path = f'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/2D_RETINAL/Dataset001_Retinal/imagesTr/patient_{patient_id}_0000.png'#'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/PADSET_BREST/input/{patient_id}_img_cropped.nii.gz'
    segmentation_path = f'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/2D_RETINAL/nnUNet_preprocessed/Dataset001_Retinal/gt_segmentations/patient_{patient_id}.png'#'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/PADSET_BREST/input/{patient_id}_seg_gt_cropped.nii.gz'
    prob_map_path = f'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/2D_RETINAL/nnUNet_results/Dataset001_Retinal/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation/patient_{patient_id}.png' #'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/PADSET_BREST/input/{patient_id}_prob_map_artery_cropped.npy'

    # Load the 3D image, segmentation, and probability map (load as 2D image)
    img = Image.open(image_path)
    seg_gt = Image.open(segmentation_path)
    seg = Image.open(prob_map_path)

    img_data = np.array(img)
    seg_gt_data = np.array(seg_gt)
    seg_data = np.array(seg)
    
    # Assuming label_mask is your labeled mask (NumPy array)
    structure = np.ones((3, 3), dtype=int)  # connectivity
    label_mask, num_features = label(seg_data, structure=structure) # seg data
    filter_sizes = np.bincount(label_mask.ravel())
    # Remove small objects (e.g., size < 10 pixels)
    min_size = 10
    for i in range(1, num_features + 1):
        if filter_sizes[i] < min_size:
            label_mask[label_mask == i] = 0
    # Re-label the mask after removing small objects
    _label_mask, num_features = label(label_mask > 0, structure=structure) 

    # Extract extremities
    print('extract all extremities ...')
    cl_ext = Extremities2D(labeled_mask=_label_mask)
    cl_ext.build_graphs_for_components()#verbose=True
    cl_ext.compute_tangents_for_all_extremities()
    cl_ext.compute_radii_for_all_extremities_mask(img_data, max_r=10)
    
    # Compute the path for one extremity ! 
    print("compute paths ...")
    img_np = img_data.astype(np.float32)
    label_np = label_mask.astype(np.int32)
    ext = tuple(map(int, cl_ext.extremities[comp_id][ext_id]))
    # radius = cl_ext.get_radius(comp_id,ext_id)
    radius = 100
    print(cl_ext.get_specs_single_extremity(comp_id=comp_id, ext_index=ext_id))

    path = GeodesicPath2D(
        comp_id, 
        img_np, 
        cl_ext.labeled_mask,
        cl_ext.extremities[comp_id][ext_id],
        ext_tan=cl_ext.tangents[comp_id][ext_id],
        ext_radius = radius)
    path.geodesic_distance_torch_2d(device = _device)
    path.to_nearest_geod_component_from_ext_2d()
    # print(path.cost_map)
    print(path.path)
    
    # Reconstruct Vessels    
    vessel = ReconstructVessels2D(path)
    # vessel.compute_tangent_at_extremity(ext)
    vessel.reconstruct(method="radius")
    print(path.min_cost,path.target_label,path.ext_coords)

if __name__=="__main__": 
    main()
    
