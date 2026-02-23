import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import warnings
import json 
from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_dilation
from tqdm import tqdm


from geodesic_vessels.extremities import *

# -------------------------------
# Geodesic distance (PyTorch, Jacobi) for 3d 
# -------------------------------

@dataclass
class GeodesicPath3D:
    """
    Direct 3D extension of your GeodesicPath2D class.
    Same structure, same logic, but extended to 3D volumes.
    """

    ext: type = Extremity3D
    img: np.ndarray = field(default_factory=lambda: np.array([]))
    label_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    prob_map_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    path: dict = field(default_factory=dict)
    path_tan: Tuple[float, float, float] = field(default_factory=tuple)
    path_valid: bool = field(default_factory=bool)
    min_cost: float = field(default_factory=float)
    target_label: int = field(default_factory=int)
    cost_map: np.ndarray = field(default_factory=lambda: np.array([]))

    # ---------------------------------------------------------------------
    # 26-neighborhood offsets + anisotropic directional metric
    # ---------------------------------------------------------------------
    def _neighbor_offsets_torch_3d(self,spacing = (1,1,1),device="cuda", lambda_dir=0.35):
        """
        26-connected neighborhood with DIP+ directional constraint (3D).
        """

        lam = float(lambda_dir)
        sx, sy, sz = spacing  # spacing along x, y, z

        # 26 neighbors
        offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    offsets.append((dx, dy, dz))

        offs = torch.tensor(offsets, device=device, dtype=torch.float32)
        dx, dy, dz = offs[:, 0], offs[:, 1], offs[:, 2]

        # step length with spacing
        step_norm = torch.sqrt((dx * sx) ** 2 + (dy * sy) ** 2 + (dz * sz) ** 2) + 1e-8

        # normalized tangent
        tx, ty, tz = self.ext.tangent  # MUST be unit norm

        # directional cosine
        cos_theta = (dx * tx + dy * ty + dz * tz) / step_norm

        # DIP+ directional penalty
        # cos(theta) > 0 -> small penalty; cos(theta) <= 0 -> inf penalty
        dir_penalty = torch.where(
            cos_theta > 0,
            (1.0 - cos_theta).clamp(min=0.0)**2,
            torch.tensor(float("inf"), device=device)
        )
    

        # anisotropic metric
        proj = dx * tx + dy * ty + dz * tz
        cos2 = (proj * proj) / (step_norm**2 + 1e-12)
        sin2 = 1.0 - cos2
        dsq = step_norm**2 * (lam + (1.0 - lam) * sin2)
        dsq = torch.clamp(dsq, min=1e-6)

        return offs.long(), dsq, dir_penalty

    # ---------------------------------------------------------------------
    # Main geodesic solver
    # ---------------------------------------------------------------------
    def geodesic_distance_torch_3d(self, 
                                   spacing=(1,1,1),
                                   alpha=1.0, 
                                   beta=1.0,
                                   n_it=50, 
                                   tol=0.0, 
                                   gamma = 1.0, 
                                   theta = 1.0,
                                   eps=1e-12, 
                                   device=None,
                                   lambda_dir=0.25,
                                   weight_map = "DIP+"):

        img_np = np.asarray(self.img, dtype=np.float32)
        lbl_np = np.asarray(self.label_mask)
        prob_np = np.asarray(self.prob_map_mask)
        
        ## local gmm
        #comp_id = self.ext["comp_id"]
        # prob_np = compute_lumen_gmm_continuous_component(self.img, self.label_mask > 0,self.)
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        img_t = torch.from_numpy(img_np).to(device=device, dtype=torch.float32)
        lbl_t = torch.from_numpy(lbl_np).to(device=device)
        prob_t = torch.from_numpy(prob_np).to(device=device, dtype=torch.float32) # add prob map

        nx, ny, nz = img_t.shape

        coord = self.ext.coord
        from_comp = int(lbl_np[coord])

        ## make the traveil trough the component impossible ; favorishing the surrounding path. 
        # allowed = (lbl_t != from_comp)
        # allowed[coord] = True
        # allowed = allowed.bool()
        
        ## make the travel of the path trough the parent compoenent possible
        allowed = torch.ones_like(lbl_t, dtype=torch.bool)

        inf = torch.tensor(float('inf'), device=device)
        dist = torch.full_like(img_t, inf)
        dist[coord] = 0.0

        offs, dsq,dir_penalty = self._neighbor_offsets_torch_3d(spacing, device=device, lambda_dir=lambda_dir)
        M = offs.shape[0]

        alpha_t = torch.tensor(alpha, device=device)
        beta_t  = torch.tensor(beta, device=device)
        gamma_t = torch.tensor(gamma, device=device)
        theta_t = torch.tensor(theta, device=device)
        img_eps = torch.tensor(eps, device=device)

        for _ in range(int(n_it)):
            best = dist.clone()

            for m in range(M):
                dx,dy,dz = offs[m]
                dx=int(dx.item()); dy=int(dy.item()); dz=int(dz.item())

                # slicing in 3D (exact analogue to your 2D code)
                x0 = max(0, dx); x1 = min(nx, nx+dx)
                y0 = max(0, dy); y1 = min(ny, ny+dy)
                z0 = max(0, dz); z1 = min(nz, nz+dz)

                cx0 = -dx if dx < 0 else 0; cx1 = cx0 + (x1 - x0)
                cy0 = -dy if dy < 0 else 0; cy1 = cy0 + (y1 - y0)
                cz0 = -dz if dz < 0 else 0; cz1 = cz0 + (z1 - z0)

                if x1<=x0 or y1<=y0 or z1<=z0:
                    continue

                dist_n = dist[x0:x1, y0:y1, z0:z1]
                img_n  = img_t[x0:x1, y0:y1, z0:z1]
                allow_n = allowed[x0:x1, y0:y1, z0:z1]
                prob_n = prob_t[x0:x1, y0:y1, z0:z1]  # add prob map
                
                dist_c = dist[cx0:cx1, cy0:cy1, cz0:cz1]
                img_c  = img_t[cx0:cx1, cy0:cy1, cz0:cz1]
                allow_c = allowed[cx0:cx1, cy0:cy1, cz0:cz1]
                prob_c = prob_t[cx0:cx1, cy0:cy1, cz0:cz1]  # add prob map

                valid = allow_n & allow_c
                if not torch.any(valid):
                    continue
                
                # add prob map to intensity difference
                # prob_diff = (prob_c - prob_n).pow(2)  # logdiff = torch.log(prob_n + prob_eps) - torch.log(prob_c + prob_eps)
                prob_diff = (torch.log(prob_c + eps) -torch.log(prob_n + eps)).pow(2)
                # The probability penalty weight γ is derived automatically from the empirical distribution of local probability differences within segmented vessels, such that typical intra-vessel variations induce a small multiplicative slowdown, while background probability variations induce a strong penalty.

                # geodesic terms (intensity diff)
                dint = (img_c - img_n).pow(2) / (dsq[m] + img_eps)
                
                # weight formulation with prob map
                if weight_map == "DIP":
                    
                    base = torch.sqrt(alpha_t * dsq[m] + beta_t * dint)
                    # w = base * (1 + gamma * prob_diff)
                    
                    # Direct probability penalty: high prob -> low penalty, low prob -> high penalty
                    # Option A: inverse probability (simple)
                    prob_penalty = 1.0 / (prob_c + eps)
                    w = base * prob_penalty
                    
                    
                elif weight_map == "DI":
                    w = torch.sqrt(alpha_t * dsq[m] + beta_t * dint)
                elif weight_map == "P": # probability only
                    # w = torch.sqrt(gamma * prob_diff)
                    w=(1 + gamma * prob_diff)
                    
                elif weight_map == "DIP+":
                    w = alpha_t*dsq[m] + beta_t * 1/dint + gamma_t*(1/prob_c+eps)
                    
                elif weight_map == "DIP+a":
                    w = alpha_t*dsq[m] + beta_t *(1-dint) + gamma_t*(1-prob_c+eps)
                    
                elif weight_map == "DIP++":
                    w = alpha_t*dsq[m] + beta_t * (1-img_c) + gamma_t*(1-prob_c)
                
                elif weight_map == "DIP+C":
                    w = alpha_t*dsq[m] + beta_t * (1-img_c) + gamma_t*(1-prob_c) + theta_t*dir_penalty[m]
                    
                elif weight_map == "DPC":
                    base = torch.sqrt(alpha_t * dsq[m] + beta_t * dint)
                    prob_penalty = 1.0 / (prob_c + eps)
                    w = base * prob_penalty + theta_t*dir_penalty[m]
                    
                cand = dist_n + w
                sub = best[cx0:cx1, cy0:cy1, cz0:cz1]
                torch.minimum(sub, torch.where(valid, cand, inf), out=sub)

            prev = dist
            dist = torch.minimum(dist, best)
            dist = torch.where(allowed, dist, inf)
            dist[coord] = 0.0

            if tol > 0.0:
                delta = torch.max(torch.abs(prev - dist))
                if not torch.isfinite(delta) or float(delta.item()) < tol:
                    break

        out = dist.detach().cpu().numpy()
        out[~allowed.detach().cpu().numpy()] = np.inf

        self.cost_map = out
        return out

    # ---------------------------------------------------------------------
    # Backtracking in 3D
    # ---------------------------------------------------------------------
    def _backtrack_min_path_3d(self, start_xyz, seeds_mask, max_steps=200000):
        x,y,z = map(int, start_xyz)
        D = self.cost_map
        nx,ny,nz = D.shape

        path = [(x,y,z)]
        last = D[x,y,z]

        # 26 neighbors
        neigh = [(dx,dy,dz)
                 for dx in (-1,0,1)
                 for dy in (-1,0,1)
                 for dz in (-1,0,1)
                 if not (dx==0 and dy==0 and dz==0)]

        for _ in range(max_steps):
            if seeds_mask[x,y,z]:
                break

            nbs = []
            vals = []
            for dx,dy,dz in neigh:
                xx,yy,zz = x+dx, y+dy, z+dz
                if 0<=xx<nx and 0<=yy<ny and 0<=zz<nz:
                    nbs.append((xx,yy,zz))
                    vals.append(D[xx,yy,zz])

            if not vals:
                break

            vals = np.array(vals)
            i_min = int(np.argmin(vals))
            v_min = vals[i_min]

            if not np.isfinite(v_min) or v_min >= last:
                break

            x,y,z = nbs[i_min]
            last = v_min
            path.append((x,y,z))

        return path


    
    def to_all_components_from_ext_3d(self, ignore_labels=(0,), cost_threshold=np.inf):
        """
        Soft version of 'to_all_components_from_ext_3d'.

        - Allows paths to reach distal components if cost map allows.
        - Uses the minimal cost voxel in each component.
        - Optionally applies a cost threshold to avoid large background leaks.
        """

        seeds_mask = np.isfinite(self.cost_map) & (self.cost_map == 0)
        nx, ny, nz = self.cost_map.shape

        coord = self.ext.coord
        from_comp = int(self.label_mask[coord])
        labels = [int(l) for l in np.unique(self.label_mask)
                if l not in ignore_labels and int(l) != from_comp]

        results = {}
        D = self.cost_map
        finite = np.isfinite(D)

        for lab in labels:
            # mask the component and make sure the voxel is reachable
            mask = (self.label_mask == lab) & finite
            if not np.any(mask):
                continue

            cand = np.where(mask, D, np.inf)
            min_cost_idx = np.argmin(cand.ravel())
            min_cost = cand.ravel()[min_cost_idx]

            # ignore if cost too high
            if min_cost > cost_threshold:
                continue

            start = np.unravel_index(min_cost_idx, D.shape)

            # backtrack along the cost map
            path = self._backtrack_min_path_3d(start, seeds_mask)

            results[lab] = {
                "start": tuple(start),
                "min_cost": float(min_cost),
                "path": path,
                "label": lab,
            }

        return results


    # ---------------------------------------------------------------------
    # Tangent & validity checks (same logic as 2D)
    # ---------------------------------------------------------------------
    def compute_path_tan(self, path_points):
        if len(path_points) < 2:
            return (0,0,0)

        pts = np.array(path_points)
        tangents = pts[1:] - pts[:-1]
        avg = np.sum(tangents, axis=0)
        avg = avg / (np.linalg.norm(avg)+1e-8)
        avg = -avg
        return tuple(avg.tolist())

    def _angle_between_tangents(self, t1, t2, degrees=True):
        t1 = np.array(t1, dtype=np.float32)
        t2 = np.array(t2, dtype=np.float32)

        n1 = np.linalg.norm(t1)
        n2 = np.linalg.norm(t2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0

        cosθ = np.clip(np.dot(t1,t2) / (n1*n2), -1,1)
        θ = np.arccos(cosθ)
        return np.degrees(θ) if degrees else θ

    def _compute_intensity_diff_along_path(self, path_points):
        """
        3D version: compute mean |I[i+1] - I[i]| along the geodesic path.
        Returns a single scalar.
        """
        if len(path_points) < 2:
            return 0.0

        values = []
        for i in range(len(path_points) - 1):
            x1, y1, z1 = path_points[i]
            x2, y2, z2 = path_points[i+1]

            if (0 <= x1 < self.img.shape[0] and 
                0 <= y1 < self.img.shape[1] and 
                0 <= z1 < self.img.shape[2] and
                0 <= x2 < self.img.shape[0] and 
                0 <= y2 < self.img.shape[1] and 
                0 <= z2 < self.img.shape[2]):

                v1 = float(self.img[x1, y1, z1])
                v2 = float(self.img[x2, y2, z2])
                values.append(abs(v2 - v1))

        return float(np.mean(values)) if len(values) else 0.0

    def _compute_intensity_diff_ortho_path(self, path_points,
                                       radius=2,
                                       center_radius=1):
        """
        3D orthogonal contrast.
        For each path point:
            - compute tangent
            - define orthogonal plane
            - sample intensities in circle around center
            - compare center vs ring
        Returns: mean contrast along the path.
        """
        if len(path_points) < 3:
            return 0.0

        H, W, D = self.img.shape
        contrasts = []

        pts = np.array(path_points, dtype=np.float32)

        for i in range(1, len(pts)-1):
            p0, p1, p2 = pts[i-1], pts[i], pts[i+1]

            # tangent
            t = p2 - p0
            n = np.linalg.norm(t)
            if n < 1e-8:
                continue
            t = t / n

            # find two orthogonal unit vectors in plane perpendicular to t
            # choose an arbitrary non-collinear vector
            if abs(t[0]) < 0.9:
                a = np.array([1, 0, 0], dtype=np.float32)
            else:
                a = np.array([0, 1, 0], dtype=np.float32)

            # first orthogonal vector
            u = np.cross(t, a)
            u = u / (np.linalg.norm(u) + 1e-8)

            # second orthogonal vector
            v = np.cross(t, u)
            v = v / (np.linalg.norm(v) + 1e-8)

            center_vals = []
            ring_vals = []

            cx, cy, cz = p1.astype(int)

            for rx in range(-radius, radius+1):
                for ry in range(-radius, radius+1):

                    # (rx, ry) in u-v plane
                    offset = rx * u + ry * v
                    xi = int(round(cx + offset[0]))
                    yi = int(round(cy + offset[1]))
                    zi = int(round(cz + offset[2]))

                    if 0 <= xi < H and 0 <= yi < W and 0 <= zi < D:
                        val = float(self.img[xi, yi, zi])

                        dist = np.sqrt(rx*rx + ry*ry)

                        if dist <= center_radius:
                            center_vals.append(val)
                        else:
                            ring_vals.append(val)

            if center_vals and ring_vals:
                c = abs(np.mean(ring_vals) - np.mean(center_vals))
                contrasts.append(c)

        return float(np.mean(contrasts)) if len(contrasts) else 0.0

    def _check_path_validity(self, path_points,
                         max_angle=75, # 60
                         min_along=10,
                         min_ortho=10):
        """
        3D version of the 2D validity logic.
        Returns True / False.
        """

        # ---------- angle check ----------
        ptan = self.compute_path_tan(path_points)
        self.path_tan = ptan

        θ = self._angle_between_tangents(ptan, self.ext.tangent)
        if θ > max_angle:
            print("[ANGLE] Reject: θ=", θ)
            return False

        # ---------- path must reach final target ----------
        coord = self.ext.coord
        if not any(np.array_equal(pt, coord) for pt in path_points):
            print("[COMPLETE] Reject: extremity missing")
            return False

        # ---------- intensity along path ----------
        # along = self._compute_intensity_diff_along_path(path_points)
        # if along < min_along:
        #     print("[ALONG] Reject: insufficient contrast =", along)
        #     return False

        # # ---------- orthogonal contrast ----------
        # ortho = self._compute_intensity_diff_ortho_path(path_points)
        # if ortho < min_ortho:
        #     print("[ORTHO] Reject: insufficient contrast =", ortho)
        #     return False

        return True

    def _select_component_for_endpoint(self, endpoint, direction, candidates):
        """
        Choose the best component for an endpoint using
        geodesic distance + directional constraint.

        Parameters
        ----------
        endpoint : np.ndarray (3,)
        direction : np.ndarray (3,)
            Local tangent direction at endpoint
        candidates : list of dict
            Each dict must contain:
                - "center": np.ndarray (3,)
                - "distance": float (geodesic distance)

        Returns
        -------
        best_candidate or None
        """

        # normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        forward_threshold = 0.0  # change to 0.3 or 0.5 for stricter gating

        valid = []

        for c in candidates:
            vec = c["center"] - endpoint
            vec = vec / (np.linalg.norm(vec) + 1e-8)

            alignment = np.dot(direction, vec)

            # keep only forward-facing components
            if alignment >= forward_threshold:
                valid.append(c)

        if not valid:
            return None

        # choose smallest geodesic distance
        best = min(valid, key=lambda x: x["distance"])

        return best
    
    def to_nearest_geod_component_from_ext_3d(self, ignore_labels=(0,)):
        results = self.to_all_components_from_ext_3d(ignore_labels)
        if not results:
            return None

        endpoint = np.array(self.ext.coord, dtype=np.float32)
        direction = np.array(self.ext.tangent, dtype=np.float32)

        # build candidate list for directional selection
        candidates = []
        for lab, info in results.items():
            pts = np.array(info["path"], dtype=np.float32)
            center = pts[0]  # start of backtracked path = hit component

            candidates.append({
                "label": lab,
                "center": center,
                "distance": info["min_cost"],
                "info": info
            })

        best = self._select_component_for_endpoint(endpoint, direction, candidates)

        if best is None:
            return None

        info = best["info"]
        points = np.array(info["path"])

        self.path = info["path"]
        self.min_cost = info["min_cost"]
        self.target_label = info["label"]
        self.path_valid = self._check_path_validity(points)


class GeodesicPaths3D():
    
    """
    GeodesicPaths2D combining all the paths from all the GeodesicPath2d
    """
    
    def __init__(self, _extremities: Extremities3D,img,label_mask,prob_mask):
        self.extremities = _extremities
        self.img = img 
        self.label = label_mask
        self.prob_mask = prob_mask
        self.paths: list = []
        self.cost_maps: list = []
        # self.graphs: Dict[int, nx.Graph] = {}
        
    def compute_all_paths(self, default_params={
        "alpha": 1,
        "beta": 1,
        "gamma": 1,
        "lambda": 0.35,
        "spacing": (1, 1, 1),
        "deg": 60,
        "max_it": 60,
        "weight_map": "DIP+",
        "device": "cuda"
        }):
        
        """Compute paths to all the possible paths (using all the identified extremities)"""

        all_paths = []
        all_cost_maps = []

        print("default params : ", default_params)

        # Compute total number of extremities for progress bar
        total_ext = sum(len(self.extremities.extremities[int(comp_id)])
                        for comp_id in self.extremities.extremities)

        with tqdm(total=total_ext, desc="Computing geodesic paths") as pbar:
            for comp_id in self.extremities.extremities:
                comp_id = int(comp_id)

                for ext_id, ext_coords in enumerate(self.extremities.extremities[comp_id]):
                    ext = self.extremities.extremities[comp_id][ext_id]

                    path_obj = GeodesicPath3D(
                        ext,
                        self.img,
                        self.label,
                        self.prob_mask
                    )

                    path_obj.geodesic_distance_torch_3d(
                        lambda_dir=default_params['lambda'],
                        alpha=default_params['alpha'],
                        beta=default_params["beta"],
                        n_it=default_params["max_it"],
                        device=default_params["device"],
                        weight_map=default_params["weight_map"]
                    )

                    path_obj.to_nearest_geod_component_from_ext_3d()

                    all_paths.append(path_obj)
                    all_cost_maps.append(path_obj.cost_map)

                    pbar.update(1)

        self.paths = all_paths
        self.cost_maps = all_cost_maps

        return None
    
    def duplicates(self)->list:
        connections = []
        for path in self.paths:
            if path.path_valid is False:
                continue
            else:
                connection = [int(path.ext.comp_id),int(path.target_label)]
                sorted_connection = sorted(connection)
                connections.append(sorted_connection)
        return connections
    
    def filter_duplicates(self)->list:
        """Return unique undirected connections keeping the shortest path.

        For every pair of component ids (undirected), keep the single path
        with the smallest `min_cost`. Replace `self.paths` with the selected
        shortest paths and return the list of unique connections.
        """
        # Select the shortest path (fewest points) for each undirected connection.
        best_for_connection: Dict[tuple, tuple] = {}

        for path in self.paths:
            if path.path_valid is False:
                continue

            a = int(path.ext.comp_id)
            b = int(path.target_label)
            
            # normalize as undirected connection
            connection = (min(a, b), max(a, b))

            # selection metric: number of points in the path (shorter is better)
            try:
                cost = int(len(path.path))
            except Exception:
                # if path.path is missing or not sized, treat as very long
                cost = int(1e9)

            if connection not in best_for_connection:
                best_for_connection[connection] = (path, cost)
            else:
                _, best_cost = best_for_connection[connection]
                if cost < best_cost:
                    best_for_connection[connection] = (path, cost)

        # assemble results preserving a deterministic order (sorted by connection)
        unique_connections = []
        unique_paths = []
        for conn in sorted(best_for_connection.keys()):
            path, _ = best_for_connection[conn]
            unique_connections.append([int(conn[0]), int(conn[1])])
            unique_paths.append(path)

        self.paths = unique_paths
        return unique_connections

    def export_cost_maps(self):
        if len(self.cost_maps) != 0:
            merged_cost_maps = np.full_like(self.img, np.inf)
            for _map in self.cost_maps:
                merged_cost_maps = np.minimum(merged_cost_maps, _map)

            return merged_cost_maps
        else:
            return None
# -------------------------------
# Geodesic distance (PyTorch, Jacobi) for 2D
# -------------------------------

