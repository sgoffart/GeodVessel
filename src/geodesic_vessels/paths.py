import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
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
        """Compute the 3D geodesic distance map using a Jacobi iteration.

        The algorithm is implemented in PyTorch to allow optional GPU
        acceleration.  A variety of weight maps are supported (``D``, ``DI``,
        ``DIP+`` etc.) combining anisotropic spatial distances, intensity
        differences and probability penalties. Directional penalties and
        anisotropic spacing are accounted for via ``_neighbor_offsets_torch_3d``.

        Parameters mirror those in the original 2D implementation:
        ``alpha``/``beta`` control spatial vs intensity weighting, ``gamma``
        handles probability penalties, ``lambda_dir`` adds directional cost,
        ``n_it`` is number of Jacobi iterations, ``tol`` stops early, and
        ``weight_map`` selects the formula. The source extremity coordinates
        are taken from ``self.ext``. After computation the resulting numpy
        array is stored in ``self.cost_map`` and returned.
        """

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
                    prob_penalty = 1.0 / (prob_c + eps)
                    w = base * prob_penalty
                elif weight_map == "DI":
                    w = torch.sqrt(alpha_t * dsq[m] + beta_t * dint)
                
                elif weight_map == "D":
                    w = dsq[m]          
                    
                elif weight_map == "P": # probability only
                    # w = torch.sqrt(gamma * prob_diff)
                    w=(1 + gamma * prob_diff)
                    
                elif weight_map == "DIP+":
                    w = alpha_t*dsq[m] + beta_t * dint + gamma_t*(1/prob_c+eps)
                    
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
        """Trace a minimal-cost path backwards from a starting voxel.

        Given a pre-computed ``self.cost_map``, the method walks voxel-by-voxel
        to the nearest low-cost seed (where ``seeds_mask`` is True) by
        selecting the neighbor with smallest cost at each step.  The search
        stops when a seed is reached, a non-finite cost is encountered, or the
        maximum number of steps is exceeded. Returns the list of coordinates
        visited.
        """
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
        """Compute minimal-cost paths from the current extremity to every other label.

        This method searches the precomputed ``self.cost_map`` for the lowest
        finite-cost voxel within each component (excluding those in
        ``ignore_labels`` or the source component). For each reachable
        component it backtracks a minimal energy path from that voxel to the
        extremity seed using :meth:`_backtrack_min_path_3d`.

        Parameters
        ----------
        ignore_labels : sequence, optional
            Labels to skip entirely (default includes 0 for background).
        cost_threshold : float, optional
            Discard target components whose minimum cost exceeds this value,
            preventing paths that would leak excessively into background.

        Returns
        -------
        dict
            Mapping label → info dict containing ``start`` (voxel coords),
            ``min_cost``, ``path`` (list of coords) and ``label``.
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
        """Return the average tangent direction of a polyline.

        The tangent is computed as the normalized sum of successive
        difference vectors along ``path_points``. The result is negated so that
        it points from the end toward the start (matching the extremity
        orientation used elsewhere).
        """
        if len(path_points) < 2:
            return (0,0,0)

        pts = np.array(path_points)
        tangents = pts[1:] - pts[:-1]
        avg = np.sum(tangents, axis=0)
        avg = avg / (np.linalg.norm(avg)+1e-8)
        avg = -avg
        return tuple(avg.tolist())

    def _angle_between_tangents(self, t1, t2, degrees=True):
        """Compute the angle between two tangent vectors.

        Returns the angle in degrees by default, or radians if
        ``degrees=False``. Handles zero-length vectors gracefully by
        returning 0.
        """
        t1 = np.array(t1, dtype=np.float32)
        t2 = np.array(t2, dtype=np.float32)

        n1 = np.linalg.norm(t1)
        n2 = np.linalg.norm(t2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0

        cosθ = np.clip(np.dot(t1,t2) / (n1*n2), -1,1)
        θ = np.arccos(cosθ)
        return np.degrees(θ) if degrees else θ

    def _check_path_validity(self, path_points,
                         max_angle=60):
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
        """Select the closest reachable component and store its path info.

        Calls :meth:`to_all_components_from_ext_3d` to compute minimal-cost
        paths to all other labels, then applies a directional filter based on
        the extremity tangent to pick the most forward-facing target.  The
        chosen path and associated metadata are stored on ``self``
        (``path``, ``min_cost``, ``target_label`` and ``path_valid``).
        Returns None if no suitable target is found.
        """
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
    """Container for computing and managing multiple geodesic paths.

    Each instance wraps an :class:`Extremities3D` object along with the
    corresponding image, label and probability volumes.  Paths are stored in
    ``self.paths`` after computation and associated cost maps in
    ``self.cost_maps``.
    """
    
    def __init__(self, _extremities: Extremities3D,img,label_mask,prob_mask):
        """Initialize with extremities handler and image volumes."""
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
        """Return list of undirected connections present in computed paths.

        Each connection is represented as a sorted two-element list of
        component IDs. Paths marked invalid are ignored.
        """
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
        """Merge all individual cost maps into a single minimum-cost volume.

        Returns a volume where each voxel holds the smallest value over all
        ``self.cost_maps`` entries, or ``None`` if no maps are available.
        """
        if len(self.cost_maps) != 0:
            merged_cost_maps = np.full_like(self.img, np.inf)
            for _map in self.cost_maps:
                merged_cost_maps = np.minimum(merged_cost_maps, _map)

            return merged_cost_maps
        else:
            return None


