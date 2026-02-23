import numpy as np
import networkx as nx
import SimpleITK as sitk

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from skimage.morphology import skeletonize
from scipy.ndimage import generate_binary_structure
from scipy import ndimage


# -------------------------------------------- 3D extremities detection classes --------------------------------------------

@dataclass
class Extremity3D:
    """
    Represents a single extremity (endpoint) of a 3D vessel component.
    Stores:
        - coordinate (z, y, x)
        - parent component ID
        - tangent vector (3D)
        - estimated radius
    """
    coord: Tuple[int, int, int]
    comp_id: int
    tangent: Optional[np.ndarray] = None
    radius: Optional[float] = None

    def as_dict(self):
        return {
            "coordinate": tuple(self.coord),
            "tangent": None if self.tangent is None else self.tangent.copy(),
            "radius": self.radius,
            "comp_id": self.comp_id
        }


class Extremities3D:
    """
    Full 3D analogue of Extremities2D.
    Handles:
        - skeletonization (3D)
        - graph construction (26-conn)
        - extremity detection (deg 1 nodes)
        - tangent computation
        - radius estimation with orthogonal sampling
    """

    def __init__(self, labeled_mask: np.ndarray):
        self.labeled_mask = labeled_mask.astype(np.uint8)
        self.extremities: Dict[int, List[Extremity3D]] = {}
        self.graphs: Dict[int, nx.Graph] = {}
        self.tangents: Dict[int, List[np.ndarray]] = {}
        self.radii: Dict[int, List[float]] = {}

    @staticmethod
    def build_skeleton_from_comp(comp: np.ndarray) -> np.ndarray:
        """3D skeletonization."""
        return skeletonize(comp)
    
    # @staticmethod
    # def build_skeleton_from_comp(binary_img):
    #     # naive iterative erosion to get "skeleton-like" voxels
    #     structure = np.ones((3,3,3))
    #     skeleton = np.zeros_like(binary_img)
    #     tmp = binary_img.copy()
    #     while tmp.sum() > 0:
    #         eroded = ndimage.binary_erosion(tmp, structure=structure)
    #         skeleton |= (tmp & ~eroded)
    #         tmp = eroded
    #     return skeleton

    @staticmethod
    def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
        """Convert 3D skeleton voxels into a 26-connected graph."""
        G = nx.Graph()
        conn = generate_binary_structure(3, 3)    # 26-connected
        coords = np.transpose(np.nonzero(skeleton))
        coord_to_id = {tuple(c): i for i, c in enumerate(coords)}

        for i, coord in enumerate(coords):
            G.add_node(i, pos=tuple(coord))
            for delta in np.argwhere(conn) - 1:
                neigh = tuple(coord + delta)
                if neigh != tuple(coord) and neigh in coord_to_id:
                    G.add_edge(i, coord_to_id[neigh])

        return G

    def build_graphs_for_components(self, verbose=False):
        labels = np.unique(self.labeled_mask)
        labels = labels[labels != 0]

        for lbl in labels:
            if verbose:
                print(f"[3D] Processing component {lbl}")
            comp = (self.labeled_mask == lbl)

            # --------------------------------------------------
            # Case 0 — empty component (should not happen, but safe)
            # --------------------------------------------------
            if np.count_nonzero(comp) == 0:
                self.graphs[lbl] = nx.Graph()
                self.extremities[lbl] = []
                continue

            # --------------------------------------------------
            # Skeletonization
            # --------------------------------------------------
            skel = self.build_skeleton_from_comp(comp)

            # --------------------------------------------------
            # Case 1 — NO skeleton voxels at all
            # --------------------------------------------------
            if np.count_nonzero(skel) == 0:
                if verbose:
                    print(f"component {lbl} has NO skeleton voxels.")

                coords = np.argwhere(comp)

                # single-voxel component
                if len(coords) == 1:
                    c = tuple(coords[0])
                    self.graphs[lbl] = nx.Graph()
                    self.extremities[lbl] = [Extremity3D(coord=c, comp_id=lbl)]
                    continue

                # centroid–farthest fallback
                center = coords.mean(axis=0)
                dists = np.linalg.norm(coords - center, axis=1)
                c = tuple(coords[np.argmax(dists)])

                self.graphs[lbl] = nx.Graph()
                self.extremities[lbl] = [Extremity3D(coord=c, comp_id=lbl)]
                continue

            # --------------------------------------------------
            # Skeleton → graph
            # --------------------------------------------------
            G = self.skeleton_to_graph(skel)
            self.graphs[lbl] = G

            # --------------------------------------------------
            # Normal endpoint detection
            # --------------------------------------------------
            endpoints = [n for n, d in G.degree() if d == 1]

            # --------------------------------------------------
            # Case 2 — skeleton exists but no endpoints (loops, blobs)
            # --------------------------------------------------
            if len(endpoints) == 0:
                if verbose:
                    print(f"component {lbl} has no endpoints yet.")

                coords = np.array([G.nodes[n]['pos'] for n in G.nodes])

                # single skeleton voxel
                if len(coords) == 1:
                    if verbose:
                        print("single voxel")
                    endpoints = [list(G.nodes)[0]]

                else:
                    if verbose:
                        print("extremity farthest strategy")
                    center = coords.mean(axis=0)
                    dists = np.linalg.norm(coords - center, axis=1)
                    farthest_node = list(G.nodes)[np.argmax(dists)]
                    endpoints = [farthest_node]

            # --------------------------------------------------
            # Store extremities
            # --------------------------------------------------
            ep_coords = [tuple(G.nodes[n]['pos']) for n in endpoints]
            self.extremities[lbl] = [
                Extremity3D(coord=c, comp_id=lbl) for c in ep_coords
            ]

            if verbose:
                print(f"[3D]   → Found {len(ep_coords)} extremities")

    # def build_graphs_for_components(self, verbose=False):
    #     labels = np.unique(self.labeled_mask)
    #     labels = labels[labels != 0]

    #     for lbl in labels:
    #         print(lbl)
    #         if verbose:
    #             print(f"[3D] Processing component {lbl}")
    #         comp = (self.labeled_mask == lbl)

    #         skel = self.build_skeleton_from_comp(comp)
    #         G = self.skeleton_to_graph(skel)
    #         self.graphs[lbl] = G

    #         endpoints = [n for n, d in G.degree() if d == 1]

    #         # --------------------------------------------------
    #         # Fallback for small / loop components
    #         # --------------------------------------------------
    #         if len(endpoints) == 0 and len(G.nodes) > 0:
    #             print(f"component {lbl} has no endpoints yet.")
    #             # get skeleton coordinates
    #             coords = np.array([G.nodes[n]['pos'] for n in G.nodes])

    #             if len(coords) == 1:
    #                 print("single voxel")
    #                 # single voxel component
    #                 endpoints = [list(G.nodes)[0]]
                    

    #             else:
    #                 print("extremity farthest strategy")
    #                 # centroid-farthest strategy
    #                 center = coords.mean(axis=0)
    #                 dists = np.linalg.norm(coords - center, axis=1)
    #                 farthest_node = list(G.nodes)[np.argmax(dists)]
    #                 endpoints = [farthest_node]

    #         ep_coords = [tuple(G.nodes[n]['pos']) for n in endpoints]
    #         self.extremities[lbl] = [Extremity3D(coord=c, comp_id=lbl) for c in ep_coords]

    #         if verbose:
    #             print(f"[3D]   → Found {len(ep_coords)} extremities")

    def compute_tangent_at_extremity(self, ext: Extremity3D, nb_neighbors=1, verbose=False):
        coord = ext.coord
        lbl = self.labeled_mask[coord]

        if lbl not in self.graphs:
            raise ValueError(f"No graph available for label {lbl}")
        
        # --------------------------------------------------
        # SAFETY: no graph or empty graph
        # --------------------------------------------------
        if lbl not in self.graphs:
            ext.tangent = np.zeros(3)
            return ext.tangent

        G = self.graphs[lbl]

        if G.number_of_nodes() == 0:
            if verbose:
                print(f"[3D] Component {lbl}: empty graph → zero tangent")
            ext.tangent = np.zeros(3)
            return ext.tangent

        G = self.graphs[lbl]
        # Find nearest graph node
        ext_node = min(G.nodes, key=lambda n: np.linalg.norm(np.array(G.nodes[n]['pos']) - np.array(coord)))
        p0 = np.array(G.nodes[ext_node]['pos'], float)

        # 1-step neighbor
        if nb_neighbors == 1:
            nbrs = list(G.neighbors(ext_node))
            if len(nbrs) == 0:
                ext.tangent = np.zeros(3)
                return ext.tangent

            neigh = min(nbrs, key=lambda n: np.linalg.norm(np.array(G.nodes[n]['pos']) - p0))
            p1 = np.array(G.nodes[neigh]['pos'], float)
            t = p1 - p0
            n = np.linalg.norm(t)
            ext.tangent = -(t / n) if n > 0 else np.zeros(3)

            return ext.tangent

        # PCA-based multi-point tangent
        visited = {ext_node}
        queue = [ext_node]
        pts = [p0]

        while queue and len(pts) < nb_neighbors + 1:
            node = queue.pop(0)
            for neigh in G.neighbors(node):
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)
                    pts.append(np.array(G.nodes[neigh]['pos'], float))
                    if len(pts) >= nb_neighbors + 1:
                        break

        pts = np.array(pts)
        if len(pts) <= 1:
            ext.tangent = np.zeros(3)
            return ext.tangent

        X = pts - pts.mean(axis=0)
        _, _, vh = np.linalg.svd(X)
        direction = vh[0]

        # orient outward
        if np.dot(direction, pts[-1] - p0) < 0:
            direction = -direction

        ext.tangent = -direction / np.linalg.norm(direction)
        return ext.tangent

    def compute_tangents_for_all_extremities(self, nb_neighbors=1):
        for lbl, eps in self.extremities.items():
            self.tangents[lbl] = []
            for ext in eps:
                t = self.compute_tangent_at_extremity(ext, nb_neighbors)
                self.tangents[lbl].append(t)

    def compute_radius_orthogonal_mask(self, ext: Extremity3D, max_r=8, n_angles=16, verbose=False):
        lbl = ext.comp_id
        coord = ext.coord
        tangent = ext.tangent

        if tangent is None or np.allclose(tangent, 0):
            ext.radius = 0.0
            return 0.0

        bin_mask = (self.labeled_mask == lbl)
        zmax, ymax, xmax = bin_mask.shape
        z0, y0, x0 = map(float, coord)

        # Build orthonormal basis
        ref = np.array([1, 0, 0], float)
        if np.allclose(abs(np.dot(ref, tangent)), 1.0):
            ref = np.array([0, 1, 0], float)

        v1 = np.cross(tangent, ref); v1 /= np.linalg.norm(v1)
        v2 = np.cross(tangent, v1);  v2 /= np.linalg.norm(v2)

        radii = []
        for k in range(n_angles):
            angle = 2 * np.pi * k / n_angles
            d = np.cos(angle) * v1 + np.sin(angle) * v2

            for r in range(1, max_r + 1):
                z = int(round(z0 + r * d[0]))
                y = int(round(y0 + r * d[1]))
                x = int(round(x0 + r * d[2]))

                if (z < 0 or z >= zmax or
                    y < 0 or y >= ymax or
                    x < 0 or x >= xmax or
                    not bin_mask[z, y, x]):
                    radii.append(r - 1)
                    break
            else:
                radii.append(max_r)

        ext.radius = float(np.median(radii))
        return ext.radius

    def compute_radii_for_all_extremities_mask(self, max_r=8):
        for lbl, eps in self.extremities.items():
            self.radii[lbl] = []
            for ext in eps:
                r = self.compute_radius_orthogonal_mask(ext, max_r=max_r)
                self.radii[lbl].append(r)

    def get_specs_single_extremity(self, comp_id, ext_id):
        return self.extremities[comp_id][ext_id].as_dict()

    def get_radius(self,comp_id, ext_id):
        return self.extremities[comp_id][ext_id].radius

    
