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
# -------------------------------------------- 2D extremities detection classes --------------------------------------------

@dataclass
class Extremity2D:
    """
    Represents a single extremity (endpoint) of a 2D vessel component.
    
    Stores:
        - coordinate on the mask
        - parent component ID
        - tangent vector (optional)
        - estimated radius (optional)
        - reference to the component's graph
    """
    
    coord: Tuple[int, int]
    comp_id: int
    tangent: Optional[np.ndarray] = None      # 2D tangent vector (unit)
    radius: Optional[float] = None            # Radius estimated at extremity

    def as_dict(self):
        """Return a clean spec dictionary compatible with your previous design."""
        return {
            "coordinate": self.coord,
            "tangent": None if self.tangent is None else self.tangent.copy(),
            "radius": self.radius,
            "comp_id": self.comp_id
        }

class Extremities2D:
    """
    Base class to hold extremity information for labeled vessel components in 2D.

    This class is designed to store the labeled mask, detected extremities, skeleton graphs,
    and tangent vectors for each connected component in a vessel segmentation mask.
    
    Attributes:
        labeled_mask (np.ndarray): 2D array with labeled connected components.
        extremities (dict[int, list[tuple]]): Mapping from label to list of extremity coordinates.
        graphs (dict[int, nx.Graph]): Mapping from label to skeleton graph (networkx).
        tangents (dict[int, list[tuple]]): Mapping from label to list of (extremity, tangent) pairs.
    """
    def __init__(self, labeled_mask: np.ndarray):
        self.labeled_mask = labeled_mask #.astype(np.uint8) # label_mask becomes components
        self.extremities: Dict[int, List[Extremity2D]] = {}
        self.graphs: Dict[int, nx.Graph] = {}

    @staticmethod
    def build_skeleton_from_comp(comp: np.ndarray) -> np.ndarray:
        """Skeletonize a binary component mask."""
        return skeletonize(comp)

    @staticmethod
    def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
        G = nx.Graph()
        connectivity = generate_binary_structure(2, 2)
        coords = np.transpose(np.nonzero(skeleton))
        coord_to_node = {tuple(c): i for i, c in enumerate(coords)}
        for i, coord in enumerate(coords):
            G.add_node(i, pos=tuple(coord))
            for delta in np.argwhere(connectivity) - 1:
                neighbor = tuple(coord + delta)
                if neighbor in coord_to_node and neighbor != tuple(coord):
                    G.add_edge(i, coord_to_node[neighbor])
        return G

    def build_graphs_for_components(self, verbose=False):
        """Compute skeleton graphs and endpoints for all labeled components."""
        labels = np.unique(self.labeled_mask)
        labels = labels[labels != 0]
        
        for label_val in labels:
            if verbose:
                print(f"Processing label {label_val}")
            component_mask = (self.labeled_mask == label_val)
            skeleton = self.build_skeleton_from_comp(component_mask)
            G = self.skeleton_to_graph(skeleton)
            self.graphs[label_val] = G
            endpoints = [n for n, deg in G.degree() if deg == 1]
            endpoint_coords = [tuple(G.nodes[n]['pos']) for n in endpoints]
            self.extremities[label_val] = [Extremity2D(coord=c,comp_id=label_val) for c in endpoint_coords] # append the Extremity2D object initiate with the coordinate (c).
            if verbose:
                print(f"[INFO] Label {label_val}: {len(endpoint_coords)} extremities found")

    def compute_tangent_at_extremity(self, ext = Extremity2D, nb_neighbors=1, verbose=False):
        """
        For each endpoint, get its neighbor and compute the tangent direction.
        """
        ext_coord = ext.coord # collect the extremity coordinate
        label = self.labeled_mask[ext_coord[0], ext_coord[1]]
        
        if self.graphs == {} or label not in self.graphs:
            raise ValueError("No graphs available; please compute them.")
        
        graph = self.graphs[label]

        # Find the nearest graph node to the given extremity coordinate
        ext_node = min(
            graph.nodes,
            key=lambda n: np.linalg.norm(np.array(graph.nodes[n]['pos']) - np.array(ext_coord))
        )
        start_pos = np.array(graph.nodes[ext_node]['pos'], dtype=float)

        if nb_neighbors == 1:
            
            nbrs = list(graph.neighbors(ext_node))
            if len(nbrs) == 0:
                if verbose:
                    print("No neighbor")
                return np.array([0, 0], dtype=float)
            # Choose the closest neighbor in Euclidean space
            neighbor = min(nbrs, key=lambda n: np.linalg.norm(np.array(graph.nodes[n]['pos'], dtype=float) - start_pos))
            neighbor_pos = np.array(graph.nodes[neighbor]['pos'], dtype=float)
            tangent = neighbor_pos - start_pos
            norm = np.linalg.norm(tangent)
            tangent = tangent / norm if norm > 0 else np.array([0, 0], dtype=float)
            if verbose:
                print("The tangent of the extremity", tuple(map(int, ext_coord)), "of the component", int(label), "is", tangent)
            ext.tangent = -tangent
            return ext.tangent

        elif nb_neighbors > 1:
            # BFS walk along the skeleton to collect the first nb_neighbors nodes
            visited = {ext_node}
            queue = [(ext_node, 0)]
            path_positions = [start_pos]

            while queue and len(path_positions) < nb_neighbors + 1:
                node, dist = queue.pop(0)
                for neigh in graph.neighbors(node):
                    if neigh not in visited:
                        visited.add(neigh)
                        queue.append((neigh, dist + 1))
                        path_positions.append(np.array(graph.nodes[neigh]['pos'], dtype=float))
                        if len(path_positions) >= nb_neighbors + 1:
                            break

            path_positions = np.array(path_positions)
            if path_positions.shape[0] <= 1:
                ext.tangent = np.array([0, 0], dtype=float)
                return ext.tangent
                

            # PCA for direction
            X = path_positions - path_positions.mean(axis=0)
            _, _, vh = np.linalg.svd(X, full_matrices=False)
            direction = vh[0]
            # print(direction)
            
            # Orient away from the extremity
            if np.dot(direction, path_positions[-1] - start_pos) < 0:
                direction = -direction

            norm = np.linalg.norm(direction)
            tangent = direction / norm if norm > 0 else np.array([0, 0], dtype=float)
            if verbose:
                print("The tangent of the extremity", tuple(map(int, ext_coord)), "of the component", int(label), "is", tangent)
            ext.tangent = -tangent
            return ext.tangent  # invert direction to point outward

        else:
            if verbose:
                print("No neighbor")
                ext.tangent = np.array([0, 0], dtype=float)
                return ext.tangent
                
    def compute_tangents_for_all_extremities(self, nb_neighbors=1):
        _tangents = {}
        for comp_id, exts in self.extremities.items():
            list_tangents = []
            for ext in exts:
                temp_tangent = self.compute_tangent_at_extremity(ext, nb_neighbors=nb_neighbors)
                list_tangents.append(temp_tangent)
            _tangents[comp_id] = list_tangents
        self.tangents = _tangents
        return None

    def compute_radius_orthogonal_mask(self, ext: Extremity2D, max_r=8, verbose=False):
        """
        Estimate vessel radius at an extremity using the orthogonal direction
        to the centerline tangent, based on the binary labeled mask only.
        """
        ext_coord = ext.coord
        label = int(self.labeled_mask[ext_coord[0], ext_coord[1]])
        if label == 0:
            if verbose:
                print("Extremity not inside any labeled region.")
            ext.radius = 0.0
            return ext.radius

        if label not in self.tangents or label not in self.extremities:
            raise ValueError(f"No tangents/extremities computed for label {label}.")

        # Find index of the extremity in the component by matching coordinates
        coords = [tuple(map(int, e.coord)) for e in self.extremities[label]]
        try:
            idx = coords.index(tuple(map(int, ext_coord)))
        except ValueError:
            if verbose:
                print(f"Extremity {ext_coord} not found in label {label}.")
            ext.radius = 0.0
            return ext.radius

        tangent = np.array(self.tangents[label][idx], dtype=float)
        if np.allclose(tangent, 0):
            if verbose:
                print(f"Tangent is zero for extremity {ext_coord}.")
            ext.radius = 0.0
            return ext.radius

        # Compute orthogonal direction
        ortho = np.array([-tangent[1], tangent[0]])
        ortho /= np.linalg.norm(ortho)

        bin_mask = (self.labeled_mask == label)
        h, w = bin_mask.shape
        y0, x0 = map(float, ext_coord)

        # Walk along orthogonal directions until mask boundary
        def walk_distance(direction):
            for r in range(1, max_r + 1):
                y_idx = int(round(float(y0) + r * float(direction[0])))
                x_idx = int(round(float(x0) + r * float(direction[1])))
                if y_idx < 0 or y_idx >= h or x_idx < 0 or x_idx >= w or not bin_mask[y_idx, x_idx]:
                    return r - 1
            return max_r

        r1 = walk_distance(ortho)
        r2 = walk_distance(-ortho)
        radius = (r1 + r2) / 2.0

        if verbose:
            print(f"Extremity {tuple(map(int, ext_coord))}: r1={r1}, r2={r2}, radius={radius:.2f}")

        ext.radius = float(radius)
        return ext.radius

    def compute_radii_for_all_extremities_mask(self, max_r=8):
        """Compute orthogonal mask-based radii for all extremities."""
        # _radii = {}
        # for comp_id, exts in self.extremities.items():
        #     list_radii = []
        #     for ext in exts:
        #         r = self.compute_radius_orthogonal_mask(ext, max_r=max_r)
        #         list_radii.append(r)
        #     _radii[comp_id] = list_radii
        # self.radii = _radii
        # return None
        for comp_id, exts in self.extremities.items():
            for ext in exts:
                self.compute_radius_orthogonal_mask(ext, max_r=max_r)

    def get_specs_single_extremity(self, comp_id, ext_id):
        """
        Retrieve the specifications of a single extremity given its component ID and index.
        
        Args:
            comp_id (int): The label of the connected component.
            ext_id (int): The index of the extremity within the component's extremities list."""
        
        if comp_id not in self.extremities:
            raise ValueError(f"Component ID {comp_id} not found.")
        if ext_id < 0 or ext_id >= len(self.extremities[comp_id]):
            raise IndexError(f"Extremity index {ext_id} out of range for component {comp_id}.")
        
        return self.extremities[comp_id][ext_id].as_dict()
    
    def get_graph_matrices(graph):
        """
        Convert NetworkX graph to different matrix representations
        Returns dict with adjacency, distance, and coordinate matrices
        """
        # Get number of nodes and create mapping
        n_nodes = len(graph.nodes())
        node_map = {node: i for i, node in enumerate(graph.nodes())}
        
        # Initialize matrices
        adj_matrix = nx.to_numpy_array(graph)
        dist_matrix = np.zeros((n_nodes, n_nodes))
        coord_matrix = np.zeros((n_nodes, 2))  # For 2D coordinates
        
        # Fill coordinate matrix
        for node in graph.nodes():
            idx = node_map[node]
            pos = graph.nodes[node]['pos']
            coord_matrix[idx] = pos
            
        # Fill distance matrix
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and adj_matrix[i,j] > 0:
                    pos_i = coord_matrix[i]
                    pos_j = coord_matrix[j]
                    dist_matrix[i,j] = np.linalg.norm(pos_i - pos_j)
        
        return {
            'adjacency': adj_matrix,
            'distances': dist_matrix,
            'coordinates': coord_matrix
        }
    
    def get_radius(self,comp_id, ext_id):
        return self.extremities[comp_id][ext_id].radius
        
if __name__ == "__main__":
    # # Example usage
    # from PIL import Image
    from scipy.ndimage import label
    import os
    import nibabel as nib

    # patient_id = '1' # 15 1
    # # Paths to the image, segmentation, and probability map files
    # # These paths should be updated to point to the actual files
    # # in your environment.

    # image_path = f'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/2D_RETINAL/Dataset001_Retinal/imagesTr/patient_{patient_id}_0000.png'#'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/PADSET_BREST/input/{patient_id}_img_cropped.nii.gz'
    # segmentation_path = f'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/2D_RETINAL/nnUNet_preprocessed/Dataset001_Retinal/gt_segmentations/patient_{patient_id}.png'#'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/PADSET_BREST/input/{patient_id}_seg_gt_cropped.nii.gz'
    # prob_map_path = f'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/2D_RETINAL/nnUNet_results/Dataset001_Retinal/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation/patient_{patient_id}.png' #'/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/PADSET_BREST/input/{patient_id}_prob_map_artery_cropped.npy'

    # # Load the 3D image, segmentation, and probability map (load as 2D image)
    # img = Image.open(image_path)
    # seg_gt = Image.open(segmentation_path)
    # seg = Image.open(prob_map_path)

    # img_data = np.array(img)
    # seg_gt_data = np.array(seg_gt)
    # seg_data = np.array(seg)
    
    # # Assuming label_mask is your labeled mask (NumPy array)
    # structure = np.ones((3, 3), dtype=int)  # connectivity
    # label_mask, num_features = label(seg_data, structure=structure) # seg data
    # filter_sizes = np.bincount(label_mask.ravel())
    # # Remove small objects (e.g., size < 10 pixels)
    # min_size = 10
    # for i in range(1, num_features + 1):
    #     if filter_sizes[i] < min_size:
    #         label_mask[label_mask == i] = 0
    # # Re-label the mask after removing small objects
    # _label_mask, num_features = label(label_mask > 0, structure=structure) 

    # # Extract extremities
    # print('extract all extremities ...')
    # cl_ext = Extremities2D(labeled_mask=_label_mask)
    # cl_ext.build_graphs_for_components()#verbose=True
    # # for nb_neighbors in [1,5,10]:
    #     # print(f'compute tangents with nb_neighbors={nb_neighbors} ...')
    #     # cl_ext.compute_tangents_for_all_extremities(nb_neighbors=nb_neighbors)
    #     # print(cl_ext.tangents[1][0])
    # cl_ext.compute_tangents_for_all_extremities(nb_neighbors=3)  
    # cl_ext.compute_radii_for_all_extremities_mask(max_r=10)
    # # print(cl_ext.extremities)
    # comp_id = 2
    # ext_id = 0
    # print(cl_ext.get_specs_single_extremity(comp_id,ext_id))
    
    
    # # specs = cl_ext.get_specs_single_extremity(comp_id=1, ext_index=0)
    # # matrices = cl_ext.get_graph_matrices(specs['graph'])
    # # print("Adjacency Matrix:\n", matrices['adjacency'])
    # # print("Distance Matrix:\n", matrices['distances'])
    # # print("Coordinate Matrix:\n", matrices['coordinates'])
    # ---------------------------------------------------------
    # Main
    # ---------------------------------------------------------

    patient_id = "109"
    outdir = "./benchmark_gmm_results"
    prob_thresh = 0.2

    print("loading images ...")
    
    img_path = f"/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/PADSET_BREST/input/nnUNet_raw/Dataset001_BRT/imagesTr/patient_{patient_id}_0000.nii.gz"
    seg_path = f"/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/PADSET_BREST/input/nnUNet_results/Dataset001_BRT/nnUNetTrainerCustomLoss__nnUNetPlans__3d_fullres/fold_0/validation/patient_{patient_id}.nii.gz"
    gt_path  = f"/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/PADSET_BREST/input/correction_cropped_GT_02122025/TOT_{patient_id}_seg_gt_cropped.nii.gz"
    seg_prob_path = f"/srv/storage/epione@storage2.sophia.grid5000.fr/sgoffart/datasets/PADSET_BREST/input/nnUNet_results/Dataset001_BRT/nnUNetTrainerCustomLoss__nnUNetPlans__3d_fullres/fold_0/validation/patient_{patient_id}.npz"

    img = nib.load(img_path).get_fdata().astype(np.float32)
    # img = (img - img.min()) / (img.max() - img.min() + 1e-8) # normalize to [0,1]

    seg = nib.load(seg_path).get_fdata() > 0
    gt  = nib.load(gt_path).get_fdata() > 0
    seg_prob_arr = np.load(seg_prob_path.format(patient_id), allow_pickle=True)['probabilities']
    seg_prob_arr = seg_prob_arr[1,:,:,:]
    seg_prob_arr_prior_tp = seg_prob_arr.copy()
    seg_prob_arr = np.transpose(seg_prob_arr, (2, 1, 0))
    
    # Label connected components
    structure = np.ones((3, 3, 3), dtype=int)
    seg_label, num_features = label(seg,structure = structure)
    
    outdir = os.path.join(outdir, f"patient_{patient_id}")
    os.makedirs(outdir, exist_ok=True)

    # compute extremities
    print("compute extremities ...")
    exts = Extremities3D(seg_label)
    exts.build_graphs_for_components()
    # exts.compute_tangents_for_all_extremities()
    
    comp_id = 4
    ext_id = 0
    ext = exts.extremities[comp_id][ext_id]
    print(ext)
    print('\n')
    # print(exts.extremities)


    
