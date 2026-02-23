import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class FastDIPOptimizer:
    """
    Ultra-fast hyperparameter optimization for DIP using:
    - Single component/region
    - Coarse resolution 
    - Analytical distance field instead of iterative solver
    - Early stopping
    
    Supports multiple weight formulations: DIP, DIP+, DIP++
    """
    
    def __init__(self, img, seg_mask, prob_map, 
                 downsample_factor=2,
                 weight_map="DIP++",
                 device="cuda"):
        """
        Args:
            img: 3D image array
            seg_mask: 3D binary segmentation  
            prob_map: 3D probability map
            downsample_factor: Reduce resolution by this factor (2 = 8x faster)
            weight_map: Weight formulation - "DIP", "DIP+", or "DIP++"
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.weight_map = weight_map
        
        # Ensure all arrays have same shape
        assert img.shape == seg_mask.shape == prob_map.shape, \
            f"Shape mismatch: img={img.shape}, seg={seg_mask.shape}, prob={prob_map.shape}"
        
        # Downsample for speed
        if downsample_factor > 1:
            img = img[::downsample_factor, ::downsample_factor, ::downsample_factor]
            seg_mask = seg_mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
            prob_map = prob_map[::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        # Store downsampling factor for coordinate conversion
        self.downsample_factor = downsample_factor
        
        # Extract target skeleton
        self.target_skel = skeletonize(seg_mask.astype(bool)).astype(bool)
        self.target_coords = np.argwhere(self.target_skel)
        
        # Convert to torch
        self.img = torch.from_numpy(img.astype(np.float32)).float().to(self.device)
        self.prob = torch.from_numpy(prob_map.astype(np.float32)).float().to(self.device)
        self.target_skel_t = torch.from_numpy(self.target_skel).float().to(self.device)
        
        # Precompute spatial distance transform (HUGE speedup)
        self.spatial_dist = torch.from_numpy(
            distance_transform_edt(~seg_mask)
        ).float().to(self.device)
        
        print(f"Optimizer ready | Weight map: {weight_map} | Resolution: {img.shape} | Skeleton points: {len(self.target_coords)}")
        print(f"Tensors shapes: img={self.img.shape}, prob={self.prob.shape}, target={self.target_skel_t.shape}")
    
    def compute_fast_cost_map(self, alpha, gamma, source_coord, beta=0.001):
        """
        Fast approximation: analytical cost field instead of iterative solver.
        
        Supports three formulations:
        - DIP:   w = sqrt(α*d² + β*ΔI²) * (1/P)
        - DIP+:  w = α*d² + β*(1/ΔI²) + γ*(1/P)
        - DIP++: w = α*d² + β*(1-I) + γ*(1-P)
        
        IMPORTANT: Returns differentiable tensor for backprop
        """
        z, y, x = source_coord
        eps = 1e-8
        
        # Spatial component (Euclidean distance from source)
        ZZ, YY, XX = torch.meshgrid(
            torch.arange(self.img.shape[0], device=self.device, dtype=torch.float32),
            torch.arange(self.img.shape[1], device=self.device, dtype=torch.float32),
            torch.arange(self.img.shape[2], device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Make sure source coordinates are tensors for gradient flow
        z_t = torch.tensor(z, device=self.device, dtype=torch.float32)
        y_t = torch.tensor(y, device=self.device, dtype=torch.float32)
        x_t = torch.tensor(x, device=self.device, dtype=torch.float32)
        
        # Spatial distance squared
        spatial_sq = (ZZ - z_t)**2 + (YY - y_t)**2 + (XX - x_t)**2
        spatial = torch.sqrt(spatial_sq + eps)
        
        # Get intensity at source
        img_source = self.img[int(z), int(y), int(x)]
        
        # Intensity difference squared (approximation: from source to each voxel)
        dint = (self.img - img_source).pow(2)
        
        # Probability penalty
        prob_penalty = 1.0 - self.prob
        
        # === WEIGHT FORMULATION ===
        if self.weight_map == "DIP":
            # DIP: w = sqrt(α*d² + β*ΔI²) * (1/P)
            base = torch.sqrt(alpha * spatial_sq + beta * dint + eps)
            prob_mult = 1.0 / (self.prob + eps)
            cost = base * prob_mult
            
        elif self.weight_map == "DIP+":
            # DIP+: w = α*d² + β*(1/ΔI²) + γ*(1/P)
            cost = (alpha * spatial_sq + 
                   beta / (dint + eps) + 
                   gamma / (self.prob + eps))
            
        elif self.weight_map == "DIP++":
            # DIP++: w = α*d² + β*(1-I) + γ*(1-P)
            cost = (alpha * spatial_sq + 
                   beta * (1.0 - self.img) + 
                   gamma * (1.0 - self.prob))
        
        else:
            raise ValueError(f"Unknown weight_map: {self.weight_map}. Use 'DIP', 'DIP+', or 'DIP++'")
        
        return cost
    
    def extract_path_from_cost(self, cost_map, target_coord, max_steps=500):
        """
        Fast gradient descent on cost map to extract path.
        Returns binary mask of the path.
        
        NOTE: This operates on detached tensors to avoid memory issues
        """
        # Detach for path extraction (no need for gradients here)
        cost_np = cost_map.detach()
        
        path_mask = torch.zeros_like(cost_map, dtype=torch.bool)
        
        z, y, x = target_coord
        current = torch.tensor([z, y, x], device=self.device)
        
        for _ in range(max_steps):
            z, y, x = current.long()
            
            if not (0 <= z < cost_np.shape[0] and 
                   0 <= y < cost_np.shape[1] and 
                   0 <= x < cost_np.shape[2]):
                break
            
            path_mask[z, y, x] = True
            
            # Stop if cost is very low (near source)
            if cost_np[z, y, x] < 1.0:
                break
            
            # Find neighbor with minimum cost
            best_cost = float('inf')
            best_next = current
            
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == dy == dx == 0:
                            continue
                        
                        nz, ny, nx = z + dz, y + dy, x + dx
                        
                        if (0 <= nz < cost_np.shape[0] and
                            0 <= ny < cost_np.shape[1] and
                            0 <= nx < cost_np.shape[2]):
                            
                            c = cost_np[nz, ny, nx]
                            if c < best_cost:
                                best_cost = c
                                best_next = torch.tensor([nz, ny, nx], device=self.device)
            
            if torch.all(best_next == current):  # No improvement
                break
            
            current = best_next
        
        return path_mask.float()
    
    def compute_loss(self, alpha, gamma, sample_points):
        """
        Compute loss based on cost map statistics instead of path extraction.
        
        Strategy: Good parameters should produce LOW cost values on the target skeleton.
        Loss = mean cost on target skeleton points
        """
        total_loss = 0.0
        eps = 1e-8
        
        for source_coord in sample_points:
            # Compute cost map (differentiable)
            cost_map = self.compute_fast_cost_map(alpha, gamma, tuple(source_coord))
            
            # Sample cost values at target skeleton points
            # Lower cost = better alignment with skeleton
            skeleton_cost = cost_map[self.target_skel_t > 0.5]
            
            if len(skeleton_cost) > 0:
                # We want LOW cost on skeleton points
                avg_skeleton_cost = skeleton_cost.mean()
                
                # Also penalize HIGH cost variance (want uniform low cost)
                cost_std = skeleton_cost.std()
                
                # Combined loss: mean cost + variance
                loss = avg_skeleton_cost + 0.1 * cost_std
                total_loss += loss
            else:
                # No skeleton points found - penalize heavily
                total_loss += 1000.0
        
        return total_loss / len(sample_points)
    
    def optimize(self, n_samples=3, n_iterations=30, lr=0.1):
        """
        Run fast optimization.
        
        Args:
            n_samples: Number of source points to sample (3-5 is enough)
            n_iterations: Max iterations (30-50 is usually enough)
            lr: Learning rate
        """
        # Initialize parameters (log space)
        log_alpha = nn.Parameter(torch.tensor(0.0, device=self.device))   # exp(0) = 1
        log_gamma = nn.Parameter(torch.tensor(2.0, device=self.device))   # exp(2) ≈ 7.4
        
        optimizer = torch.optim.Adam([log_alpha, log_gamma], lr=lr)
        
        # Sample source points (uniformly from skeleton)
        indices = np.random.choice(len(self.target_coords), size=min(n_samples, len(self.target_coords)), replace=False)
        sample_points = [self.target_coords[i] for i in indices]
        
        print(f"\nOptimizing with {len(sample_points)} source points, {n_iterations} iterations")
        print("Loss = average cost on target skeleton (lower is better)")
        print("-" * 60)
        
        history = {'loss': [], 'alpha': [], 'gamma': []}
        best_loss = float('inf')
        patience = 10
        no_improve = 0
        
        for it in range(n_iterations):
            optimizer.zero_grad()
            
            # Get actual values
            alpha = torch.exp(log_alpha).clamp(0.1, 10.0)
            gamma = torch.exp(log_gamma).clamp(0.1, 100.0)
            
            # Compute loss
            loss = self.compute_loss(alpha, gamma, sample_points)
            
            # Check if loss is valid
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at iteration {it}, skipping...")
                continue
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            # Track
            loss_val = loss.item()
            history['loss'].append(loss_val)
            history['alpha'].append(alpha.item())
            history['gamma'].append(gamma.item())
            
            # Early stopping
            if loss_val < best_loss:
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"\nEarly stopping at iteration {it}")
                break
            
            if it % 5 == 0 or it == n_iterations - 1:
                print(f"Iter {it:3d} | Loss: {loss_val:.4f} | α={alpha.item():.3f} | γ={gamma.item():.3f}")
        
        # Final values
        final_alpha = torch.exp(log_alpha).clamp(0.1, 10.0).item()
        final_gamma = torch.exp(log_gamma).clamp(0.1, 100.0).item()
        
        best_params = {
            'alpha': final_alpha,
            'gamma': final_gamma,
            'lambda': 0.1  # Can be optimized separately if needed
        }
        
        self.plot_history(history)
        
        return best_params, history
    
    def plot_history(self, history):
        """Plot optimization convergence."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(history['loss'])
        axes[0].set_title('Loss (1 - Dice)')
        axes[0].set_xlabel('Iteration')
        axes[0].grid(True)
        
        axes[1].plot(history['alpha'])
        axes[1].set_title('Alpha (spatial weight)')
        axes[1].set_xlabel('Iteration')
        axes[1].grid(True)
        
        axes[2].plot(history['gamma'])
        axes[2].set_title('Gamma (probability weight)')
        axes[2].set_xlabel('Iteration')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('dip_optimization_fast.png', dpi=100)
        print(f"\nSaved plot to 'dip_optimization_fast.png'")
        plt.close()


# ============================================
# USAGE - INTEGRATION AVEC TON CODE
# ============================================

def optimize_dip_params(img_arr, seg_arr, prob_arr, 
                       weight_map="DIP++",
                       n_samples=25, 
                       n_iterations=40, 
                       downsample=2):
    """
    Optimize DIP hyperparameters quickly.
    
    Args:
        img_arr: 3D image
        seg_arr: 3D segmentation mask
        prob_arr: 3D probability map
        weight_map: "DIP", "DIP+", or "DIP++" formulation
        n_samples: Number of source points (3-30)
                   Impact: Coverage of skeleton diversity
                   Recommended: 20-25 for 30s budget
        n_iterations: Number of optimization steps (20-50)
                      Impact: Fine-tuning precision
                      Recommended: 40-50 for 30s budget
        downsample: Downsampling factor (2 or 4)
                    2 = good quality, 4 = very fast
    
    Returns:
        dict with optimized alpha, gamma, lambda
    """
    optimizer = FastDIPOptimizer(
        img=img_arr,
        seg_mask=seg_arr,
        prob_map=prob_arr,
        downsample_factor=downsample,
        weight_map=weight_map
    )
    
    best_params, history = optimizer.optimize(
        n_samples=n_samples,
        n_iterations=n_iterations,
        lr=0.1
    )
    
    print("\n" + "="*60)
    print(f"OPTIMIZED HYPERPARAMETERS ({weight_map}):")
    print("="*60)
    print(f"Alpha (spatial):      {best_params['alpha']:.4f}")
    print(f"Gamma (probability):  {best_params['gamma']:.4f}")
    print(f"Lambda (anisotropy):  {best_params['lambda']:.4f}")
    print(f"\nOptimization quality:")
    print(f"  - Weight formulation: {weight_map}")
    print(f"  - Skeleton coverage: {n_samples} source points")
    print(f"  - Convergence: {len(history['loss'])} iterations")
    print(f"  - Final loss: {history['loss'][-1]:.4f}")
    print(f"  - Initial loss: {history['loss'][0]:.4f}")
    print(f"  - Improvement: {((history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100):.1f}%")
    print("="*60 + "\n")
    
    return best_params
