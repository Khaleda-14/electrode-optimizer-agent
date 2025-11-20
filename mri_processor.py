import numpy as np
import os
import time
from scipy.ndimage import zoom, gaussian_filter

class MRIProcessor:
    """
    Simulates loading and preprocessing of NIfTI MRI data.
    Implements advanced synthetic data generation including isotropic resampling 
    and simulated Fuzzy C-Means segmentation to produce a White Matter Probability Map.
    """
    def __init__(self, config):
        self.config = config
        self.ANATOMICAL_SIZE_MM = 200.0
        
    def load_and_process(self, mri_path=None):
        """
        Simulates the entire processing pipeline: Loading -> Resampling -> Segmentation -> Slicing.
        """
        if mri_path and os.path.exists(mri_path):
            print(f"INFO: Attempting to load real MRI data from {mri_path}...")
        else:
            print("WARN: No valid MRI path provided or file not found. Generating advanced synthetic data.")
        
        return self._generate_synthetic_data()

    def _generate_synthetic_data(self, modality="Synthetic T1w (Resampled)"):
        """
        Generates dummy NumPy arrays with simulated isotropic resampling and
        advanced tissue segmentation (White Matter Probability Map).
        """
        # 1. Determine Target Voxel Dimensions based on Configuration
        target_res = self.config.get('target_mri_resolution', 2.0)
        target_dim = int(self.ANATOMICAL_SIZE_MM / target_res)
        
        # Ensure minimum size for visualization stability
        target_dim = max(target_dim, 64) 
        
        print(f"INFO: Simulating resampling to isotropic {target_res}mm -> Volume Size: {target_dim}x{target_dim}x{target_dim}")
        
        X, Y, Z = target_dim, target_dim, target_dim
        rng = np.random.default_rng(int(time.time()))
        
        # 2. Simulate T1 Volume
        # Base T1 data (normalized grayscale 0-1) - slightly higher signal in center (WM area)
        full_volume = rng.uniform(0.3, 0.7, (X, Y, Z))
        
        # Simulate a central elliptical structure (White Matter) with higher intensity
        center_x, center_y, center_z = X // 2, Y // 2, Z // 2
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    # Distance from center, weighted for an elliptical WM mass
                    dist_sq = ((i - center_x) * 0.8)**2 + ((j - center_y) * 1.2)**2 + ((k - center_z) * 1.0)**2
                    if dist_sq < (target_dim / 4)**2:
                        full_volume[i, j, k] += 0.3 * rng.uniform(0.8, 1.0)
        
        # Normalize the final volume
        full_volume = (full_volume - full_volume.min()) / (full_volume.max() - full_volume.min())
        
        # 3. Simulate Advanced Segmentation (White Matter Probability Map - WMPM)
        
        # Action 1: Simulate Multi-Scale GMM/FCM: Use Gaussian blur to smooth the intensity differences, 
        # mimicking the soft tissue classification of Fuzzy C-Means or GMM.
        # This creates a 'probability' map by blurring the boundary.
        wm_probability_map = full_volume.copy()
        wm_probability_map = gaussian_filter(wm_probability_map, sigma=target_res) 
        
        # Action 2: Apply soft threshold (Simulated Tanh/Sigmoid function)
        # WM typically has intensity > 0.6. We squash the values to create a probability.
        threshold = 0.65
        soft_threshold = 1.0 / (1.0 + np.exp(-15 * (wm_probability_map - threshold)))

        # The segmentation mask is now the WMPM (continuous values from 0 to 1)
        segmentation_mask = soft_threshold

        # 4. Implement Slicing based on n_representative_slices
        n_slices = self.config.get('n_representative_slices', 5)
        
        # Select slices evenly spaced across the Z (axial) axis
        slice_indices = np.linspace(0, Z - 1, n_slices, dtype=int)
        representative_slices = [full_volume[:, :, i] for i in slice_indices]
        
        # Derived metrics
        derived_metrics = {
            "resolution_mm": target_res,
            "tissue_density_score": np.mean(segmentation_mask), # Average WM probability
            "volume_size": target_dim
        }
        
        print(f"INFO: Generated WMPM (Avg Probability: {derived_metrics['tissue_density_score']:.3f})")
        
        return {
            "modality": modality,
            "full_volume": full_volume,
            "segmentation_mask": segmentation_mask, # WMPM is returned here
            "representative_slices": representative_slices,
            "derived_metrics": derived_metrics
        }