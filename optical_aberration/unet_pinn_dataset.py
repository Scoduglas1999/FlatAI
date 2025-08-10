import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
import random

class DeconvolutionPINNDataset(Dataset):
    """
    Physics-Informed Neural Network Dataset for deconvolution.
    Loads triplets of (blurry, sharp, psf) images for training.
    """
    
    def __init__(self, sharp_dir: str, blurry_dir: str, psf_dir: str):
        """
        Args:
            sharp_dir: Directory containing sharp/ground truth images (.npy files)
            blurry_dir: Directory containing blurry input images (.npy files)
            psf_dir: Directory containing PSF kernels (.npy files)
        """
        self.sharp_dir = sharp_dir
        self.blurry_dir = blurry_dir
        self.psf_dir = psf_dir
        
        # Get all .npy files from sharp directory and sort them
        self.filenames = sorted([f for f in os.listdir(sharp_dir) if f.endswith('.npy')])
        
        # Verify that corresponding files exist in all three directories
        for filename in self.filenames:
            sharp_path = os.path.join(sharp_dir, filename)
            blurry_path = os.path.join(blurry_dir, filename)
            psf_path = os.path.join(psf_dir, filename)
            
            if not os.path.exists(sharp_path):
                raise FileNotFoundError(f"Sharp image not found: {sharp_path}")
            if not os.path.exists(blurry_path):
                raise FileNotFoundError(f"Blurry image not found: {blurry_path}")
            if not os.path.exists(psf_path):
                raise FileNotFoundError(f"PSF not found: {psf_path}")
        
        print(f"Loaded dataset with {len(self.filenames)} triplets from:")
        print(f"  Sharp: {sharp_dir}")
        print(f"  Blurry: {blurry_dir}")
        print(f"  PSFs: {psf_dir}")
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and return a triplet of (blurry, sharp, psf) images.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Tuple of (blurry_tensor, sharp_tensor, psf_tensor)
            - blurry_tensor: Shape [1, H, W], normalized to [-1, 1]
            - sharp_tensor: Shape [1, H, W], normalized to [-1, 1]
            - psf_tensor: Shape [1, psf_H, psf_W], normalized (sum to 1)
        """
        filename = self.filenames[idx]
        
        # Load the three components
        sharp_path = os.path.join(self.sharp_dir, filename)
        blurry_path = os.path.join(self.blurry_dir, filename)
        psf_path = os.path.join(self.psf_dir, filename)
        
        # Load numpy arrays
        sharp_img = np.load(sharp_path).astype(np.float32)
        blurry_img = np.load(blurry_path).astype(np.float32)
        psf_kernel = np.load(psf_path).astype(np.float32)
        
        # Normalize images to [-1, 1] range (assuming input is [0, 1])
        sharp_img = sharp_img * 2.0 - 1.0
        blurry_img = blurry_img * 2.0 - 1.0
        
        # Normalize PSF to sum to 1 (important for convolution)
        psf_sum = np.sum(psf_kernel)
        if psf_sum > 0:
            psf_kernel = psf_kernel / psf_sum
        
        # Convert to tensors and add channel dimension
        sharp_tensor = torch.from_numpy(sharp_img).unsqueeze(0)  # [1, H, W]
        blurry_tensor = torch.from_numpy(blurry_img).unsqueeze(0)  # [1, H, W]
        psf_tensor = torch.from_numpy(psf_kernel).unsqueeze(0)  # [1, psf_H, psf_W]
        
        # --- NEW: On-the-Fly Data Augmentation ---
        # 1. Random Horizontal Flip
        if random.random() < 0.5:
            blurry_tensor = torch.flip(blurry_tensor, dims=[2])
            sharp_tensor = torch.flip(sharp_tensor, dims=[2])
            psf_tensor = torch.flip(psf_tensor, dims=[2])

        # 2. Random Vertical Flip
        if random.random() < 0.5:
            blurry_tensor = torch.flip(blurry_tensor, dims=[1])
            sharp_tensor = torch.flip(sharp_tensor, dims=[1])
            psf_tensor = torch.flip(psf_tensor, dims=[1])
            
        # 3. Random 90-degree Rotation
        k = random.randint(0, 3)
        if k > 0:
            blurry_tensor = torch.rot90(blurry_tensor, k, dims=[1, 2])
            sharp_tensor = torch.rot90(sharp_tensor, k, dims=[1, 2])
            psf_tensor = torch.rot90(psf_tensor, k, dims=[1, 2])

        return blurry_tensor, sharp_tensor, psf_tensor

# Test the dataset if run directly
if __name__ == "__main__":
    # Test dataset loading
    test_data_dir = "./deconvolution_dataset/"
    
    try:
        train_sharp_dir = os.path.join(test_data_dir, 'train', 'sharp')
        train_blurry_dir = os.path.join(test_data_dir, 'train', 'blurry')
        train_psf_dir = os.path.join(test_data_dir, 'train', 'psfs')
        
        dataset = DeconvolutionPINNDataset(
            sharp_dir=train_sharp_dir,
            blurry_dir=train_blurry_dir,
            psf_dir=train_psf_dir
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            blurry, sharp, psf = dataset[0]
            print(f"Sample shapes:")
            print(f"  Blurry: {blurry.shape}, range: [{blurry.min():.3f}, {blurry.max():.3f}]")
            print(f"  Sharp: {sharp.shape}, range: [{sharp.min():.3f}, {sharp.max():.3f}]")
            print(f"  PSF: {psf.shape}, range: [{psf.min():.3f}, {psf.max():.3f}], sum: {psf.sum():.6f}")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure the dataset directory structure exists with train/sharp, train/blurry, and train/psfs subdirectories.") 