import os
import random
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from artifact_generator import synthesize_fields, apply_poisson_read_noise

class FlatFieldDataset(Dataset):
    """
    A hybrid dataset class for flat-field correction.

    - In 'train' mode, it generates samples dynamically on-the-fly.
      It loads a pre-processed sharp image and applies newly synthesized
      artifacts to it in real-time. This creates a virtually infinite dataset.

    - In 'val' mode, it loads a static, pre-generated set of samples.
      This ensures a consistent and reliable validation metric.
    """

    def __init__(self, root_dir: str, mode: str = 'train', epoch_size: int = 10000, max_train_side: Optional[int] = 1024, fixed_train_side: Optional[int] = None):
        if mode not in ['train', 'train_fixed', 'val']:
            raise ValueError("Mode must be 'train', 'train_fixed', or 'val'")
        self.mode = mode
        self.root_dir = root_dir
        # --- CRITICAL FIX ---
        # Data loading and augmentation MUST happen on the CPU in worker processes.
        # Only the main training loop should move data to the GPU.
        self.device = torch.device("cpu")
        # Limit the spatial size of dynamically generated training samples to control VRAM/CPU RAM.
        # Validation remains full-size and deterministic.
        self.max_train_side = max_train_side if mode == 'train' else None
        # If provided, force dynamic training samples to this exact square side via crop/pad
        self.fixed_train_side = fixed_train_side if mode == 'train' else None

        if self.mode == 'train':
            self.sharp_dir = os.path.join(root_dir, 'train', 'sharp_processed')
            self.sharp_files = [os.path.join(self.sharp_dir, f) for f in os.listdir(self.sharp_dir) if f.endswith('.npy')]
            if not self.sharp_files:
                raise FileNotFoundError(f"No pre-processed sharp files found in {self.sharp_dir}. "
                                      "Did you run create_flat_dataset.py first?")
            self.epoch_size = epoch_size  # Number of dynamic samples per epoch
            print(f"Loaded on-the-fly training dataset with {len(self.sharp_files)} base sharp images.")
            # Small in-memory cache of sharp .npy to avoid disk stalls
            self._sharp_cache: dict[str, np.ndarray] = {}
            self._cache_capacity: int = 32
        elif self.mode == 'train_fixed':
            self.sharp_dir = os.path.join(root_dir, 'train', 'sharp')
            self.affected_dir = os.path.join(root_dir, 'train', 'affected')
            self.flat_dir = os.path.join(root_dir, 'train', 'flat')
            self.grad_dir = os.path.join(root_dir, 'train', 'grad')
            self.filenames = sorted([f for f in os.listdir(self.sharp_dir) if f.endswith('.npy') or f.endswith('.npz')])
            print(f"Loaded fixed training dataset with {len(self.filenames)} samples.")
        else: # mode == 'val'
            self.sharp_dir = os.path.join(root_dir, 'val', 'sharp')
            self.affected_dir = os.path.join(root_dir, 'val', 'affected')
            self.flat_dir = os.path.join(root_dir, 'val', 'flat')
            self.grad_dir = os.path.join(root_dir, 'val', 'grad')
            self.filenames = sorted([f for f in os.listdir(self.sharp_dir) if f.endswith('.npy') or f.endswith('.npz')])
            print(f"Loaded static validation dataset with {len(self.filenames)} samples.")

    def __len__(self) -> int:
        if self.mode == 'train':
            return self.epoch_size
        else:
            return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == 'train':
            return self._get_dynamic_train_item()
        elif self.mode == 'train_fixed':
            return self._get_static_item(idx, split='train')
        else:
            return self._get_static_val_item(idx)

    def _get_static_val_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        def _load(path):
            if path.endswith('.npz'):
                return np.load(path)['arr']
            return np.load(path)
        sharp = _load(os.path.join(self.sharp_dir, filename)).astype(np.float32)
        affected = _load(os.path.join(self.affected_dir, filename)).astype(np.float32)
        flat = _load(os.path.join(self.flat_dir, filename)).astype(np.float32)
        grad = _load(os.path.join(self.grad_dir, filename)).astype(np.float32)
        
        # Data is saved as [0,1], normalize to [-1, 1] for the model
        sharp = sharp * 2.0 - 1.0
        affected = affected * 2.0 - 1.0

        return torch.from_numpy(affected).unsqueeze(0), \
               torch.from_numpy(sharp).unsqueeze(0), \
               torch.from_numpy(flat).unsqueeze(0), \
               torch.from_numpy(grad).unsqueeze(0)

    def _get_static_item(self, idx: int, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        def _load(path):
            if path.endswith('.npz'):
                return np.load(path)['arr']
            return np.load(path)
        sharp = _load(os.path.join(self.sharp_dir, filename)).astype(np.float32)
        affected = _load(os.path.join(self.affected_dir, filename)).astype(np.float32)
        # Optional GT fields
        flat_path = os.path.join(self.flat_dir, filename)
        grad_path = os.path.join(self.grad_dir, filename)
        if os.path.exists(flat_path):
            flat = _load(flat_path).astype(np.float32)
        else:
            flat = np.ones_like(sharp, dtype=np.float32)
        if os.path.exists(grad_path):
            grad = _load(grad_path).astype(np.float32)
        else:
            grad = np.zeros_like(sharp, dtype=np.float32)

        # Normalize to [-1,1] for the model on inputs only
        sharp_t = torch.from_numpy(sharp * 2.0 - 1.0).unsqueeze(0)
        affected_t = torch.from_numpy(affected * 2.0 - 1.0).unsqueeze(0)
        flat_t = torch.from_numpy(flat).unsqueeze(0)
        grad_t = torch.from_numpy(grad).unsqueeze(0)
        return affected_t, sharp_t, flat_t, grad_t

    def _get_dynamic_train_item(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load a random pre-processed sharp image
        sharp_path = random.choice(self.sharp_files)
        arr = self._sharp_cache.get(sharp_path)
        if arr is None:
            sharp_norm_01 = np.load(sharp_path)  # stored as float16 to save disk
            # Keep as numpy array (float16/float32) in cache; convert to torch later
            self._sharp_cache[sharp_path] = sharp_norm_01
            # Evict older entries if needed
            if len(self._sharp_cache) > self._cache_capacity:
                try:
                    self._sharp_cache.pop(next(iter(self._sharp_cache)))
                except Exception:
                    pass
        else:
            sharp_norm_01 = arr
        
        # Convert to tensor. This will be on the CPU.
        gt_lin = torch.from_numpy(sharp_norm_01.astype(np.float32, copy=False)).to(self.device)
        # Optionally crop very large images to keep compute/memory bounded.
        if self.max_train_side is not None and (self.fixed_train_side is None):
            gt_lin = self._random_crop_to_limit(gt_lin, self.max_train_side)
        # If a fixed side is requested, enforce exact square via crop/pad.
        if self.fixed_train_side is not None:
            gt_lin = self._random_crop_or_pad_to_side(gt_lin, self.fixed_train_side)
        h, w = gt_lin.shape
        
        # Randomly determine difficulty for this sample
        difficulty = random.uniform(0.1, 1.0)

        # Synthesize fields on-the-fly
        F_mul, G_add = synthesize_fields(h, w, difficulty, self.device)

        # Compose affected image and apply noise
        affected = torch.clamp(gt_lin * F_mul + G_add, 0.0, 1.0)
        affected = apply_poisson_read_noise(affected, difficulty)
        
        # --- Data Augmentation ---
        # Apply the same random transformations to all tensors to maintain alignment.
        # This is critical for making the model robust to tiling artifacts.
        # 50% chance of horizontal flip
        if random.random() < 0.5:
            gt_lin = torch.flip(gt_lin, dims=[-1])
            F_mul = torch.flip(F_mul, dims=[-1])
            G_add = torch.flip(G_add, dims=[-1])
            affected = torch.flip(affected, dims=[-1])

        # 50% chance of vertical flip
        if random.random() < 0.5:
            gt_lin = torch.flip(gt_lin, dims=[-2])
            F_mul = torch.flip(F_mul, dims=[-2])
            G_add = torch.flip(G_add, dims=[-2])
            affected = torch.flip(affected, dims=[-2])

        # Random 90-degree rotations
        k = random.randint(0, 3)  # 0, 1, 2, or 3 rotations
        if k > 0:
            gt_lin = torch.rot90(gt_lin, k, dims=[-2, -1])
            F_mul = torch.rot90(F_mul, k, dims=[-2, -1])
            G_add = torch.rot90(G_add, k, dims=[-2, -1])
            affected = torch.rot90(affected, k, dims=[-2, -1])

        # Final tensors should be on CPU, ready for the collate_fn.
        # The main training loop will move the batch to the GPU.
        affected_cpu = (affected * 2.0 - 1.0).unsqueeze(0)
        sharp_cpu = (gt_lin * 2.0 - 1.0).unsqueeze(0)
        flat_cpu = F_mul.unsqueeze(0)
        grad_cpu = G_add.unsqueeze(0)

        return affected_cpu, sharp_cpu, flat_cpu, grad_cpu

    @staticmethod
    def _random_crop_to_limit(img_2d: torch.Tensor, max_side: int) -> torch.Tensor:
        """
        Randomly crop to a square with side <= max_side (divisible by 16) for shape stability.
        If the image is smaller than max_side, returns a centered crop of the largest square possible.
        """
        h, w = img_2d.shape
        target = min(max_side, h, w)
        # Make divisible by 16
        target = max(16, (target // 16) * 16)
        if h == target and w == target:
            return img_2d
        top = 0 if h == target else random.randint(0, h - target)
        left = 0 if w == target else random.randint(0, w - target)
        return img_2d[top:top+target, left:left+target]

    @staticmethod
    def _random_crop_or_pad_to_side(img_2d: torch.Tensor, side: int) -> torch.Tensor:
        """
        Return a square tensor of exactly `side` by random crop if larger, or reflect-pad if smaller.
        Side is snapped to a multiple of 16 for U-Net down/up sampling.
        """
        side = max(16, (int(side) // 16) * 16)
        h, w = img_2d.shape
        # If larger, random crop
        if h >= side and w >= side:
            top = 0 if h == side else random.randint(0, h - side)
            left = 0 if w == side else random.randint(0, w - side)
            return img_2d[top:top+side, left:left+side]
        # Else, pad to reach side using replication (safe for any pad size)
        pad_h = max(0, side - h)
        pad_w = max(0, side - w)
        # Symmetric padding split
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img4 = img_2d.unsqueeze(0).unsqueeze(0)
        padded4 = torch.nn.functional.pad(img4, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        return padded4.squeeze(0).squeeze(0)


