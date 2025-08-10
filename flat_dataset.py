import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FlatFieldDataset(Dataset):
    """
    Loads triplets for flat-field correction training:
      - affected: observed input with multiplicative (flat) and additive (grad) artifacts, [0,1]
      - sharp: ground truth clean target, [0,1]
      - flat: multiplicative field F(x,y), mean-normalized around 1.0
      - grad: additive field G(x,y), typically [0, ~0.3]

    Returns tensors:
      - affected_tensor: [1, H, W], normalized to [-1, 1]
      - sharp_tensor:    [1, H, W], normalized to [-1, 1]
      - flat_tensor:     [1, H, W], raw domain (positive, mean ~1)
      - grad_tensor:     [1, H, W], raw domain [0, 1]
    """

    def __init__(self, sharp_dir: str, affected_dir: str, flat_dir: str, grad_dir: str):
        self.sharp_dir = sharp_dir
        self.affected_dir = affected_dir
        self.flat_dir = flat_dir
        self.grad_dir = grad_dir

        self.filenames = sorted([f for f in os.listdir(sharp_dir) if f.endswith('.npy')])

        # Verify correspondence
        for filename in self.filenames:
            for d in [sharp_dir, affected_dir, flat_dir, grad_dir]:
                path = os.path.join(d, filename)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing file: {path}")

        print(f"Loaded FlatFieldDataset with {len(self.filenames)} samples from:")
        print(f"  Sharp:    {sharp_dir}")
        print(f"  Affected: {affected_dir}")
        print(f"  Flat:     {flat_dir}")
        print(f"  Grad:     {grad_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        sharp = np.load(os.path.join(self.sharp_dir, filename)).astype(np.float32)
        affected = np.load(os.path.join(self.affected_dir, filename)).astype(np.float32)
        flat = np.load(os.path.join(self.flat_dir, filename)).astype(np.float32)
        grad = np.load(os.path.join(self.grad_dir, filename)).astype(np.float32)

        # Normalize images to [-1, 1]
        sharp = sharp * 2.0 - 1.0
        affected = affected * 2.0 - 1.0

        sharp_t = torch.from_numpy(sharp).unsqueeze(0)
        affected_t = torch.from_numpy(affected).unsqueeze(0)
        flat_t = torch.from_numpy(flat).unsqueeze(0)
        grad_t = torch.from_numpy(grad).unsqueeze(0)

        # Random augmentations applied consistently
        if random.random() < 0.5:
            sharp_t = torch.flip(sharp_t, dims=[2])
            affected_t = torch.flip(affected_t, dims=[2])
            flat_t = torch.flip(flat_t, dims=[2])
            grad_t = torch.flip(grad_t, dims=[2])

        if random.random() < 0.5:
            sharp_t = torch.flip(sharp_t, dims=[1])
            affected_t = torch.flip(affected_t, dims=[1])
            flat_t = torch.flip(flat_t, dims=[1])
            grad_t = torch.flip(grad_t, dims=[1])

        k = random.randint(0, 3)
        if k > 0:
            sharp_t = torch.rot90(sharp_t, k, dims=[1, 2])
            affected_t = torch.rot90(affected_t, k, dims=[1, 2])
            flat_t = torch.rot90(flat_t, k, dims=[1, 2])
            grad_t = torch.rot90(grad_t, k, dims=[1, 2])

        return affected_t, sharp_t, flat_t, grad_t


