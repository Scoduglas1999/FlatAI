"""
create_flat_dataset.py - Hybrid Dataset Generator

This script now serves two purposes for a more robust and scalable training pipeline:

1.  **On-the-Fly Training Set:** It processes all images from the `SHARP_IMAGE_DIR`,
    resizing and padding them correctly. It saves these "base" sharp images
    to a `train/sharp_processed` directory. The live `FlatFieldDataset` will
    use these to generate unique training samples in real-time.

2.  **Fixed Validation Set:** It generates a static, pre-calculated set of
    validation samples (`sharp`, `affected`, `flat`, `grad`). This ensures
    that the validation metric is stable and reliable from epoch to epoch,
    as the model is always tested against the exact same data.
"""

import os
import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from tqdm import tqdm

from artifact_generator import synthesize_fields, apply_poisson_read_noise

try:
    from PIL import Image
except ImportError:
    Image = None


# =============================
# 1) Configuration
# =============================

SHARP_IMAGE_DIR = "./sharp_images/"
OUTPUT_DATA_DIR = "./randomized_flat_dataset/"

# For fixed training, we will generate a set of realistic samples.
# For validation, we create a fixed, smaller set.
NUM_TRAIN_SAMPLES = 20000
NUM_VALIDATION_SAMPLES = 1000

# Practical limits to keep memory usage reasonable.
MAX_IMAGE_DIM = 1024
SAVE_DTYPE = np.float16
# Compress per-array files to save disk space. Loader supports both .npy and .npz.
SAVE_COMPRESSED = True  # True → .npz (zip/deflate), False → .npy

# Curriculum for sample generation
CURRICULUM_WARMUP_SAMPLES = 500


# =============================
# 2) Image Loading & Preparation
# =============================

def load_astro_image(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        print(f"Missing file: {image_path}")
        return None
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    try:
        if ext in [".fits", ".fit"]:
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data
        elif ext in [".jpg", ".jpeg", ".png"]:
            if Image is None:
                raise ImportError("Pillow is required to load JPG/PNG images")
            with Image.open(image_path) as img:
                image_data = np.array(img.convert("L"))
        else:
            return None
        if image_data is None:
            return None
        if image_data.ndim == 3:
            image_data = image_data.mean(axis=2)
        return image_data.astype(np.float32)
    except Exception as exc:
        print(f"Failed to load {image_path}: {exc}")
        return None

def resize_image(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize image if its largest dimension exceeds max_dim."""
    h, w = img.shape
    if max(h, w) <= max_dim:
        return img
    
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
        
    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    new_h = new_h if new_h % 2 == 0 else new_h - 1
    new_w = new_w if new_w % 2 == 0 else new_w - 1
    
    resized_t = F.interpolate(img_t, size=(new_h, new_w), mode='area')
    return resized_t.squeeze(0).squeeze(0).numpy()

def pad_to_sixteen(img: np.ndarray) -> np.ndarray:
    """Pads a 2D numpy array so its dimensions are divisible by 16."""
    h, w = img.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    if pad_h == 0 and pad_w == 0:
        return img
    
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')


def _save_array(out_dir: str, base_name: str, array: np.ndarray):
    """Save array as .npz (compressed) or .npy based on SAVE_COMPRESSED. base_name may end with .npy."""
    os.makedirs(out_dir, exist_ok=True)
    if base_name.endswith('.npy'):
        base_root = base_name[:-4]
    else:
        base_root = base_name
    if SAVE_COMPRESSED:
        path = os.path.join(out_dir, base_root + '.npz')
        np.savez_compressed(path, arr=array.astype(SAVE_DTYPE))
    else:
        path = os.path.join(out_dir, base_root + '.npy')
        np.save(path, array.astype(SAVE_DTYPE))


# =============================
# 3) Main generation loop
# =============================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare output directories
    # Train dir (fixed) needs full sets; keep dynamic preprocessed for future use
    for sub in [
        "train/sharp", "train/affected", "train/flat", "train/grad",
        "train/sharp_processed",
        "val/sharp", "val/affected", "val/flat", "val/grad"
    ]:
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, sub), exist_ok=True)

    # Enumerate images
    valid_ext = [".fits", ".fit", ".jpg", ".jpeg", ".png"]
    files = [f for f in os.listdir(SHARP_IMAGE_DIR) if os.path.splitext(f)[1].lower() in valid_ext]
    if not files:
        print(f"No valid images found in {SHARP_IMAGE_DIR}")
        raise SystemExit(1)
    print(f"Found {len(files)} source images for processing.")

    # --- Process training images (save base sharp files for optional dynamic use) ---
    pbar_train = tqdm(files, desc="Processing sharp images for base set")
    for fname in pbar_train:
        path = os.path.join(SHARP_IMAGE_DIR, fname)
        img = load_astro_image(path)
        if img is None or img.size == 0 or img.ndim != 2 or min(img.shape) == 0:
            continue
        
        img = resize_image(img, MAX_IMAGE_DIM)
        img = pad_to_sixteen(img)
        if img is None or img.size == 0 or img.ndim != 2 or min(img.shape) == 0:
            continue
        
        # Normalize to [0,1] before saving
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)

        save_path = os.path.join(OUTPUT_DATA_DIR, "train", "sharp_processed", f"{os.path.splitext(fname)[0]}.npy")
        np.save(save_path, img_norm.astype(SAVE_DTYPE))

    print(f"\nProcessed {len(files)} sharp images for base set (optional dynamic use).")

    # --- Generate fixed training set ---
    pbar_fixed = tqdm(range(NUM_TRAIN_SAMPLES), desc="Generating fixed training set")
    for i in pbar_fixed:
        fname = random.choice(files)
        path = os.path.join(SHARP_IMAGE_DIR, fname)
        img = load_astro_image(path)
        if img is None or img.size == 0 or img.ndim != 2 or min(img.shape) == 0:
            continue
        img = resize_image(img, MAX_IMAGE_DIM)
        img = pad_to_sixteen(img)
        if img is None or img.size == 0 or img.ndim != 2 or min(img.shape) == 0:
            continue
        h, w = img.shape
        img_t = torch.from_numpy(img)
        # Curriculum difficulty ramp
        difficulty = min(1.0, 0.2 + 0.8 * (i / max(1, NUM_TRAIN_SAMPLES - 1)))
        # Global normalization to [0,1]
        img_min = img_t.min()
        img_max = img_t.max()
        if float(img_max) > float(img_min):
            gt_lin = (img_t - img_min) / (img_max - img_min)
        else:
            gt_lin = torch.zeros_like(img_t)
        # Synthesize fields and affected
        device_cpu = torch.device("cpu")
        F_mul, G_add = synthesize_fields(h, w, difficulty, device_cpu)
        affected = torch.clamp(gt_lin * F_mul + G_add, 0.0, 1.0)
        affected = apply_poisson_read_noise(affected, difficulty)
        # Save
        gt_np = gt_lin.numpy().astype(SAVE_DTYPE)
        aff_np = affected.numpy().astype(SAVE_DTYPE)
        flat_np = F_mul.numpy().astype(SAVE_DTYPE)
        grad_np = G_add.numpy().astype(SAVE_DTYPE)
        base = f"train_sample_{i:06d}"
        _save_array(os.path.join(OUTPUT_DATA_DIR, "train", "sharp"), base, gt_np)
        _save_array(os.path.join(OUTPUT_DATA_DIR, "train", "affected"), base, aff_np)
        _save_array(os.path.join(OUTPUT_DATA_DIR, "train", "flat"), base, flat_np)
        _save_array(os.path.join(OUTPUT_DATA_DIR, "train", "grad"), base, grad_np)
    
    # --- Generate fixed validation set ---
    pbar_val = tqdm(range(NUM_VALIDATION_SAMPLES), desc="Generating fixed validation set")
    for i in pbar_val:
        # Pick a random source image each time
        fname = random.choice(files)
        path = os.path.join(SHARP_IMAGE_DIR, fname)
        img = load_astro_image(path)
        if img is None or img.size == 0 or img.ndim != 2 or min(img.shape) == 0:
            continue

        img = resize_image(img, MAX_IMAGE_DIM)
        img = pad_to_sixteen(img)
        if img is None or img.size == 0 or img.ndim != 2 or min(img.shape) == 0:
            continue
        h, w = img.shape
        # --- BUG FIX ---
        # All data generation must happen on the CPU to match the on-the-fly generator.
        # Moving to GPU here was causing the artifact synthesis to fail silently.
        device_cpu = torch.device("cpu")
        img_t = torch.from_numpy(img).to(device_cpu)
        
        difficulty = min(1.0, 0.1 + 0.9 * (i / CURRICULUM_WARMUP_SAMPLES))

        # Normalize sharp image to [0,1]
        img_min = img_t.min()
        img_max = img_t.max()
        if float(img_max) > float(img_min):
            gt_lin = (img_t - img_min) / (img_max - img_min)
        else:
            gt_lin = torch.zeros_like(img_t)

        F_mul, G_add = synthesize_fields(h, w, difficulty, device_cpu)
        affected = torch.clamp(gt_lin * F_mul + G_add, 0.0, 1.0)
        affected = apply_poisson_read_noise(affected, difficulty)

        gt_np = gt_lin.detach().cpu().numpy().astype(SAVE_DTYPE)
        aff_np = affected.detach().cpu().numpy().astype(SAVE_DTYPE)
        flat_np = F_mul.detach().cpu().numpy().astype(SAVE_DTYPE)
        grad_np = G_add.detach().cpu().numpy().astype(SAVE_DTYPE)

        base = f"val_sample_{i:04d}"
        _save_array(os.path.join(OUTPUT_DATA_DIR, "val", "sharp"), base, gt_np)
        _save_array(os.path.join(OUTPUT_DATA_DIR, "val", "affected"), base, aff_np)
        _save_array(os.path.join(OUTPUT_DATA_DIR, "val", "flat"), base, flat_np)
        _save_array(os.path.join(OUTPUT_DATA_DIR, "val", "grad"), base, grad_np)

    print(f"\nGenerated {NUM_VALIDATION_SAMPLES} fixed samples for the validation set.")
    print(f"Dataset preparation complete. Saved to: {OUTPUT_DATA_DIR}")


