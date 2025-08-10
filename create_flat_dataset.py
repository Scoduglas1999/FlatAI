"""
create_flat_dataset.py - Synthetic Flat-Field Artifact Dataset Generator

This script generates a dataset for training a neural network to correct flat-field
artifacts in astrophotography images. It synthesizes:
 - Multiplicative field F(x,y): vignetting, PRNU, dust motes (dark/light, in/out of focus)
 - Additive field G(x,y): illumination gradients, amp glow

Observed image model:
    I_obs = clip(I_true * F + G, 0, 1)

Output directory structure:
  randomized_flat_dataset/
    train/
      sharp/      (ground-truth, [0,1])
      affected/   (flat-affected input, [0,1])
      flat/       (multiplicative field F, mean-normalized around 1)
      grad/       (additive field G, [0, ~0.3])
    val/
      ... same as train

Notes:
 - Uses only NumPy and PyTorch to avoid extra dependencies. Heavy lifting is done
   via torch for convenience and speed (CUDA if available).
 - The generator employs curriculum learning to gradually increase artifact severity.
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

try:
    from PIL import Image
except ImportError:
    Image = None


# =============================
# 1) Configuration
# =============================

SHARP_IMAGE_DIR = "./sharp_images/"
OUTPUT_DATA_DIR = "./randomized_flat_dataset/"

PATCH_SIZE = 256
# Target around ~30-40k samples depending on number of source images
PATCHES_PER_IMAGE = 400
TRAIN_VAL_SPLIT = 0.9
SAVE_DTYPE = np.float16  # reduce disk usage ~2x without affecting training (loader casts to float32)

# Curriculum: increase severity from easy to hard over N patches
CURRICULUM_WARMUP_PATCHES = 10000


# =============================
# 2) Image Loading
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
            # Skip unsupported formats silently to keep generator resilient
            return None
        if image_data is None:
            return None
        if image_data.ndim == 3:
            image_data = image_data.mean(axis=2)
        return image_data.astype(np.float32)
    except Exception as exc:
        print(f"Failed to load {image_path}: {exc}")
        return None


# =============================
# 3) Synthetic Field Generators
# =============================

def make_meshgrid(h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device),
        torch.linspace(0, 1, w, device=device),
        indexing="ij",
    )
    return yy, xx


def generate_vignetting_field(h: int, w: int, strength: float, order: float, center_jitter: float, device: torch.device) -> torch.Tensor:
    """
    Generate a multiplicative vignetting field around 1.0.
      strength: negative darkens corners, positive brightens
      order:    controls radial falloff shape (2..6 typical)
      center_jitter: relative offset of center (0..0.2)
    """
    yy, xx = make_meshgrid(h, w, device)
    # Randomized optical center
    cx = 0.5 + random.uniform(-center_jitter, center_jitter)
    cy = 0.5 + random.uniform(-center_jitter, center_jitter)
    rr = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rr = rr / torch.sqrt(torch.tensor(0.5 ** 2 + 0.5 ** 2, device=device))  # normalize to [0, ~1]
    rr = torch.clamp(rr, 0.0, 1.0)
    field = 1.0 + strength * (rr ** order)
    return field


def generate_prnu_field(h: int, w: int, amplitude: float, cell: int, device: torch.device) -> torch.Tensor:
    """
    PRNU-like low-frequency multiplicative noise around 1.0.
    Create a coarse random grid and upsample with bicubic to smoothness.
    """
    coarse_h = max(2, h // cell)
    coarse_w = max(2, w // cell)
    coarse = torch.rand((1, 1, coarse_h, coarse_w), device=device)
    up = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=True)
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    # Map to [1 - a, 1 + a]
    field = 1.0 + amplitude * (up.squeeze(0).squeeze(0) * 2.0 - 1.0)
    return field


def gaussian_ring(h: int, w: int, y0: float, x0: float, r0: float, sigma_r: float, gain: float, device: torch.device) -> torch.Tensor:
    """
    Return multiplicative map for a donut-shaped mote.
      gain < 1 -> dark ring, gain > 1 -> bright ring
    """
    yy, xx = make_meshgrid(h, w, device)
    r = torch.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    ring = torch.exp(-0.5 * ((r - r0) / (sigma_r + 1e-6)) ** 2)
    # Blend ring around 1.0 with amplitude (gain - 1)
    return 1.0 + (gain - 1.0) * ring


def gaussian_disc(h: int, w: int, y0: float, x0: float, sigma: float, gain: float, device: torch.device) -> torch.Tensor:
    yy, xx = make_meshgrid(h, w, device)
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    disc = torch.exp(-0.5 * r2 / (sigma ** 2 + 1e-6))
    return 1.0 + (gain - 1.0) * disc


def _make_lowfreq_noise(h: int, w: int, scale: int, device: torch.device) -> torch.Tensor:
    """Perlin-like low-frequency noise in [-1,1] via bicubic upsampling of a coarse grid."""
    coarse_h = max(2, h // max(1, scale))
    coarse_w = max(2, w // max(1, scale))
    grid = torch.rand((1, 1, coarse_h, coarse_w), device=device)
    up = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=True)
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    return up.squeeze(0).squeeze(0) * 2.0 - 1.0


def _elliptical_radius(xx: torch.Tensor, yy: torch.Tensor, cx: float, cy: float,
                       a: float, b: float, theta: float) -> torch.Tensor:
    """
    Normalized elliptical radius r such that r=1 lies on the ellipse boundary.
    a and b are semi-axes (relative to [0,1] image coordinates).
    """
    # Shift to center
    x = xx - cx
    y = yy - cy
    c, s = math.cos(theta), math.sin(theta)
    xr = c * x + s * y
    yr = -s * x + c * y
    r = torch.sqrt((xr / (a + 1e-8)) ** 2 + (yr / (b + 1e-8)) ** 2)
    return r


def _softstep(x: torch.Tensor, k: float) -> torch.Tensor:
    """Fast smooth step using sigmoid; k controls edge softness."""
    return torch.sigmoid(x / (k + 1e-6))


def _radial_distance(h: int, w: int, cx: float, cy: float, device: torch.device) -> torch.Tensor:
    yy, xx = make_meshgrid(h, w, device)
    return torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def _circular_mask_jittered(h: int, w: int, cx: float, cy: float, radius: float,
                            edge: float, rough_amp: float, device: torch.device) -> torch.Tensor:
    rr = _radial_distance(h, w, cx, cy, device)
    jitter = _make_lowfreq_noise(h, w, scale=48, device=device)
    boundary = radius * (1.0 + rough_amp * jitter)
    return _softstep(boundary - rr, edge)


def _donut_mask_jittered(h: int, w: int, cx: float, cy: float,
                         r_inner: float, r_outer: float, edge: float,
                         rough_amp: float, device: torch.device) -> torch.Tensor:
    rr = _radial_distance(h, w, cx, cy, device)
    jitter1 = _make_lowfreq_noise(h, w, scale=44, device=device)
    jitter2 = _make_lowfreq_noise(h, w, scale=44, device=device)
    b_in = r_inner * (1.0 + 0.6 * rough_amp * jitter1)
    b_out = r_outer * (1.0 + rough_amp * jitter2)
    outer = _softstep(b_out - rr, edge)
    inner = _softstep(b_in - rr, edge)
    return torch.clamp(outer - inner, 0.0, 1.0)


def _irregular_disc_mask(h: int, w: int, cx: float, cy: float, a: float, b: float,
                         theta: float, edge: float, rough_amp: float, device: torch.device) -> torch.Tensor:
    yy, xx = make_meshgrid(h, w, device)
    r = _elliptical_radius(xx, yy, cx, cy, a, b, theta)
    rough = _make_lowfreq_noise(h, w, scale=int(32), device=device)
    r = r + rough_amp * rough
    # Inside ellipse -> r < 1
    mask = _softstep(1.0 - r, edge)
    return torch.clamp(mask, 0.0, 1.0)


def _irregular_ring_mask(h: int, w: int, cx: float, cy: float, a: float, b: float,
                         theta: float, r0: float, thickness: float, edge: float,
                         rough_amp: float, device: torch.device) -> torch.Tensor:
    yy, xx = make_meshgrid(h, w, device)
    r = _elliptical_radius(xx, yy, cx, cy, a, b, theta)
    rough = _make_lowfreq_noise(h, w, scale=int(28), device=device)
    r = r + rough_amp * rough
    band = torch.exp(-0.5 * ((r - r0) / (thickness / 2.355 + 1e-6)) ** 2)
    # Slightly soften the band edges
    band = band * _softstep(1.2 - r, edge)
    return torch.clamp(band, 0.0, 1.0)


def _soft_disc_mask(h: int, w: int, cy: float, cx: float, radius: float, edge: float, device: torch.device) -> torch.Tensor:
    """Soft-edged circular mask in [0,1]; 1 inside the disc, 0 outside."""
    yy, xx = make_meshgrid(h, w, device)
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # mask ~ 1 for r < radius, soft transition with width 'edge'
    return torch.sigmoid((radius - r) / (edge + 1e-6))


def _soft_ring_mask(h: int, w: int, cy: float, cx: float, r_inner: float, r_outer: float, edge: float, device: torch.device) -> torch.Tensor:
    """Soft-edged ring/band mask in [0,1]."""
    yy, xx = make_meshgrid(h, w, device)
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = torch.sigmoid((r_inner - r) / (edge + 1e-6))  # 1 inside inner radius
    outer = torch.sigmoid((r_outer - r) / (edge + 1e-6))  # 1 inside outer radius
    # Ring band: inside outer but outside inner
    return torch.clamp(outer - inner, 0.0, 1.0)


def generate_dust_field(h: int, w: int, num_motes: int, device: torch.device, difficulty: float) -> torch.Tensor:
    """
    Strong, realistic dust occlusions (rings and discs) as multiplicative field.
    - Elliptical, rotated shapes with rough edges
    - Interior non-uniformity using low-frequency noise
    - Occasional multi-ring structure for 3D-like donuts
    """
    field = torch.ones((h, w), device=device)
    for _ in range(num_motes):
        cx = random.random()
        cy = random.random()
        # Circular base (limit eccentricity)
        base_r = random.uniform(0.03, 0.10 + 0.15 * difficulty)
        edge = random.uniform(0.004, 0.015 + 0.015 * difficulty)
        rough_amp = random.uniform(0.0, 0.04 + 0.05 * difficulty)

        interior_noise = _make_lowfreq_noise(h, w, scale=56, device=device)
        interior = (interior_noise * 0.5 + 0.5)

        mote_type = random.choices(["disc", "ring", "multi"], weights=[0.55, 0.35, 0.10])[0]
        very_dark = random.random() < (0.30 + 0.35 * difficulty)

        if mote_type == "disc":
            mask = _circular_mask_jittered(h, w, cx, cy, base_r, edge, rough_amp, device)
            depth = (random.uniform(0.45, 0.75) if very_dark else random.uniform(0.25, 0.55))
            # Slight edge darkening to emulate shadow boundary
            rr = _radial_distance(h, w, cx, cy, device)
            boundary = base_r
            edge_enhance = torch.clamp(torch.exp(-((rr - boundary) ** 2) / (2 * (edge * 2.5 + 1e-6) ** 2)), 0.0, 1.0)
            shading = 0.85 + 0.5 * interior
            mote = 1.0 - depth * mask * (0.7 + 0.6 * edge_enhance) * shading

        elif mote_type == "ring":
            r_outer = base_r * random.uniform(0.95, 1.25)
            r_inner = r_outer * random.uniform(0.55, 0.85)
            mask = _donut_mask_jittered(h, w, cx, cy, r_inner, r_outer, edge, rough_amp, device)
            depth = (random.uniform(0.35, 0.65) if very_dark else random.uniform(0.18, 0.45))
            shading = 0.9 + 0.4 * interior
            mote = 1.0 - depth * mask * shading

        else:  # multi
            r_outer = base_r * random.uniform(1.0, 1.35)
            r_inner = r_outer * random.uniform(0.55, 0.85)
            ring_outer = _donut_mask_jittered(h, w, cx, cy, r_outer * 0.85, r_outer, edge, rough_amp, device)
            inner = _circular_mask_jittered(h, w, cx, cy, r_inner * 0.8, edge, rough_amp, device)
            d_outer = random.uniform(0.12, 0.35)
            d_inner = random.uniform(0.25, 0.55)
            shading = 0.9 + 0.4 * interior
            mote = 1.0 - (d_outer * ring_outer + d_inner * inner) * shading

        field = field * torch.clamp(mote, 0.05, 3.0)

    return torch.clamp(field, 0.02, 3.0)


def generate_additive_gradient(h: int, w: int, magnitude: float, device: torch.device) -> torch.Tensor:
    yy, xx = make_meshgrid(h, w, device)
    theta = random.uniform(0.0, math.tau)
    # Unit direction
    dx = math.cos(theta)
    dy = math.sin(theta)
    # Project onto direction, normalize to [0,1]
    proj = (xx - 0.5) * dx + (yy - 0.5) * dy
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    grad = magnitude * proj
    return grad


def generate_amp_glow(h: int, w: int, magnitude: float, device: torch.device) -> torch.Tensor:
    yy, xx = make_meshgrid(h, w, device)
    corner = random.choice([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
    cx, cy = corner[1], corner[0]
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Exponential falloff from the corner
    glow = magnitude * torch.exp(-3.0 * (r / (math.sqrt(2))) )
    return glow


def synthesize_fields(h: int, w: int, difficulty: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (F_multiplicative, G_additive)
    """
    # Vignetting
    vig_strength = random.uniform(-0.6, 0.4) * difficulty
    vig_order = random.uniform(2.0, 5.5)
    vig_center_jitter = random.uniform(0.0, 0.12) * difficulty
    F_vig = generate_vignetting_field(h, w, vig_strength, vig_order, vig_center_jitter, device)

    # PRNU
    prnu_amp = random.uniform(0.0, 0.08) * difficulty
    prnu_cell = random.choice([8, 12, 16, 24, 32])
    F_prnu = generate_prnu_field(h, w, prnu_amp, prnu_cell, device)

    # Dust motes
    max_motes = int(2 + 10 * difficulty)
    num_motes = random.randint(0, max_motes)
    F_dust = generate_dust_field(h, w, num_motes, device, difficulty)

    # Combine multiplicative fields
    F_mul = torch.clamp(F_vig * F_prnu * F_dust, 0.1, 3.0)
    # Normalize to mean 1.0 like calibrated flats
    F_mul = F_mul / (F_mul.mean() + 1e-8)

    # Additive gradient + amp glow
    grad_mag = random.uniform(0.0, 0.20) * difficulty
    G_grad = generate_additive_gradient(h, w, grad_mag, device)
    glow_mag = random.uniform(0.0, 0.15) * difficulty
    G_glow = generate_amp_glow(h, w, glow_mag, device)
    G_add = torch.clamp(G_grad + G_glow, 0.0, 1.0)

    return F_mul, G_add


def apply_poisson_read_noise(image_lin: torch.Tensor, difficulty: float) -> torch.Tensor:
    """
    Optional shot + read noise model. image_lin is [0,1].
    """
    if difficulty <= 0.05:
        return image_lin
    # Scale to electrons
    e_min, e_max = 5000.0, 60000.0
    electrons = random.uniform(e_min, e_max)
    lam = torch.clamp(image_lin, 0.0, 1.0) * electrons
    # Poisson noise in float via gaussian approx when electrons large
    noisy = lam + torch.randn_like(lam) * torch.sqrt(torch.clamp(lam, 1.0, None))
    noisy = torch.clamp(noisy, 0.0, None)
    # Add read noise (e-)
    read_sigma = random.uniform(1.0, 10.0) * math.sqrt(difficulty)
    noisy = noisy + torch.randn_like(noisy) * read_sigma
    noisy = torch.clamp(noisy, 0.0, None)
    # Back to [0,1]
    return torch.clamp(noisy / (electrons + 1e-8), 0.0, 1.0)


# =============================
# 4) Main generation loop
# =============================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare output directories
    for sub in ["train/sharp", "train/affected", "train/flat", "train/grad",
                "val/sharp", "val/affected", "val/flat", "val/grad"]:
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, sub), exist_ok=True)

    # Enumerate images
    valid_ext = [".fits", ".fit", ".jpg", ".jpeg", ".png"]
    files = [f for f in os.listdir(SHARP_IMAGE_DIR) if os.path.splitext(f)[1].lower() in valid_ext]
    if not files:
        print(f"No valid images found in {SHARP_IMAGE_DIR}")
        raise SystemExit(1)
    print(f"Found {len(files)} source images")

    patch_counter = 0

    for fname in tqdm(files, desc="Generating flat-field dataset"):
        path = os.path.join(SHARP_IMAGE_DIR, fname)
        img = load_astro_image(path)
        if img is None:
            continue
        h, w = img.shape
        if h < PATCH_SIZE or w < PATCH_SIZE:
            continue

        # Preconvert to torch to avoid repeated transfers
        img_t = torch.from_numpy(img).to(device)

        for _ in range(PATCHES_PER_IMAGE):
            # Curriculum difficulty from 0.1 to 1.0
            difficulty = min(1.0, 0.1 + 0.9 * (patch_counter / CURRICULUM_WARMUP_PATCHES))

            y0 = random.randint(0, h - PATCH_SIZE)
            x0 = random.randint(0, w - PATCH_SIZE)
            gt_patch = img_t[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]

            # Normalize sharp patch to [0,1] per patch
            gt_min = gt_patch.min()
            gt_max = gt_patch.max()
            if float(gt_max) > float(gt_min):
                gt_lin = (gt_patch - gt_min) / (gt_max - gt_min)
            else:
                gt_lin = torch.zeros_like(gt_patch)

            # Synthesize fields
            F_mul, G_add = synthesize_fields(PATCH_SIZE, PATCH_SIZE, difficulty, device)

            # Compose affected image in linear domain
            affected = torch.clamp(gt_lin * F_mul + G_add, 0.0, 1.0)

            # Optional noise
            affected = apply_poisson_read_noise(affected, difficulty)

            # Move to CPU numpy
            gt_np = gt_lin.detach().cpu().numpy().astype(SAVE_DTYPE)
            aff_np = affected.detach().cpu().numpy().astype(SAVE_DTYPE)
            flat_np = F_mul.detach().cpu().numpy().astype(SAVE_DTYPE)
            grad_np = G_add.detach().cpu().numpy().astype(SAVE_DTYPE)

            # Save
            subset = "train" if random.random() < TRAIN_VAL_SPLIT else "val"
            base = f"patch_{patch_counter:06d}.npy"
            np.save(os.path.join(OUTPUT_DATA_DIR, subset, "sharp", base), gt_np)
            np.save(os.path.join(OUTPUT_DATA_DIR, subset, "affected", base), aff_np)
            np.save(os.path.join(OUTPUT_DATA_DIR, subset, "flat", base), flat_np)
            np.save(os.path.join(OUTPUT_DATA_DIR, subset, "grad", base), grad_np)

            patch_counter += 1

    print(f"\nFlat-field dataset complete. Generated {patch_counter} samples.")
    print(f"Saved to: {OUTPUT_DATA_DIR}")


