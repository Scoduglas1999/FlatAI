import math
import random
from typing import Tuple

import torch
import torch.nn.functional as F


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


def _make_lowfreq_noise(h: int, w: int, scale: int, device: torch.device) -> torch.Tensor:
    """Perlin-like low-frequency noise in [-1,1] via bicubic upsampling of a coarse grid."""
    coarse_h = max(2, h // max(1, scale))
    coarse_w = max(2, w // max(1, scale))
    grid = torch.rand((1, 1, coarse_h, coarse_w), device=device)
    up = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=True)
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    return up.squeeze(0).squeeze(0) * 2.0 - 1.0


def _radial_distance(h: int, w: int, cx: float, cy: float, device: torch.device) -> torch.Tensor:
    yy, xx = make_meshgrid(h, w, device)
    return torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def _softstep(x: torch.Tensor, k: float) -> torch.Tensor:
    """Fast smooth step using sigmoid; k controls edge softness."""
    return torch.sigmoid(x / (k + 1e-6))


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


def _common_donut_mote(h: int, w: int, cx: float, cy: float,
                       r0: float, thickness: float, edge: float,
                       rough_amp: float, depth: float,
                       center_rel: float, center_scale: float,
                       device: torch.device) -> torch.Tensor:
    """
    Generate a typical dust mote donut: a darker soft ring with a slightly
    lighter interior. Kept circular with modest boundary roughness.
    Returns multiplicative field contribution in [0.05, 1.0+]. Will be clamped later.
    """
    rr = _radial_distance(h, w, cx, cy, device)
    jitter = _make_lowfreq_noise(h, w, scale=48, device=device) * rough_amp
    rr_j = rr * (1.0 + jitter)
    sigma_ring = max(1e-4, thickness / 2.355)
    ring = torch.exp(-0.5 * ((rr_j - r0) / (sigma_ring + 1e-6)) ** 2)
    ring = ring * _softstep(r0 + thickness - rr_j, edge)
    ring = ring * _softstep(rr_j - (r0 - thickness), edge)
    sigma_center = max(1e-4, r0 * center_rel)
    center = torch.exp(-0.5 * (rr / (sigma_center + 1e-6)) ** 2)
    center_depth = depth * center_scale
    mote = 1.0 - depth * ring - center_depth * center
    mote = torch.clamp(mote, 0.05, 1.05)
    return mote


def generate_dust_field(
    h: int,
    w: int,
    num_motes: int,
    device: torch.device,
    difficulty: float,
    mote_weights: tuple | list | None = None,
    common_params_override: dict | None = None,
) -> torch.Tensor:
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
        base_r = random.uniform(0.03, 0.10 + 0.15 * difficulty)
        edge = random.uniform(0.004, 0.015 + 0.015 * difficulty)
        rough_amp = random.uniform(0.0, 0.04 + 0.05 * difficulty)
        interior_noise = _make_lowfreq_noise(h, w, scale=56, device=device)
        interior = (interior_noise * 0.5 + 0.5)
        if mote_weights is None:
            weights = [0.65, 0.18, 0.12, 0.05]
        else:
            weights = list(mote_weights)
            if len(weights) != 4:
                raise ValueError("mote_weights must be a sequence of 4 numbers: (common, disc, ring, multi)")
        mote_type = random.choices(["common", "disc", "ring", "multi"], weights=weights)[0]
        very_dark = random.random() < (0.30 + 0.35 * difficulty)

        if mote_type == "common":
            tr = (0.28, 0.55)
            cr = (0.18, 0.45)
            cs = (0.35, 0.70)
            es = (1.2, 2.2)
            ra = 0.8
            if common_params_override:
                tr = common_params_override.get('thickness_range', tr)
                cr = common_params_override.get('center_rel_range', cr)
                cs = common_params_override.get('center_scale_range', cs)
                es = common_params_override.get('edge_softness_scale', es)
                ra = common_params_override.get('rough_amp_scale', ra)

            thickness = base_r * random.uniform(*tr)
            depth = (random.uniform(0.35, 0.75) if very_dark else random.uniform(0.22, 0.55))
            center_rel = random.uniform(*cr)
            center_scale = random.uniform(*cs)
            edge_eff = edge * random.uniform(*es)
            mote = _common_donut_mote(
                h, w, cx, cy, base_r, thickness, edge_eff, rough_amp * ra,
                depth, center_rel, center_scale, device
            )

        elif mote_type == "disc":
            mask = _circular_mask_jittered(h, w, cx, cy, base_r, edge, rough_amp, device)
            depth = (random.uniform(0.45, 0.75) if very_dark else random.uniform(0.25, 0.55))
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
    dx = math.cos(theta)
    dy = math.sin(theta)
    proj = (xx - 0.5) * dx + (yy - 0.5) * dy
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    grad = magnitude * proj
    return grad


def generate_amp_glow(h: int, w: int, magnitude: float, device: torch.device) -> torch.Tensor:
    yy, xx = make_meshgrid(h, w, device)
    corner = random.choice([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
    cx, cy = corner[1], corner[0]
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glow = magnitude * torch.exp(-3.0 * (r / (math.sqrt(2))) )
    return glow


def synthesize_fields(h: int, w: int, difficulty: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (F_multiplicative, G_additive)
    """
    vig_strength = random.uniform(-0.6, 0.4) * difficulty
    vig_order = random.uniform(2.0, 5.5)
    vig_center_jitter = random.uniform(0.0, 0.12) * difficulty
    F_vig = generate_vignetting_field(h, w, vig_strength, vig_order, vig_center_jitter, device)

    prnu_amp = random.uniform(0.0, 0.08) * difficulty
    prnu_cell = random.choice([8, 12, 16, 24, 32])
    F_prnu = generate_prnu_field(h, w, prnu_amp, prnu_cell, device)

    max_motes = int(2 + 10 * difficulty)
    num_motes = random.randint(0, max_motes)
    F_dust = generate_dust_field(h, w, num_motes, device, difficulty)

    F_mul = torch.clamp(F_vig * F_prnu * F_dust, 0.1, 3.0)
    F_mul = F_mul / (F_mul.mean() + 1e-8)

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
    e_min, e_max = 5000.0, 60000.0
    electrons = random.uniform(e_min, e_max)
    lam = torch.clamp(image_lin, 0.0, 1.0) * electrons
    noisy = lam + torch.randn_like(lam) * torch.sqrt(torch.clamp(lam, 1.0, None))
    noisy = torch.clamp(noisy, 0.0, None)
    read_sigma = random.uniform(1.0, 10.0) * math.sqrt(difficulty)
    noisy = noisy + torch.randn_like(noisy) * read_sigma
    noisy = torch.clamp(noisy, 0.0, None)
    return torch.clamp(noisy / (electrons + 1e-8), 0.0, 1.0)
