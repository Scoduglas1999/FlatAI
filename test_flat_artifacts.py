import os
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from create_flat_dataset import (
    load_astro_image,
    generate_vignetting_field,
    generate_prnu_field,
    generate_dust_field,
    generate_additive_gradient,
    generate_amp_glow,
    synthesize_fields,
)


OUTPUT_DIR = "./flat_demo_samples/"
PATCH_SIZE = 256
SOURCE_DIR = "./sharp_images/"


def generate_synthetic_gt(h: int, w: int, num_stars: int = 200) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    img = np.zeros((h, w), dtype=np.float32)
    for _ in range(num_stars):
        x0 = np.random.rand()
        y0 = np.random.rand()
        amp = 0.2 + 0.8 * np.random.rand()
        sigma = 0.003 + 0.02 * np.random.rand()
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        img += amp * np.exp(-0.5 * r2 / (sigma**2))
    # mild background gradient
    grad = 0.05 * (xx * 0.7 + yy * 0.3)
    img = np.clip(img + grad, 0.0, None)
    img = img / (img.max() + 1e-8)
    return img.astype(np.float32)


def load_any_patch() -> np.ndarray:
    files = sorted(os.listdir(SOURCE_DIR)) if os.path.exists(SOURCE_DIR) else []
    for fname in files:
        if os.path.splitext(fname)[1].lower() not in [".fits", ".fit", ".jpg", ".jpeg", ".png"]:
            continue
        arr = load_astro_image(os.path.join(SOURCE_DIR, fname))
        if arr is None:
            continue
        h, w = arr.shape
        if h < PATCH_SIZE or w < PATCH_SIZE:
            continue
        y0 = np.random.randint(0, h - PATCH_SIZE + 1)
        x0 = np.random.randint(0, w - PATCH_SIZE + 1)
        patch = arr[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE].astype(np.float32)
        pmin, pmax = patch.min(), patch.max()
        if pmax > pmin:
            patch = (patch - pmin) / (pmax - pmin)
        else:
            patch = np.zeros_like(patch, dtype=np.float32)
        return patch
    # Fallback synthetic
    return generate_synthetic_gt(PATCH_SIZE, PATCH_SIZE)


def visualize_case(tag: str, gt01: np.ndarray, F_map: torch.Tensor, G_map: torch.Tensor, device: torch.device):
    gt_t = torch.from_numpy(gt01).to(device)
    F = F_map
    G = G_map
    # Compose
    mul_only = torch.clamp(gt_t * F, 0.0, 1.0)
    add_only = torch.clamp(gt_t + G, 0.0, 1.0)
    affected = torch.clamp(gt_t * F + G, 0.0, 1.0)
    # To CPU numpy
    F_viz = F.detach().cpu().numpy().astype(np.float32)
    # Normalize F for display to [0,1], keeping relative contrast around 1
    F_show = (F_viz - F_viz.min()) / (F_viz.max() - F_viz.min() + 1e-8)
    G_show = G.detach().cpu().numpy().astype(np.float32)
    gt_show = gt01
    mul_show = mul_only.detach().cpu().numpy().astype(np.float32)
    add_show = add_only.detach().cpu().numpy().astype(np.float32)
    aff_show = affected.detach().cpu().numpy().astype(np.float32)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0,0].imshow(gt_show, cmap='gray'); axs[0,0].set_title('Ground Truth'); axs[0,0].axis('off')
    axs[0,1].imshow(F_show, cmap='magma'); axs[0,1].set_title('Multiplicative F'); axs[0,1].axis('off')
    axs[0,2].imshow(G_show, cmap='inferno'); axs[0,2].set_title('Additive G'); axs[0,2].axis('off')
    axs[1,0].imshow(mul_show, cmap='gray'); axs[1,0].set_title('GT * F (only)'); axs[1,0].axis('off')
    axs[1,1].imshow(add_show, cmap='gray'); axs[1,1].set_title('GT + G (only)'); axs[1,1].axis('off')
    axs[1,2].imshow(aff_show, cmap='gray'); axs[1,2].set_title('Affected = GT*F + G'); axs[1,2].axis('off')
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, f"demo_{tag}.png")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Generate demo flat-field artifacts")
    parser.add_argument("--seed", type=str, default="random",
                        help="Seed for RNG: integer, 'random' for a random seed, or 'none' to leave RNG unseeded")
    args = parser.parse_args()

    # Seeding policy
    seed_arg = args.seed.lower() if isinstance(args.seed, str) else args.seed
    if seed_arg == "random":
        seed_val = int.from_bytes(os.urandom(8), "little")
        print(f"Using random seed: {seed_val}")
        random.seed(seed_val)
        np.random.seed(seed_val % (2**32 - 1))
        torch.manual_seed(seed_val % (2**31 - 1))
    elif seed_arg == "none":
        print("Not setting seeds (non-deterministic run)")
    else:
        try:
            seed_val = int(args.seed)
            print(f"Using fixed seed: {seed_val}")
            random.seed(seed_val)
            np.random.seed(seed_val % (2**32 - 1))
            torch.manual_seed(seed_val % (2**31 - 1))
        except Exception:
            print("Unrecognized seed value; proceeding without explicit seeding")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt01 = load_any_patch()
    h, w = gt01.shape

    # Vignetting (dark corners)
    F_vig = generate_vignetting_field(h, w, strength=-0.5, order=3.5, center_jitter=0.02, device=device)
    G_zero = torch.zeros((h, w), device=device)
    visualize_case("vignetting_dark", gt01, F_vig / (F_vig.mean() + 1e-8), G_zero, device)

    # Vignetting (bright corners)
    F_vig_b = generate_vignetting_field(h, w, strength=+0.4, order=3.0, center_jitter=0.0, device=device)
    visualize_case("vignetting_bright", gt01, F_vig_b / (F_vig_b.mean() + 1e-8), G_zero, device)

    # Dust motes (dominant common donuts, thicker rings, smaller fuzzy centers)
    override = {
        'thickness_range': (0.35, 0.65),
        'center_rel_range': (0.12, 0.35),
        'center_scale_range': (0.45, 0.8),
        'edge_softness_scale': (1.6, 2.6),
        'rough_amp_scale': 1.0,
    }
    F_dust_soft = generate_dust_field(
        h, w, num_motes=6, device=device, difficulty=0.5,
        mote_weights=(0.85, 0.12, 0.02, 0.01),
        common_params_override=override,
    )
    F_dust_strong = generate_dust_field(
        h, w, num_motes=10, device=device, difficulty=1.0,
        mote_weights=(0.85, 0.10, 0.03, 0.02),
        common_params_override=override,
    )
    visualize_case("dust_motes_soft", gt01, F_dust_soft / (F_dust_soft.mean() + 1e-8), G_zero, device)
    visualize_case("dust_motes_strong", gt01, F_dust_strong / (F_dust_strong.mean() + 1e-8), G_zero, device)

    # PRNU only
    F_prnu = generate_prnu_field(h, w, amplitude=0.06, cell=16, device=device)
    visualize_case("prnu", gt01, F_prnu / (F_prnu.mean() + 1e-8), G_zero, device)

    # Gradient only
    G_grad = generate_additive_gradient(h, w, magnitude=0.15, device=device)
    F_one = torch.ones((h, w), device=device)
    visualize_case("gradient", gt01, F_one, G_grad, device)

    # Amp glow only
    G_glow = generate_amp_glow(h, w, magnitude=0.18, device=device)
    visualize_case("amp_glow", gt01, F_one, G_glow, device)

    # Combined cocktail via synthesize_fields
    F_mul, G_add = synthesize_fields(h, w, difficulty=1.0, device=device)
    visualize_case("combined", gt01, F_mul, G_add, device)


if __name__ == "__main__":
    main()


