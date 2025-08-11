# run_correction.py - Apply Trained U-Net Model to Correct Astronomical Images

import os
import torch
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import time

# We assume the model architecture is defined in unet_model.py
from unet_model import AttentionResUNet

# Handle potential missing imports
try:
    import xisf
except ImportError:
    print("Warning: The 'xisf' library is not installed. XISF support is disabled.")
    print("Install it with: 'pip install xisf'")
    xisf = None

try:
    from PIL import Image
except ImportError:
    print("Warning: The 'Pillow' library is not installed. JPG/PNG support is disabled.")
    print("Install it with: 'pip install Pillow'")
    Image = None

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
# --- IMPORTANT: User must set these paths ---
# Path to the trained U-Net model file (.pth)
MODEL_PATH = "./unet_pinn_checkpoint.pth"  # default; adjust to flat model as needed
# Path to the large astronomical image you want to correct
IMAGE_PATH_TO_CORRECT = "./Image34.fit"  # Test with FITS image
# Path to save the final, corrected image
CORRECTED_IMAGE_SAVE_PATH = "./test_corrected_rgb.fits"

# -- Processing Parameters --
TILE_SIZE = 256  # The model was trained on 256x256 patches
TILE_OVERLAP = 48  # How much the tiles should overlap to avoid edge artifacts
USE_GLOBAL_NORMALIZATION = True  # use global robust normalization to reduce seams/brightness drift
ENABLE_CONSERVATIVE_BLEND = True  # blend output with input to make surgical corrections
CONSERVATIVE_STRENGTH = 0.7  # 0..1, higher = stronger corrections

# =============================================================================
# --- 2. HELPER FUNCTIONS ---
# =============================================================================

def load_trained_unet(model_path, device):
    """Loads the trained U-Net model from a .pth file."""
    print(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = AttentionResUNet()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"   Model loaded successfully to {device}.")
    return model

def load_astro_image(image_path):
    """Loads an image from FITS, XISF, JPG, or PNG format."""
    if not os.path.exists(image_path):
        return None
    
    _, file_extension = os.path.splitext(image_path)
    file_ext = file_extension.lower()

    try:
        if file_ext in ['.fits', '.fit']:
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data
        elif file_ext == '.xisf':
            if xisf is None:
                raise ImportError("XISF file provided, but 'xisf' library is not installed.")
            image_data = xisf.read(image_path)[0].data
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            if Image is None:
                raise ImportError("JPG/PNG file provided, but 'Pillow' library is not installed.")
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                image_data = np.array(img)
                # Move channel axis to first position for consistency: (H,W,3) -> (3,H,W)
                image_data = np.transpose(image_data, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Handle different data structures
        if image_data.ndim == 2:
            # Grayscale image
            print(f"   Loaded grayscale image: {image_data.shape}")
        elif image_data.ndim == 3:
            if image_data.shape[0] == 3:
                # RGB image with channels first (3, H, W)
                print(f"   Loaded RGB image: {image_data.shape}")
            elif image_data.shape[2] == 3:
                # RGB image with channels last (H, W, 3) - move to first
                image_data = np.transpose(image_data, (2, 0, 1))
                print(f"   Loaded RGB image (transposed): {image_data.shape}")
            else:
                # Uncertain 3D structure - flatten to mean
                print(f"   Warning: Uncertain 3D structure {image_data.shape}, converting to grayscale")
                image_data = np.mean(image_data, axis=0)
        else:
            raise ValueError(f"Unsupported image dimensions: {image_data.shape}")
        
        return image_data.astype(np.float32)

    except Exception as e:
        print(f"   [ERROR] Could not load image: {e}")
        return None

def preprocess_tile(tile_np, global_min=None, global_max=None):
    """Preprocess a tile with global or per-tile normalization. Returns (tensor, used_min, used_max)."""
    if (global_min is not None) and (global_max is not None) and (global_max > global_min):
        tile01 = (tile_np - global_min) / (global_max - global_min)
        tile01 = np.clip(tile01, 0.0, 1.0)
        used_min, used_max = global_min, global_max
    else:
        tmin, tmax = tile_np.min(), tile_np.max()
        used_min, used_max = float(tmin), float(tmax)
        if tmax > tmin:
            tile01 = (tile_np - tmin) / (tmax - tmin)
        else:
            tile01 = np.zeros_like(tile_np)
    tile = tile01 * 2.0 - 1.0
    tensor = torch.from_numpy(tile.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor, used_min, used_max

def postprocess_tile(tile_tensor):
    """Map model output from [-1,1] to [0,1] float."""
    tile_np = tile_tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
    return np.clip((tile_np + 1.0) / 2.0, 0.0, 1.0)

def create_blending_window(size):
    """
    Creates a 2D Hanning window for smooth tile blending.
    """
    window_1d = np.hanning(size)
    window_2d = window_1d[:, None] * window_1d[None, :]
    return window_2d

def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)

def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    kernel = _gaussian_kernel1d(sigma)
    # horizontal
    pad_w = len(kernel) // 2
    tmp = np.pad(image, ((0, 0), (pad_w, pad_w)), mode='reflect')
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 1, tmp)
    tmp = tmp[:, pad_w:-pad_w]
    # vertical
    tmp = np.pad(tmp, ((pad_w, pad_w), (0, 0)), mode='reflect')
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 0, tmp)
    out = out[pad_w:-pad_w, :]
    return out.astype(np.float32)

def _gradient_magnitude(img01: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img01.astype(np.float32))
    g = np.sqrt(gx * gx + gy * gy)
    return g.astype(np.float32)

def conservative_blend(original: np.ndarray, corrected: np.ndarray, aggressiveness: float = 0.7) -> np.ndarray:
    # Robustly map to [0,1] using original's range
    p_lo, p_hi = np.percentile(original, [0.5, 99.5])
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        p_lo, p_hi = float(np.min(original)), float(np.max(original))
        if p_hi <= p_lo:
            return corrected
    orig01 = np.clip((original - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)
    corr01 = np.clip((corrected - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)

    diff = corr01 - orig01
    blur_sigma = 3.0 + 7.0 * float(np.clip(aggressiveness, 0.0, 1.0))
    m0 = _gaussian_blur(np.abs(diff), sigma=blur_sigma)

    # Suppress changes near strong edges/structures
    g = _gradient_magnitude(_gaussian_blur(orig01, sigma=1.0))
    g_norm = g / (np.percentile(g, 99.0) + 1e-6)
    g_norm = np.clip(g_norm, 0.0, 1.0)

    power = 0.5 + 1.5 * float(np.clip(aggressiveness, 0.0, 1.0))
    mask = m0 * np.power(1.0 - g_norm, power)

    # Emphasize only meaningful differences
    t = np.percentile(m0, 75.0 + 20.0 * float(np.clip(aggressiveness, 0.0, 1.0)))
    if np.isfinite(t) and t > 0:
        mask = np.where(m0 >= t, mask, mask * 0.2)
    mask = np.clip(mask * (0.5 + 0.5 * aggressiveness), 0.0, 1.0).astype(np.float32)

    return (original + mask * (corrected - original)).astype(np.float32)

def rescale_tile_to_original_range(corrected_tile, original_tile):
    """
    Rescales the corrected tile to match the brightness range of the original tile.
    This helps maintain the overall brightness characteristics of the image.
    """
    original_min, original_max = original_tile.min(), original_tile.max()
    if original_max > original_min:
        # Scale corrected tile from [0,1] to original range
        rescaled_tile = corrected_tile * (original_max - original_min) + original_min
    else:
        rescaled_tile = np.full_like(corrected_tile, original_min)
    
    return rescaled_tile

def _process_single_channel(image_2d, model, device):
    """
    Processes a single 2D grayscale channel. Contains the core tiling and blending logic.
    """
    # Optionally compute global robust range for normalization
    print(f"Processing image with shape: {image_2d.shape}, range: [{image_2d.min():.3f}, {image_2d.max():.3f}]")
    
    # Store original range for final rescaling (if not using global min/max)
    original_min, original_max = float(np.min(image_2d)), float(np.max(image_2d))
    global_min = global_max = None
    if USE_GLOBAL_NORMALIZATION:
        # Preserve faint background by anchoring to the true minimum
        # and using a high percentile to ignore only extreme highlights
        global_min = float(np.min(image_2d))
        global_max = float(np.percentile(image_2d, 99.99))
        if not np.isfinite(global_max) or global_max <= global_min:
            global_min, global_max = float(np.min(image_2d)), float(np.max(image_2d))
    
    # 2. Tiled Processing
    corrected_image = np.zeros_like(image_2d, dtype=np.float32)
    weight_map = np.zeros_like(image_2d, dtype=np.float32)
    
    TILE_SIZE, TILE_OVERLAP = 256, 48
    window = np.hanning(TILE_SIZE)[:, None] * np.hanning(TILE_SIZE)[None, :]
    step = TILE_SIZE - TILE_OVERLAP
    
    x_coords = list(range(0, image_2d.shape[1] - TILE_SIZE + 1, step))
    y_coords = list(range(0, image_2d.shape[0] - TILE_SIZE + 1, step))
    if x_coords[-1] < image_2d.shape[1] - TILE_SIZE: x_coords.append(image_2d.shape[1] - TILE_SIZE)
    if y_coords[-1] < image_2d.shape[0] - TILE_SIZE: y_coords.append(image_2d.shape[0] - TILE_SIZE)
            
    for y in y_coords:
        for x in x_coords:
            tile = image_2d[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.std() < 1e-8:
                corrected = tile.astype(np.float32)
            else:
                input_tensor, used_min, used_max = preprocess_tile(tile, global_min, global_max)
                input_tensor = input_tensor.to(device)
                with torch.no_grad():
                    corrected_tensor = model(input_tensor)
                    # Clamp model output to the trained range to avoid saturation/clipping artifacts
                    corrected_tensor = torch.clamp(corrected_tensor, -1.0, 1.0)
                corrected01 = postprocess_tile(corrected_tensor)
                corrected = corrected01 * (used_max - used_min) + used_min
            corrected_image[y:y+TILE_SIZE, x:x+TILE_SIZE] += corrected * window
            weight_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += window

    # 3. Final Blending
    weight_map[weight_map == 0] = 1.0
    corrected_image /= weight_map
    if ENABLE_CONSERVATIVE_BLEND:
        corrected_image = conservative_blend(image_2d.astype(np.float32), corrected_image.astype(np.float32), CONSERVATIVE_STRENGTH)
    return corrected_image.astype(np.float32)

# =============================================================================
# --- 3. MAIN EXECUTION LOGIC ---
# =============================================================================

def main():
    print("=== U-Net Astronomical Image Correction ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Input Image: {IMAGE_PATH_TO_CORRECT}")
    print(f"Output Image: {CORRECTED_IMAGE_SAVE_PATH}")
    print(f"Tile Size: {TILE_SIZE}x{TILE_SIZE}, Overlap: {TILE_OVERLAP}")
    print("=" * 50)
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Image ---
    try:
        model = load_trained_unet(MODEL_PATH, device)
        raw_image = load_astro_image(IMAGE_PATH_TO_CORRECT)
        if raw_image is None:
            print("Failed to load image. Exiting.")
            return
    except Exception as e:
        print(f"Error during initialization: {e}")
        return
    
    start_time = time.time()
    
    if raw_image.ndim == 2:
        print("-> Processing grayscale image...")
        final_image = _process_single_channel(raw_image, model, device)
    elif raw_image.ndim == 3 and raw_image.shape[0] == 3:
        print("-> Processing RGB image (channel by channel)...")
        corrected_channels = []
        for i in range(3):
            print(f"   Processing channel {i+1}/3...")
            channel_data = raw_image[i]  # Extract channel from (3, H, W)
            corrected_channel = _process_single_channel(channel_data, model, device)
            corrected_channels.append(corrected_channel)
        # Stack channels back to (3, H, W) format for FITS
        final_image = np.stack(corrected_channels, axis=0)
    else:
        print(f"[ERROR] Unsupported image shape for processing: {raw_image.shape}")
        return

    # 5. Save Final Image
    print(f"-> Saving corrected image to: {CORRECTED_IMAGE_SAVE_PATH}")
    try:
        def _sanitize_header(header):
            if header is None:
                return None
            hdr = header.copy()
            for key in ['BSCALE', 'BZERO', 'BLANK', 'DATAMIN', 'DATAMAX', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2']:
                if key in hdr:
                    del hdr[key]
            return hdr

        original_header = None
        if IMAGE_PATH_TO_CORRECT.lower().endswith(('.fits', '.fit')):
            with fits.open(IMAGE_PATH_TO_CORRECT) as hdul:
                original_header = _sanitize_header(hdul[0].header)

        hdu = fits.PrimaryHDU(final_image.astype(np.float32), header=original_header)
        hdu.header.add_history('Corrected using U-Net neural network')
        hdu.header.add_history(f'Original file: {os.path.basename(IMAGE_PATH_TO_CORRECT)}')
        hdu.header.add_history(f'Model used: {os.path.basename(MODEL_PATH)}')
        fits.HDUList([hdu]).writeto(CORRECTED_IMAGE_SAVE_PATH, overwrite=True)
        
        print(f"✓ Correction complete! Final image saved to: {CORRECTED_IMAGE_SAVE_PATH}")
        
        # Print some statistics
        improvement_ratio = np.std(final_image) / np.std(raw_image)
        print(f"\nImage Statistics:")
        print(f"  Original std: {np.std(raw_image):.2f}")
        print(f"  Corrected std: {np.std(final_image):.2f}")
        print(f"  Improvement ratio: {improvement_ratio:.3f}")
        
    except Exception as e:
        print(f"Error saving final image: {e}")

if __name__ == '__main__':
    main() 