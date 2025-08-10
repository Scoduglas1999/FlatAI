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
MODEL_PATH = "./unet_pinn_checkpoint.pth"  # Use the available PINN model
# Path to the large astronomical image you want to correct
IMAGE_PATH_TO_CORRECT = "./Image34.fit"  # Test with FITS image
# Path to save the final, corrected image
CORRECTED_IMAGE_SAVE_PATH = "./test_corrected_rgb.fits"

# -- Processing Parameters --
TILE_SIZE = 256  # The model was trained on 256x256 patches
TILE_OVERLAP = 48  # How much the tiles should overlap to avoid edge artifacts

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

def preprocess_tile(tile_np):
    """Preprocesses a NumPy tile for the U-Net model using the same normalization as training."""
    # Apply the EXACT same normalization logic as training:
    # 1. Normalize tile to [0, 1] using tile's own min/max (per-patch normalization)
    tile_min, tile_max = tile_np.min(), tile_np.max()
    if tile_max > tile_min:
        tile_normalized_01 = (tile_np - tile_min) / (tile_max - tile_min)
    else:
        tile_normalized_01 = tile_np - tile_np.mean()  # Handle flat tiles
    
    # 2. Convert [0, 1] to [-1, 1] (exactly like the dataset loader)
    tile_normalized = tile_normalized_01 * 2.0 - 1.0
    
    tile_tensor = torch.from_numpy(tile_normalized.astype(np.float32))
    tile_tensor = tile_tensor.unsqueeze(0).unsqueeze(0)
    return tile_tensor

def postprocess_tile(tile_tensor, original_tile):
    """Post-processes a corrected tensor from the model back to a NumPy array in [0, 1] range."""
    tile_np = tile_tensor.squeeze(0).squeeze(0)
    tile_np = tile_np.cpu().detach().numpy()
    tile_np = (tile_np + 1.0) / 2.0
    return np.clip(tile_np, 0, 1)

def create_blending_window(size):
    """
    Creates a 2D Hanning window for smooth tile blending.
    """
    window_1d = np.hanning(size)
    window_2d = window_1d[:, None] * window_1d[None, :]
    return window_2d

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
    # No global normalization needed - preprocessing handles per-tile normalization
    print(f"Processing image with shape: {image_2d.shape}, range: [{image_2d.min():.3f}, {image_2d.max():.3f}]")
    
    # Store original range for final rescaling
    original_min, original_max = np.min(image_2d), np.max(image_2d)
    
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
            # Extract raw tile from original data
            tile = image_2d[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            # Skip processing if tile has no dynamic range
            if tile.std() < 1e-8:
                corrected_tile = tile.copy()
            else:
                # Preprocessing now handles per-tile normalization to [-1, 1]
                input_tensor = preprocess_tile(tile).to(device)
                with torch.no_grad():
                    corrected_tensor = model(input_tensor)
                
                # Postprocess: Convert from [-1, 1] back to tile's original range
                corrected_tile = postprocess_tile(corrected_tensor, tile)
            
            corrected_image[y:y+TILE_SIZE, x:x+TILE_SIZE] += corrected_tile * window
            weight_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += window

    # 3. Final Blending
    weight_map[weight_map == 0] = 1.0
    corrected_image /= weight_map
    
    return corrected_image

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
        original_header = None
        if IMAGE_PATH_TO_CORRECT.lower().endswith(('.fits', '.fit')):
            with fits.open(IMAGE_PATH_TO_CORRECT) as hdul:
                original_header = hdul[0].header
        
        hdu = fits.PrimaryHDU(final_image.astype(np.float32), header=original_header)
        hdu.header['COMMENT'] = 'Corrected using U-Net neural network'
        hdu.header['COMMENT'] = f'Original file: {os.path.basename(IMAGE_PATH_TO_CORRECT)}'
        hdu.header['COMMENT'] = f'Model used: {os.path.basename(MODEL_PATH)}'
        hdul = fits.HDUList([hdu])
        hdul.writeto(CORRECTED_IMAGE_SAVE_PATH, overwrite=True)
        hdul.close()
        
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