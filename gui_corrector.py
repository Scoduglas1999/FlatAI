# gui_corrector.py - FlatAI: Modern Flat-Field Correction Interface

# =============================================================================
# --- CONFIGURATION FOR TESTING ---
# =============================================================================
# Processing mode control
# None = auto-detect (preferred); True = force linear; False = force non-linear
FORCE_LINEAR_MODE = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
from tqdm import tqdm
import time

# Handle potential missing imports
try:
    import xisf
except ImportError:
    xisf = None

try:
    from PIL import Image
except ImportError:
    Image = None

# --- Import the correct model architecture ---
from unet_model import AttentionResUNet

# Default model paths
PREFERRED_MODEL = "./unet_flat_model_final.pth"
FALLBACK_MODEL = "./unet_flat_checkpoint.pth"

# =============================================================================
# --- BACKEND PROCESSING LOGIC (keeping existing functions) ---
# =============================================================================

# Correct UNet Model Definition (matches the trained model)
def double_conv(in_channels, out_channels):
    """
    A helper function that creates a block of two convolutional layers,
    each followed by a ReLU activation function.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    """
    A standard U-Net architecture for image-to-image translation tasks.
    This matches the exact architecture used in training.
    """
    def __init__(self):
        super(UNet, self).__init__()

        # --- Encoder (Down-sampling Path) ---
        self.d_conv1 = double_conv(1, 64)
        self.d_conv2 = double_conv(64, 128)
        self.d_conv3 = double_conv(128, 256)
        self.d_conv4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = double_conv(512, 1024)

        # --- Decoder (Up-sampling Path) ---
        self.up_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.u_conv1 = double_conv(1024, 512) # 512 from skip + 512 from up_trans

        self.up_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.u_conv2 = double_conv(512, 256) # 256 from skip + 256 from up_trans

        self.up_trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.u_conv3 = double_conv(256, 128) # 128 from skip + 128 from up_trans

        self.up_trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.u_conv4 = double_conv(128, 64) # 64 from skip + 64 from up_trans

        # --- Output Layer ---
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.tanh = nn.Tanh() # Tanh activation to map output to [-1, 1]

    def forward(self, x):
        # Encoder
        x1 = self.d_conv1(x)    # Skip connection 1
        x2 = self.d_conv2(self.pool(x1)) # Skip connection 2
        x3 = self.d_conv3(self.pool(x2)) # Skip connection 3
        x4 = self.d_conv4(self.pool(x3)) # Skip connection 4

        # Bottleneck
        b = self.bottleneck(self.pool(x4))

        # Decoder
        u1 = self.up_trans1(b)
        u1 = torch.cat([u1, x4], dim=1) # Concatenate skip connection
        u1 = self.u_conv1(u1)

        u2 = self.up_trans2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.u_conv2(u2)

        u3 = self.up_trans3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.u_conv3(u3)

        u4 = self.up_trans4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.u_conv4(u4)

        # Output
        output = self.out(u4)
        return self.tanh(output)

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
                # Convert to RGB to handle palettes and remove alpha channel
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
        print(f"Error loading image: {e}")
        return None

def preprocess_tile(tile_np, global_min=None, global_max=None):
    """
    Preprocesses a NumPy tile for the U-Net model using adaptive normalization.
    Uses global normalization for linear data, per-tile for non-linear data.
    """
    if global_min is not None and global_max is not None:
        # Global normalization mode (for linear astronomical data)
        if global_max > global_min:
            tile_normalized_01 = (tile_np - global_min) / (global_max - global_min)
            tile_normalized_01 = np.clip(tile_normalized_01, 0, 1)  # Ensure [0,1] range
        else:
            tile_normalized_01 = np.zeros_like(tile_np)
        
        # Convert [0, 1] to [-1, 1]
        tile_normalized = tile_normalized_01 * 2.0 - 1.0
        
        tile_tensor = torch.from_numpy(tile_normalized.astype(np.float32))
        tile_tensor = tile_tensor.unsqueeze(0).unsqueeze(0)
        return tile_tensor, global_min, global_max
    else:
        # Per-tile normalization mode (for non-linear data like JPG/PNG)
        tile_min, tile_max = tile_np.min(), tile_np.max()
        if tile_max > tile_min:
            tile_normalized_01 = (tile_np - tile_min) / (tile_max - tile_min)
        else:
            tile_normalized_01 = tile_np - tile_np.mean()  # Handle flat tiles
        
        # Convert [0, 1] to [-1, 1]
        tile_normalized = tile_normalized_01 * 2.0 - 1.0
        
        tile_tensor = torch.from_numpy(tile_normalized.astype(np.float32))
        tile_tensor = tile_tensor.unsqueeze(0).unsqueeze(0)
        return tile_tensor, tile_min, tile_max

def postprocess_tile(tile_tensor, tile_min, tile_max):
    """Post-processes a corrected tensor from the model back to a NumPy array in original range."""
    tile_np = tile_tensor.squeeze(0).squeeze(0)
    tile_np = tile_np.cpu().detach().numpy()
    
    # Convert from [-1, 1] back to [0, 1]
    tile_np = (tile_np + 1.0) / 2.0
    tile_np = np.clip(tile_np, 0, 1)
    
    # Scale back to original range
    if tile_max > tile_min:
        tile_np = tile_np * (tile_max - tile_min) + tile_min
    else:
        tile_np = np.full_like(tile_np, tile_min)
    
    return tile_np

def detect_image_type(image_data):
    """
    Detect if image is linear or non-linear based purely on statistical analysis.
    Returns True for linear, False for non-linear.
    """
    # Statistical analysis of data distribution
    # Linear images typically have:
    # 1. Most values concentrated in lower range with long tail toward higher values
    # 2. High dynamic range with extreme outliers
    # 3. Highly skewed distribution
    
    # Calculate key statistics
    data_flat = image_data.flatten()
    
    # Remove extreme outliers for more robust analysis
    p1 = np.percentile(data_flat, 1)
    p99 = np.percentile(data_flat, 99)
    trimmed_data = data_flat[(data_flat >= p1) & (data_flat <= p99)]
    
    if len(trimmed_data) == 0:
        return False  # Fallback for edge cases
    
    # Calculate distribution metrics
    percentile_5 = np.percentile(trimmed_data, 5)
    percentile_50 = np.percentile(trimmed_data, 50)  # median
    percentile_95 = np.percentile(trimmed_data, 95)
    percentile_99 = np.percentile(trimmed_data, 99)
    
    # Metric 1: High dynamic range ratio
    # Linear data often has huge gaps between median and maximum
    if percentile_95 > percentile_5:
        dynamic_range_ratio = (percentile_99 - percentile_95) / (percentile_95 - percentile_5)
    else:
        dynamic_range_ratio = 0
    
    # Metric 2: Skewness of lower portion of data
    # Linear data tends to have most values clustered near the minimum
    lower_quartile_ratio = (percentile_50 - percentile_5) / (percentile_95 - percentile_5 + 1e-10)
    
    # Metric 3: Concentration in lower values
    # What percentage of pixels are in the lower 50% of the range?
    threshold = percentile_5 + 0.5 * (percentile_95 - percentile_5)
    lower_concentration = np.mean(trimmed_data <= threshold)
    
    # Decision logic combining multiple metrics
    is_linear_votes = 0
    
    # Vote 1: High dynamic range suggests linear
    if dynamic_range_ratio > 1.5:
        is_linear_votes += 1
    
    # Vote 2: Low median position suggests linear
    if lower_quartile_ratio < 0.3:
        is_linear_votes += 1
        
    # Vote 3: High concentration in lower values suggests linear
    if lower_concentration > 0.75:
        is_linear_votes += 1
    
    # Require at least 2 out of 3 votes for linear
    is_linear = is_linear_votes >= 2
    
    print(f"   Image analysis: dynamic_ratio={dynamic_range_ratio:.2f}, "
          f"lower_quartile_ratio={lower_quartile_ratio:.2f}, "
          f"lower_concentration={lower_concentration:.2f}, "
          f"votes={is_linear_votes}/3 -> {'LINEAR' if is_linear else 'NON-LINEAR'}")
    
    return is_linear

def detect_image_type_improved(image_data):
    """
    Improved detection for normalized linear data (common in stacked astronomical images).
    """
    # First try the original detection
    is_linear_original = detect_image_type(image_data)
    
    # Additional checks for normalized linear astronomical data
    # These images often have been normalized to [0,1] but retain linear characteristics
    
    # Check for very high star concentration (astronomical hallmark)
    # Count pixels that are significantly above the median
    median_val = np.median(image_data)
    bright_threshold = median_val + 3 * np.std(image_data)
    bright_pixel_ratio = np.mean(image_data > bright_threshold)
    
    # Check background uniformity (astronomical images have relatively uniform backgrounds)
    # Sample random patches and check background consistency
    h, w = image_data.shape
    patch_size = min(64, h//10, w//10)
    if patch_size > 10:
        num_samples = min(20, (h//patch_size) * (w//patch_size))
        background_values = []
        
        for _ in range(num_samples):
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)
            patch = image_data[y:y+patch_size, x:x+patch_size]
            # Use 10th percentile as background estimate
            background_values.append(np.percentile(patch, 10))
        
        background_std = np.std(background_values)
        background_mean = np.mean(background_values)
        background_consistency = background_std / (background_mean + 1e-10)
    else:
        background_consistency = 1.0  # Can't reliably measure
    
    # Decision logic for normalized linear astronomical data
    is_astronomical = False
    
    # Low bright pixel ratio suggests astronomical (most pixels are background)
    if bright_pixel_ratio < 0.01:  # Less than 1% bright pixels
        is_astronomical = True
        
    # Consistent background suggests astronomical
    if background_consistency < 0.3:  # Relatively uniform background
        is_astronomical = True
    
    # For now, force linear mode for astronomical-looking data
    # TODO: Remove this once we're confident in detection
    final_decision = is_linear_original or is_astronomical
    
    print(f"   Improved detection: original={is_linear_original}, "
          f"bright_ratio={bright_pixel_ratio:.4f}, bg_consistency={background_consistency:.3f}, "
          f"astronomical={is_astronomical} -> {'LINEAR' if final_decision else 'NON-LINEAR'}")
    
    return final_decision

def _process_single_channel(image_2d, model, device, progress_callback, status_callback, image_path=None, force_linear=None):
    """
    Processes a single 2D grayscale channel using adaptive normalization strategy.
    Uses global normalization for linear data, per-tile for non-linear data.
    
    Args:
        force_linear: None for auto-detect, True to force linear, False to force non-linear
    """
    print(f"Processing image with shape: {image_2d.shape}, range: [{image_2d.min():.3f}, {image_2d.max():.3f}]")
    
    # Determine processing mode
    if force_linear is not None:
        is_linear = force_linear
        print(f"   Forced mode: {'LINEAR' if is_linear else 'NON-LINEAR'}")
    else:
        # Auto-detect with improved logic for normalized linear data
        is_linear = detect_image_type_improved(image_2d)
    
    if is_linear:
        # Linear data: use robust global normalization
        status_callback("Using LINEAR mode - global normalization...")
        # Use percentiles to handle outliers robustly
        global_min = np.percentile(image_2d, 0.1)  # Slightly above true minimum to handle noise
        global_max = np.percentile(image_2d, 99.9)  # Slightly below true maximum to handle hot pixels
        print(f"   Using global range: [{global_min:.3f}, {global_max:.3f}]")
    else:
        # Non-linear data: use per-tile normalization
        status_callback("Using NON-LINEAR mode - per-tile normalization...")
        global_min = global_max = None
        print("   Using per-tile normalization")
    
    status_callback("Beginning tiled processing...")
    
    # Initialize output arrays
    corrected_image = np.zeros_like(image_2d, dtype=np.float32)
    weight_map = np.zeros_like(image_2d, dtype=np.float32)
    
    TILE_SIZE = 256
    TILE_OVERLAP = 48
    step = TILE_SIZE - TILE_OVERLAP
    
    # Create improved blending window with more gradual transitions
    # Use a cosine window which provides smoother blending than Hanning
    def create_blend_window(size, overlap):
        """Create a blending window with smooth transitions at edges."""
        window = np.ones((size, size), dtype=np.float32)
        
        # Apply fade-in/fade-out at edges
        fade_size = min(overlap, size // 4)  # Limit fade size
        
        for i in range(fade_size):
            # Cosine fade for smoother transition
            fade_val = 0.5 * (1 - np.cos(np.pi * i / fade_size))
            
            # Top edge
            window[i, :] *= fade_val
            # Bottom edge  
            window[size-1-i, :] *= fade_val
            # Left edge
            window[:, i] *= fade_val
            # Right edge
            window[:, size-1-i] *= fade_val
            
        return window
    
    # Generate tile coordinates with proper edge handling
    y_coords = []
    x_coords = []
    
    # Generate coordinates ensuring we cover the entire image
    y = 0
    while y < image_2d.shape[0]:
        if y + TILE_SIZE > image_2d.shape[0]:
            y = max(0, image_2d.shape[0] - TILE_SIZE)  # Align last tile to image edge
        y_coords.append(y)
        if y + TILE_SIZE >= image_2d.shape[0]:
            break
        y += step
    
    x = 0  
    while x < image_2d.shape[1]:
        if x + TILE_SIZE > image_2d.shape[1]:
            x = max(0, image_2d.shape[1] - TILE_SIZE)  # Align last tile to image edge
        x_coords.append(x)
        if x + TILE_SIZE >= image_2d.shape[1]:
            break
        x += step
    
    # Remove duplicates while preserving order
    y_coords = list(dict.fromkeys(y_coords))
    x_coords = list(dict.fromkeys(x_coords))
    
    total_tiles = len(x_coords) * len(y_coords)
    tile_count = 0
    
    print(f"   Processing {total_tiles} tiles ({len(y_coords)} rows × {len(x_coords)} cols)")

    for y in y_coords:
        for x in x_coords:
            tile_count += 1
            progress = (tile_count / total_tiles) * 100
            progress_callback(progress)
            
            # Ensure we don't go outside image boundaries
            actual_tile_h = min(TILE_SIZE, image_2d.shape[0] - y)
            actual_tile_w = min(TILE_SIZE, image_2d.shape[1] - x)
            
            # Extract raw tile from original data
            tile = image_2d[y:y+actual_tile_h, x:x+actual_tile_w]
            
            # Pad tile to TILE_SIZE if necessary (for edge tiles)
            if tile.shape != (TILE_SIZE, TILE_SIZE):
                padded_tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=tile.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            # Skip processing if tile has no dynamic range
            if tile.std() < 1e-8:
                corrected_tile = tile.copy()
            else:
                # Apply adaptive normalization based on image type
                input_tensor, norm_min, norm_max = preprocess_tile(tile, global_min, global_max)
                input_tensor = input_tensor.to(device)
                with torch.no_grad():
                    corrected_tensor = model(input_tensor)
                
                # Postprocess: Convert from [-1, 1] back to original range
                corrected_tile = postprocess_tile(corrected_tensor, norm_min, norm_max)
            
            # Create blending window for this tile
            window = create_blend_window(TILE_SIZE, TILE_OVERLAP)
            
            # Only use the part of the tile that fits in the image
            corrected_tile_cropped = corrected_tile[:actual_tile_h, :actual_tile_w]
            window_cropped = window[:actual_tile_h, :actual_tile_w]
            
            # Accumulate with blending weights
            corrected_image[y:y+actual_tile_h, x:x+actual_tile_w] += corrected_tile_cropped * window_cropped
            weight_map[y:y+actual_tile_h, x:x+actual_tile_w] += window_cropped

    # Final Blending
    status_callback("Blending tiles...")
    # Avoid division by zero
    weight_map[weight_map == 0] = 1.0
    corrected_image /= weight_map
    
    return corrected_image

def pick_model_path(user_path: str | None) -> str:
    if user_path and os.path.exists(user_path):
        return user_path
    if os.path.exists(PREFERRED_MODEL):
        return PREFERRED_MODEL
    if os.path.exists(FALLBACK_MODEL):
        return FALLBACK_MODEL
    raise FileNotFoundError("No model file selected and no default model found.")


def run_correction_logic(image_path, model_path, output_path, progress_callback, status_callback, app_instance, force_mode, overwrite):
    """
    Core image processing logic. Now handles both grayscale and RGB images with proper data type preservation.
    """
    try:
        status_callback("Setting up...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_path = pick_model_path(model_path)
        status_callback(f"Loading model: {os.path.basename(model_path)}")
        model = AttentionResUNet()
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        status_callback("Model loaded successfully!")

        status_callback(f"Loading image: {os.path.basename(image_path)}")
        
        # Store original data type before loading (to preserve exact type information)
        original_dtype = None
        if image_path.lower().endswith(('.fits', '.fit')):
            with fits.open(image_path) as hdul:
                original_dtype = hdul[0].data.dtype
        elif image_path.lower().endswith('.xisf'):
            if xisf is not None:
                original_dtype = xisf.read(image_path)[0].data.dtype
        elif image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Image formats are typically uint8
            original_dtype = np.uint8
        
        raw_image = load_astro_image(image_path)
        if raw_image is None:
            raise ValueError("Failed to load image.")
        
        # Fallback if we couldn't determine original dtype
        if original_dtype is None:
            original_dtype = np.float32
            
        print(f"Original image dtype: {original_dtype}, loaded as: {raw_image.dtype}")
        
        start_time = time.time()
        
        if raw_image.ndim == 2:
            # It's grayscale
            status_callback("Processing grayscale image...")
            final_image = _process_single_channel(raw_image, model, device, progress_callback, status_callback, image_path, force_mode)
        elif raw_image.ndim == 3 and raw_image.shape[0] == 3:
            # It's RGB (3, H, W)
            status_callback("Processing RGB image (channel by channel)...")
            corrected_channels = []
            for i in range(3):
                def channel_status(msg):
                    status_callback(f"[Channel {i+1}/3] {msg}")
                def channel_progress(val):
                    base_progress = i * (100 / 3)
                    progress_callback(base_progress + val / 3)
                
                channel_status("Starting processing...")
                channel_data = raw_image[i]  # Extract channel from (3, H, W)
                corrected_channel = _process_single_channel(channel_data, model, device, channel_progress, channel_status, image_path, force_mode)
                corrected_channels.append(corrected_channel)
            
            # Stack channels back to (3, H, W) format
            final_image = np.stack(corrected_channels, axis=0)
        else:
            raise ValueError(f"Unsupported image shape for processing: {raw_image.shape}")

        # Preserve type/range with option to force float32 to avoid banding
        status_callback("Converting to output data type...")
        if app_instance.save_float32_var.get():
            # Always save as float32 FITS. Avoids quantization and block artifacts.
            final_image = final_image.astype(np.float32)
        else:
            if np.issubdtype(original_dtype, np.integer):
                # Preserve original integer range using robust percentile scaling
                # This minimizes banding when the corrected image range changed slightly.
                src_min = np.percentile(final_image, 0.01)
                src_max = np.percentile(final_image, 99.99)
                if src_max > src_min:
                    scaled = (final_image - src_min) / (src_max - src_min)
                else:
                    scaled = np.zeros_like(final_image)
                # Map back to original min/max
                orig_min = np.min(raw_image)
                orig_max = np.max(raw_image)
                mapped = scaled * (orig_max - orig_min) + orig_min
                final_image = np.clip(mapped, orig_min, orig_max)
                final_image = final_image.astype(original_dtype)
            elif np.issubdtype(original_dtype, np.floating):
                final_image = final_image.astype(original_dtype)
            else:
                print(f"Warning: Unknown data type {original_dtype}, using float32")
                final_image = final_image.astype(np.float32)

        status_callback(f"Saving corrected image to: {output_path}")
        original_header = None
        if image_path.lower().endswith(('.fits', '.fit')):
            with fits.open(image_path) as hdul:
                original_header = hdul[0].header

        # If overwriting input, create a safety backup alongside
        if overwrite and os.path.abspath(output_path) == os.path.abspath(image_path):
            backup_path = output_path + ".bak"
            try:
                if os.path.exists(output_path):
                    os.replace(output_path, backup_path)
                    status_callback(f"Backup created: {backup_path}")
            except Exception as be:
                status_callback(f"Backup warning: {be}")

        fits.writeto(output_path, final_image, header=original_header, overwrite=True)
        
        end_time = time.time()
        status_callback(f"Correction complete in {end_time - start_time:.2f} seconds.")
        print(f"Output image dtype: {final_image.dtype}")
        progress_callback(100)

    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
        status_callback(f"Error: {str(e)}")
        progress_callback(0)
    finally:
        # This will run whether there was an error or not
        # Ensure UI is re-enabled in the main thread
        app_instance.after(100, app_instance.reset_ui)

# =============================================================================
# --- 2. GUI FRONTEND LOGIC ---
# =============================================================================

class FlatAICorrectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FlatAI - Flat-Field Correction")
        self.geometry("800x700")
        self.resizable(True, True)
        
        # Configure modern styling
        self.configure(bg='#f0f0f0')
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors for modern look
        style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Segoe UI', 12), foreground='#34495e')
        style.configure('Info.TLabel', font=('Segoe UI', 10), foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Segoe UI', 10, 'bold'), foreground='#27ae60')
        style.configure('Warning.TLabel', font=('Segoe UI', 10, 'bold'), foreground='#e67e22')
        
        # Configure accent button style
        style.configure('Accent.TButton', font=('Segoe UI', 12, 'bold'), padding=(20, 10))
        
        # Variables
        self.image_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.is_processing = False
        self.force_mode_var = tk.StringVar(value="auto")  # auto|linear|nonlinear

        self.save_float32_var = tk.BooleanVar(value=True)  # default to float32 to avoid banding
        self.overwrite_var = tk.BooleanVar(value=False)    # overwrite input file (destructive)

        self.setup_ui()
        self.update_run_button_state()

    def setup_ui(self):
        """Set up the modern, purpose-focused user interface."""
        # Main container with padding
        main_container = ttk.Frame(self, padding="20")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)

        # ===== HEADER SECTION =====
        header_frame = ttk.Frame(main_container)
        header_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Title and subtitle
        title_label = ttk.Label(header_frame, text="FlatAI", style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        subtitle_label = ttk.Label(header_frame, text="Advanced Flat-Field Correction", style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Purpose description
        purpose_text = "Remove dust motes, vignetting, and uneven illumination from astronomical images using AI"
        purpose_label = ttk.Label(header_frame, text=purpose_text, style='Info.TLabel', wraplength=600)
        purpose_label.grid(row=2, column=0, sticky=tk.W, pady=(10, 0))

        # ===== INPUT SECTION =====
        input_frame = ttk.LabelFrame(main_container, text="Input Configuration", padding="15")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(1, weight=1)

        # Input Image Selection
        ttk.Label(input_frame, text="Input Image:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Label(input_frame, text="Select your astronomical image (FITS, XISF, JPG, PNG)", style='Info.TLabel').grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        input_entry_frame = ttk.Frame(input_frame)
        input_entry_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_entry_frame.columnconfigure(0, weight=1)
        
        self.input_entry = ttk.Entry(input_entry_frame, textvariable=self.image_path, font=('Segoe UI', 10))
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(input_entry_frame, text="Browse...", command=self.browse_input_image).grid(row=0, column=1)

        # Model File Selection
        ttk.Label(input_frame, text="Model File (Optional):", style='Subtitle.TLabel').grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        ttk.Label(input_frame, text="AI model for correction. Auto-detects if not specified.", style='Info.TLabel').grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        
        model_entry_frame = ttk.Frame(input_frame)
        model_entry_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        model_entry_frame.columnconfigure(0, weight=1)
        
        self.model_entry = ttk.Entry(model_entry_frame, textvariable=self.model_path, font=('Segoe UI', 10))
        self.model_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(model_entry_frame, text="Browse...", command=self.browse_model_file).grid(row=0, column=1)

        # ===== OUTPUT SECTION =====
        output_frame = ttk.LabelFrame(main_container, text="Output Configuration", padding="15")
        output_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        output_frame.columnconfigure(1, weight=1)

        # Output Path Selection
        ttk.Label(output_frame, text="Output Path:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Label(output_frame, text="Where to save the corrected image", style='Info.TLabel').grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_entry_frame.columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_entry_frame, textvariable=self.output_path, font=('Segoe UI', 10))
        self.output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.output_browse_btn = ttk.Button(output_entry_frame, text="Browse...", command=self.browse_output_path)
        self.output_browse_btn.grid(row=0, column=1)

        # ===== PROCESSING OPTIONS SECTION =====
        options_frame = ttk.LabelFrame(main_container, text="Processing Options", padding="15")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        # Processing Mode
        mode_frame = ttk.Frame(options_frame)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        ttk.Label(mode_frame, text="Processing Mode:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Label(mode_frame, text="How to handle image normalization", style='Info.TLabel').grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        mode_buttons_frame = ttk.Frame(mode_frame)
        mode_buttons_frame.grid(row=2, column=0, sticky=tk.W)
        
        ttk.Radiobutton(mode_buttons_frame, text="Auto-detect (Recommended)", value="auto", 
                       variable=self.force_mode_var).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(mode_buttons_frame, text="Force Linear", value="linear", 
                       variable=self.force_mode_var).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(mode_buttons_frame, text="Force Non-linear", value="nonlinear", 
                       variable=self.force_mode_var).pack(side=tk.LEFT)

        # Output Options
        output_options_frame = ttk.Frame(options_frame)
        output_options_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Checkbutton(output_options_frame, text="Save as float32 (prevents banding artifacts)", 
                       variable=self.save_float32_var).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Checkbutton(output_options_frame, text="Overwrite input file", 
                       variable=self.overwrite_var, command=self.on_overwrite_toggle).pack(side=tk.LEFT)

        # ===== PROCESSING SECTION =====
        processing_frame = ttk.LabelFrame(main_container, text="Processing", padding="15")
        processing_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        processing_frame.columnconfigure(1, weight=1)

        # Run Button
        self.run_button = ttk.Button(processing_frame, text="🚀 Start Flat-Field Correction", 
                                    command=self.start_correction, style='Accent.TButton')
        self.run_button.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        # Progress Bar
        ttk.Label(processing_frame, text="Progress:", style='Subtitle.TLabel').grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(processing_frame, variable=self.progress_var, 
                                          maximum=100, length=600, mode='determinate')
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # Status Display
        ttk.Label(processing_frame, text="Status:", style='Subtitle.TLabel').grid(row=3, column=0, sticky=(tk.W, tk.N), pady=(0, 5))
        self.status_var = tk.StringVar(value="Ready to process. Select an input image and click 'Start Flat-Field Correction'.")
        self.status_label = ttk.Label(processing_frame, textvariable=self.status_var, 
                                     wraplength=600, justify=tk.LEFT, style='Info.TLabel')
        self.status_label.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # ===== FOOTER SECTION =====
        footer_frame = ttk.Frame(main_container)
        footer_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Device Info
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        device_label = ttk.Label(footer_frame, text=f"Processing Device: {device}", style='Info.TLabel')
        device_label.pack(side=tk.LEFT)
        
        # Version info
        version_label = ttk.Label(footer_frame, text="FlatAI v1.0 - 40M Parameter U-Net", style='Info.TLabel')
        version_label.pack(side=tk.RIGHT)

    def browse_input_image(self):
        """Browse for input image file."""
        filetypes = [
            ("All Supported", "*.fits;*.fit;*.xisf;*.jpg;*.jpeg;*.png"),
            ("FITS files", "*.fits;*.fit"),
            ("XISF files", "*.xisf"),
            ("Image files", "*.jpg;*.jpeg;*.png"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(title="Select Input Image", filetypes=filetypes)
        if filename:
            self.image_path.set(filename)
            self.update_run_button_state()

    def browse_model_file(self):
        """Browse for model file."""
        filetypes = [("PyTorch Model", "*.pth"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Select Model File", filetypes=filetypes)
        if filename:
            self.model_path.set(filename)
            self.update_run_button_state()

    def browse_output_path(self):
        """Browse for output path."""
        filetypes = [("FITS files", "*.fits"), ("All files", "*.*")]
        filename = filedialog.asksaveasfilename(title="Save Corrected Image As", 
                                              filetypes=filetypes,
                                              defaultextension=".fits")
        if filename:
            self.output_path.set(filename)
            self.update_run_button_state()

    def update_run_button_state(self):
        """Enable/disable run button based on file selections."""
        has_image = bool(self.image_path.get())
        has_model = bool(self.model_path.get())
        has_output = bool(self.output_path.get())
        if (has_image and has_model and ((self.overwrite_var.get()) or has_output) and not self.is_processing):
            self.run_button.config(state='normal')
        else:
            self.run_button.config(state='disabled')

    def start_correction(self):
        self.is_processing = True
        self.run_button.config(text="⏳ Processing...", state='disabled')
        self.progress_bar['value'] = 0
        self.status_var.set("Starting flat-field correction...")
        
        # Start the processing in a separate thread to keep the GUI responsive
        thread = threading.Thread(
            target=self.run_correction_thread,
            daemon=True
        )
        thread.start()

    def run_correction_thread(self):
        """Wrapper to call the backend logic from a thread."""
        mode = self.force_mode_var.get()
        force_mode = None if mode == "auto" else (True if mode == "linear" else False)
        # Determine output path (overwrite uses input path)
        out_path = self.image_path.get() if self.overwrite_var.get() else self.output_path.get()
        run_correction_logic(
            self.image_path.get(),
            self.model_path.get(),
            out_path,
            self.update_progress,
            self.update_status,
            self,  # Pass the app instance itself
            force_mode,
            self.overwrite_var.get(),
        )

    def on_overwrite_toggle(self):
        """Enable/disable output path widgets when overwrite is toggled."""
        overwrite = self.overwrite_var.get()
        state = 'disabled' if overwrite else 'normal'
        self.output_entry.config(state=state)
        self.output_browse_btn.config(state=state)
        # If overwriting, mirror output path to input for display clarity
        if overwrite and self.image_path.get():
            self.output_path.set(self.image_path.get())
        self.update_run_button_state()

    def update_progress(self, value):
        """Updates the progress bar from any thread."""
        self.after(0, lambda: self.progress_var.set(value))

    def update_status(self, message):
        """Update status label (thread-safe)."""
        self.after(0, lambda: self.status_var.set(message))

    def reset_ui(self):
        """Reset UI after processing is complete."""
        self.is_processing = False
        self.run_button.config(text="🚀 Start Flat-Field Correction")
        self.update_run_button_state()

# =============================================================================
# --- 3. MAIN EXECUTION ---
# =============================================================================

def main():
    app = FlatAICorrectorApp()
    app.mainloop()

if __name__ == '__main__':
    main() 