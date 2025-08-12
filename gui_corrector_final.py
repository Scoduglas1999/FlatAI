import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
import time

# Force astropy to use a more forgiving encoding for FITS headers
# to prevent crashes on non-standard metadata.
fits.conf.STR_ENCODING = 'latin-1'

try:
    from xisf import XISF
except ImportError:
    XISF = None

try:
    from PIL import Image
except ImportError:
    Image = None

from unet_model import AttentionResUNet, AttentionResUNetFG

# =============================================================================
# --- BACKEND PROCESSING LOGIC (Restored from original) ---
# =============================================================================

def load_astro_image(image_path):
    if not os.path.exists(image_path):
        return None, "File not found."
    
    _, file_extension = os.path.splitext(image_path)
    file_ext = file_extension.lower()

    try:
        if file_ext in ['.fits', '.fit']:
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data
        elif file_ext == '.xisf':
            if XISF is None:
                raise ImportError("XISF file provided, but 'xisf' library is not installed.")
            xisf_file = XISF(image_path)
            image_data = xisf_file.read_image(0)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            if Image is None:
                raise ImportError("JPG/PNG file provided, but 'Pillow' library is not installed.")
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                image_data = np.array(img)
                image_data = np.transpose(image_data, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        if image_data.ndim == 3 and image_data.shape[0] != 3 and image_data.shape[2] == 3:
            image_data = np.transpose(image_data, (2, 0, 1))
        
        # Squeeze out single-channel dimension if present (e.g., from XISF grayscale)
        if image_data.ndim == 3 and image_data.shape[2] == 1:
            image_data = np.squeeze(image_data, axis=2)

        return image_data.astype(np.float32), None

    except Exception as e:
        return None, f"Error loading image: {e}"

def preprocess_tile(tile_np, global_min=None, global_max=None):
    if global_min is not None and global_max is not None:
        if global_max > global_min:
            tile_normalized_01 = (tile_np - global_min) / (global_max - global_min)
        else:
            tile_normalized_01 = np.zeros_like(tile_np)
        tile_normalized_01 = np.clip(tile_normalized_01, 0, 1)
    else:
        tile_min, tile_max = tile_np.min(), tile_np.max()
        if tile_max > tile_min:
            tile_normalized_01 = (tile_np - tile_min) / (tile_max - tile_min)
        else:
            tile_normalized_01 = np.zeros_like(tile_np)

    tile_normalized = tile_normalized_01 * 2.0 - 1.0
    tile_tensor = torch.from_numpy(tile_normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tile_tensor, (global_min, global_max) if global_min is not None else (tile_np.min(), tile_np.max())

def postprocess_tile(tile_tensor, norm_range):
    tile_np = tile_tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
    tile_np = (tile_np + 1.0) / 2.0
    tile_np = np.clip(tile_np, 0, 1)
    
    tile_min, tile_max = norm_range
    if tile_max > tile_min:
        tile_np = tile_np * (tile_max - tile_min) + tile_min
    else:
        tile_np = np.full_like(tile_np, tile_min)
    return tile_np

def detect_image_type_improved(image_data):
    """
    Detect if image is linear or non-linear based on statistical analysis.
    This is the robust implementation from the original script.
    """
    data_flat = image_data.flatten()
    p1 = np.percentile(data_flat, 1)
    p99 = np.percentile(data_flat, 99)
    trimmed_data = data_flat[(data_flat >= p1) & (data_flat <= p99)]
    
    if len(trimmed_data) == 0: return False

    percentile_5 = np.percentile(trimmed_data, 5)
    percentile_95 = np.percentile(trimmed_data, 95)
    
    if percentile_95 <= percentile_5: return False

    percentile_50 = np.percentile(trimmed_data, 50)
    percentile_99 = np.percentile(trimmed_data, 99)
    
    dynamic_range_ratio = (percentile_99 - percentile_95) / (percentile_95 - percentile_5)
    lower_quartile_ratio = (percentile_50 - percentile_5) / (percentile_95 - percentile_5 + 1e-10)
    
    is_linear_votes = 0
    if dynamic_range_ratio > 1.5: is_linear_votes += 1
    if lower_quartile_ratio < 0.3: is_linear_votes += 1
    
    return is_linear_votes >= 1

def _process_single_channel(image_2d, model, device, progress_callback, status_callback, force_linear):
    is_linear = force_linear if force_linear is not None else detect_image_type_improved(image_2d)
    
    if is_linear:
        status_callback("Using LINEAR mode - global normalization.")
        # Use true minimum to preserve faint background and high percentile for highlights
        global_min, global_max = float(np.min(image_2d)), float(np.percentile(image_2d, 99.99))
        if not np.isfinite(global_max) or global_max <= global_min:
            global_min, global_max = float(np.min(image_2d)), float(np.max(image_2d))
    else:
        status_callback("Using NON-LINEAR mode - per-tile normalization.")
        global_min, global_max = None, None

    corrected_image = np.zeros_like(image_2d, dtype=np.float32)
    weight_map = np.zeros_like(image_2d, dtype=np.float32)
    
    TILE_SIZE = 256
    TILE_OVERLAP = 48
    step = TILE_SIZE - TILE_OVERLAP

    window = np.hanning(TILE_SIZE)
    window = np.outer(window, window)

    y_coords = list(range(0, image_2d.shape[0], step))
    x_coords = list(range(0, image_2d.shape[1], step))
    if (image_2d.shape[0] % step) != 0: y_coords.append(image_2d.shape[0] - TILE_SIZE)
    if (image_2d.shape[1] % step) != 0: x_coords.append(image_2d.shape[1] - TILE_SIZE)
    y_coords = sorted(list(set([max(0, y) for y in y_coords])))
    x_coords = sorted(list(set([max(0, x) for x in x_coords])))

    total_tiles = len(x_coords) * len(y_coords)
    tile_count = 0
    
    for y in y_coords:
        for x in x_coords:
            tile_count += 1
            progress_callback((tile_count / total_tiles) * 100)
            
            h, w = image_2d.shape
            tile = image_2d[y:min(y + TILE_SIZE, h), x:min(x + TILE_SIZE, w)]
            
            tile_h, tile_w = tile.shape
            padded_tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=tile.dtype)
            padded_tile[:tile_h, :tile_w] = tile

            if padded_tile.std() < 1e-8:
                corrected_padded_tile = padded_tile.copy()
            else:
                input_tensor, norm_range = preprocess_tile(padded_tile, global_min, global_max)
                input_tensor = input_tensor.to(device)
                with torch.no_grad():
                    out = model(input_tensor)
                    if isinstance(out, tuple) and len(out) >= 2:
                        if len(out) == 3:
                            F_pred, G_pred, M_pred = out
                        else:
                            F_pred, G_pred = out
                            M_pred = None
                        pmin, pmax = norm_range
                        base_min = pmin if pmin is not None else float(padded_tile.min())
                        base_max = pmax if pmax is not None else float(padded_tile.max())
                        tile01 = np.clip((padded_tile - base_min) / (base_max - base_min + 1e-6), 0.0, 1.0)
                        tile01_t = torch.from_numpy(tile01).unsqueeze(0).unsqueeze(0).to(device)
                        y_clean01_u = torch.clamp((tile01_t - G_pred) / torch.clamp(F_pred, 1e-3, None), 0.0, 1.0)
                        if M_pred is not None:
                            y_clean01 = torch.clamp(tile01_t + M_pred * (y_clean01_u - tile01_t), 0.0, 1.0)
                        else:
                            y_clean01 = y_clean01_u
                        corrected_padded_tile = y_clean01.squeeze().cpu().numpy().astype(np.float32)
                    else:
                        corrected_tensor = out
                        corrected_tensor = torch.clamp(corrected_tensor, -1.0, 1.0)
                        corrected_padded_tile = postprocess_tile(corrected_tensor, norm_range)
            
            corrected_tile = corrected_padded_tile[:tile_h, :tile_w]
            current_window = window[:tile_h, :tile_w]

            corrected_image[y:y+tile_h, x:x+tile_w] += corrected_tile * current_window
            weight_map[y:y+tile_h, x:x+tile_w] += current_window

    weight_map[weight_map == 0] = 1.0
    corrected_image /= weight_map

    # Optional conservative blending to preserve structures and only correct artifacts
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
        pad_w = len(kernel) // 2
        tmp = np.pad(image, ((0, 0), (pad_w, pad_w)), mode='reflect')
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 1, tmp)
        tmp = tmp[:, pad_w:-pad_w]
        tmp = np.pad(tmp, ((pad_w, pad_w), (0, 0)), mode='reflect')
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 0, tmp)
        out = out[pad_w:-pad_w, :]
        return out.astype(np.float32)

    def _gradient_magnitude(img01: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(img01.astype(np.float32))
        return np.sqrt(gx * gx + gy * gy).astype(np.float32)

    app_ref = getattr(status_callback, '__self__', None)
    blend_mode = None
    aggr = 0.7
    if app_ref is not None and hasattr(app_ref, 'blend_mode_var'):
        blend_mode = app_ref.blend_mode_var.get()
        aggr = float(np.clip(getattr(app_ref, 'conservative_strength', lambda: 0.7)(), 0.0, 1.0)) if callable(getattr(app_ref, 'conservative_strength', None)) else float(np.clip(app_ref.conservative_strength.get(), 0.0, 1.0))

    if blend_mode and blend_mode != 'none':
        # Map to [0,1] and compute difference mask
        p_lo, p_hi = np.percentile(image_2d, [0.5, 99.5])
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
            p_lo, p_hi = float(np.min(image_2d)), float(np.max(image_2d))
        orig01 = np.clip((image_2d - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0)
        corr01 = np.clip((corrected_image - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0)

        diff = corr01 - orig01
        if blend_mode == 'conservative':
            blur_sigma = 3.0 + 7.0 * aggr
            m0 = _gaussian_blur(np.abs(diff), sigma=blur_sigma)
            g = _gradient_magnitude(_gaussian_blur(orig01, sigma=1.0))
            g_norm = g / (np.percentile(g, 99.0) + 1e-6)
            g_norm = np.clip(g_norm, 0.0, 1.0)
            power = 0.5 + 1.5 * aggr
            mask = m0 * np.power(1.0 - g_norm, power)
            # Exclude stellar cores/bright structures
            bright_thr = np.percentile(orig01, 98.5 - 5.0 * aggr)
            star_core = (orig01 >= bright_thr).astype(np.float32)
            star_core = _gaussian_blur(star_core, sigma=1.5)
            star_edge = (g_norm > 0.6).astype(np.float32)
            star_edge = _gaussian_blur(star_edge, sigma=1.0)
            star_mask = np.clip(star_core + 0.5 * star_edge, 0.0, 1.0)
            mask = mask * (1.0 - star_mask)
            t = np.percentile(m0, 75.0 + 20.0 * aggr)
            if np.isfinite(t) and t > 0:
                mask = np.where(m0 >= t, mask, mask * 0.2)
            mask = np.clip(mask * (0.5 + 0.5 * aggr), 0.0, 1.0).astype(np.float32)
        else:  # lowfreq: only apply low-frequency component of correction
            # Extract low-frequency correction with a broad blur
            lf = _gaussian_blur(corr01 - orig01, sigma=8.0 + 16.0 * aggr)
            mask = np.clip(np.abs(lf) / (np.percentile(np.abs(lf), 95.0) + 1e-6), 0.0, 1.0).astype(np.float32)
            mask = _gaussian_blur(mask, sigma=3.0)

        # Residual gating so only significant per-pixel changes pass
        mu = _gaussian_blur(orig01, sigma=1.0)
        hf = orig01 - mu
        var = _gaussian_blur(hf * hf, sigma=1.0)
        local_sigma = np.sqrt(np.clip(var, 0.0, None))
        thr = np.clip((2.5 + 2.0 * aggr) * local_sigma, 1e-4, None)
        soft = np.clip(0.4 + 0.2 * aggr, 0.1, 1.0) * thr
        resid = np.abs(corr01 - orig01)
        gate = np.clip((resid - thr) / (soft + 1e-6), 0.0, 1.0).astype(np.float32)
        final_mask = np.clip(mask * gate, 0.0, 1.0)
        corrected_image = (image_2d.astype(np.float32) + final_mask * (corrected_image.astype(np.float32) - image_2d.astype(np.float32))).astype(np.float32)

    return corrected_image

def run_correction_logic(image_path, model_path, output_path, progress_callback, status_callback, app_instance, force_mode_str, overwrite):
    try:
        status_callback("Setting up...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if force_mode_str == "auto": force_mode = None
        elif force_mode_str == "linear": force_mode = True
        else: force_mode = False
        
        try:
            model = AttentionResUNetFG()
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            model = AttentionResUNet()
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        original_header = None
        source_dtype = None  # Native dtype of the input file, before our float conversion
        if image_path.lower().endswith(('.fits', '.fit')):
            with fits.open(image_path) as hdul:
                original_header = hdul[0].header
                source_dtype = hdul[0].data.dtype
        elif image_path.lower().endswith('.xisf') and XISF is not None:
            try:
                xmeta = XISF(image_path).get_images_metadata()[0]
                source_dtype = xmeta.get('dtype')
            except Exception:
                source_dtype = None
        elif image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            source_dtype = np.uint8
        
        raw_image, error = load_astro_image(image_path)
        if error: raise ValueError(error)

        # raw_image has been converted to float32 in load_astro_image().
        # Keep a handle to the native input dtype for correct scaling decisions.
        original_dtype = source_dtype if source_dtype is not None else raw_image.dtype
        start_time = time.time()
        
        if raw_image.ndim == 2:
            final_image = _process_single_channel(raw_image, model, device, progress_callback, status_callback, force_mode)
        elif raw_image.ndim == 3 and raw_image.shape[0] == 3:
            corrected_channels = []
            for i in range(3):
                channel_progress = lambda p, chan=i: progress_callback((chan * 100 + p) / 3)
                channel_status = lambda msg, chan=i: status_callback(f"[Channel {chan+1}/3] {msg}")
                corrected_channels.append(_process_single_channel(raw_image[i], model, device, channel_progress, channel_status, force_mode))
            final_image = np.stack(corrected_channels, axis=0)
        else:
            raise ValueError(f"Unsupported image shape: {raw_image.shape}")

        if app_instance.save_float32_var.get():
            final_image = final_image.astype(np.float32)
        else:
            if np.issubdtype(original_dtype, np.integer):
                src_min, src_max = np.percentile(final_image, 0.01), np.percentile(final_image, 99.99)
                if src_max > src_min:
                    scaled = (final_image - src_min) / (src_max - src_min)
                else:
                    scaled = np.zeros_like(final_image)
                
                iinfo = np.iinfo(original_dtype)
                mapped = scaled * (iinfo.max - iinfo.min) + iinfo.min
                final_image = np.clip(mapped, iinfo.min, iinfo.max).astype(original_dtype)
            else:
                final_image = final_image.astype(original_dtype)
        
        if overwrite and os.path.abspath(output_path) == os.path.abspath(image_path):
            backup_path = output_path + ".bak"
            if os.path.exists(output_path):
                os.replace(output_path, backup_path)

        # Use the correct writer based on file extension
        if output_path.lower().endswith('.xisf'):
            if XISF is None:
                raise ImportError("Cannot write XISF file; 'xisf' library is not installed.")

            # XISF module expects a channels-last ndarray. Add an explicit channel
            # dimension for grayscale images and ensure contiguous layout.
            xisf_image = final_image
            if xisf_image.ndim == 2:
                xisf_image = xisf_image[:, :, None]
            elif xisf_image.ndim == 3 and xisf_image.shape[0] in (1, 3):
                # our pipeline is channels-first → convert to channels-last
                xisf_image = np.transpose(xisf_image, (1, 2, 0))
            xisf_image = np.ascontiguousarray(xisf_image)

            # Preserve linear float data without percentile remapping.
            # Just ensure float32 dtype and contiguous memory.
            if np.issubdtype(xisf_image.dtype, np.floating):
                xisf_image = xisf_image.astype(np.float32, copy=False)

            # Build FITSKeywords dict in the structure expected by xisf.XISF.write
            fits_keywords = {}
            if original_header is not None:
                for card in original_header.cards:
                    key = str(card.keyword)
                    if key in (None, 'END', 'COMMENT', 'HISTORY'):
                        continue
                    # Convert to safe Python strings (utf-8 encodable)
                    def to_text(v):
                        if isinstance(v, bytes):
                            try:
                                return v.decode('utf-8')
                            except UnicodeDecodeError:
                                return v.decode('latin-1', 'replace')
                        return str(v)
                    val = to_text(card.value)
                    com = to_text(card.comment or '')
                    fits_keywords.setdefault(key, []).append({'value': val, 'comment': com})

            image_metadata = {
                'FITSKeywords': fits_keywords,
                'XISFProperties': {}
            }
            xisf_metadata = {}

            # Write a proper monolithic XISF file using the official API
            # Choose a sensible default codec for good compatibility/size
            XISF.write(
                output_path,
                xisf_image,
                creator_app='FlatAI',
                image_metadata=image_metadata,
                xisf_metadata=xisf_metadata,
                codec='lz4hc',
                shuffle=True,
                level=9,
            )

        else:  # Default to FITS
            # Sanitize legacy scaling keywords so float data isn't re-scaled on read
            def _sanitize_header(header):
                if header is None:
                    return None
                hdr = header.copy()
                for key in ['BSCALE', 'BZERO', 'BLANK', 'DATAMIN', 'DATAMAX', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2']:
                    if key in hdr:
                        del hdr[key]
                return hdr

            clean_header = _sanitize_header(original_header)
            fits.writeto(output_path, final_image.astype(np.float32, copy=False), header=clean_header, overwrite=True)

        end_time = time.time()
        status_callback(f"Correction complete in {end_time - start_time:.2f} seconds.")
        progress_callback(100)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        status_callback(f"Error: {str(e)}")
    finally:
        app_instance.after(100, app_instance.reset_ui)

class FlatAICorrectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("FlatAI - Flat-Field Correction")
        self.geometry("800x850")
        
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        self.image_path = ctk.StringVar()
        self.model_path = ctk.StringVar(value=r"C:\Users\scdou\Documents\NeuralNet\unet_flat_checkpoint.pth")
        self.output_path = ctk.StringVar()
        self.force_mode_var = ctk.StringVar(value="auto")
        self.save_float32_var = ctk.BooleanVar(value=True)
        self.overwrite_var = ctk.BooleanVar(value=False)
        self.is_processing = False
        self.conservative_var = ctk.BooleanVar(value=True)
        self.conservative_strength = ctk.DoubleVar(value=0.7)
        self.blend_mode_var = ctk.StringVar(value="none")  # none | conservative | lowfreq

        self.setup_ui()
        self.update_run_button_state()

    def setup_ui(self):
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        ctk.CTkLabel(title_frame, text="FlatAI", font=ctk.CTkFont(size=24, weight="bold")).pack(side="left")
        
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(input_frame, text="Input Image").grid(row=0, column=0, padx=10, pady=10)
        self.input_entry = ctk.CTkEntry(input_frame, textvariable=self.image_path)
        self.input_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(input_frame, text="Browse...", width=100, command=self.browse_input_image).grid(row=0, column=2, padx=10, pady=10)

        ctk.CTkLabel(input_frame, text="Model File").grid(row=1, column=0, padx=10, pady=10)
        self.model_entry = ctk.CTkEntry(input_frame, textvariable=self.model_path)
        self.model_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(input_frame, text="Browse...", width=100, command=self.browse_model_file).grid(row=1, column=2, padx=10, pady=10)

        output_frame = ctk.CTkFrame(self)
        output_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        output_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(output_frame, text="Output Path").grid(row=0, column=0, padx=10, pady=10)
        self.output_entry = ctk.CTkEntry(output_frame, textvariable=self.output_path)
        self.output_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(output_frame, text="Browse...", width=100, command=self.browse_output_path).grid(row=0, column=2, padx=10, pady=10)
        
        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(options_frame, text="Processing Mode:").pack(side="left", padx=10, pady=10)
        ctk.CTkRadioButton(options_frame, text="Auto", variable=self.force_mode_var, value="auto").pack(side="left", padx=5)
        ctk.CTkRadioButton(options_frame, text="Linear", variable=self.force_mode_var, value="linear").pack(side="left", padx=5)
        ctk.CTkRadioButton(options_frame, text="Non-linear", variable=self.force_mode_var, value="nonlinear").pack(side="left", padx=5)
        
        ctk.CTkCheckBox(options_frame, text="Save as float32", variable=self.save_float32_var).pack(side="left", padx=20)
        self.overwrite_checkbox = ctk.CTkCheckBox(options_frame, text="Overwrite input file", variable=self.overwrite_var, command=self.on_overwrite_toggle)
        self.overwrite_checkbox.pack(side="left", padx=5)

        # Separate row for surgical blend controls to avoid clipping off-screen
        blend_frame = ctk.CTkFrame(self)
        blend_frame.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")
        blend_frame.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(blend_frame, text="Blend mode").grid(row=0, column=0, padx=(10, 5), pady=10, sticky="w")
        ctk.CTkOptionMenu(blend_frame, variable=self.blend_mode_var, values=["none", "conservative", "lowfreq"]).grid(row=0, column=1, padx=(5, 10), pady=10, sticky="w")
        ctk.CTkLabel(blend_frame, text="Aggressiveness").grid(row=0, column=2, padx=(10, 5), pady=10, sticky="e")
        ctk.CTkSlider(blend_frame, from_=0.0, to=1.0, number_of_steps=100, variable=self.conservative_strength).grid(row=0, column=3, padx=(5, 10), pady=10, sticky="ew")

        processing_frame = ctk.CTkFrame(self)
        processing_frame.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
        processing_frame.grid_columnconfigure(0, weight=1)

        self.run_button = ctk.CTkButton(processing_frame, text="Start Flat-Field Correction", command=self.start_correction)
        self.run_button.pack(pady=20, padx=20, fill="x")

        self.progress_bar = ctk.CTkProgressBar(processing_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10, padx=20, fill="x")

        self.status_label = ctk.CTkLabel(processing_frame, text="Ready.", wraplength=700)
        self.status_label.pack(pady=10, padx=20, fill="x")

        footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        footer_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        ctk.CTkLabel(footer_frame, text=f"Device: {device}").pack(side="left")
        ctk.CTkLabel(footer_frame, text="FlatAI v1.2").pack(side="right")


    def browse_input_image(self):
        filename = filedialog.askopenfilename(filetypes=[("All Supported", "*.fits *.fit *.xisf *.jpg *.jpeg *.png"), ("All files", "*.*")])
        if filename:
            self.image_path.set(filename)
            self.update_run_button_state()
            if not self.output_path.get() or self.overwrite_var.get():
                base, ext = os.path.splitext(filename)
                self.output_path.set(f"{base}_corrected.fits")

    def browse_model_file(self):
        filename = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")])
        if filename:
            self.model_path.set(filename)
            self.update_run_button_state()

    def browse_output_path(self):
        filename = filedialog.asksaveasfilename(defaultextension=".fits", filetypes=[("FITS files", "*.fits"), ("All files", "*.*")])
        if filename:
            self.output_path.set(filename)
            self.update_run_button_state()

    def on_overwrite_toggle(self):
        if self.overwrite_var.get():
            self.output_entry.configure(state="disabled")
            self.output_path.set(self.image_path.get())
        else:
            self.output_entry.configure(state="normal")
        self.update_run_button_state()

    def update_run_button_state(self):
        state = "normal" if self.image_path.get() and self.model_path.get() and (self.output_path.get() or self.overwrite_var.get()) and not self.is_processing else "disabled"
        self.run_button.configure(state=state)

    def start_correction(self):
        self.is_processing = True
        self.run_button.configure(text="Processing...", state="disabled")
        self.progress_bar.set(0)
        
        thread = threading.Thread(
            target=run_correction_logic,
            args=(
                self.image_path.get(), self.model_path.get(),
                self.output_path.get(), self.update_progress,
                self.update_status, self,
                self.force_mode_var.get(), self.overwrite_var.get()
            ),
            daemon=True
        )
        thread.start()

    def update_progress(self, value):
        self.progress_bar.set(value / 100)

    def update_status(self, message):
        self.status_label.configure(text=message)

    def reset_ui(self):
        self.is_processing = False
        self.run_button.configure(text="Start Flat-Field Correction")
        self.update_run_button_state()

if __name__ == "__main__":
    app = FlatAICorrectorApp()
    app.mainloop()
