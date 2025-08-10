# correct_image_final.py - Final version with corrected detect_threshold call

import os
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
from skimage.restoration import richardson_lucy
from tqdm import tqdm

try:
    import xisf
except ImportError:
    xisf = None

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
IMAGE_PATH_TO_CORRECT = "C:/Users/scdou/Downloads/06-04/Lights/LIGHT-20250605T150020Z-1-003/LIGHT/2025-06-04_22-14-42_Lum_-15.00_60.00s_0031.fits"
CORRECTED_IMAGE_SAVE_PATH = "./corrected_image_final.fits"
IMAGE_WIDTH = 4656
IMAGE_HEIGHT = 3520

COMA_X_COEFFS = {'a': 7.2660, 'b': -1.5037, 'c': 0.4187}
COMA_Y_COEFFS = {'a': 1.6457, 'b': 5.9629, 'c': 1.5828}

STAR_DETECTION_NSIGMA = 5.0
DECONVOLUTION_ITERATIONS = 3
CUTOUT_SIZE = 128
BLEND_SIGMA_FACTOR = 2.0
SATURATION_THRESHOLD = 65000

WAVELENGTH = 550e-9
APERTURE_DIAMETER = 0.254
OBSTRUCTION_RATIO = 0.33
PUPIL_GRID_SIZE = 256

# =============================================================================
# --- 2. HELPER FUNCTIONS (SELF-CONTAINED) ---
# =============================================================================

def load_astro_image(image_path):
    """Loads an astronomical image from a FITS or XISF file."""
    print(f"Loading astronomical image from {image_path}...")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at: {image_path}")
        return None
    _, file_extension = os.path.splitext(image_path)
    try:
        if file_extension.lower() in ['.fits', '.fit']:
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data.astype(np.float32)
        elif file_extension.lower() == '.xisf':
            if xisf is None: raise ImportError("XISF file provided, but 'xisf' library is not installed.")
            image_data = xisf.read(image_path)[0].astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        if image_data.ndim == 3: image_data = image_data.mean(axis=2)
        print(f"Image loaded with shape: {image_data.shape}")
        return image_data
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return None

def predict_zernikes_at_position(x, y, width, height, z7_coeffs, z8_coeffs):
    """Predicts Zernike coefficients for a pixel coordinate using the fitted model."""
    x_norm = (x - width / 2) / (width / 2)
    y_norm = (y - height / 2) / (height / 2)
    pred_z7 = z7_coeffs['a'] * x_norm + z7_coeffs['b'] * y_norm + z7_coeffs['c']
    pred_z8 = z8_coeffs['a'] * x_norm + z8_coeffs['b'] * y_norm + z8_coeffs['c']
    return {4: 0.0, 7: pred_z7, 8: pred_z8}

def create_pupil_mask(grid_size, radius, obstruction_ratio=0.0):
    """Creates a circular pupil mask with an optional central obstruction."""
    y, x = np.mgrid[-grid_size//2:grid_size//2, -grid_size//2:grid_size//2]
    rho = np.sqrt(x**2 + y**2)
    pupil = (rho <= radius).astype(np.float32)
    if obstruction_ratio > 0:
        pupil[rho < radius * obstruction_ratio] = 0
    return pupil

def generate_zernike_polynomial(noll_index, grid_size, pupil_mask):
    """Generates a normalized Zernike polynomial map for a given Noll index."""
    radius_pixels = PUPIL_GRID_SIZE / 2
    y, x = np.mgrid[-radius_pixels:radius_pixels, -radius_pixels:radius_pixels] / radius_pixels
    rho, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    zernike = np.zeros_like(rho)
    if noll_index == 4: zernike = np.sqrt(3) * (2 * rho**2 - 1)
    elif noll_index == 7: zernike = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(theta)
    elif noll_index == 8: zernike = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.sin(theta)
    zernike *= pupil_mask
    rms = np.sqrt(np.mean(zernike[pupil_mask > 0]**2))
    if rms > 1e-9: zernike /= rms
    return zernike

def generate_psf_from_zernikes(zernike_coeffs, psf_size):
    """Generates a PSF from a set of Zernike coefficients."""
    pupil_radius = PUPIL_GRID_SIZE / 2
    pupil_mask = create_pupil_mask(PUPIL_GRID_SIZE, pupil_radius, OBSTRUCTION_RATIO)
    zernike_maps = {n: generate_zernike_polynomial(n, PUPIL_GRID_SIZE, pupil_mask) for n in zernike_coeffs}
    phase_map = np.zeros_like(pupil_mask, dtype=np.float32)
    for n, c in zernike_coeffs.items():
        if n in zernike_maps: phase_map += c * zernike_maps[n]
    wavefront = pupil_mask * np.exp(1j * 2 * np.pi * phase_map)
    psf_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wavefront)))
    psf_intensity = np.abs(psf_fft)**2
    center, half_size = PUPIL_GRID_SIZE // 2, psf_size // 2
    psf_cropped = psf_intensity[center-half_size:center+half_size, center-half_size:center+half_size]
    return psf_cropped / psf_cropped.sum()

# =============================================================================
# --- 3. MAIN EXECUTION LOGIC ---
# =============================================================================
if __name__ == '__main__':
    raw_image = load_astro_image(IMAGE_PATH_TO_CORRECT)
    if raw_image is None: exit()
    h, w = raw_image.shape
    
    corrected_image = raw_image.copy()
    
    print("Finding stars using segmentation...")
    # --- CORRECTED CODE BLOCK ---
    # The detect_threshold function calculates the sigma internally from the data.
    # We do not need to calculate it separately and pass it in.
    threshold = detect_threshold(raw_image, nsigma=STAR_DETECTION_NSIGMA)
    
    segment_map = detect_sources(raw_image, threshold, npixels=10)
    if segment_map is None:
        print(f"No stars found at {STAR_DETECTION_NSIGMA}-sigma. Try lowering the value."); exit()
    cat = SourceCatalog(raw_image, segment_map)
    stars = cat.to_table()
    print(f"Found {len(stars)} sources (stars) to correct.")

    with tqdm(total=len(stars), desc="Correcting Stars") as pbar:
        for star in stars:
            x, y = int(round(star['xcentroid'])), int(round(star['ycentroid']))
            star_peak_value = star['max_value']

            half_size = CUTOUT_SIZE // 2
            if not (half_size <= x < w - half_size and half_size <= y < h - half_size):
                pbar.update(1); continue

            tile = raw_image[y-half_size:y+half_size, x-half_size:x+half_size]
            
            if star_peak_value < SATURATION_THRESHOLD:
                zernike_coeffs = predict_zernikes_at_position(x, y, w, h, COMA_X_COEFFS, COMA_Y_COEFFS)
                local_psf = generate_psf_from_zernikes(zernike_coeffs, CUTOUT_SIZE)
                
                tile_min, tile_max = tile.min(), tile.max()
                if not tile_max > tile_min: pbar.update(1); continue
                    
                tile_norm = (tile - tile_min) / (tile_max - tile_min)
                deconvolved_tile_norm = richardson_lucy(tile_norm, local_psf, num_iter=DECONVOLUTION_ITERATIONS)
                corrected_tile = deconvolved_tile_norm * (tile_max - tile_min) + tile_min
            
            else:
                perfect_zernikes = {4: 0.0, 7: 0.0, 8: 0.0}
                perfect_psf = generate_psf_from_zernikes(perfect_zernikes, CUTOUT_SIZE)
                original_star_flux = np.sum(tile)
                perfect_psf_scaled = perfect_psf * original_star_flux
                corrected_tile = perfect_psf_scaled

            blend_sigma = star['semimajor_sigma'].value * BLEND_SIGMA_FACTOR
            yy_b, xx_b = np.mgrid[-half_size:half_size, -half_size:half_size]
            blend_mask = np.exp(-((xx_b**2 + yy_b**2) / (2 * blend_sigma**2)))
            
            original_tile = corrected_image[y-half_size:y+half_size, x-half_size:x+half_size]
            blended_tile = original_tile * (1 - blend_mask) + corrected_tile * blend_mask
            corrected_image[y-half_size:y+half_size, x-half_size:x+half_size] = blended_tile
            
            pbar.update(1)
            
    print(f"Saving final corrected image to {CORRECTED_IMAGE_SAVE_PATH}")
    hdu = fits.PrimaryHDU(corrected_image.astype(np.float32))
    hdul = fits.HDUList([hdu])
    hdul.writeto(CORRECTED_IMAGE_SAVE_PATH, overwrite=True)
    hdul.close()
    print("Process complete.")