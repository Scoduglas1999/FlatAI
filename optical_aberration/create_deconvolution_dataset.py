# create_deconvolution_dataset.py

import os
import numpy as np
import random
from astropy.io import fits
from scipy.signal import convolve2d
from tqdm import tqdm

# It is better to handle potential missing imports
try:
    from PIL import Image
except ImportError:
    print("Warning: The 'Pillow' library is not installed. JPG/PNG support is disabled.")
    print("Install it with: 'pip install Pillow'")
    Image = None

try:
    import xisf
except ImportError:
    print("Warning: The 'xisf' library is not installed. XISF support is disabled.")
    xisf = None

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
# --- Directory and Dataset Parameters ---
SHARP_IMAGE_DIR = "./sharp_images/"  # Directory containing high-quality source images
OUTPUT_DATA_DIR = "./deconvolution_dataset/"
PATCH_SIZE = 256  # Size of the square patches to extract
PATCHES_PER_IMAGE = 200  # Number of random patches to extract from each source image
TRAIN_TEST_SPLIT = 0.9  # 90% of data for training, 10% for testing

# --- Aberration Model & PSF Generation Parameters ---
# These define the optical system whose aberrations we are simulating.
# Image dimensions are for the system the aberration model was fitted to.
IMAGE_WIDTH = 4656
IMAGE_HEIGHT = 3520

# Aberration Model Coefficients (from fit_aberration_model.py)
COMA_X_COEFFS = {'a': 7.2660, 'b': -1.5037, 'c': 0.4187}
COMA_Y_COEFFS = {'a': 1.6457, 'b': 5.9629, 'c': 1.5828}

# PSF Simulation Constants
PSF_IMAGE_SIZE = 64
PUPIL_DIAMETER = 64
WAVELENGTH = 1.0


# =============================================================================
# --- 2. SELF-CONTAINED HELPER FUNCTIONS ---
# =============================================================================

# --- Image Loading ---
def load_astro_image(image_path):
    """
    Loads an image from FITS, XISF, JPG, or PNG format.
    Ensures the output is a 2D grayscale NumPy array with float32 type.
    """
    print(f"Loading image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: File not found at: {image_path}"); return None
    
    _, file_extension = os.path.splitext(image_path)
    file_ext = file_extension.lower()

    try:
        if file_ext in ['.fits', '.fit']:
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data
        elif file_ext == '.xisf':
            if xisf is None: raise ImportError("XISF file provided, but 'xisf' library is not installed.")
            image_data = xisf.read(image_path)[0].data
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            if Image is None: raise ImportError("JPG/PNG file provided, but 'Pillow' library is not installed.")
            with Image.open(image_path) as img:
                image_data = np.array(img.convert('L'))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Ensure data is 2D and float32
        if image_data.ndim == 3: image_data = image_data.mean(axis=2) # Basic RGB to grayscale
        return image_data.astype(np.float32)

    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return None

# --- PSF Generation (copied from previous work for self-containment) ---
def create_pupil_mask(image_size, pupil_diameter):
    radius = pupil_diameter / 2
    x = np.linspace(-image_size / 2, image_size / 2, image_size)
    y = np.linspace(-image_size / 2, image_size / 2, image_size)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx**2 + yy**2)
    return (rho <= radius).astype(float)

def generate_zernike_polynomial(index, rho, phi):
    if index == 4: return np.sqrt(3) * (2 * rho**2 - 1)
    if index == 7: return np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(phi)
    if index == 8: return np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.sin(phi)
    raise ValueError(f"Zernike index {index} not implemented.")

def construct_wavefront(coeffs, pupil_mask):
    image_size = pupil_mask.shape[0]
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    xx, yy = np.meshgrid(x, y)
    rho, phi = np.sqrt(xx**2 + yy**2), np.arctan2(yy, xx)
    wavefront = np.zeros_like(pupil_mask, dtype=float)
    for index, magnitude in coeffs.items():
        if magnitude != 0:
            wavefront += magnitude * generate_zernike_polynomial(index, rho, phi)
    return wavefront * pupil_mask

def calculate_psf(wavefront, pupil_mask):
    pupil_function = pupil_mask * np.exp(2j * np.pi * wavefront / WAVELENGTH)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil_function)))**2
    return psf / np.sum(psf)

# --- Aberration Modeling ---
def predict_zernikes_at_position(x, y, width, height):
    x_norm = (x - width / 2) / (width / 2)
    y_norm = (y - height / 2) / (height / 2)
    pred_z7 = COMA_X_COEFFS['a'] * x_norm + COMA_X_COEFFS['b'] * y_norm + COMA_X_COEFFS['c']
    pred_z8 = COMA_Y_COEFFS['a'] * x_norm + COMA_Y_COEFFS['b'] * y_norm + COMA_Y_COEFFS['c']
    return {4: 0.0, 7: pred_z7, 8: pred_z8}

def generate_local_psf(x, y, width, height):
    zernike_coeffs = predict_zernikes_at_position(x, y, width, height)
    pupil_mask = create_pupil_mask(PSF_IMAGE_SIZE, PUPIL_DIAMETER)
    wavefront = construct_wavefront(zernike_coeffs, pupil_mask)
    return calculate_psf(wavefront, pupil_mask)


# =============================================================================
# --- 3. MAIN EXECUTION LOGIC ---
# =============================================================================
if __name__ == '__main__':
    # 1. Set up output directories
    print("Setting up output directories...")
    subdirs = ['train/sharp', 'train/blurry', 'test/sharp', 'test/blurry']
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, subdir), exist_ok=True)

    # 2. Find all valid source images
    valid_extensions = ['.fits', '.fit', '.xisf', '.jpg', '.jpeg', '.png']
    source_files = [
        f for f in os.listdir(SHARP_IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    if not source_files:
        print(f"Error: No valid images found in '{SHARP_IMAGE_DIR}'.")
        exit()
    print(f"Found {len(source_files)} source images.")

    patch_counter = 0

    # 3. Loop through each source image and generate patches
    for filename in tqdm(source_files, desc="Processing Source Images"):
        image_path = os.path.join(SHARP_IMAGE_DIR, filename)
        sharp_image = load_astro_image(image_path)
        if sharp_image is None:
            continue

        h, w = sharp_image.shape
        if h < PATCH_SIZE or w < PATCH_SIZE:
            print(f"Skipping '{filename}': smaller than patch size.")
            continue

        for i in range(PATCHES_PER_IMAGE):
            # a. Extract a random sharp patch
            x_start = random.randint(0, w - PATCH_SIZE)
            y_start = random.randint(0, h - PATCH_SIZE)
            sharp_patch = sharp_image[y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE]
            
            # b. Generate the local PSF for the patch's center
            patch_center_x = x_start + PATCH_SIZE / 2
            patch_center_y = y_start + PATCH_SIZE / 2
            local_psf = generate_local_psf(patch_center_x, patch_center_y, IMAGE_WIDTH, IMAGE_HEIGHT)

            # c. Create the blurry patch by convolving the sharp patch with the PSF
            blurry_patch = convolve2d(sharp_patch, local_psf, mode='same', boundary='symm')

            # d. Normalize both patches to a standard [0, 1] float range for saving
            sharp_min, sharp_max = sharp_patch.min(), sharp_patch.max()
            blurry_min, blurry_max = blurry_patch.min(), blurry_patch.max()

            if sharp_max > sharp_min:
                sharp_patch = (sharp_patch - sharp_min) / (sharp_max - sharp_min)
            if blurry_max > blurry_min:
                blurry_patch = (blurry_patch - blurry_min) / (blurry_max - blurry_min)

            # e. Decide whether to save in train or test set
            subset = 'train' if random.random() < TRAIN_TEST_SPLIT else 'test'
            
            # f. Save the blurry/sharp pair as .npy files
            sharp_save_path = os.path.join(OUTPUT_DATA_DIR, subset, 'sharp', f'patch_{patch_counter}.npy')
            blurry_save_path = os.path.join(OUTPUT_DATA_DIR, subset, 'blurry', f'patch_{patch_counter}.npy')
            
            np.save(sharp_save_path, sharp_patch.astype(np.float32))
            np.save(blurry_save_path, blurry_patch.astype(np.float32))
            
            patch_counter += 1
            
    print(f"\nDataset creation complete. Generated {patch_counter} pairs.")
