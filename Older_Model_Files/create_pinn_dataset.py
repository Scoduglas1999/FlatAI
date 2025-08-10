# create_pinn_dataset.py - Physics-Informed Neural Network Dataset Generator
# CUDA-ACCELERATED VERSION

import os
import numpy as np
import random
from astropy.io import fits
# from scipy.signal import convolve2d # No longer needed
import torch
import torch.nn.functional as F
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
OUTPUT_DATA_DIR = "./randomized_pinn_dataset/"  # NEW: Updated output directory for randomized dataset
PATCH_SIZE = 256  # Size of the square patches to extract
PATCHES_PER_IMAGE = 200  # Number of random patches to extract from each source image
TRAIN_VAL_SPLIT = 0.9  # 90% of data for training, 10% for validation

# --- Aberration Model & PSF Generation Parameters ---
# These define the optical system whose aberrations we are simulating.
# Image dimensions are for the system the aberration model was fitted to.
IMAGE_WIDTH = 4656
IMAGE_HEIGHT = 3520

# --- NEW: The Prometheus Stochastic Aberration Configuration ---
# This dictionary defines the "cocktail" of possible aberrations.
# Each aberration has an 'activation_probability' (0.0 to 1.0) and ranges
# for its coefficients. This allows for a diverse and realistic training set.
ULTIMATE_ABERRATION_COCKTAIL = {
    'coma': {
        'activation_probability': 0.9,  # Coma is very common
        'x_ranges': {'a': (6.0, 8.5), 'b': (-2.5, -0.5), 'c': (-0.5, 0.5)},
        'y_ranges': {'a': (0.5, 2.5), 'b': (5.0, 7.0), 'c': (-2.0, 2.0)}
    },
    'astigmatism': {
        'activation_probability': 0.75,
        'oblique_ranges': {'a': (-1.5, 1.5), 'b': (-1.5, 1.5), 'c': (-0.5, 0.5)},
        'vertical_ranges': {'a': (-1.5, 1.5), 'b': (-1.5, 1.5), 'c': (-0.5, 0.5)}
    },
    'defocus_curvature': {
        'activation_probability': 0.95,  # Some focus variation is almost always present
        'ranges': {'a': (-0.3, 0.3), 'b': (-0.3, 0.3), 'c': (-0.2, 0.2)}
    },
    'trefoil': {
        'activation_probability': 0.5,  # Less common mechanical aberration
        'x_range': (-0.4, 0.4),
        'y_range': (-0.4, 0.4)
    },
    'spherical': {
        'activation_probability': 0.6,
        'range': (-0.3, 0.3)
    }
}

# PSF Simulation Constants
PSF_IMAGE_SIZE = 256 # Increased PSF size for higher fidelity representation of complex aberrations
PUPIL_DIAMETER = 256
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

# --- NEW: PSF Generation rewritten for PyTorch and CUDA ---
def create_pupil_mask_torch(image_size, pupil_diameter, device):
    radius = pupil_diameter / 2
    x = torch.linspace(-image_size / 2, image_size / 2, image_size, device=device)
    y = torch.linspace(-image_size / 2, image_size / 2, image_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    rho = torch.sqrt(xx**2 + yy**2)
    return (rho <= radius).float()

def generate_zernike_polynomial_torch(index, rho, phi):
    """Generates the specified Zernike polynomial on the GPU."""
    # Defocus (Z4)
    if index == 4: return torch.sqrt(torch.tensor(3.0)) * (2 * rho**2 - 1)
    # Astigmatism (Z5, Z6)
    if index == 5: return torch.sqrt(torch.tensor(6.0)) * rho**2 * torch.cos(2 * phi)
    if index == 6: return torch.sqrt(torch.tensor(6.0)) * rho**2 * torch.sin(2 * phi)
    # Coma (Z7, Z8)
    if index == 7: return torch.sqrt(torch.tensor(8.0)) * (3 * rho**3 - 2 * rho) * torch.cos(phi)
    if index == 8: return torch.sqrt(torch.tensor(8.0)) * (3 * rho**3 - 2 * rho) * torch.sin(phi)
    # Trefoil (Z9, Z10)
    if index == 9: return torch.sqrt(torch.tensor(8.0)) * rho**3 * torch.cos(3 * phi)
    if index == 10: return torch.sqrt(torch.tensor(8.0)) * rho**3 * torch.sin(3 * phi)
    # Spherical Aberration (Z11)
    if index == 11: return torch.sqrt(torch.tensor(5.0)) * (6 * rho**4 - 6 * rho**2 + 1)

    raise ValueError(f"Zernike index {index} not implemented.")

def construct_wavefront_torch(coeffs, pupil_mask, device):
    image_size = pupil_mask.shape[0]
    x = torch.linspace(-1, 1, image_size, device=device)
    y = torch.linspace(-1, 1, image_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    rho, phi = torch.sqrt(xx**2 + yy**2), torch.atan2(yy, xx)
    wavefront = torch.zeros_like(pupil_mask, dtype=torch.float32)
    for index, magnitude in coeffs.items():
        if magnitude != 0:
            wavefront += magnitude * generate_zernike_polynomial_torch(index, rho, phi)
    return wavefront * pupil_mask

def calculate_psf_torch(wavefront, pupil_mask):
    # Ensure wavefront is complex for the exponentiation
    wavefront_complex = wavefront.to(torch.complex64)
    pupil_function = pupil_mask.to(torch.complex64) * torch.exp(2j * np.pi * wavefront_complex / WAVELENGTH)
    
    # Perform FFT
    psf = torch.abs(torch.fft.fftshift(torch.fft.fft2(pupil_function)))**2
    return psf / torch.sum(psf)

# --- Aberration Modeling ---
def predict_zernikes_at_position(x, y, width, height, random_coeffs):
    """
    Predicts the Zernike coefficients for a given position on the sensor based on a
    randomly generated aberration field model.
    """
    x_norm = (x - width / 2) / (width / 2)
    y_norm = (y - height / 2) / (height / 2)

    # This function now simply returns the coefficients passed into it.
    # The dictionary 'random_coeffs' already contains the final values for this patch,
    # including which aberrations are active (non-zero) or inactive (zero).
    return random_coeffs

# --- NEW: Wrapper function to drive the torch-based PSF generation ---
def generate_local_psf_torch_wrapper(x, y, width, height, random_coeffs, device):
    zernike_coeffs = predict_zernikes_at_position(x, y, width, height, random_coeffs)
    pupil_mask = create_pupil_mask_torch(PSF_IMAGE_SIZE, PUPIL_DIAMETER, device)
    wavefront = construct_wavefront_torch(zernike_coeffs, pupil_mask, device)
    return calculate_psf_torch(wavefront, pupil_mask)

# =============================================================================
# --- 3. MAIN EXECUTION LOGIC ---
# =============================================================================
if __name__ == '__main__':
    # --- NEW: Setup CUDA device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU. This will be slow.")

    # 1. Set up output directories
    print("Setting up output directories...")
    # NEW: Enhanced directory structure for PINN dataset
    subdirs = [
        'train/sharp', 'train/blurry', 'train/psfs',
        'val/sharp', 'val/blurry', 'val/psfs'
    ]
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
            
            # --- Prometheus Stochastic Aberration Logic ---
            # For each patch, decide which aberrations are active based on probability.
            # This creates a diverse "curriculum" of simple and complex problems.
            
            zernike_coeffs = {4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0} # Initialize all to zero

            # 1. Coma (Z7, Z8)
            if random.random() < ULTIMATE_ABERRATION_COCKTAIL['coma']['activation_probability']:
                cfg = ULTIMATE_ABERRATION_COCKTAIL['coma']
                # Coma X (Z7) coefficients
                a7 = random.uniform(*cfg['x_ranges']['a'])
                b7 = random.uniform(*cfg['x_ranges']['b'])
                c7 = random.uniform(*cfg['x_ranges']['c'])
                # Coma Y (Z8) coefficients
                a8 = random.uniform(*cfg['y_ranges']['a'])
                b8 = random.uniform(*cfg['y_ranges']['b'])
                c8 = random.uniform(*cfg['y_ranges']['c'])
                # Calculate final Zernike values for this position
                x_norm = (i * PATCH_SIZE / 2 - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
                y_norm = (i * PATCH_SIZE / 2 - IMAGE_HEIGHT / 2) / (IMAGE_HEIGHT / 2)
                zernike_coeffs[7] = a7 * x_norm + b7 * y_norm + c7
                zernike_coeffs[8] = a8 * x_norm + b8 * y_norm + c8

            # 2. Astigmatism (Z5, Z6)
            if random.random() < ULTIMATE_ABERRATION_COCKTAIL['astigmatism']['activation_probability']:
                cfg = ULTIMATE_ABERRATION_COCKTAIL['astigmatism']
                # Oblique Astigmatism (Z5) coefficients
                a5 = random.uniform(*cfg['oblique_ranges']['a'])
                b5 = random.uniform(*cfg['oblique_ranges']['b'])
                c5 = random.uniform(*cfg['oblique_ranges']['c'])
                # Vertical Astigmatism (Z6) coefficients
                a6 = random.uniform(*cfg['vertical_ranges']['a'])
                b6 = random.uniform(*cfg['vertical_ranges']['b'])
                c6 = random.uniform(*cfg['vertical_ranges']['c'])
                x_norm = (i * PATCH_SIZE / 2 - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
                y_norm = (i * PATCH_SIZE / 2 - IMAGE_HEIGHT / 2) / (IMAGE_HEIGHT / 2)
                zernike_coeffs[5] = a5 * x_norm + b5 * y_norm + c5
                zernike_coeffs[6] = a6 * x_norm + b6 * y_norm + c6
                
            # 3. Defocus / Field Curvature (Z4)
            if random.random() < ULTIMATE_ABERRATION_COCKTAIL['defocus_curvature']['activation_probability']:
                cfg = ULTIMATE_ABERRATION_COCKTAIL['defocus_curvature']
                a4 = random.uniform(*cfg['ranges']['a'])
                b4 = random.uniform(*cfg['ranges']['b'])
                c4 = random.uniform(*cfg['ranges']['c'])
                x_norm = (i * PATCH_SIZE / 2 - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
                y_norm = (i * PATCH_SIZE / 2 - IMAGE_HEIGHT / 2) / (IMAGE_HEIGHT / 2)
                # Field curvature is modeled as a plane in Z4
                zernike_coeffs[4] = a4 * x_norm + b4 * y_norm + c4
                
            # 4. Trefoil (Z9, Z10)
            if random.random() < ULTIMATE_ABERRATION_COCKTAIL['trefoil']['activation_probability']:
                cfg = ULTIMATE_ABERRATION_COCKTAIL['trefoil']
                # Trefoil is often not strongly field-dependent, so we model it as a constant offset
                zernike_coeffs[9] = random.uniform(*cfg['x_range'])
                zernike_coeffs[10] = random.uniform(*cfg['y_range'])

            # 5. Spherical Aberration (Z11)
            if random.random() < ULTIMATE_ABERRATION_COCKTAIL['spherical']['activation_probability']:
                cfg = ULTIMATE_ABERRATION_COCKTAIL['spherical']
                # Spherical is constant across the field
                zernike_coeffs[11] = random.uniform(*cfg['range'])

            # a. Extract a random sharp patch
            x_start = random.randint(0, w - PATCH_SIZE)
            y_start = random.randint(0, h - PATCH_SIZE)
            sharp_patch = sharp_image[y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE]
            
            # b. Generate the local PSF for the patch's center using the stochastically generated coefficients
            patch_center_x = x_start + PATCH_SIZE / 2
            patch_center_y = y_start + PATCH_SIZE / 2
            
            # NOTE: We pass the final calculated zernike_coeffs directly now
            local_psf_tensor = generate_local_psf_torch_wrapper(patch_center_x, patch_center_y, IMAGE_WIDTH, IMAGE_HEIGHT, zernike_coeffs, device)

            # c. Create the blurry patch by convolving on the GPU
            # Move sharp patch to GPU, unsqueeze to add batch and channel dims [1, 1, H, W]
            sharp_patch_tensor = torch.from_numpy(sharp_patch.copy()).to(device).unsqueeze(0).unsqueeze(0)
            
            # Unsqueeze PSF tensor to [1, 1, H, W] for conv2d
            psf_reshaped = local_psf_tensor.unsqueeze(0).unsqueeze(0)

            # Use 'same' padding for convolution
            padding = (psf_reshaped.shape[2] // 2, psf_reshaped.shape[3] // 2)
            blurry_patch_tensor = F.conv2d(sharp_patch_tensor, psf_reshaped, padding=padding)

            # --- FIX: Crop to original size to handle even kernel padding ---
            # The convolution with manual 'same' padding for an even-sized kernel (64x64)
            # results in an output of size 257x257. We crop it back to 256x256.
            h_in, w_in = sharp_patch_tensor.shape[2], sharp_patch_tensor.shape[3]
            blurry_patch_tensor = blurry_patch_tensor[:, :, :h_in, :w_in]

            # d. Move tensors back to CPU and convert to numpy for saving and normalization
            local_psf = local_psf_tensor.cpu().numpy()
            blurry_patch = blurry_patch_tensor.squeeze().cpu().numpy()
            
            # The original sharp_patch is already a numpy array

            # e. Normalize all three components to a standard [0, 1] float range for saving
            sharp_min, sharp_max = sharp_patch.min(), sharp_patch.max()
            blurry_min, blurry_max = blurry_patch.min(), blurry_patch.max()
            psf_min, psf_max = local_psf.min(), local_psf.max()

            if sharp_max > sharp_min:
                sharp_patch = (sharp_patch - sharp_min) / (sharp_max - sharp_min)
            if blurry_max > blurry_min:
                blurry_patch = (blurry_patch - blurry_min) / (blurry_max - blurry_min)
            if psf_max > psf_min:
                local_psf = (local_psf - psf_min) / (psf_max - psf_min)

            # f. Decide whether to save in train or val set
            subset = 'train' if random.random() < TRAIN_VAL_SPLIT else 'val'
            
            # g. Save all three components with consistent naming
            base_filename = f'patch_{patch_counter:06d}.npy'
            sharp_save_path = os.path.join(OUTPUT_DATA_DIR, subset, 'sharp', base_filename)
            blurry_save_path = os.path.join(OUTPUT_DATA_DIR, subset, 'blurry', base_filename)
            psf_save_path = os.path.join(OUTPUT_DATA_DIR, subset, 'psfs', base_filename)  # NEW: Save PSF
            
            np.save(sharp_save_path, sharp_patch.astype(np.float32))
            np.save(blurry_save_path, blurry_patch.astype(np.float32))
            np.save(psf_save_path, local_psf.astype(np.float32))  # NEW: Save the PSF used
            
            patch_counter += 1
            
    print(f"\nPINN dataset creation complete. Generated {patch_counter} triplets (sharp, blurry, psf).")
    print(f"Dataset saved to: {OUTPUT_DATA_DIR}")