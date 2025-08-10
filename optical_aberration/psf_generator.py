import numpy as np
import torch

# --- Optical Simulation Constants ---
# These parameters should ideally match those used for the original
# training data generation to ensure consistency.

# Size of the PSF image to generate. This should match the cutout size
# used for training the model.
PSF_IMAGE_SIZE = 64

# Diameter of the telescope's pupil in pixels for the simulation.
PUPIL_DIAMETER = 64  # Should be <= PSF_IMAGE_SIZE

# Wavelength of light in waves (can be normalized to 1)
WAVELENGTH = 1.0


def create_pupil_mask(image_size, pupil_diameter):
    """
    Creates a circular mask representing the telescope pupil.
    
    Args:
        image_size (int): The size of the square image (e.g., 64).
        pupil_diameter (int): The diameter of the pupil within the image.
        
    Returns:
        np.ndarray: A 2D array with 1s inside the pupil and 0s outside.
    """
    radius = pupil_diameter / 2
    x = np.linspace(-image_size / 2, image_size / 2, image_size)
    y = np.linspace(-image_size / 2, image_size / 2, image_size)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx**2 + yy**2)
    mask = (rho <= radius).astype(float)
    return mask

def generate_zernike_polynomial(index, rho, phi):
    """
    Generates a specific Zernike polynomial on a grid.
    (Noll indexing)
    
    Args:
        index (int): The Noll index of the Zernike polynomial (e.g., 4, 7, 8).
        rho (np.ndarray): 2D array of radial coordinates.
        phi (np.ndarray): 2D array of angular coordinates.
        
    Returns:
        np.ndarray: The calculated Zernike polynomial.
    """
    if index == 4:  # Z(2, 0) - Defocus
        return np.sqrt(3) * (2 * rho**2 - 1)
    if index == 7:  # Coma X, consistent with project's Z7 = Coma X convention
        return np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(phi)
    if index == 8:  # Coma Y, consistent with project's Z8 = Coma Y convention
        return np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.sin(phi)
    raise ValueError(f"Zernike polynomial for index {index} is not implemented.")

def construct_wavefront(coeffs, pupil_mask):
    """
    Constructs a wavefront error map from Zernike coefficients.
    
    Args:
        coeffs (dict): A dictionary mapping Zernike indices to their
                       magnitudes (e.g., {4: 0.5, 7: -0.2}).
        pupil_mask (np.ndarray): The pupil mask.
        
    Returns:
        np.ndarray: The combined wavefront error map.
    """
    image_size = pupil_mask.shape[0]
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    
    wavefront = np.zeros_like(pupil_mask, dtype=float)
    for index, magnitude in coeffs.items():
        if magnitude != 0:
            wavefront += magnitude * generate_zernike_polynomial(index, rho, phi)
            
    return wavefront * pupil_mask

def calculate_psf(wavefront, pupil_mask):
    """
    Calculates the Point Spread Function (PSF) from a wavefront.
    
    Args:
        wavefront (np.ndarray): The wavefront error map.
        pupil_mask (np.ndarray): The pupil mask.
        
    Returns:
        np.ndarray: The calculated, normalized, noise-free PSF.
    """
    # Create the complex pupil function
    pupil_function = pupil_mask * np.exp(2j * np.pi * wavefront / WAVELENGTH)
    
    # Calculate the PSF via Fourier Transform
    psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil_function)))**2
    
    # Normalize the PSF
    psf /= np.sum(psf)
    
    return psf 