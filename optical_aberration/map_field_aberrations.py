import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import our custom model architecture
from PSFNet_Model import PSFNetCNN

# Astronomy-specific imports
try:
    import xisf
    from astropy.io import fits
    from astropy.stats import mad_std
    from photutils.detection import DAOStarFinder
except ImportError:
    print("Error: Required astronomy libraries not found.")
    print("Please install them by running:")
    print("pip install astropy photutils xisf-python pandas tqdm")
    exit()

# --- I. SCRIPT CONFIGURATION ---

# Path to the trained model file
MODEL_PATH = "./psfnet_model_final.pth"

# --- IMPORTANT ---
# Path to the large-format astronomical image to be analyzed.
# CHANGE THIS TO THE ACTUAL PATH OF YOUR IMAGE.
IMAGE_PATH = "C:/Users/scdou/Downloads/06-04/Lights/LIGHT-20250605T150020Z-1-003/LIGHT/2025-06-04_22-14-42_Lum_-15.00_60.00s_0031.fits" # Example: "C:/Astro/Images/M31_L.fits"

# Path for the output CSV file containing the aberration map
OUTPUT_CSV_PATH = "./aberration_map.csv"

# The size of the square cutout to extract for each star (should match model input)
CUTOUT_SIZE = 64

# The significance threshold for star detection (in standard deviations above background)
STAR_DETECTION_THRESHOLD_SIGMA = 5.0

# --- II. HELPER FUNCTIONS ---

def load_trained_model(model_path, device):
    """
    Loads the trained PSFNetCNN model from a .pth file.
    
    Args:
        model_path (str): The path to the saved model state dictionary.
        device (torch.device): The device to load the model onto ('cpu' or 'cuda').
        
    Returns:
        torch.nn.Module: The loaded and initialized model in evaluation mode.
    """
    print(f"Loading trained model from {model_path}...")
    model = PSFNetCNN(num_output_coeffs=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    return model

def load_astro_image(image_path):
    """
    Loads an astronomical image from a FITS or XISF file.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        np.ndarray: A 2D NumPy array containing the image data.
    """
    print(f"Loading astronomical image from {image_path}...")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    _, file_extension = os.path.splitext(image_path)
    
    if file_extension.lower() in ['.fits', '.fit']:
        with fits.open(image_path) as hdul:
            image_data = hdul[0].data
    elif file_extension.lower() == '.xisf':
        xisf_list = xisf.read(image_path)
        image_data = xisf_list[0].data
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please use FITS or XISF.")
    
    # Ensure data is 2D (e.g., convert color to grayscale by averaging)
    if image_data.ndim == 3:
        print("3D image data detected, converting to grayscale by averaging.")
        image_data = image_data.mean(axis=2)
        
    print(f"Image loaded with shape: {image_data.shape}")
    return image_data.astype(np.float32)

def find_stars(image_data, sigma_threshold):
    """
    Detects stars in the image data using DAOStarFinder.
    
    Args:
        image_data (np.ndarray): The 2D image data.
        sigma_threshold (float): The detection threshold in sigma.
        
    Returns:
        astropy.table.Table or None: A table of found stars with their coordinates.
    """
    print("Finding stars in the image...")
    bkg_sigma = mad_std(image_data)
    daofind = DAOStarFinder(fwhm=4.0, threshold=sigma_threshold * bkg_sigma)
    sources = daofind(image_data)
    
    if sources is None:
        print("No stars found.")
        return None
    
    print(f"Found {len(sources)} stars.")
    return sources

def extract_cutout(image_data, x, y, size):
    """
    Extracts a square cutout from the image centered on the given coordinates.
    
    Args:
        image_data (np.ndarray): The full 2D image.
        x (float): The x-coordinate of the center.
        y (float): The y-coordinate of the center.
        size (int): The width and height of the cutout.
        
    Returns:
        np.ndarray or None: The cutout image data, or None if the cutout is
                            out of bounds.
    """
    half_size = size // 2
    x_int, y_int = int(round(x)), int(round(y))
    
    # Check boundaries
    if (x_int - half_size < 0 or y_int - half_size < 0 or
        x_int + half_size >= image_data.shape[1] or
        y_int + half_size >= image_data.shape[0]):
        return None
        
    cutout = image_data[y_int - half_size : y_int + half_size,
                        x_int - half_size : x_int + half_size]
    return cutout

def preprocess_for_model(image_cutout):
    """
    Preprocesses a single image cutout to match the model's training data format.
    
    Args:
        image_cutout (np.ndarray): The 2D NumPy array of the star cutout.
        
    Returns:
        torch.Tensor: A tensor ready to be input to the model.
    """
    # 1. Ensure dtype is float32
    processed_cutout = image_cutout.astype(np.float32)
    
    # 2. Normalize to [0, 1] range (assuming original data is 16-bit)
    processed_cutout = processed_cutout / 65535.0
    
    # 3. Convert to PyTorch tensor
    tensor = torch.from_numpy(processed_cutout)
    
    # 4. Add channel and batch dimensions: [H, W] -> [1, 1, H, W]
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor

# --- III. MAIN EXECUTION LOGIC ---

if __name__ == "__main__":
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 2. Load Model
        model = load_trained_model(MODEL_PATH, device)

        # 3. Load Image
        image_data = load_astro_image(IMAGE_PATH)

        # 4. Find Stars
        star_sources = find_stars(image_data, STAR_DETECTION_THRESHOLD_SIGMA)
        
        if star_sources is None:
            print("Exiting as no stars were found to process.")
            exit()

        # 5. Process Stars and Predict Aberrations
        aberration_results = []
        print(f"\nProcessing {len(star_sources)} stars to map aberrations...")
        
        for star in tqdm(star_sources, desc="Analyzing Stars"):
            x, y = star['xcentroid'], star['ycentroid']
            
            # Extract a 64x64 cutout for the star
            cutout = extract_cutout(image_data, x, y, CUTOUT_SIZE)
            
            if cutout is None:
                continue  # Skip stars too close to the edge

            # Preprocess the cutout and send to the model
            input_tensor = preprocess_for_model(cutout).to(device)
            
            with torch.no_grad():
                # Get the model's prediction
                predicted_coeffs = model(input_tensor)
            
            # Move prediction to CPU and convert to a flat numpy array
            coeffs = predicted_coeffs.cpu().numpy().flatten()
            
            # Store the results
            aberration_results.append({
                'x_coord': x,
                'y_coord': y,
                'pred_z4': coeffs[0],
                'pred_z7': coeffs[1],
                'pred_z8': coeffs[2]
            })

        # 6. Save Results
        if not aberration_results:
            print("\nProcessing complete, but no valid star cutouts could be analyzed.")
        else:
            print(f"\nProcessing complete. Analyzed {len(aberration_results)} stars.")
            results_df = pd.DataFrame(aberration_results)
            results_df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"Aberration map saved successfully to: {OUTPUT_CSV_PATH}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure MODEL_PATH and IMAGE_PATH are set correctly.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}") 