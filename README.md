# FlatAI: An Advanced Deep Learning Approach to Astrophotography Image Correction

**FlatAI** is a sophisticated, 40-million-parameter U-Net based neural network designed to tackle complex image correction tasks in astrophotography. The project has evolved through two primary phases:

1.  **Synthetic Flat-Frame Generation:** The current focus of the project. The model learns to remove artifacts like dust motes, vignetting, and uneven illumination from astronomical images, effectively creating a perfect "flat frame" on the fly.
2.  **Optical Aberration Correction (Physics-Informed):** The original goal. This version of the model was a Physics-Informed Neural Network (PINN) trained to correct optical aberrations like coma by simulating and then reversing them.

This repository contains the code for both approaches, representing a comprehensive toolkit for deep-learning-based astrophotography processing.

## The "Flat-Field" Problem in Astrophotography

A "flat frame" is a calibration image used to correct for imperfections in the optical train of a telescope (e.g., dust on the sensor, vignetting from the telescope optics). These imperfections create a fixed pattern of "shadows" and brightness variations on every image taken. By dividing the astronomical image by a flat frame, these artifacts can be removed.

However, taking high-quality flat frames can be difficult and time-consuming. **FlatAI** aims to solve this by learning the *characteristics* of these artifacts and removing them directly from the science image, eliminating the need for a separate flat frame.

## The "Optical Aberration" Problem

Optical aberrations are distortions in an image caused by the physical properties of the lenses and mirrors in a telescope. Aberrations like coma, astigmatism, and chromatic aberration can deform stars and reduce image sharpness, especially towards the edges of the field of view.

The initial version of this project (now located in the `optical_aberration` directory) was designed as a Physics-Informed Neural Network (PINN) to correct these aberrations. It used Zernike polynomials to mathematically model the aberrations, generate a dataset of "perfect" vs. "aberrated" images, and then train a network to reverse the distortion.

### Project Evolution and Challenges

While the PINN approach for aberration correction showed promise, it encountered several challenges:

*   **Noise Amplification:** The model would sometimes amplify background noise in an attempt to sharpen stars.
*   **Artifact Generation:** It could create unnatural-looking dark rings or other artifacts around stars.
*   **"Cheating" the Loss Function:** The model found ways to slightly improve its loss score by making non-physical changes to the image, rather than truly correcting the aberrations.

Due to these challenges, the project pivoted to focus on the more constrained, but equally important, problem of flat-field correction.

## Models and Training

The core of the project is an Attention-based Residual U-Net (`unet_model.py`). Two separate training workflows are provided, one for each of the project's goals.

### 1. Flat-Field Correction Model

*   **Training Script:** `train_unet_flat.py`
*   **Dataset Generator:** `create_flat_dataset.py`

This model is trained on a synthetic dataset of images with procedurally generated flat-field artifacts. The `create_flat_dataset.py` script takes a directory of "perfect" astronomical images and applies a variety of effects:

*   **Multiplicative Artifacts (Flat Field):**
    *   Vignetting (darker/brighter corners)
    *   Dust motes (dark, light, in-focus, out-of-focus)
    *   Photo-Response Non-Uniformity (PRNU)
*   **Additive Artifacts (Gradient Field):**
    *   Linear gradients
    *   Amplifier glow

The model is then trained to take an "affected" image and reproduce the "perfect" ground-truth image.

#### Loss Function

The flat-field model uses a sophisticated multi-part loss function to ensure high-fidelity results:

*   **L1 Loss:** For pixel-level accuracy.
*   **LPIPS Loss:** A perceptual loss that encourages the output to be visually similar to the ground truth.
*   **Style Loss (VGG-based):** A texture-based loss that helps prevent the model from creating unnatural patterns.
*   **Physics-Consistency Loss:** The model's output is fed back into the forward physics model (i.e., the flat/gradient fields are re-applied). The result is compared to the original input image, ensuring the correction is physically plausible.

### 2. Optical Aberration (PINN) Model

*   **Training Script:** `optical_aberration/train_unet_pinn.py`
*   **Dataset Generator:** `optical_aberration/create_pinn_dataset.py`

This model attempts to deconvolve the Point Spread Function (PSF) caused by optical aberrations. The `create_pinn_dataset.py` script works as follows:

1.  **Aberration Modeling:** It uses Zernike polynomials to model optical aberrations (specifically, coma) across the field of view of a simulated telescope.
2.  **PSF Generation:** For any given point in the image, it can generate a unique PSF that represents the aberration at that location.
3.  **Dataset Creation:** It takes "perfect" images, convolves them with the generated PSFs to create "blurry" versions, and saves the sharp/blurry/PSF triplets.

#### Loss Function

The PINN model also uses a multi-part loss function:

*   **L1 Loss, LPIPS Loss, Style Loss:** Similar to the flat-field model, these ensure the de-blurred image is accurate and visually correct.
*   **Physics-Informed Loss (Re-convolution):** The model's "corrected" (sharpened) output is convolved with the original PSF. This "re-blurred" image is then compared to the original blurry input. This forces the network to learn a deconvolution that is the inverse of the physical blurring process.

## Usage

### 1. Setup

Clone the repository and install the required Python packages:
```bash
git clone https://github.com/your-username/FlatAI.git
cd FlatAI
pip install -r requirements.txt
```

### 2. Dataset Generation

First, you need to populate the `sharp_images` directory with high-quality, artifact-free astronomical images. FITS, XISF, JPG, and PNG formats are supported.

**To generate the flat-field dataset:**
```bash
python create_flat_dataset.py
```

**To generate the optical aberration (PINN) dataset:**
```bash
python optical_aberration/create_pinn_dataset.py
```
These scripts will create `randomized_flat_dataset` and `randomized_pinn_dataset` directories, respectively.

### 3. Training

You can train either model using its corresponding script. Both scripts support resuming from a checkpoint.

**To train the flat-field model:**
```bash
python train_unet_flat.py
```

**To train the PINN model:**
```bash
python optical_aberration/train_unet_pinn.py
```
Training progress (including sample images and loss curves) will be saved to the `flat_training_samples` or `pinn_training_samples` directories.

### 4. Inference

*   **Flat-Field Correction:** Use `gui_corrector.py` for a graphical interface to correct images with the flat-field model.
*   **Aberration Correction:** Use `optical_aberration/correct_image.py` to apply the aberration correction model.

## Future Work

*   **Combined Model:** A single model could be trained to correct both flat-field artifacts and optical aberrations simultaneously.
*   **Real-World Data:** The models are currently trained purely on synthetic data. Fine-tuning on real-world "before and after" images could improve performance.
*   **Hyperparameter Optimization:** A more systematic search for optimal loss weights and other hyperparameters could yield better results.
*   **User Interface:** The existing GUI for the flat-field model could be expanded to include the aberration correction model.
