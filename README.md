# FlatAI: An Advanced Deep Learning Approach to Astrophotography Image Correction

**FlatAI** is a sophisticated, 40-million-parameter U-Net based neural network designed to tackle complex image correction tasks in astrophotography. The project has evolved through two primary phases:

1.  **Synthetic Flat-Frame Generation:** The current focus of the project. The model learns to remove artifacts like dust motes, vignetting, and uneven illumination from astronomical images, effectively creating a perfect "flat frame" on the fly.
2.  **Optical Aberration Correction (Physics-Informed):** The original goal. This version of the model was a Physics-Informed Neural Network (PINN) trained to correct optical aberrations like coma by simulating and then reversing them.

This repository contains the code for both approaches, but do note that a significant portion of this project is for personal learning and experimentation.

While looking at the project you may realize that some files seem disjointed, superfluous, or you may notice that there is bad code in some. This is because I have not been culling files as they become unnecessary like I would on a real public project. In the end, I may release the final model once done training, but that day is not today, so browse this repo with caution understanding much of it can seem random at times.

Training samples will be added in batches as I continue working. I likely will not be uploading each one as they finish, but likely there will be more at least once a day if improvement warrants it.

## Table of Contents
* [The "Flat-Field" Problem in Astrophotography](#the-flat-field-problem-in-astrophotography)
* [The "Optical Aberration" Problem](#the-optical-aberration-problem)
* [Project Evolution and Challenges](#project-evolution-and-challenges)
* [Model Architecture](#model-architecture)
* [Models and Training](#models-and-training)
  * [1. Flat-Field Correction Model](#1-flat-field-correction-model)
  * [2. Optical Aberration (PINN) Model](#2-optical-aberration-pinn-model)
* [Usage](#usage)
  * [1. Setup](#1-setup)
  * [2. Dataset Generation](#2-dataset-generation)
  * [3. Training](#3-training)
  * [4. Inference](#4-inference)
* [Future Work](#future-work)

## The "Flat-Field" Problem in Astrophotography

A "flat frame" is a calibration image used to correct for imperfections in the optical train of a telescope (e.g., dust on the sensor, vignetting from the telescope optics). These imperfections create a fixed pattern of "shadows" and brightness variations on every image taken. By dividing the astronomical image by a flat frame, these artifacts can be removed.

However, taking high-quality flat frames can be difficult and time-consuming. **FlatAI** aims to solve this by learning the *characteristics* of these artifacts and removing them directly from the science image, eliminating the need for a separate flat frame.

The image formation model for flat-fielding is as follows:
\[ I_{observed} = (I_{true} \times F_{mult}) + G_{add} \]
Where:
- \(I_{observed}\) is the final image captured by the sensor.
- \(I_{true}\) is the "perfect" image of the astronomical object.
- \(F_{mult}\) is the multiplicative flat-field, containing artifacts like vignetting and dust motes.
- \(G_{add}\) is the additive gradient field, containing artifacts like amplifier glow and linear gradients.

## The "Optical Aberration" Problem

Optical aberrations are distortions in an image caused by the physical properties of the lenses and mirrors in a telescope. Aberrations like coma, astigmatism, and chromatic aberration can deform stars and reduce image sharpness, especially towards the edges of the field of view.

The initial version of this project (now located in the `optical_aberration` directory) was designed as a Physics-Informed Neural Network (PINN) to correct these aberrations. It used Zernike polynomials to mathematically model the aberrations, generate a dataset of "perfect" vs. "aberrated" images, and then train a network to reverse the distortion.

The image formation model for optical aberrations is a convolution:
\[ I_{observed} = I_{true} * PSF \]
Where:
- \(I_{observed}\) is the blurry image captured by the sensor.
- \(I_{true}\) is the "perfect" sharp image.
- \(PSF\) is the Point Spread Function, which represents the optical aberration.

### Project Evolution and Challenges

While the PINN approach for aberration correction showed promise, it encountered several challenges:

*   **Noise Amplification:** The model would sometimes amplify background noise in an attempt to sharpen stars.
*   **Artifact Generation:** It could create unnatural-looking dark rings or other artifacts around stars.
*   **"Cheating" the Loss Function:** The model found ways to slightly improve its loss score by making non-physical changes to the image, rather than truly correcting the aberrations.

Due to these challenges, the project pivoted to focus on the more constrained, but equally important, problem of flat-field correction.

## Model Architecture

The core of the project is an Attention-based Residual U-Net (`unet_model.py`). This architecture uses residual blocks to prevent vanishing gradients and attention gates to focus on salient features from the skip connections.

![AttentionResUNet Architecture]
graph TD
    subgraph Encoder
        A[Input] --> B(enc1: ResBlock 1->64);
        B --> C{pool1: MaxPool2d};
        C --> D(enc2: ResBlock 64->128);
        D --> E{pool2: MaxPool2d};
        E --> F(enc3: ResBlock 128->256);
        F --> G{pool3: MaxPool2d};
        G --> H(enc4: ResBlock 256->512);
        H --> I{pool4: MaxPool2d};
    end

    subgraph Bottleneck
        I --> J(bottleneck: ResBlock 512->1024);
    end

    subgraph Decoder
        J --> K(up4: Upsample);
        K --> L(dec_conv4: ResBlock 1024->512);
        subgraph Skip Connection 4
            H --> M(att4: AttentionGate);
            L --> M;
        end
        M --> N(dec_combine4: ResBlock 1024->512);
        N --> O(up3: Upsample);
        O --> P(dec_conv3: ResBlock 512->256);
        subgraph Skip Connection 3
            F --> Q(att3: AttentionGate);
            P --> Q;
        end
        Q --> R(dec_combine3: ResBlock 512->256);
        R --> S(up2: Upsample);
        S --> T(dec_conv2: ResBlock 256->128);
        subgraph Skip Connection 2
            D --> U(att2: AttentionGate);
            T --> U;
        end
        U --> V(dec_combine2: ResBlock 256->128);
        V --> W(up1: Upsample);
        W --> X(dec_conv1: ResBlock 128->64);
        subgraph Skip Connection 1
            B --> Y(att1: AttentionGate);
            X --> Y;
        end
        Y --> Z(dec_combine1: ResBlock 128->64);
        Z --> AA(out_conv: Conv2d 64->1);
        AA --> BB[Output];
    end

## Models and Training

Two separate training workflows are provided, one for each of the project's goals.

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
\[ \mathcal{L}_{total} = \lambda_{L1}\mathcal{L}_{L1} + \lambda_{LPIPS}\mathcal{L}_{LPIPS} + \lambda_{Style}\mathcal{L}_{Style} + \lambda_{Physics}\mathcal{L}_{Physics} \]
Where:
-   **\(\mathcal{L}_{L1}\):** A pixel-level L1 loss for basic reconstruction accuracy.
-   **\(\mathcal{L}_{LPIPS}\):** The Learned Perceptual Image Patch Similarity (LPIPS) loss, which better captures human perception of image similarity.
-   **\(\mathcal{L}_{Style}\):** A VGG-based style loss calculated from the Gram matrix of feature maps. The Gram matrix is defined as:
    \[ G_{l}(I) = \frac{1}{C_l H_l W_l} F_l(I) F_l(I)^T \]
    where \(F_l(I)\) is the flattened feature map from layer \(l\) of a VGG network.
-   **\(\mathcal{L}_{Physics}\):** A physics-consistency loss. The model's output \(I_{pred}\) is fed back into the forward physics model to reconstruct the observed image: \(I_{reconstructed} = (I_{pred} \times F_{mult}) + G_{add}\). The loss is the L1 distance between \(I_{reconstructed}\) and the original \(I_{observed}\).

### 2. Optical Aberration (PINN) Model

*   **Training Script:** `optical_aberration/train_unet_pinn.py`
*   **Dataset Generator:** `optical_aberration/create_pinn_dataset.py`

This model attempts to deconvolve the Point Spread Function (PSF) caused by optical aberrations. The `create_pinn_dataset.py` script works as follows:

1.  **Aberration Modeling:** It uses Zernike polynomials to model optical aberrations (specifically, coma) across the field of view of a simulated telescope.
2.  **PSF Generation:** For any given point in the image, it can generate a unique PSF that represents the aberration at that location.
3.  **Dataset Creation:** It takes "perfect" images, convolves them with the generated PSFs to create "blurry" versions, and saves the sharp/blurry/PSF triplets.

#### Loss Function

The PINN model also uses a multi-part loss function, similar to the flat-field model, but with a different physics-informed component:
\[ \mathcal{L}_{total} = \lambda_{L1}\mathcal{L}_{L1} + \lambda_{LPIPS}\mathcal{L}_{LPIPS} + \lambda_{Style}\mathcal{L}_{Style} + \lambda_{Physics}\mathcal{L}_{Physics} \]
-   **\(\mathcal{L}_{Physics}\):** The physics-informed loss here is a re-convolution loss. The model's "corrected" (sharpened) output \(I_{pred}\) is convolved with the original PSF: \(I_{reblurred} = I_{pred} * PSF\). The loss is the L1 distance between \(I_{reblurred}\) and the original blurry input \(I_{observed}\).

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
