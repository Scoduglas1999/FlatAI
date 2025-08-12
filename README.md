# FlatAI - Synthetic Flat Frames Done In Seconds Using Neural Networks

**Flat-Correction-AI** is a deep learning-based framework for advanced correction of optical artifacts in astrophotography images. This repository details a state-of-the-art approach that goes beyond simple image-to-image translation, instead modeling the physical processes of image degradation to achieve superior correction quality. The system is designed to remove a wide range of common artifacts, including vignetting, dust motes, sensor non-uniformity (PRNU), and additive gradients from sources like amplifier glow, effectively eliminating the need for traditional flat-field calibration frames.

## Abstract

Traditional flat-field correction in astrophotography relies on capturing dedicated calibration frames, a process that is often tedious and prone to error. This project introduces a novel deep learning paradigm that circumvents this requirement. We propose a specially designed U-Net architecture, `AttentionResUNetFG`, which, instead of mapping a corrupted image directly to a clean one, learns to deconstruct the image into its constituent physical components. The model explicitly predicts the multiplicative flat-field ($F_{mul}$) and the additive gradient field ($G_{add}$) that corrupt the true astronomical signal ($I_{true}$). This is achieved by training the network with a sophisticated, multi-component loss function that includes perceptual, style, and physics-informed terms, ensuring both high-fidelity reconstruction and physical consistency. The result is a robust system capable of producing high-quality, artifact-free images from a single input frame.

## The Physical Model of Image Degradation

The image formation process in the presence of flat-field artifacts can be described by the following equation:

$I_{observed} = (I_{true} \times F_{mult}) + G_{add}$

Where:
-   $I_{observed}$ is the final image captured by the sensor.
-   $I_{true}$ is the "perfect" image of the astronomical object.
-   $F_{mult}$ is the **multiplicative flat-field**, a 2D map representing artifacts like vignetting, dust motes, and photo-response non-uniformity (PRNU). Values in this field are typically centered around 1.0.
-   $G_{add}$ is the **additive gradient field**, a 2D map representing artifacts like amplifier glow and linear gradients, which are independent of the image signal.

Our network is designed to predict $F_{mult}$ and $G_{add}$ directly, allowing for a physically-motivated correction.

## Model Architecture: `AttentionResUNetFG`

The core of this project is the `AttentionResUNetFG`, a variant of the popular U-Net architecture enhanced with several key features:

1.  **Residual Blocks:** The standard convolutional blocks are replaced with residual blocks (`ResidualBlock`) to facilitate the training of a deeper network and mitigate the vanishing gradient problem.
2.  **Attention Gates:** Attention gates are integrated into the decoder's skip connections. This allows the model to selectively focus on the most relevant features from the encoder path, improving reconstruction accuracy.
3.  **Multi-Headed Output:** The network has three output heads, predicting:
    -   $F_{mul}$: The multiplicative field.
    -   $G_{add}$: The additive field.
    -   $M$: A **confidence map**, which the model uses to modulate the strength of the correction. This prevents the model from altering parts of the image that are already clean.
4.  **Spatially-Aware Input:** The model receives four input channels: the affected image, two normalized coordinate channels (`xx`, `yy`), and a dynamically generated local noise map. This provides the network with global spatial context and helps it differentiate between signal, noise, and artifacts.

<details>
<summary>Click to expand a diagram of the model architecture</summary>

```mermaid
graph TD
    subgraph Input
        A[Image + Coords + Noise]
    end

    subgraph Encoder
        A --> B(enc1: ResBlock 4->64);
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
    end
    
    subgraph Output Heads
        Z --> AA(out_conv: Conv2d 64->3);
        AA --> BB[F_mul prediction];
        AA --> CC[G_add prediction];
        AA --> DD[M (confidence) prediction];
    end
```
</details>

## Synthetic Data Generation

A robust model requires a diverse and realistic dataset. The `create_flat_dataset.py` script generates a large-scale synthetic dataset by applying procedurally generated artifacts to a collection of clean, high-quality astronomical images.

The generation process includes:
-   **Multiplicative Artifacts ($F_{mult}$):**
    -   **Vignetting:** Parametrized radial gradients.
    -   **Dust Motes:** A sophisticated generator for creating realistic dust motes, including soft, hard, and donut-shaped occlusions with irregular edges.
    -   **PRNU:** Low-frequency, noise-like patterns simulating sensor non-uniformity.
-   **Additive Artifacts ($G_{add}$):**
    -   **Linear Gradients:** Simulating uneven skyglow or light pollution.
    -   **Amplifier Glow:** Exponential glow originating from the corners of the image.

The script also employs a **curriculum learning** strategy, gradually increasing the intensity and complexity of the artifacts as more data is generated. This helps the model learn basic corrections before tackling more challenging examples.

## Training and Loss Function

The model is trained to minimize a composite loss function, carefully designed to balance reconstruction accuracy with perceptual quality and physical realism.

$L_{total} = \lambda_{L1} L_{L1} + \lambda_{LPIPS} L_{LPIPS} + \lambda_{Style} L_{Style} + \lambda_{Physics} L_{Physics} + L_{Priors}$

-   **$L_{L1}$:** Standard L1 pixel-wise loss between the corrected and ground-truth images.
-   **$L_{LPIPS}$:** The Learned Perceptual Image Patch Similarity (LPIPS) loss, which better aligns with human perception of image quality.
-   **$L_{Style}$:** A VGG-based style loss that preserves the texture of the original image by comparing the Gram matrices of feature maps.
-   **$L_{Physics}$:** A **physics-informed consistency loss**. After the model predicts a clean image $I_{pred}$, this image is fed *back* into the forward physics model using the predicted fields $F_{pred}$ and $G_{pred}$. The result is compared to the original input image $I_{observed}$. This forces the model to learn the true degradation process, not just an arbitrary image-to-image mapping.
-   **$L_{Priors}$:** A set of regularization terms that impose physically-motivated constraints, such as smoothness on the predicted fields and penalties for altering already-clean regions of the image.

## Usage

### 1. Setup
Clone the repository and install the necessary packages. A `requirements.txt` is not yet provided, but key dependencies include `torch`, `torchvision`, `numpy`, `astropy`, `matplotlib`, `tqdm`, and `lpips`.

```bash
git clone https://github.com/your-username/Flat-Correction-AI.git
cd Flat-Correction-AI
pip install torch torchvision numpy astropy matplotlib tqdm lpips
```

### 2. Dataset Generation
1.  Populate the `sharp_images/` directory with high-quality, artifact-free astronomical images (FITS, PNG, or JPG).
2.  Run the dataset generation script:
    ```bash
    python create_flat_dataset.py
    ```
    This will create the `randomized_flat_dataset/` directory containing training and validation data.

### 3. Training
To train the model, run the main training script. Training will automatically resume from a checkpoint if one is found.
```bash
python train_unet_flat.py
```
Training progress, including sample images and a loss curve, will be saved to the `flat_training_samples/` directory.

### 4. Inference
A graphical user interface is provided for applying the trained model to new images.
```bash
python gui_corrector_final.py
```

## Future Work
-   **Hyperparameter Optimization:** A systematic search for the optimal weights ($\lambda$) of the loss components could further improve results.
-   **Combined Aberration and Flat Correction:** The architecture could be extended to simultaneously correct for optical aberrations (like coma and astigmatism) in addition to flat-field artifacts.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
