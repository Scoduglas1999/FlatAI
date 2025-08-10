# train_unet_pinn.py - Final version for High-Fidelity Replication

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import lpips # <-- Import LPIPS for perceptual loss
import time # <-- Import time for profiling
import warnings # <-- Import warnings module
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

# --- MONKEY-PATCH FOR LPIPS STABILITY ---
# The LPIPS library has a bug where it can cause a NaN gradient error (SqrtBackward0)
# if a feature map inside its network becomes all zeros. This happens surprisingly often.
# To fix this, we replace its internal normalization function with a numerically stable version.
def safe_normalize_tensor(in_feat, eps=1e-10):
    """
    Stabilized version of lpips.normalize_tensor to prevent sqrt(0) gradients.
    """
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True) + eps)
    return in_feat / norm_factor

lpips.normalize_tensor = safe_normalize_tensor
# --- END MONKEY-PATCH ---

# Suppress specific UserWarning from torchvision used by lpips
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated*")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated*")

# =============================================================================
# --- Style Loss Implementation (VGG-based) ---
# =============================================================================

class VGG19(nn.Module):
    """A VGG19 model wrapper to extract intermediate feature maps."""
    def __init__(self, requires_grad=False):
        super().__init__()
        # Use the recommended 'weights' parameter for modern torchvision
        vgg_features = vgg19(weights=VGG19_Weights.DEFAULT).features
        
        # We use layers relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 for style
        self.slice1 = nn.Sequential(*vgg_features[0:2])
        self.slice2 = nn.Sequential(*vgg_features[2:7])
        self.slice3 = nn.Sequential(*vgg_features[7:12])
        self.slice4 = nn.Sequential(*vgg_features[12:21])
        self.slice5 = nn.Sequential(*vgg_features[21:30])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]

def gram_matrix(input_tensor):
    """Calculates the Gram matrix for a batch of feature maps."""
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b, c, h * w)
    # Batch-wise matrix multiplication
    gram = torch.bmm(features, features.transpose(1, 2))
    # Normalize by the number of elements in each feature map.
    # Add a small epsilon for numerical stability.
    return gram.div(c * h * w + 1e-8)

class StyleLoss(nn.Module):
    """Computes the style loss between two images using a VGG network."""
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        # VGG networks expect images normalized with ImageNet stats
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, y_pred, y_true):
        # Input images are [-1, 1], 1-channel. VGG expects [0, 1], 3-channel, ImageNet-normalized.
        y_pred_rgb = y_pred.repeat(1, 3, 1, 1)
        y_true_rgb = y_true.repeat(1, 3, 1, 1)
        
        y_pred_rescaled = (y_pred_rgb + 1) / 2
        y_true_rescaled = (y_true_rgb + 1) / 2
        
        y_pred_normalized = self.normalize(y_pred_rescaled)
        y_true_normalized = self.normalize(y_true_rescaled)

        y_pred_features = self.vgg(y_pred_normalized)
        y_true_features = self.vgg(y_true_normalized)

        loss = 0.0
        for y_pred_feat, y_true_feat in zip(y_pred_features, y_true_features):
            gram_pred = gram_matrix(y_pred_feat)
            gram_true = gram_matrix(y_true_feat)
            loss += self.criterion(gram_pred, gram_true)
            
        return loss

# Import custom classes from other project files
from unet_pinn_dataset import DeconvolutionPINNDataset
from unet_model import AttentionResUNet

# =============================================================================
# --- 1. CONFIGURATION / HYPERPARAMETERS ---
# =============================================================================
# -- Paths and General Config --
DATA_DIR = "./randomized_pinn_dataset/"
MODEL_CHECKPOINT_PATH = "./unet_pinn_checkpoint.pth"
FINAL_MODEL_SAVE_PATH = "./unet_pinn_model_final.pth"
SAMPLE_IMAGE_DIR = "./pinn_training_samples/"
LOSS_CURVE_PATH = "./pinn_loss_curve.png"

# -- Training Hyperparameters --
RESUME_TRAINING = True
FINE_TUNING_MODE = True     # NEW: Set to True to reset history for a new task
LEARNING_RATE = 1e-5 # Drastically lowered for fine-tuning
BATCH_SIZE = 8
NUM_EPOCHS = 200

# --- Final Loss Configuration: The Quartet of Fidelity ---
LAMBDA_PHYSICS = 1.5   # Weight for physical consistency
LAMBDA_LPIPS = 1.0     # The primary driver for perceptual replication
LAMBDA_L1 = 0.15       # A foundational weight for pixel-level accuracy
LAMBDA_STYLE = 1e5     # Reduced weight for stability during fine-tuning

# =============================================================================
# --- 2. HELPER FUNCTIONS ---
# =============================================================================

def torch_convolve2d(image, psf):
    """
    Performs 2D convolution of a batch of images with a batch of PSFs efficiently
    using grouped convolution. Assumes psf is already centered.
    """
    # image shape: (B, 1, H, W)
    # psf shape:   (B, 1, kH, kW)
    batch_size = image.shape[0]

    # Pad the image to maintain size after convolution.
    # This handles both even and odd kernel sizes correctly.
    _, _, h_psf, w_psf = psf.shape
    pad_h_top = (h_psf - 1) // 2
    pad_h_bottom = h_psf - 1 - pad_h_top
    pad_w_left = (w_psf - 1) // 2
    pad_w_right = w_psf - 1 - pad_w_left
    
    padded_image = F.pad(image, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode='reflect')

    # For grouped convolution to work per-sample, we need to restructure the tensors.
    # The weight tensor (psf) is shaped as (out_channels, in_channels_per_group, kH, kW).
    # Our psf is (B, 1, kH, kW). With groups=B, this implies out_channels=B and in_channels_per_group=1.
    # This requires the input tensor to have B input channels. Our image is (B, 1, H, W).
    # We satisfy this by reshaping the input from (B, 1, H, W) to (1, B, H, W).
    padded_image_reshaped = padded_image.view(1, batch_size, padded_image.shape[2], padded_image.shape[3])

    # Prepare PSF for convolution. F.conv2d expects a kernel, so we need to flip the PSF.
    psf_flipped = torch.flip(psf, [2, 3])

    # Use grouped convolution.
    # input: (1, B, H_padded, W_padded)
    # weight: (B, 1, kH, kW) --> interpreted as (out_channels=B, in_channels/groups=1, kH, kW)
    # groups: B
    reblurred = F.conv2d(
        padded_image_reshaped,
        psf_flipped,
        groups=batch_size
    )

    # Reshape the output from (1, B, H_out, W_out) back to (B, 1, H_out, W_out)
    return reblurred.view(batch_size, 1, reblurred.shape[2], reblurred.shape[3])

def save_sample_images(epoch, blurry, sharp, corrected, save_dir):
    """Saves a grid of sample images to monitor training progress."""
    blurry = (blurry[0].cpu().detach() + 1) / 2
    sharp = (sharp[0].cpu().detach() + 1) / 2
    corrected = (corrected[0].cpu().detach() + 1) / 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(blurry.squeeze(), cmap='gray')
    axs[0].set_title("Blurry Input")
    axs[0].axis('off')
    
    axs[1].imshow(corrected.squeeze(), cmap='gray')
    axs[1].set_title("AI Corrected Output")
    axs[1].axis('off')

    axs[2].imshow(sharp.squeeze(), cmap='gray')
    axs[2].set_title("Sharp Ground Truth")
    axs[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch+1:04d}_sample.png")
    plt.savefig(save_path)
    plt.close(fig)

def save_checkpoint(epoch, model, optimizer, best_loss, filename="unet_pinn_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
    }, filename)

def main():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(SAMPLE_IMAGE_DIR, exist_ok=True)

    # --- Datasets and DataLoaders ---
    try:
        train_dataset = DeconvolutionPINNDataset(
            sharp_dir=os.path.join(DATA_DIR, 'train', 'sharp'),
            blurry_dir=os.path.join(DATA_DIR, 'train', 'blurry'),
            psf_dir=os.path.join(DATA_DIR, 'train', 'psfs')
        )
        val_dataset = DeconvolutionPINNDataset(
            sharp_dir=os.path.join(DATA_DIR, 'val', 'sharp'),
            blurry_dir=os.path.join(DATA_DIR, 'val', 'blurry'),
            psf_dir=os.path.join(DATA_DIR, 'val', 'psfs')
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    except FileNotFoundError as e:
        print(f"Error: Dataset not found at {DATA_DIR}. {e}")
        return

    # --- Model, Loss, and Optimizer ---
    model = AttentionResUNet().to(device)
    l1_criterion = nn.L1Loss()
    lpips_loss_fn = lpips.LPIPS(net='alex', spatial=False).to(device)
    style_criterion = StyleLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    start_epoch = 0
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    if RESUME_TRAINING and os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Loading checkpoint: {MODEL_CHECKPOINT_PATH}")
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH)

        if FINE_TUNING_MODE:
            print("  -> FINE-TUNING MODE ACTIVATED")
            # Load only the model's weights, as it's the only relevant part.
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  -> Model weights loaded for fine-tuning.")

            # Reset the scoreboard and history for the new training task.
            start_epoch = 0
            best_val_loss = float('inf')
            train_losses, val_losses = [], []
            print("  -> Training history and loss targets have been reset.")
            print(f"  -> Using fresh optimizer for fine-tuning with LR = {LEARNING_RATE:.1e}")

        else: # Original resume logic for continuing the same run
            print("  -> Resuming previous training session.")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            best_val_loss = checkpoint.get('loss', float('inf'))
            print(f"Resumed from epoch {start_epoch-1}. Best val loss: {best_val_loss:.6f}")

    print(f"\nStarting High-Fidelity Replication training (Fine-Tuning Mode)...")

    # --- Enable anomaly detection to find the source of NaNs ---
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for blurry, sharp, psfs in pbar_train:
            blurry, sharp, psfs = blurry.to(device), sharp.to(device), psfs.to(device)
            
            optimizer.zero_grad()
            
            predicted_sharp = model(blurry)

            # --- Loss Stabilization ---
            # Clamp the output to the expected [-1, 1] range to prevent out-of-bounds values.
            predicted_sharp = torch.clamp(predicted_sharp, -1.0, 1.0)
            
            # 1. Pixel Fidelity
            loss_l1 = l1_criterion(predicted_sharp, sharp)

            # 2. Perceptual Fidelity
            loss_lpips = lpips_loss_fn(predicted_sharp, sharp).mean()

            # 3. Physical Fidelity
            reblurred = torch_convolve2d(predicted_sharp, psfs)
            loss_physics = l1_criterion(reblurred, blurry)

            # 4. Style Fidelity
            loss_style = style_criterion(predicted_sharp, sharp)

            # 5. Total Combined Loss
            total_loss = ((LAMBDA_L1 * loss_l1) + 
                          (LAMBDA_LPIPS * loss_lpips) + 
                          (LAMBDA_PHYSICS * loss_physics) +
                          (LAMBDA_STYLE * loss_style))
            
            total_loss.backward()
            # --- Gradient Clipping to prevent explosion ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += total_loss.item()
            pbar_train.set_postfix_str(f"loss={total_loss.item():.6f}, lr={LEARNING_RATE:.1e}")
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for i, (blurry, sharp, psfs) in enumerate(pbar_val):
                blurry, sharp, psfs = blurry.to(device), sharp.to(device), psfs.to(device)
                
                predicted_sharp = model(blurry)
                
                # --- Loss Stabilization (also needed for validation) ---
                predicted_sharp = torch.clamp(predicted_sharp, -1.0, 1.0)

                # Validation loss uses the same Quartet for consistency
                loss_l1 = l1_criterion(predicted_sharp, sharp)
                loss_lpips = lpips_loss_fn(predicted_sharp, sharp).mean()
                reblurred = torch_convolve2d(predicted_sharp, psfs)
                loss_physics = l1_criterion(reblurred, blurry)
                loss_style = style_criterion(predicted_sharp, sharp)

                total_val_loss = ((LAMBDA_L1 * loss_l1) + 
                                  (LAMBDA_LPIPS * loss_lpips) + 
                                  (LAMBDA_PHYSICS * loss_physics) +
                                  (LAMBDA_STYLE * loss_style))
                                     
                running_val_loss += total_val_loss.item()
                pbar_val.set_postfix_str(f"val_loss={total_val_loss.item():.6f}")

                if i == 0:
                    save_sample_images(epoch, blurry, sharp, predicted_sharp, SAMPLE_IMAGE_DIR)
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {LEARNING_RATE:.1e}")

        # --- Save Checkpoint ---
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"  -> New best validation loss: {best_val_loss:.6f}. Saving checkpoint.")
            save_checkpoint(epoch, model, optimizer, best_val_loss)

    # --- Final Model Save ---
    print(f"\nTraining complete. Saving final model to {FINAL_MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)

    # --- Plot and Save Loss Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")

if __name__ == '__main__':
    main()