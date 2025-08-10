import os
import warnings
import contextlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lpips
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

from flat_dataset import FlatFieldDataset
from unet_model import AttentionResUNet


# --- Stabilize lpips normalization ---
def safe_normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True) + eps)
    return in_feat / norm_factor

lpips.normalize_tensor = safe_normalize_tensor
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated*")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated*")


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_features = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*vgg_features[0:2])
        self.slice2 = nn.Sequential(*vgg_features[2:7])
        self.slice3 = nn.Sequential(*vgg_features[7:12])
        self.slice4 = nn.Sequential(*vgg_features[12:21])
        self.slice5 = nn.Sequential(*vgg_features[21:30])
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]


def gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram.div(c * h * w + 1e-8)


class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, y_pred, y_true):
        # Compute style loss in full precision to avoid AMP/half overflows
        if y_pred.is_cuda:
            autocast_ctx = torch.amp.autocast('cuda', enabled=False)
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
            y_pred_rgb = y_pred.repeat(1, 3, 1, 1).float()
            y_true_rgb = y_true.repeat(1, 3, 1, 1).float()
            y_pred_rescaled = (y_pred_rgb + 1.0) / 2.0
            y_true_rescaled = (y_true_rgb + 1.0) / 2.0
            y_pred_normalized = self.normalize(y_pred_rescaled)
            y_true_normalized = self.normalize(y_true_rescaled)
            y_pred_features = self.vgg(y_pred_normalized)
            y_true_features = self.vgg(y_true_normalized)
            loss = 0.0
            for p, t in zip(y_pred_features, y_true_features):
                gram_p = gram_matrix(p.float())
                gram_t = gram_matrix(t.float())
                loss = loss + self.criterion(gram_p, gram_t)
            return loss


def compute_lpips_fp32(lpips_fn, y_pred_norm: torch.Tensor, y_true_norm: torch.Tensor) -> torch.Tensor:
    """
    Compute LPIPS in FP32 and 3 channels to avoid AMP/conv NaNs.
    Inputs are in [-1,1], shape [B,1,H,W].
    """
    if y_pred_norm.is_cuda:
        autocast_ctx = torch.amp.autocast('cuda', enabled=False)
    else:
        autocast_ctx = contextlib.nullcontext()
    with autocast_ctx:
        yp = torch.clamp(y_pred_norm, -1.0, 1.0).repeat(1, 3, 1, 1).float()
        yt = torch.clamp(y_true_norm, -1.0, 1.0).repeat(1, 3, 1, 1).float()
        val = lpips_fn(yp, yt).mean()
        if not torch.isfinite(val):
            # Fallback to zero to keep training stable
            val = torch.zeros((), device=y_pred_norm.device, dtype=torch.float32)
        return val


# =====================
# Config
# =====================

DATA_DIR = "./randomized_flat_dataset/"
MODEL_CHECKPOINT_PATH = "./unet_flat_checkpoint.pth"
FINAL_MODEL_SAVE_PATH = "./unet_flat_model_final.pth"
SAMPLE_IMAGE_DIR = "./flat_training_samples/"
LOSS_CURVE_PATH = "./flat_loss_curve.png"

RESUME_TRAINING = True
FINE_TUNING_MODE = True
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
NUM_EPOCHS = 150
USE_AMP = True  # mixed precision for speed/stability on CUDA

# Loss weights (base)
LAMBDA_PHYSICS = 2.0     # consistency with flat-field forward model
LAMBDA_LPIPS = 1.0
LAMBDA_L1 = 0.15
LAMBDA_STYLE = 2e3

# Warmup (steps)
WARMUP_PHYS_STEPS = 500
WARMUP_LPIPS_STEPS = 1000
WARMUP_STYLE_STEPS = 5000


def save_sample_images(epoch, affected, sharp, corrected, save_dir):
    affected = (affected[0].cpu().detach() + 1) / 2
    sharp = (sharp[0].cpu().detach() + 1) / 2
    corrected = (corrected[0].cpu().detach() + 1) / 2
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(affected.squeeze(), cmap='gray')
    axs[0].set_title("Affected Input")
    axs[0].axis('off')
    axs[1].imshow(corrected.squeeze(), cmap='gray')
    axs[1].set_title("AI Corrected Output")
    axs[1].axis('off')
    axs[2].imshow(sharp.squeeze(), cmap='gray')
    axs[2].set_title("Sharp Ground Truth")
    axs[2].axis('off')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1:04d}_sample.png"))
    plt.close(fig)


def forward_flat_model(y_pred_norm: torch.Tensor, F_mul: torch.Tensor, G_add: torch.Tensor) -> torch.Tensor:
    """
    Given y_pred in [-1,1], reconstruct the expected observed image:
      y_obs = clamp(to01(y_pred) * F + G, 0, 1) -> then map to [-1,1]
    """
    y01 = torch.clamp((y_pred_norm + 1.0) / 2.0, 0.0, 1.0)
    obs01 = torch.clamp(y01 * F_mul + G_add, 0.0, 1.0)
    return obs01 * 2.0 - 1.0


def save_checkpoint(epoch, model, optimizer, best_loss, filename=MODEL_CHECKPOINT_PATH, train_losses=None, val_losses=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'train_losses': train_losses or [],
        'val_losses': val_losses or [],
    }, filename)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    os.makedirs(SAMPLE_IMAGE_DIR, exist_ok=True)

    # Data
    train_dataset = FlatFieldDataset(
        sharp_dir=os.path.join(DATA_DIR, 'train', 'sharp'),
        affected_dir=os.path.join(DATA_DIR, 'train', 'affected'),
        flat_dir=os.path.join(DATA_DIR, 'train', 'flat'),
        grad_dir=os.path.join(DATA_DIR, 'train', 'grad'),
    )
    val_dataset = FlatFieldDataset(
        sharp_dir=os.path.join(DATA_DIR, 'val', 'sharp'),
        affected_dir=os.path.join(DATA_DIR, 'val', 'affected'),
        flat_dir=os.path.join(DATA_DIR, 'val', 'flat'),
        grad_dir=os.path.join(DATA_DIR, 'val', 'grad'),
    )
    num_workers = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )

    # Model & losses
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
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
        if FINE_TUNING_MODE:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Fine-tuning: loaded model weights; resetting optimizer and history.")
            start_epoch = 0
            best_val_loss = float('inf')
            train_losses, val_losses = [], []
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            best_val_loss = checkpoint.get('loss', float('inf'))

    torch.autograd.set_detect_anomaly(True)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=(USE_AMP and device.type == 'cuda'))

    global_step = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        max_train_steps = int(os.environ.get('MAX_TRAIN_STEPS', '0'))
        step_count = 0
        for affected, sharp, flat, grad in pbar:
            affected = affected.to(device)
            sharp = sharp.to(device)
            flat = flat.to(device)
            grad = grad.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, dtype=torch.float16, enabled=(USE_AMP and device.type == 'cuda')):
                pred = model(affected)
                pred = torch.clamp(pred, -1.0, 1.0)

                # Basic reconstruction losses (pred vs sharp)
                loss_l1 = l1_criterion(pred, sharp)
            # LPIPS and Style in FP32 for stability
            loss_lpips = compute_lpips_fp32(lpips_loss_fn, pred, sharp)
            loss_style = style_criterion(pred, sharp)

            # Physics consistency: forward model must reproduce affected
            with torch.amp.autocast(device_type, dtype=torch.float16, enabled=(USE_AMP and device.type == 'cuda')):
                re_observed = forward_flat_model(pred, flat, grad)
                loss_physics = l1_criterion(re_observed, affected)

            # Warmup schedules
            wp = min(1.0, global_step / max(1, WARMUP_PHYS_STEPS))
            wl = min(1.0, global_step / max(1, WARMUP_LPIPS_STEPS))
            ws = min(1.0, global_step / max(1, WARMUP_STYLE_STEPS))

            total = (LAMBDA_L1 * loss_l1 +
                     (LAMBDA_LPIPS * wl) * loss_lpips +
                     (LAMBDA_PHYSICS * wp) * loss_physics +
                     (LAMBDA_STYLE * ws) * loss_style)

            # NaN/Inf guard
            if not torch.isfinite(total):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running += total.item()
            pbar.set_postfix_str(
                f"loss={total.item():.6f} (L1 {loss_l1.item():.4f} | LPIPS {loss_lpips.item():.4f} | PHY {loss_physics.item():.4f} | ST {loss_style.item():.4f})"
            )
            step_count += 1
            global_step += 1
            if max_train_steps and step_count >= max_train_steps:
                break

        avg_train = running / len(train_loader)
        train_losses.append(avg_train)

        # Val
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            max_val_steps = int(os.environ.get('MAX_VAL_STEPS', '0'))
            for i, (affected, sharp, flat, grad) in enumerate(pbar_val):
                affected = affected.to(device)
                sharp = sharp.to(device)
                flat = flat.to(device)
                grad = grad.to(device)

                with torch.amp.autocast(device_type, dtype=torch.float16, enabled=(USE_AMP and device.type == 'cuda')):
                    pred = model(affected)
                    pred = torch.clamp(pred, -1.0, 1.0)
                    loss_l1 = l1_criterion(pred, sharp)
                loss_lpips = compute_lpips_fp32(lpips_loss_fn, pred, sharp)
                loss_style = style_criterion(pred, sharp)
                with torch.amp.autocast(device_type, dtype=torch.float16, enabled=(USE_AMP and device.type == 'cuda')):
                    re_observed = forward_flat_model(pred, flat, grad)
                    loss_physics = l1_criterion(re_observed, affected)

                total_val = (LAMBDA_L1 * loss_l1 +
                             LAMBDA_LPIPS * loss_lpips +
                             LAMBDA_PHYSICS * loss_physics +
                             LAMBDA_STYLE * loss_style)
                running_val += total_val.item()
                pbar_val.set_postfix_str(f"val_loss={total_val.item():.6f}")

                if i == 0:
                    save_sample_images(epoch, affected, sharp, pred, SAMPLE_IMAGE_DIR)
                if max_val_steps and (i + 1) >= max_val_steps:
                    break

        avg_val = running_val / len(val_loader)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}: Train {avg_train:.6f}, Val {avg_val:.6f}")

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            print(f"  -> New best val {best_val_loss:.6f}; saving checkpoint")
            save_checkpoint(epoch, model, optimizer, best_val_loss, train_losses=train_losses, val_losses=val_losses)

    print(f"Training complete. Saving final model to {FINAL_MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)

    # Plot losses
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss (Flat Field)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")


if __name__ == '__main__':
    main()


