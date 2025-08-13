import os
import warnings
import contextlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lpips
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

from flat_dataset import FlatFieldDataset
from unet_model import AttentionResUNet, AttentionResUNetFG


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

RESUME_TRAINING = False
FINE_TUNING_MODE = False
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
NUM_EPOCHS = 150
USE_AMP = True  # mixed precision for speed/stability on CUDA

# Loss weights (base)
LAMBDA_PHYSICS = 2.0     # consistency with flat-field forward model
LAMBDA_LPIPS = 1.5
LAMBDA_L1 = 1.0
LAMBDA_STYLE = 0.0 # Disabled

# Warmup (steps)
WARMUP_PHYS_STEPS = 500
WARMUP_LPIPS_STEPS = 1000
WARMUP_STYLE_STEPS = 0 # Disabled

# Scheduler settings
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-7


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


def save_checkpoint(epoch, model, optimizer, scheduler, best_loss, filename=MODEL_CHECKPOINT_PATH, train_losses=None, val_losses=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
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
    # Switch to FG model that predicts multiplicative (F) and additive (G) fields
    model = AttentionResUNetFG().to(device)
    l1_criterion = nn.L1Loss()
    lpips_loss_fn = lpips.LPIPS(net='alex', spatial=False).to(device)
    style_criterion = StyleLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    scheduler = ReduceLROnPlateau(
        optimizer, 'min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
        min_lr=SCHEDULER_MIN_LR
    )

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
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            best_val_loss = checkpoint.get('loss', float('inf'))

    torch.autograd.set_detect_anomaly(False)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=(USE_AMP and device.type == 'cuda'))

    global_step = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train
        model.train()
        running = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train] LR: {current_lr:.2e}")
        max_train_steps = int(os.environ.get('MAX_TRAIN_STEPS', '0'))
        step_count = 0
        for affected, sharp, flat, grad in pbar:
            affected = affected.to(device)
            sharp = sharp.to(device)
            flat = flat.to(device)
            grad = grad.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, dtype=torch.float16, enabled=(USE_AMP and device.type == 'cuda')):
                # Build absolute coordinate channels for global context
                b, _, h, w = affected.shape
                # Approximate local noise sigma from high-frequency energy of the normalized input
                with torch.no_grad():
                    y_obs_for_noise = torch.clamp((affected + 1.0) / 2.0, 0.0, 1.0)
                    mu = F.avg_pool2d(y_obs_for_noise, kernel_size=3, stride=1, padding=1)
                    hf = y_obs_for_noise - mu
                    var = F.avg_pool2d(hf * hf, kernel_size=3, stride=1, padding=1)
                    sigma = torch.sqrt(torch.clamp(var, 1e-6))
                yy = torch.linspace(0.0, 1.0, steps=h, device=device).view(1, 1, h, 1).expand(b, -1, -1, w)
                xx = torch.linspace(0.0, 1.0, steps=w, device=device).view(1, 1, 1, w).expand(b, -1, h, -1)
                affected_in = torch.cat([affected, xx, yy, sigma], dim=1)
                F_pred, G_pred, M_pred = model(affected_in)
                # Reconstruct clean target using forward model inversion
                # Inputs/targets in [-1,1] => map to [0,1] first
                y_obs = torch.clamp((affected + 1.0) / 2.0, 0.0, 1.0)
                y_true = torch.clamp((sharp + 1.0) / 2.0, 0.0, 1.0)
                # Inverse model: y_clean = clamp((y_obs - G) / F, 0, 1)
                y_clean_ungated = torch.clamp((y_obs - G_pred) / torch.clamp(F_pred, 1e-3, None), 0.0, 1.0)
                # Confidence-gated blend with observed to prevent over-correction
                y_clean = torch.clamp(y_obs + M_pred * (y_clean_ungated - y_obs), 0.0, 1.0)
                pred = y_clean * 2.0 - 1.0

                # Basic reconstruction losses (pred vs sharp)
                loss_l1 = l1_criterion(pred, sharp)
            # LPIPS and Style in FP32 for stability
            loss_lpips = compute_lpips_fp32(lpips_loss_fn, pred, sharp)
            loss_style = torch.tensor(0.0, device=device)

            # Physics consistency: forward model must reproduce affected
            with torch.amp.autocast(device_type, dtype=torch.float16, enabled=(USE_AMP and device.type == 'cuda')):
                # Physics consistency: recompose observed with predicted F,G
                y_clean01 = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
                re_obs01 = torch.clamp(y_clean01 * F_pred + G_pred, 0.0, 1.0)
                re_observed = re_obs01 * 2.0 - 1.0
                loss_physics = l1_criterion(re_observed, affected)

            # Warmup schedules
            wp = min(1.0, global_step / max(1, WARMUP_PHYS_STEPS))
            wl = min(1.0, global_step / max(1, WARMUP_LPIPS_STEPS))
            ws = min(1.0, global_step / max(1, WARMUP_STYLE_STEPS))

            # Identity loss outside artifacts: penalize changes where affected≈sharp
            with torch.no_grad():
                delta = torch.abs(affected - sharp)
                ident_mask = (delta < 0.05).float()
                ident_mask = torch.nn.functional.avg_pool2d(ident_mask, kernel_size=3, stride=1, padding=1)
            # Penalize changes where mask says clean; also regularize M to be low there
            loss_identity = (torch.abs(pred - affected) * ident_mask).mean()
            loss_conf_clean = (M_pred * ident_mask).mean()

            # Gentle priors: mean(F) ≈ 1, smooth F and G
            loss_F_mean = (F_pred.mean(dim=[2,3]) - 1.0).abs().mean()
            def _smoothness(t):
                dy = (t[:,:,1:,:] - t[:,:,:-1,:]).abs()
                dx = (t[:,:,:,1:] - t[:,:,:,:-1]).abs()
                return (dx.mean() + dy.mean())
            loss_smooth = _smoothness(F_pred) + 0.5 * _smoothness(G_pred)

            total = (LAMBDA_L1 * loss_l1 +
                     (LAMBDA_LPIPS * wl) * loss_lpips +
                     (LAMBDA_PHYSICS * wp) * loss_physics +
                     (LAMBDA_STYLE * ws) * loss_style +
                     0.5 * loss_identity +  # Reduced from 5.0
                     0.2 * loss_conf_clean +  # Reduced from 2.0
                     0.05 * loss_F_mean +
                     0.1 * loss_smooth)

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
                    b, _, h, w = affected.shape
                    with torch.no_grad():
                        y_obs_for_noise = torch.clamp((affected + 1.0) / 2.0, 0.0, 1.0)
                        mu = F.avg_pool2d(y_obs_for_noise, kernel_size=3, stride=1, padding=1)
                        hf = y_obs_for_noise - mu
                        var = F.avg_pool2d(hf * hf, kernel_size=3, stride=1, padding=1)
                        sigma = torch.sqrt(torch.clamp(var, 1e-6))
                    yy = torch.linspace(0.0, 1.0, steps=h, device=device).view(1, 1, h, 1).expand(b, -1, -1, w)
                    xx = torch.linspace(0.0, 1.0, steps=w, device=device).view(1, 1, 1, w).expand(b, -1, h, -1)
                    affected_in = torch.cat([affected, xx, yy, sigma], dim=1)
                    F_pred, G_pred, M_pred = model(affected_in)
                    y_obs = torch.clamp((affected + 1.0) / 2.0, 0.0, 1.0)
                    y_true = torch.clamp((sharp + 1.0) / 2.0, 0.0, 1.0)
                    y_clean_ungated = torch.clamp((y_obs - G_pred) / torch.clamp(F_pred, 1e-3, None), 0.0, 1.0)
                    y_clean = torch.clamp(y_obs + M_pred * (y_clean_ungated - y_obs), 0.0, 1.0)
                    pred = y_clean * 2.0 - 1.0
                    loss_l1 = l1_criterion(pred, sharp)
                loss_lpips = compute_lpips_fp32(lpips_loss_fn, pred, sharp)
                loss_style = torch.tensor(0.0, device=device)
                with torch.amp.autocast(device_type, dtype=torch.float16, enabled=(USE_AMP and device.type == 'cuda')):
                    y_clean01 = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
                    re_obs01 = torch.clamp(y_clean01 * F_pred + G_pred, 0.0, 1.0)
                    re_observed = re_obs01 * 2.0 - 1.0
                    loss_physics = l1_criterion(re_observed, affected)

                with torch.no_grad():
                    mu = torch.nn.functional.avg_pool2d(y_obs, 3, 1, 1)
                    hf = y_obs - mu
                    var = torch.nn.functional.avg_pool2d(hf * hf, 3, 1, 1) + 1e-6
                    inv_var = 1.0 / var
                    delta = torch.abs(affected - sharp)
                    ident_mask = (delta < 0.05).float()
                    ident_mask = torch.nn.functional.avg_pool2d(ident_mask, kernel_size=3, stride=1, padding=1)
                loss_identity = (torch.abs(pred - affected) * ident_mask).mean()
                loss_conf_clean = (M_pred * ident_mask).mean()

                loss_F_mean = (F_pred.mean(dim=[2,3]) - 1.0).abs().mean()
                loss_smooth = _smoothness(F_pred) + 0.5 * _smoothness(G_pred)

                total_val = (LAMBDA_L1 * loss_l1 +
                             LAMBDA_LPIPS * loss_lpips +
                             LAMBDA_PHYSICS * loss_physics +
                             LAMBDA_STYLE * loss_style +
                             0.5 * loss_identity + # Reduced from 5.0
                             0.2 * loss_conf_clean + # Reduced from 2.0
                             0.05 * loss_F_mean +
                             0.1 * loss_smooth)
                running_val += total_val.item()
                pbar_val.set_postfix_str(f"val_loss={total_val.item():.6f}")

                if i == 0:
                    save_sample_images(epoch, affected, sharp, pred, SAMPLE_IMAGE_DIR)
                if max_val_steps and (i + 1) >= max_val_steps:
                    break

        avg_val = running_val / len(val_loader)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}: Train {avg_train:.6f}, Val {avg_val:.6f}")

        # Update LR
        scheduler.step(avg_val)

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            print(f"  -> New best val {best_val_loss:.6f}; saving checkpoint")
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, train_losses=train_losses, val_losses=val_losses)

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


