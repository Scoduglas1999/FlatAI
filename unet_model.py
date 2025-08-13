# unet_model.py - State-of-the-Art Attention Residual U-Net

import torch
import torch.nn as nn
import torch.nn.functional as F


def icnr_init(tensor, upscale_factor=2, a=0.0, b=1.0):
    """
    ICNR (Initialized - Checkerboard-artifact-free - N - Right) initialization.
    Fills the tensor with values that mimic a nearest-neighbor interpolation,
    which helps prevent checkerboard artifacts in upsampling layers.
    """
    o, i, h, w = tensor.shape
    assert h == w and h % 2 == 1, "Kernel must be square with odd dimensions."
    assert o % (upscale_factor ** 2) == 0, "Output channels must be divisible by upscale_factor squared."
    
    sub_kernel = torch.empty(o // (upscale_factor ** 2), i, h, w)
    nn.init.uniform_(sub_kernel, a=a, b=b) # Or any other init
    
    kernel = sub_kernel.repeat_interleave(upscale_factor ** 2, dim=0)
    
    # Corrected logic for shuffling channels
    c_out, c_in, k_h, k_w = kernel.shape
    kernel = kernel.view(c_out // (upscale_factor**2), (upscale_factor**2), c_in, k_h, k_w)
    kernel = kernel.permute(1, 0, 2, 3, 4).contiguous()
    kernel = kernel.view(c_out, c_in, k_h, k_w)
    
    tensor.data.copy_(kernel)


class ResidualBlock(nn.Module):
    """
    The fundamental building block of the Attention Res-UNet.
    It includes two convolutional layers, LeakyReLU activations, and a
    residual "shortcut" connection. A 1x1 convolution is used in the
    shortcut if the number of input/output channels is different.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2): # Add dropout_rate
        super().__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut path
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.final_activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.final_activation(self.main_path(x) + self.shortcut(x))

class AttentionGate(nn.Module):
    """
    An Attention Gate module to focus on salient features from the skip connection.
    Includes BatchNorm layers for enhanced numerical stability.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # Convolution and normalization for the gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Convolution and normalization for the skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Final convolution, normalization, and sigmoid for the attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class PixelShuffleUpsample(nn.Module):
    """
    A modern upsampling block using PixelShuffle.
    This avoids the checkerboard artifacts associated with ConvTranspose2d
    or simple bilinear upsampling followed by convolution.
    It consists of a convolution that increases channels by a factor of 4,
    followed by PixelShuffle which rearranges those channels into a 2x
    spatial upscaling.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use a 3x3 conv for more spatial context before shuffling
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activ = nn.LeakyReLU(0.2, inplace=True)
        # Initialize weights to prevent learned upsampling artifacts
        icnr_init(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        x = self.activ(x)
        return x


class AttentionResUNet(nn.Module):
    """
    A deep, wide Attention Residual U-Net.
    - Deeper & Wider: 4 down-sampling levels, ~20M parameters.
    - Residual Connections: Uses ResidualBlocks.
    - Attention Gates: On all skip connections.
    - Artifact-Free Up-Sampling: Uses Upsample + Conv (in a ResBlock).
    """
    def __init__(self):
        super().__init__()

        # --- Encoder Path ---
        self.enc1 = ResidualBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(512, 1024)

        # --- Decoder Path (with artifact-free up-sampling) ---
        # Up-sampling stage 4
        self.up4 = PixelShuffleUpsample(1024, 512)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec_combine4 = ResidualBlock(1024, 512)

        # Up-sampling stage 3
        self.up3 = PixelShuffleUpsample(512, 256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec_combine3 = ResidualBlock(512, 256)

        # Up-sampling stage 2
        self.up2 = PixelShuffleUpsample(256, 128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec_combine2 = ResidualBlock(256, 128)

        # Up-sampling stage 1
        self.up1 = PixelShuffleUpsample(128, 64)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec_combine1 = ResidualBlock(128, 64)

        # --- Output Layer ---
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Standard forward pass returning only the final linear output."""
        # --- Encoder ---
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(self.pool1(enc1_out))
        enc3_out = self.enc3(self.pool2(enc2_out))
        enc4_out = self.enc4(self.pool3(enc3_out))

        # --- Bottleneck ---
        bottleneck_out = self.bottleneck(self.pool4(enc4_out))

        # --- Decoder ---
        d4 = self.up4(bottleneck_out)
        attn4 = self.att4(g=d4, x=enc4_out)
        d4 = self.dec_combine4(torch.cat([attn4, d4], dim=1))
        
        d3 = self.up3(d4)
        attn3 = self.att3(g=d3, x=enc3_out)
        d3 = self.dec_combine3(torch.cat([attn3, d3], dim=1))
        
        d2 = self.up2(d3)
        attn2 = self.att2(g=d2, x=enc2_out)
        d2 = self.dec_combine2(torch.cat([attn2, d2], dim=1))

        d1 = self.up1(d2)
        attn1 = self.att1(g=d1, x=enc1_out)
        d1 = self.dec_combine1(torch.cat([attn1, d1], dim=1))

        # --- Output ---
        out = self.out_conv(d1)
        return out


class AttentionResUNetFG(nn.Module):
    """
    Variant that predicts multiplicative (F) and additive (G) fields instead of a clean image.
    - F is parameterized via a bounded log-scale: logF in [-log(F_max), +log(F_max)], F = exp(logF)
    - G is parameterized in [0, 1] via a sigmoid
    """
    def __init__(self, max_flat_scale: float = 3.0):
        super().__init__()

        # --- Encoder Path --- (identical to AttentionResUNet)
        # Accepts image plus coordinate channels (x,y) and a noise map → 4 channels
        self.enc1 = ResidualBlock(4, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(512, 1024)

        # --- Decoder Path ---
        self.up4 = PixelShuffleUpsample(1024, 512)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec_combine4 = ResidualBlock(1024, 512)

        self.up3 = PixelShuffleUpsample(512, 256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec_combine3 = ResidualBlock(512, 256)

        self.up2 = PixelShuffleUpsample(256, 128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec_combine2 = ResidualBlock(256, 128)

        self.up1 = PixelShuffleUpsample(128, 64)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec_combine1 = ResidualBlock(128, 64)

        # Heads: predict 3 channels (logF, G, M)
        self.out_conv_fg = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # Store max log-scale for multiplicative field
        import math
        self.register_buffer('max_log_F', torch.tensor(float(math.log(max_flat_scale))), persistent=False)

    def forward(self, x):
        # Accepts a [B,4,H,W] tensor with img, xx, yy, sigma
        if x.dim() != 4 or x.shape[1] != 4:
            raise ValueError("Input must be [B,4,H,W] with img, xx, yy, sigma")

        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(self.pool1(enc1_out))
        enc3_out = self.enc3(self.pool2(enc2_out))
        enc4_out = self.enc4(self.pool3(enc3_out))

        # Bottleneck
        bottleneck_out = self.bottleneck(self.pool4(enc4_out))

        # Decoder
        d4 = self.up4(bottleneck_out)
        attn4 = self.att4(g=d4, x=enc4_out)
        d4 = self.dec_combine4(torch.cat([attn4, d4], dim=1))

        d3 = self.up3(d4)
        attn3 = self.att3(g=d3, x=enc3_out)
        d3 = self.dec_combine3(torch.cat([attn3, d3], dim=1))

        d2 = self.up2(d3)
        attn2 = self.att2(g=d2, x=enc2_out)
        d2 = self.dec_combine2(torch.cat([attn2, d2], dim=1))
        
        d1 = self.up1(d2)
        attn1 = self.att1(g=d1, x=enc1_out)
        d1 = self.dec_combine1(torch.cat([attn1, d1], dim=1))

        out = self.out_conv_fg(d1)
        logF_raw = self.tanh(out[:, 0:1, :, :]) * self.max_log_F  # [-max_log_F, +max_log_F]
        F_mul = torch.exp(logF_raw)  # (0, max_flat_scale]
        G = self.sigmoid(out[:, 1:2, :, :])  # [0,1]
        M = self.sigmoid(out[:, 2:3, :, :])  # [0,1], confidence/strength
        return F_mul, G, M

# Optional test block to verify the architecture and parameter count
if __name__ == '__main__':
    # You need the torch-summary library for this: pip install torch-summary
    try:
        from torchsummary import summary
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AttentionResUNet().to(device)
        # Input size: (Channel, Height, Width)
        summary(model, input_size=(1, 256, 256))
    except ImportError:
        # Fallback if torchsummary is not installed
        model = AttentionResUNet()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Attention-Res-UNet Model Summary:")
        print(f"Total trainable parameters: {total_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 256, 256)
        output = model(dummy_input)
        print(f"Input shape:  {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("✓ Model test passed!") 