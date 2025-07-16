# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for bottleneck.
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            Swish(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    """
    Residual block with optional SE attention and GroupNorm.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_se (bool): Whether to use SE attention block.
    """
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        h = self.conv(x)
        h = self.se(h)
        return h + self.skip(x)

class DenoiseUNet(nn.Module):
    """
    U-Net architecture with Swish, GroupNorm, SE attention, and timestep embedding for diffusion denoising.
    Args:
        in_channels (int): Number of input channels.
        base_channels (int): Number of base feature maps.
        time_embed_dim (int): Dimension of timestep embedding.
    """
    def __init__(self, in_channels, base_channels=64, time_embed_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            Swish(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        self.enc1 = ResidualBlock(in_channels, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)

        self.middle = ResidualBlock(base_channels * 4, base_channels * 4)

        self.dec3 = ResidualBlock(base_channels * 4, base_channels * 2)
        self.dec2 = ResidualBlock(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, t_embed):
        t = self.time_embed(t_embed)
        h1 = self.enc1(x)
        h2 = self.enc2(self.down(h1))
        h3 = self.enc3(self.down(h2))

        mid = self.middle(h3 + t[:, None, None, None])

        d2 = self.dec3(self.up(mid) + h2)
        d1 = self.dec2(self.up(d2) + h1)
        return self.out_conv(d1)
