"""Candidate density neural network architectures for shootout and distillation."""

import torch
import torch.nn as nn


INPUT_CH = 64  # Noise features
OUTPUT_CH = 1  # Density output


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VoxelMLP(nn.Module):
    """Per-voxel MLP: fastest lower bound, no spatial mixing."""

    def __init__(self, in_ch: int, out_ch: int, width: int = 96, depth: int = 6):
        super().__init__()
        layers = [nn.Conv3d(in_ch, width, 1), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv3d(width, width, 1), nn.GELU()]
        layers += [nn.Conv3d(width, out_ch, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SepResBlock(nn.Module):
    """Depthwise-separable residual block."""

    def __init__(self, ch: int, expansion: int = 2):
        super().__init__()
        hidden = ch * expansion
        self.net = nn.Sequential(
            nn.Conv3d(ch, hidden, 1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm3d(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, ch, 1, bias=False),
            nn.BatchNorm3d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class SeparableResNet3D(nn.Module):
    """Depthwise-separable residual 3D net: strong runtime/accuracy tradeoff."""

    def __init__(self, in_ch: int, out_ch: int, width: int = 32, blocks: int = 6):
        super().__init__()
        body = [nn.Conv3d(in_ch, width, 1), nn.GELU()]
        for _ in range(blocks):
            body.append(SepResBlock(width))
        body.append(nn.Conv3d(width, out_ch, 1))
        self.net = nn.Sequential(*body)

    def forward(self, x):
        return self.net(x)


class AxialResBlock(nn.Module):
    """Axial residual block: separable convolutions along each axis."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(ch),
            nn.GELU(),
            nn.Conv3d(ch, ch, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False),
            nn.BatchNorm3d(ch),
            nn.GELU(),
            nn.Conv3d(ch, ch, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class AxialMixer3D(nn.Module):
    """Axial residual mixer: cheap receptive field expansion."""

    def __init__(self, in_ch: int, out_ch: int, width: int = 32, blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_ch, width, 1), nn.GELU())
        self.blocks = nn.Sequential(*[AxialResBlock(width) for _ in range(blocks)])
        self.head = nn.Conv3d(width, out_ch, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class DoubleConv(nn.Module):
    """Double convolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet3D(nn.Module):
    """Tiny anisotropic U-Net: accuracy-first teacher."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 24):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.down1 = nn.MaxPool3d((2, 2, 2))
        self.enc2 = DoubleConv(base, base * 2)
        self.down2 = nn.MaxPool3d((1, 2, 1))
        self.bottleneck = DoubleConv(base * 2, base * 4)
        self.up2 = nn.Upsample(scale_factor=(1, 2, 1), mode="trilinear", align_corners=False)
        self.dec2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.up1 = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=False)
        self.dec1 = DoubleConv(base * 2 + base, base)
        self.head = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        xb = self.bottleneck(self.down2(x2))
        y2 = self.up2(xb)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        return self.head(y1)


# Candidate architectures dictionary for shootout and distillation
CANDIDATES = {
    "mlp": {
        "label": "VoxelMLP(96x6)",
        "build": lambda: VoxelMLP(INPUT_CH, OUTPUT_CH, width=96, depth=6),
        "note": "Fastest lower bound; no local neighborhood mixing",
    },
    "sep": {
        "label": "SeparableResNet3D(width=32, blocks=6)",
        "build": lambda: SeparableResNet3D(INPUT_CH, OUTPUT_CH, width=32, blocks=6),
        "note": "Likely Pareto sweet spot",
    },
    "axial": {
        "label": "AxialMixer3D(width=32, blocks=6)",
        "build": lambda: AxialMixer3D(INPUT_CH, OUTPUT_CH, width=32, blocks=6),
        "note": "Cheap receptive field expansion",
    },
    "unet": {
        "label": "TinyUNet3D(base=24)",
        "build": lambda: TinyUNet3D(INPUT_CH, OUTPUT_CH, base=24),
        "note": "Accuracy-first teacher / upper bound",
    },
}
