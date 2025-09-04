"""
Flexible 3D U-Net Model Architecture for Multi-LOD Voxel Super-Resolution

This module implements a flexible 3D U-Net architecture that can handle
multiple LOD transitions in a single model:
- LOD4→LOD3: 1³ → 2³
- LOD3→LOD2: 2³ → 4³
- LOD2→LOD1: 4³ → 8³
- LOD1→LOD0: 8³ → 16³

The model uses LOD embeddings to adapt its processing to the specific
scale transition being performed.
"""

import math
from dataclasses import dataclass
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings for LOD conditioning.
    """
    assert len(timesteps.shape) == 1  # Should be 1D
    assert embedding_dim >= 2, "embedding_dim must be at least 2 to avoid division by zero."

    half_dim = embedding_dim // 2
    log_base = math.log(10000) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -log_base
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    if embedding_dim % 2 == 1:  # Zero pad for odd dimensions
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

    return emb


class AdaptiveConv3D(nn.Module):
    """
    3D convolution that adapts its processing based on input size and LOD.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, lod_level=None):
        # For now, standard convolution - could add LOD-specific processing here
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class FlexibleDoubleConv3D(nn.Module):
    """
    Double convolution block that adapts to different spatial scales.
    """

    def __init__(self, in_channels, out_channels, lod_embed_dim=64):
        super().__init__()
        self.conv1 = AdaptiveConv3D(in_channels, out_channels)
        self.conv2 = AdaptiveConv3D(out_channels, out_channels)

        # LOD conditioning
        self.lod_proj = nn.Linear(lod_embed_dim, out_channels * 2)  # scale and bias

    def forward(self, x, lod_embedding):
        x = self.conv1(x)

        # Apply LOD conditioning via FiLM
        if lod_embedding is not None:
            scale_bias = self.lod_proj(lod_embedding)  # (B, 2*C)
            scale, bias = scale_bias.chunk(2, dim=1)  # (B, C) each
            scale = scale.view(-1, x.size(1), 1, 1, 1)
            bias = bias.view(-1, x.size(1), 1, 1, 1)
            x = x * (1 + scale) + bias

        x = self.conv2(x)
        return x


class FlexibleUpBlock(nn.Module):
    """
    Upsampling block that adapts scale factor based on target size.
    """

    def __init__(self, in_channels, out_channels, lod_embed_dim=64):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = FlexibleDoubleConv3D(in_channels, out_channels, lod_embed_dim)

    def forward(self, x, skip, lod_embedding, target_size=None):
        # Adaptive upsampling
        if target_size is not None:
            current_size = x.shape[-1]
            scale_factor = target_size / current_size
            if scale_factor != 2.0:
                # Use interpolation for non-2x upsampling
                x = F.interpolate(
                    x,
                    size=(target_size, target_size, target_size),
                    mode="trilinear",
                    align_corners=False,
                )
            else:
                x = self.upconv(x)
        else:
            x = self.upconv(x)

        # Handle skip connection size mismatch
        if skip is not None:
            if x.shape != skip.shape:
                skip = F.interpolate(skip, size=x.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x, lod_embedding)
        return x


class FlexibleDownBlock(nn.Module):
    """
    Downsampling block that preserves features across scales.
    """

    def __init__(self, in_channels, out_channels, lod_embed_dim=64):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = FlexibleDoubleConv3D(in_channels, out_channels, lod_embed_dim)

    def forward(self, x, lod_embedding):
        x = self.pool(x)
        x = self.conv(x, lod_embedding)
        return x


@dataclass
class FlexibleUNet3DConfig:
    """Configuration for flexible multi-LOD UNet."""

    # Model architecture
    base_channels: int = 32
    depth: int = 3
    max_channels: int = 256

    # Input conditioning
    biome_vocab_size: int = 256
    biome_embed_dim: int = 32
    heightmap_channels: int = 1
    river_channels: int = 1
    y_embed_dim: int = 16
    lod_embed_dim: int = 64

    # Output heads
    block_vocab_size: int = 1104

    # Training
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    activation: Literal["relu", "gelu", "swish"] = "relu"


class FlexibleConditioningFusion(nn.Module):
    """
    Fuses biome, heightmap, and other conditioning information.
    Adapts to different spatial scales.
    """

    def __init__(self, config: FlexibleUNet3DConfig):
        super().__init__()

        # Biome embedding
        self.biome_embedding = nn.Embedding(config.biome_vocab_size, config.biome_embed_dim)

        # Y-level embedding
        self.y_embedding = nn.Embedding(384, config.y_embed_dim)  # 384 Y levels

        # Spatial feature processing
        spatial_in_channels = (
            config.biome_embed_dim
            + config.heightmap_channels
            + config.river_channels
            + config.y_embed_dim
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(spatial_in_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.base_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, biome_patch, heightmap_patch, river_patch, y_index, lod):
        # Process biomes - handle both one-hot and index formats
        if biome_patch.dim() == 5:  # One-hot format (B, NB, 16, 16, 1)
            biome_indices = biome_patch.argmax(dim=1).squeeze(-1)  # (B, 16, 16)
        else:  # Index format (B, 16, 16)
            biome_indices = biome_patch

        biome_emb = self.biome_embedding(biome_indices)  # (B, 16, 16, embed_dim)
        biome_emb = biome_emb.permute(0, 3, 1, 2)  # (B, embed_dim, 16, 16)

        # Y-level embedding (broadcast to spatial dims)
        y_emb = self.y_embedding(y_index)  # (B, y_embed_dim)
        y_emb = y_emb.unsqueeze(-1).unsqueeze(-1)  # (B, y_embed_dim, 1, 1)
        y_emb = y_emb.expand(-1, -1, 16, 16)  # (B, y_embed_dim, 16, 16)

        # Squeeze spatial dimensions from height/river if needed
        if heightmap_patch.dim() == 5:
            heightmap_patch = heightmap_patch.squeeze(-1)  # (B, 1, 16, 16)
        if river_patch.dim() == 5:
            river_patch = river_patch.squeeze(-1)  # (B, 1, 16, 16)

        # Concatenate all spatial features
        spatial_features = torch.cat([biome_emb, heightmap_patch, river_patch, y_emb], dim=1)

        # Process through spatial convolutions
        conditioning = self.spatial_conv(spatial_features)

        return conditioning


class FlexibleVoxelUNet3D(nn.Module):
    """
    Flexible 3D U-Net that handles multiple LOD transitions in a single model.

    Supports:
    - LOD4→LOD3: 1³ → 2³
    - LOD3→LOD2: 2³ → 4³
    - LOD2→LOD1: 4³ → 8³
    - LOD1→LOD0: 8³ → 16³
    """

    def __init__(self, config: FlexibleUNet3DConfig):
        super().__init__()

        self.config = config

        # LOD embedding with enhanced processing
        self.lod_embed_dim = config.lod_embed_dim
        self.lod_projection = nn.Sequential(
            nn.Linear(config.lod_embed_dim, config.lod_embed_dim),
            nn.ReLU(),
            nn.Linear(config.lod_embed_dim, config.lod_embed_dim),
        )

        # Conditioning fusion
        self.conditioning_fusion = FlexibleConditioningFusion(config)

        # Initial convolution - handles variable input sizes
        self.initial_conv = FlexibleDoubleConv3D(1, config.base_channels, config.lod_embed_dim)

        # Encoder path
        self.encoder = nn.ModuleList()
        in_channels = config.base_channels
        for i in range(config.depth):
            out_channels = min(config.base_channels * (2 ** (i + 1)), config.max_channels)
            self.encoder.append(FlexibleDownBlock(in_channels, out_channels, config.lod_embed_dim))
            in_channels = out_channels

        # Bottleneck
        bottleneck_channels = min(config.base_channels * (2**config.depth), config.max_channels)
        self.bottleneck = FlexibleDoubleConv3D(
            in_channels, bottleneck_channels, config.lod_embed_dim
        )

        # Decoder path
        self.decoder = nn.ModuleList()
        in_channels = bottleneck_channels
        for i in range(config.depth):
            out_channels = min(
                config.base_channels * (2 ** (config.depth - i - 1)), config.max_channels
            )
            self.decoder.append(FlexibleUpBlock(in_channels, out_channels, config.lod_embed_dim))
            in_channels = out_channels

        # Final processing with conditioning
        self.final_conv = nn.Conv3d(
            config.base_channels + config.base_channels,  # features + conditioning
            config.base_channels,
            3,
            padding=1,
        )

        # Output heads
        self.air_mask_head = nn.Conv3d(config.base_channels, 1, 1)
        self.block_type_head = nn.Conv3d(config.base_channels, config.block_vocab_size, 1)

    def forward(
        self,
        parent_voxel: torch.Tensor,
        biome_patch: torch.Tensor,
        heightmap_patch: torch.Tensor,
        river_patch: torch.Tensor,
        y_index: torch.Tensor,
        lod: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with flexible LOD processing.

        Args:
            parent_voxel: (B, 1, S, S, S) where S can be 1, 2, 4, or 8
            biome_patch: (B, NB, 16, 16, 1) or (B, 16, 16)
            heightmap_patch: (B, 1, 16, 16, 1) or (B, 1, 16, 16)
            river_patch: (B, 1, 16, 16, 1) or (B, 1, 16, 16)
            y_index: (B,) Y-level indices
            lod: (B,) LOD level (4, 3, 2, 1)

        Returns:
            Dictionary with 'air_mask_logits' and 'block_type_logits'
        """
        # Determine target size based on input size
        input_size = parent_voxel.shape[-1]
        target_size = input_size * 2

        # Create enhanced LOD embeddings
        lod_sinusoidal = get_timestep_embedding(lod, self.lod_embed_dim)
        lod_enhanced = self.lod_projection(lod_sinusoidal)

        # Get spatial conditioning (always at 16x16 resolution)
        conditioning = self.conditioning_fusion(
            biome_patch, heightmap_patch, river_patch, y_index, lod
        )

        # Pad input to minimum processing size if needed
        min_size = 4  # Minimum size for U-Net processing
        if input_size < min_size:
            pad_size = min_size - input_size
            parent_voxel = F.pad(parent_voxel, (0, pad_size, 0, pad_size, 0, pad_size))
            processing_size = min_size
        else:
            processing_size = input_size

        # Initial convolution
        x = self.initial_conv(parent_voxel, lod_enhanced)

        # Encoder path with skip connections
        skip_connections = [x]
        for encoder_block in self.encoder:
            x = encoder_block(x, lod_enhanced)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x, lod_enhanced)

        # Decoder path
        skip_connections = skip_connections[:-1]  # Remove bottleneck input
        for i, decoder_block in enumerate(self.decoder):
            skip = skip_connections[-(i + 1)] if i < len(skip_connections) else None

            # Calculate target size for this decoder level
            decoder_target_size = processing_size * (2 ** (i + 1))
            if decoder_target_size > target_size:
                decoder_target_size = target_size

            x = decoder_block(x, skip, lod_enhanced, decoder_target_size)

        # Final upsampling to exact target size
        if x.shape[-1] != target_size:
            x = F.interpolate(
                x,
                size=(target_size, target_size, target_size),
                mode="trilinear",
                align_corners=False,
            )  # Expand conditioning to 3D and match target size
        batch_size, cond_channels, _, _ = conditioning.shape
        if conditioning.shape[2] != target_size:
            conditioning = F.interpolate(
                conditioning, size=(target_size, target_size), mode="bilinear", align_corners=False
            )

        conditioning_3d = conditioning.unsqueeze(2).expand(-1, -1, target_size, -1, -1)

        # Combine features with conditioning
        combined = torch.cat([x, conditioning_3d], dim=1)
        features = self.final_conv(combined)

        # Generate outputs
        air_mask_logits = self.air_mask_head(features)
        block_type_logits = self.block_type_head(features)

        return {"air_mask_logits": air_mask_logits, "block_type_logits": block_type_logits}
