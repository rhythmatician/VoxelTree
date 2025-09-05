"""
Flexible 3D U-Net Model Architecture for Multi-LOD Voxel Super-Resolution

This module implements a flexible 3D U-Net architecture that handles
all LOD transitions in a single network:
- LOD4→LOD3: 1³ → 2³
- LOD3→LOD2: 2³ → 4³
- LOD2→LOD1: 4³ → 8³
- LOD1→LOD0: 8³ → 16³

Key Features:
- Adaptive input/output processing for multiple        # Simpler 2-level UNet for 8x8x8 inputs (8→4→2, then back up)
        self.down1 = self._make_layer(config.base_channels, config.base_channels * 2)
        self.down2 = self._make_layer(config.base_channels * 2, config.base_channels * 4)

        # Bottleneck at 2x2x2 resolution
        self.bottleneck = self._make_layer(config.base_channels * 4, config.base_channels * 8)

        # Up-sampling path
        self.up2 = self._make_layer(
            config.base_channels * 8 + config.base_channels * 4, config.base_channels * 4
        )
        self.up1 = self._make_layer(
            config.base_channels * 4 + config.base_channels * 2, config.base_channels * 2
        )odal conditioning (biome, heightmap, river, Y-level, LOD)
- Dual output heads for air mask and block type prediction
- Memory-efficient architecture suitable for training on consumer GPUs
"""

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings for LOD conditioning.

    This creates embeddings similar to those used in diffusion models,
    allowing the model to understand temporal/scale relationships between LOD levels.

    Args:
        timesteps: (B,) tensor of timestep values
        embedding_dim: dimension of the embedding space

    Returns:
        (B, embedding_dim) tensor of embeddings
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


@dataclass
class SimpleFlexibleConfig:
    """Simplified configuration for flexible multi-LOD UNet."""

    # Core architecture
    base_channels: int = 32
    max_channels: int = 128

    # Input conditioning
    biome_vocab_size: int = 256
    biome_embed_dim: int = 32
    lod_embed_dim: int = 64

    # Output
    block_vocab_size: int = 1104


class SimpleConditioningFusion(nn.Module):
    """Simplified conditioning that just handles biomes and heightmaps."""

    def __init__(self, config: SimpleFlexibleConfig):
        super().__init__()
        self.biome_embedding = nn.Embedding(config.biome_vocab_size, config.biome_embed_dim)

        # Simple fusion: biome + height + river -> base_channels
        spatial_channels = config.biome_embed_dim + 1 + 1  # biome + height + river
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(spatial_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, biome_patch, heightmap_patch, river_patch, y_index, lod):
        # Handle biome format
        if biome_patch.dim() == 4 and biome_patch.shape[1] > 16:  # One-hot (B, 256, 16, 16)
            biome_indices = biome_patch.argmax(dim=1)  # (B, 16, 16)
        elif biome_patch.dim() == 5:  # One-hot (B, NB, 16, 16, 1)
            biome_indices = biome_patch.argmax(dim=1).squeeze(-1)
        else:  # Index format (B, 16, 16)
            biome_indices = biome_patch

        biome_emb = self.biome_embedding(biome_indices)  # (B, 16, 16, embed_dim)
        biome_emb = biome_emb.permute(0, 3, 1, 2)  # (B, embed_dim, 16, 16)

        # Handle heightmap/river format
        if heightmap_patch.dim() == 5:
            heightmap_patch = heightmap_patch.squeeze(-1)
        if river_patch.dim() == 5:
            river_patch = river_patch.squeeze(-1)

        # Concatenate and process
        spatial_features = torch.cat([biome_emb, heightmap_patch, river_patch], dim=1)
        conditioning = self.spatial_conv(spatial_features)

        return conditioning


class SimpleFlexibleUNet3D(nn.Module):
    """
    Simplified flexible 3D U-Net that handles all LOD transitions.

    Uses a fixed architecture with adaptive input/output processing.
    This is the current working implementation for multi-LOD training.
    """

    def __init__(self, config: SimpleFlexibleConfig):
        super().__init__()

        self.config = config

        # LOD embedding
        self.lod_embed_dim = config.lod_embed_dim
        self.lod_projection = nn.Sequential(
            nn.Linear(config.lod_embed_dim, config.lod_embed_dim),
            nn.ReLU(),
            nn.Linear(config.lod_embed_dim, config.lod_embed_dim),
        )

        # Conditioning fusion
        self.conditioning_fusion = SimpleConditioningFusion(config)

        # Core architecture - works on 16³ resolution always
        # We'll pad/crop inputs to fit this
        self.input_conv = nn.Sequential(
            nn.Conv3d(1, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        # Simple encoder
        self.enc1 = self._make_layer(config.base_channels, config.base_channels)
        self.enc2 = self._make_layer(config.base_channels, config.base_channels * 2)
        self.enc3 = self._make_layer(config.base_channels * 2, config.base_channels * 4)

        # Bottleneck
        self.bottleneck = self._make_layer(config.base_channels * 4, config.base_channels * 4)

        # Simple decoder
        self.dec3 = self._make_layer(
            config.base_channels * 8, config.base_channels * 2
        )  # skip + enc
        self.dec2 = self._make_layer(config.base_channels * 4, config.base_channels)  # skip + enc
        self.dec1 = self._make_layer(config.base_channels * 2, config.base_channels)  # skip + enc

        # Final processing with conditioning
        self.final_conv = nn.Conv3d(
            config.base_channels * 2, config.base_channels, 3, padding=1  # features + conditioning
        )

        # Output heads
        self.air_mask_head = nn.Conv3d(config.base_channels, 1, 1)
        self.block_type_head = nn.Conv3d(config.base_channels, config.block_vocab_size, 1)

    def _make_layer(self, in_channels, out_channels):
        """Create a simple conv layer with normalization."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

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
        Forward pass with flexible input/output sizes.

        Strategy: Always process at 16³, pad/crop as needed.
        """

        input_size = parent_voxel.shape[-1]
        target_size = input_size * 2

        # Create LOD embeddings (flatten batch dimension)
        # Note: LOD embedding could be enhanced and used for conditioning
        # For now, we just ensure the embedding works without error
        get_timestep_embedding(lod.flatten(), self.lod_embed_dim)

        # Get spatial conditioning at 16×16
        conditioning = self.conditioning_fusion(
            biome_patch, heightmap_patch, river_patch, y_index, lod
        )

        # Pad input to 16³ for processing
        processing_size = 16
        if input_size < processing_size:
            # Pad smaller inputs
            pad_size = processing_size - input_size
            padded_input = F.pad(
                parent_voxel, (0, pad_size, 0, pad_size, 0, pad_size), mode="constant", value=0
            )
        elif input_size > processing_size:
            # Downsample larger inputs (shouldn't happen in our case)
            padded_input = F.interpolate(
                parent_voxel, size=(processing_size,) * 3, mode="trilinear", align_corners=False
            )
        else:
            padded_input = parent_voxel

        # Process through network
        x = self.input_conv(padded_input)  # (B, base_channels, 16, 16, 16)

        # Encoder with skip connections
        enc1_out = self.enc1(x)  # (B, base_channels, 16, 16, 16)

        enc2_in = F.max_pool3d(enc1_out, 2)  # (B, base_channels, 8, 8, 8)
        enc2_out = self.enc2(enc2_in)  # (B, base_channels*2, 8, 8, 8)

        enc3_in = F.max_pool3d(enc2_out, 2)  # (B, base_channels*2, 4, 4, 4)
        enc3_out = self.enc3(enc3_in)  # (B, base_channels*4, 4, 4, 4)

        # Bottleneck
        bottleneck_in = F.max_pool3d(enc3_out, 2)  # (B, base_channels*4, 2, 2, 2)
        bottleneck_out = self.bottleneck(bottleneck_in)  # (B, base_channels*4, 2, 2, 2)

        # Decoder with skip connections
        dec3_in = F.interpolate(
            bottleneck_out, size=(4, 4, 4), mode="trilinear", align_corners=False
        )
        dec3_in = torch.cat([dec3_in, enc3_out], dim=1)  # (B, base_channels*8, 4, 4, 4)
        dec3_out = self.dec3(dec3_in)  # (B, base_channels*2, 4, 4, 4)

        dec2_in = F.interpolate(dec3_out, size=(8, 8, 8), mode="trilinear", align_corners=False)
        dec2_in = torch.cat([dec2_in, enc2_out], dim=1)  # (B, base_channels*4, 8, 8, 8)
        dec2_out = self.dec2(dec2_in)  # (B, base_channels, 8, 8, 8)

        dec1_in = F.interpolate(dec2_out, size=(16, 16, 16), mode="trilinear", align_corners=False)
        dec1_in = torch.cat([dec1_in, enc1_out], dim=1)  # (B, base_channels*2, 16, 16, 16)
        dec1_out = self.dec1(dec1_in)  # (B, base_channels, 16, 16, 16)

        # Add spatial conditioning
        conditioning_3d = conditioning.unsqueeze(2).expand(-1, -1, 16, -1, -1)

        # Resize conditioning to target size if needed
        if target_size != 16:
            conditioning_3d = F.interpolate(
                conditioning_3d,
                size=(target_size, target_size, target_size),
                mode="trilinear",
                align_corners=False,
            )

        # Final upsampling to target size
        if target_size != 16:
            features = F.interpolate(
                dec1_out,
                size=(target_size, target_size, target_size),
                mode="trilinear",
                align_corners=False,
            )
        else:
            features = dec1_out

        # Combine with conditioning
        combined = torch.cat([features, conditioning_3d], dim=1)
        final_features = self.final_conv(combined)

        # Generate outputs
        air_mask_logits = self.air_mask_head(final_features)
        block_type_logits = self.block_type_head(final_features)

        return {"air_mask_logits": air_mask_logits, "block_type_logits": block_type_logits}


class V2ConditioningFusion(nn.Module):
    """V2 conditioning for LODiffusion inputs: x_height + x_biomefeat."""

    def __init__(self, config: SimpleFlexibleConfig):
        super().__init__()

        # x_biomefeat has 6 channels: temp, prec_one_hot(3), has_prec, is_cold
        # x_height has 1 channel
        # Total input: 6 + 1 = 7 channels at 8×8 resolution
        input_channels = 7

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(input_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
            # Upsample from 8×8 to 16×16 for UNet processing
            nn.Upsample(size=(16, 16), mode="bilinear", align_corners=False),
        )

    def forward(self, x_height, x_biomefeat, x_lod):
        """
        Args:
            x_height: [B,1,8,8,1] -> squeeze to [B,1,8,8]
            x_biomefeat: [B,6,8,8,1] -> squeeze to [B,6,8,8]
            x_lod: [B,1] (not used spatially yet)
        """
        # Remove the trailing dimension
        height_2d = x_height.squeeze(-1)  # [B,1,8,8]
        biome_2d = x_biomefeat.squeeze(-1)  # [B,6,8,8]

        # Concatenate height + biome features
        spatial_features = torch.cat([height_2d, biome_2d], dim=1)  # [B,7,8,8]

        # Process and upsample to 16×16
        conditioning = self.spatial_conv(spatial_features)  # [B,base_channels,16,16]

        return conditioning


class SimpleFlexibleUNet3D_V2(nn.Module):
    """V2 model that accepts LODiffusion contract inputs."""

    def __init__(self, config: SimpleFlexibleConfig):
        super().__init__()
        self.config = config
        self.lod_embed_dim = config.lod_embed_dim

        # V2 conditioning fusion for new input format
        self.conditioning_fusion = V2ConditioningFusion(config)

        # Same 3D processing as original model
        self.input_conv = nn.Sequential(
            nn.Conv3d(1, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        # Down-sampling path
        self.down1 = self._make_layer(config.base_channels, config.base_channels * 2)
        self.down2 = self._make_layer(config.base_channels * 2, config.base_channels * 4)
        self.down3 = self._make_layer(config.base_channels * 4, config.base_channels * 8)

        # Bottleneck
        self.bottleneck = self._make_layer(config.base_channels * 8, config.base_channels * 16)

        # Up-sampling path
        self.up3 = self._make_layer(
            config.base_channels * 16 + config.base_channels * 8, config.base_channels * 8
        )
        self.up2 = self._make_layer(
            config.base_channels * 8 + config.base_channels * 4, config.base_channels * 4
        )
        self.up1 = self._make_layer(
            config.base_channels * 4 + config.base_channels * 2, config.base_channels * 2
        )

        # Final convolution combining spatial conditioning
        self.final_conv = nn.Conv3d(
            config.base_channels * 2 + config.base_channels, config.base_channels, 1
        )  # Same heads as original
        self.air_mask_head = nn.Conv3d(config.base_channels, 1, 1)
        self.block_type_head = nn.Conv3d(config.base_channels, config.block_vocab_size, 1)

    def _make_layer(self, in_channels, out_channels):
        """Create a simple conv layer with normalization."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x_parent: torch.Tensor,
        x_height: torch.Tensor,
        x_biomefeat: torch.Tensor,
        x_lod: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with v2 LODiffusion inputs.

        Args:
            x_parent: [B,1,8,8,8] parent occupancy
            x_height: [B,1,16,16] height features at full resolution
            x_biomefeat: [B,6,16,16] biome climate features at full resolution
            x_lod: [B,1] LOD level
        """

        # Super-resolution approach: Start with upsampled parent at target resolution
        # Upsample parent from 8x8x8 to 16x16x16 using trilinear interpolation
        x_parent_upsampled = F.interpolate(
            x_parent, size=(16, 16, 16), mode="trilinear", align_corners=False
        )  # [B, 1, 16, 16, 16]

        # Process 2D conditioning at full 16x16 resolution
        # Add depth dimension to 2D features (broadcast along Y axis)
        x_height_3d = x_height.unsqueeze(-1).expand(-1, -1, -1, -1, 16)  # [B, 1, 16, 16, 16]
        x_biomefeat_3d = x_biomefeat.unsqueeze(-1).expand(-1, -1, -1, -1, 16)  # [B, 6, 16, 16, 16]

        # Combine all inputs at 16x16x16 resolution
        x = torch.cat(
            [x_parent_upsampled, x_height_3d, x_biomefeat_3d], dim=1
        )  # [B, 8, 16, 16, 16]

        # Encoder path (16x16x16 -> 8x8x8 -> 4x4x4, stop before 2x2x2)
        skip1 = x  # [B, 8, 16, 16, 16]
        x = self.encoder1(x)  # [B, 64, 16, 16, 16]
        x = self.pool1(x)  # [B, 64, 8, 8, 8]

        skip2 = x  # [B, 64, 8, 8, 8]
        x = self.encoder2(x)  # [B, 128, 8, 8, 8]
        x = self.pool2(x)  # [B, 128, 4, 4, 4]

        # Bottleneck at 4x4x4 (safe for batch norm)
        x = self.bottleneck(x)  # [B, 256, 4, 4, 4]

        # Decoder path with skip connections
        x = self.upconv2(x)  # [B, 128, 8, 8, 8]
        x = torch.cat([x, skip2], dim=1)  # [B, 192, 8, 8, 8]
        x = self.decoder2(x)  # [B, 128, 8, 8, 8]

        x = self.upconv1(x)  # [B, 64, 16, 16, 16]
        x = torch.cat([x, skip1], dim=1)  # [B, 72, 16, 16, 16]
        x = self.decoder1(x)  # [B, 64, 16, 16, 16]

        # Output heads
        air_mask_logits = self.air_head(x)  # [B, 1, 16, 16, 16]
        block_type_logits = self.block_head(x)  # [B, num_blocks, 16, 16, 16]

        return {
            "air_mask_logits": air_mask_logits,
            "block_type_logits": block_type_logits,
        }
        x = self.input_conv(x)  # (B, base_channels, 16, 16, 16)

        # Encoder path with skip connections
        skip1 = x
        x = F.max_pool3d(x, 2)
        x = self.down1(x)  # (B, 128, 8, 8, 8)

        skip2 = x
        x = F.max_pool3d(x, 2)
        x = self.down2(x)  # (B, 256, 4, 4, 4)

        skip3 = x
        x = F.max_pool3d(x, 2)
        x = self.down3(x)  # (B, 512, 2, 2, 2)

        # Bottleneck - now safely at 2x2x2
        x = F.max_pool3d(x, 2)  # (B, 512, 1, 1, 1)
        x = self.bottleneck(x)

        # Decoder path with skip connections
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.up3(x)

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x)

        # No final upsampling needed - we're already at 16x16x16

        # Output heads
        air_mask_logits = self.air_head(x)
        block_type_logits = self.block_head(x)

        return {
            "air_mask_logits": air_mask_logits,
            "block_type_logits": block_type_logits,
        }

        # Decoder path with skip connections
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.up3(x)

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x)

        # No final upsampling needed - we're already at 16x16x16

        # Output heads
        air_mask_logits = self.air_head(x)
        block_type_logits = self.block_head(x)

        return {
            "air_mask_logits": air_mask_logits,
            "block_type_logits": block_type_logits,
        }
