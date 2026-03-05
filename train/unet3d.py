"""
Flexible 3D U-Net Model Architecture for Multi-LOD Voxel Super-Resolution

This module implements a flexible 3D U-Net architecture that handles
all LOD transitions in a single network:
- LOD4→LOD3: 1³ → 2³
- LOD3→LOD2: 2³ → 4³
- LOD2→LOD1: 4³ → 8³
- LOD1→LOD0: 8³ → 16³

Key Features:
- Adaptive input/output processing for multiple LODs
- Simple spatial conditioning (biome + heightmap)
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
    """Simplified conditioning that handles biomes and heightmaps (no river)."""

    def __init__(self, config: SimpleFlexibleConfig):
        super().__init__()
        self.biome_embedding = nn.Embedding(config.biome_vocab_size, config.biome_embed_dim)

        # Simple fusion: biome + height -> base_channels
        spatial_channels = config.biome_embed_dim + 1  # biome + height
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(spatial_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, biome_patch, heightmap_patch, y_index, lod):
        # Handle biome format
        if biome_patch.dim() == 4 and biome_patch.shape[1] > 16:  # One-hot (B, 256, 16, 16)
            biome_indices = biome_patch.argmax(dim=1)  # (B, 16, 16)
        elif biome_patch.dim() == 5:  # One-hot (B, NB, 16, 16, 1)
            biome_indices = biome_patch.argmax(dim=1).squeeze(-1)
        else:  # Index format (B, 16, 16)
            biome_indices = biome_patch

        biome_emb = self.biome_embedding(biome_indices)  # (B, 16, 16, embed_dim)
        biome_emb = biome_emb.permute(0, 3, 1, 2)  # (B, embed_dim, 16, 16)

        # Handle heightmap format
        if heightmap_patch.dim() == 5:
            heightmap_patch = heightmap_patch.squeeze(-1)

        # Concatenate and process
        spatial_features = torch.cat([biome_emb, heightmap_patch], dim=1)
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
        # Map LOD embedding to a per-channel bias we can add to features
        self.lod_to_channels = nn.Sequential(
            nn.Linear(config.lod_embed_dim, config.base_channels),
            nn.ReLU(inplace=True),
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
        y_index: torch.Tensor,
        lod: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with flexible input/output sizes.

        Strategy: Always process at 16³, pad/crop as needed.
        """
        input_size = parent_voxel.shape[-1]
        target_size = input_size * 2

        # Create LOD embeddings and map to feature bias
        lod_emb = get_timestep_embedding(lod.flatten(), self.lod_embed_dim)
        lod_emb = self.lod_projection(lod_emb)  # (B, D)

        # Get spatial conditioning at 16×16
        conditioning = self.conditioning_fusion(biome_patch, heightmap_patch, y_index, lod)

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

        # Inject LOD conditioning as a per-channel bias
        lod_bias = self.lod_to_channels(lod_emb).view(-1, self.config.base_channels, 1, 1, 1)
        dec1_out = dec1_out + lod_bias

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


# Note: V2 experimental classes removed for clarity and to keep the file aligned with
# the current input contract (no river). If needed later, re-introduce under separate module.
