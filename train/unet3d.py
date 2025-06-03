"""
3D U-Net Model Architecture for VoxelTree LOD Super-Resolution

This module implements a 3D U-Net architecture specifically designed for
voxel super-resolution in Minecraft terrain generation. The model takes
8³ parent voxels and upsamples them to 16³ target voxels, conditioned on
biome data, heightmaps, and LOD embeddings.

Key Features:
- 3D U-Net with skip connections for spatial feature preservation
- Multi-modal conditioning (biome, heightmap, river, Y-level, LOD)
- Dual output heads for air mask and block type prediction
- Memory-efficient architecture suitable for training on consumer GPUs
"""

from typing import Dict, Literal, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNet3DConfig:
    """Configuration for VoxelUNet3D model architecture."""

    # Core architecture
    input_channels: int = 1
    output_channels: int = 2  # Air mask + block types (will be split)
    base_channels: int = 32
    depth: int = 3

    # Conditioning inputs
    biome_vocab_size: int = 50
    biome_embed_dim: int = 16
    heightmap_channels: int = 1
    river_channels: int = 1
    y_embed_dim: int = 8
    lod_embed_dim: int = 8

    # Training hyperparameters
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    activation: Literal["relu", "leaky_relu", "gelu"] = "relu"

    # Output configuration
    air_mask_channels: int = 1
    block_type_channels: int = 10  # Number of different block types

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.output_channels < 2:
            raise ValueError("output_channels must be at least 2")
        if self.base_channels < 16:
            raise ValueError("base_channels must be at least 16")
        if self.depth < 1:
            raise ValueError("depth must be at least 1")
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.biome_vocab_size <= 0:
            raise ValueError("biome_vocab_size must be positive")
        if self.activation not in ["relu", "leaky_relu", "gelu"]:
            raise ValueError("activation must be one of: relu, leaky_relu, gelu")


class Conv3DBlock(nn.Module):
    """Basic 3D convolution block with normalization and activation."""

    # Explicit type annotations to handle conditional assignments
    norm: Union[nn.GroupNorm, nn.Identity]
    activation: Union[nn.ReLU, nn.LeakyReLU, nn.GELU]
    dropout: Union[nn.Dropout3d, nn.Identity]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not use_batch_norm,
        )

        if use_batch_norm:
            # Use GroupNorm instead of BatchNorm to handle single-element batches
            # GroupNorm with num_groups=8 works well for channels >= 16
            num_groups = min(8, out_channels // 2) if out_channels >= 16 else 1
            self.norm = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm = nn.Identity()

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if dropout_rate > 0:
            self.dropout = nn.Dropout3d(dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DoubleConv3D(nn.Module):
    """Double convolution block commonly used in U-Net architectures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.conv1 = Conv3DBlock(
            in_channels,
            out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.conv2 = Conv3DBlock(
            out_channels,
            out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block with max pooling and double convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(
            in_channels,
            out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with transposed convolution and skip connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Combine upsampled features with skip connection
        self.conv = DoubleConv3D(
            in_channels // 2 + skip_channels,
            out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatches by padding
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)

        x = F.pad(
            x,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
                diff_d // 2,
                diff_d - diff_d // 2,
            ],
        )

        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class ConditioningFusion(nn.Module):
    """Fuses multiple conditioning inputs into spatial features."""

    def __init__(
        self,
        biome_vocab_size: int,
        biome_embed_dim: int,
        heightmap_channels: int,
        river_channels: int,
        y_embed_dim: int,
        lod_embed_dim: int,
        spatial_size: int = 16,  # Target spatial resolution
        output_channels: int = 32,
    ):
        super().__init__()

        self.spatial_size = spatial_size

        # Embedding layers for discrete inputs
        self.biome_embedding = nn.Embedding(biome_vocab_size, biome_embed_dim)
        self.y_embedding = nn.Embedding(24, y_embed_dim)  # 24 vertical subchunks
        self.lod_embedding = nn.Embedding(5, lod_embed_dim)  # LOD levels 0-4

        # Process heightmap and river data
        self.heightmap_conv = nn.Conv2d(heightmap_channels, 8, kernel_size=3, padding=1)
        self.river_conv = nn.Conv2d(river_channels, 8, kernel_size=3, padding=1)

        # Compute total conditioning channels
        total_channels = (
            biome_embed_dim  # Biome embeddings
            + 8  # Processed heightmap
            + 8  # Processed river
            + y_embed_dim  # Y-level embedding
            + lod_embed_dim  # LOD embedding
        )

        # Final fusion layer
        self.fusion_conv = nn.Conv2d(total_channels, output_channels, kernel_size=1)

    def forward(
        self,
        biome_patch: torch.Tensor,
        heightmap_patch: torch.Tensor,
        river_patch: torch.Tensor,
        y_index: torch.Tensor,
        lod: torch.Tensor,
    ) -> torch.Tensor:
        # Process biome embeddings
        biome_embeds = self.biome_embedding(biome_patch)  # (B, 16, 16, E)
        biome_embeds = biome_embeds.permute(0, 3, 1, 2)  # (B, E, 16, 16)

        # Process heightmap and river
        heightmap_features = F.relu(self.heightmap_conv(heightmap_patch))
        river_features = F.relu(self.river_conv(river_patch))

        # Process scalar embeddings and broadcast
        y_embeds = self.y_embedding(y_index)  # (B, Y_DIM)
        lod_embeds = self.lod_embedding(lod)  # (B, LOD_DIM)

        # Broadcast scalar embeddings to spatial dimensions
        y_spatial = y_embeds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16)
        lod_spatial = lod_embeds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16)

        # Concatenate all conditioning features
        conditioning = torch.cat(
            [biome_embeds, heightmap_features, river_features, y_spatial, lod_spatial], dim=1
        )

        # Fuse into final conditioning representation
        conditioning = self.fusion_conv(conditioning)
        return conditioning


class VoxelUNet3D(nn.Module):
    """
    3D U-Net for voxel super-resolution with multi-modal conditioning.

    Takes 8³ parent voxels and upsamples to 16³ target voxels,
    conditioned on biome, heightmap, river, Y-level, and LOD information.
    """

    def __init__(self, config: UNet3DConfig):
        super().__init__()

        self.config = config

        # Conditioning fusion module
        self.conditioning_fusion = ConditioningFusion(
            biome_vocab_size=config.biome_vocab_size,
            biome_embed_dim=config.biome_embed_dim,
            heightmap_channels=config.heightmap_channels,
            river_channels=config.river_channels,
            y_embed_dim=config.y_embed_dim,
            lod_embed_dim=config.lod_embed_dim,
            output_channels=config.base_channels,
        )

        # U-Net encoder (downsampling path)
        self.encoder = nn.ModuleList()

        # Initial convolution
        self.initial_conv = DoubleConv3D(
            config.input_channels,
            config.base_channels,
            use_batch_norm=config.use_batch_norm,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
        )

        # Encoder blocks
        encoder_channels = []
        in_channels = config.base_channels
        for i in range(config.depth):
            out_channels = config.base_channels * (2 ** (i + 1))
            encoder_channels.append(in_channels)

            self.encoder.append(
                DownBlock(
                    in_channels,
                    out_channels,
                    use_batch_norm=config.use_batch_norm,
                    activation=config.activation,
                    dropout_rate=config.dropout_rate,
                )
            )
            in_channels = out_channels
        # Bottleneck
        self.bottleneck = DoubleConv3D(
            in_channels,
            in_channels * 2,
            use_batch_norm=config.use_batch_norm,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
        )

        # U-Net decoder (upsampling path)
        self.decoder = nn.ModuleList()
        in_channels = in_channels * 2

        for i in range(config.depth):
            skip_channels = encoder_channels[-(i + 1)]
            out_channels = config.base_channels * (2 ** (config.depth - i - 1))

            self.decoder.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    skip_channels,
                    use_batch_norm=config.use_batch_norm,
                    activation=config.activation,
                    dropout_rate=config.dropout_rate,
                )
            )
            in_channels = out_channels

        # Additional upsampling layer to go from 8³ to 16³
        # After decoder, we have (B, base_channels, 8, 8, 8)
        # Need one more 2x upsample to reach (B, base_channels, 16, 16, 16)
        self.final_upsample = nn.ConvTranspose3d(
            config.base_channels, config.base_channels, kernel_size=2, stride=2
        )

        # Conditioning injection layer
        # Projects 2D conditioning to 3D and combines with decoder features
        self.conditioning_projection = nn.Conv3d(
            config.base_channels + config.base_channels,  # Features + conditioning
            config.base_channels,
            kernel_size=1,
        )

        # Output heads
        self.air_mask_head = nn.Conv3d(
            config.base_channels, config.air_mask_channels, kernel_size=1
        )

        self.block_type_head = nn.Conv3d(
            config.base_channels, config.block_type_channels, kernel_size=1
        )

        # Store embedding modules for test access
        self.biome_embedding = self.conditioning_fusion.biome_embedding
        self.y_embedding = self.conditioning_fusion.y_embedding
        self.lod_embedding = self.conditioning_fusion.lod_embedding

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
        Forward pass through the 3D U-Net.

        Args:
            parent_voxel: (B, 1, 8, 8, 8) - Parent voxel data
            biome_patch: (B, 16, 16) - Biome IDs per spatial location
            heightmap_patch: (B, 1, 16, 16) - Height data
            river_patch: (B, 1, 16, 16) - River noise data
            y_index: (B,) - Vertical subchunk index
            lod: (B,) - Level of detail

        Returns:
            Dictionary containing:
                - air_mask_logits: (B, 1, 16, 16, 16) - Air/solid predictions
                - block_type_logits: (B, 10, 16, 16, 16) - Block type predictions
        """

        # Get conditioning features (B, C, 16, 16)
        conditioning = self.conditioning_fusion(
            biome_patch, heightmap_patch, river_patch, y_index, lod
        )

        # Initial convolution on parent voxel
        x = self.initial_conv(parent_voxel)  # (B, base_channels, 8, 8, 8)

        # Encoder path with skip connections
        skip_connections = [x]
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        skip_connections = skip_connections[:-1]  # Remove last (bottleneck input)
        for i, decoder_block in enumerate(self.decoder):
            skip = skip_connections[-(i + 1)]
            x = decoder_block(x, skip)

        # Final upsampling from 8³ to 16³
        x = self.final_upsample(x)  # (B, base_channels, 16, 16, 16)

        # At this point x should be (B, base_channels, 16, 16, 16)
        # We need to inject 2D conditioning into 3D features

        # Expand conditioning to 3D by replicating across depth dimension
        batch_size, cond_channels, height, width = conditioning.shape
        depth = x.size(2)

        # Expand conditioning: (B, C, 16, 16) -> (B, C, 16, 16, 16)
        conditioning_3d = conditioning.unsqueeze(2).expand(-1, -1, depth, -1, -1)

        # Concatenate features and conditioning
        combined = torch.cat([x, conditioning_3d], dim=1)

        # Project to final feature representation
        features = self.conditioning_projection(combined)

        # Generate outputs
        air_mask_logits = self.air_mask_head(features)
        block_type_logits = self.block_type_head(features)

        return {"air_mask_logits": air_mask_logits, "block_type_logits": block_type_logits}
