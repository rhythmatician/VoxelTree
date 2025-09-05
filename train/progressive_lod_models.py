"""
Progressive LOD Models following the 4-model I/O contract.

This implements the complete LOD4→LOD3→LOD2→LOD1→LOD0 progression
with proper vanilla-parity inputs from Fabric API.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .unet3d import SimpleFlexibleConfig


class VanillaConditioningFusion(nn.Module):
    """
    Conditioning fusion for vanilla-parity inputs.
    Handles height planes, biome quart, router6, and optional features.
    """

    def __init__(
        self,
        height_channels: int = 5,  # surface, ocean_floor, slope_x, slope_z, curvature
        biome_quart_channels: int = 6,  # temp, precip_onehot[3], isCold
        router6_channels: int = 6,  # temperature, vegetation, continents, erosion, depth, ridges
        coord_channels: int = 2,  # chunk x,z
        lod_embed_dim: int = 16,
        hidden_dim: int = 32,
        # Optional features
        barrier_channels: int = 1,
        aquifer_channels: int = 3,  # fluidLevelFloodedness, fluidLevelSpread, lava
        cave_prior_channels: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim  # Store for later use

        # Process height planes (16×16 → broadcast to 3D)
        self.height_processor = nn.Sequential(
            nn.Conv2d(height_channels, hidden_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Process router6 slices (16×16 → broadcast to 3D)
        self.router6_processor = nn.Sequential(
            nn.Conv2d(router6_channels, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Process biomes at quart resolution (4×4×4 → upsample to output size)
        self.biome_processor = nn.Sequential(
            nn.Conv3d(biome_quart_channels, hidden_dim // 8, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Optional features (16×16 → broadcast to 3D)
        self.barrier_processor = nn.Sequential(
            nn.Conv2d(barrier_channels, hidden_dim // 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.aquifer_processor = nn.Sequential(
            nn.Conv2d(aquifer_channels, hidden_dim // 8, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Cave prior at 4×4×4 resolution
        self.cave_processor = nn.Sequential(
            nn.Conv3d(cave_prior_channels, hidden_dim // 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # LOD embedding
        self.lod_embedding = nn.Embedding(8, lod_embed_dim)

        # Coordinate processing
        self.coord_processor = nn.Linear(coord_channels + lod_embed_dim, hidden_dim // 8)

        # Calculate total channels for fusion
        # Required: height(8) + router6(16) + biome(4) + coord(4) = 32
        # Optional: barrier(2) + aquifer(4) + cave(2) = 8
        # We'll create the fusion layer dynamically based on actual inputs
        self.base_fusion = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),  # 32 → 32 (base case)
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Optional features fusion (will be created dynamically if needed)
        self.full_fusion = nn.Sequential(
            nn.Conv3d(
                hidden_dim + hidden_dim // 4, hidden_dim, 3, padding=1
            ),  # 40 → 32 (with optional)
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x_height_planes: torch.Tensor,  # [B,5,1,16,16]
        x_biome_quart: torch.Tensor,  # [B,6,4,4,4]
        x_router6: torch.Tensor,  # [B,6,1,16,16]
        x_chunk_pos: torch.Tensor,  # [B,2]
        x_lod: torch.Tensor,  # [B,1]
        output_size: int,  # Target 3D size (2, 4, 8, or 16)
        # Optional inputs
        x_barrier: Optional[torch.Tensor] = None,  # [B,1,1,16,16]
        x_aquifer3: Optional[torch.Tensor] = None,  # [B,3,1,16,16]
        x_cave_prior4: Optional[torch.Tensor] = None,  # [B,1,4,4,4]
    ) -> torch.Tensor:
        """Fuse vanilla inputs into 3D conditioning features."""
        B = x_height_planes.shape[0]

        # Process 2D features (remove singleton height dimension first)
        height_2d = x_height_planes.squeeze(2)  # [B,5,16,16]
        router6_2d = x_router6.squeeze(2)  # [B,6,16,16]

        height_feat = self.height_processor(height_2d)  # [B,8,16,16]
        router6_feat = self.router6_processor(router6_2d)  # [B,16,16,16]

        # Process 3D biome features (upsample from 4×4×4 to output size)
        biome_feat = self.biome_processor(x_biome_quart)  # [B,4,4,4,4]
        if output_size != 4:
            scale_factor = output_size / 4
            biome_feat = nn.functional.interpolate(
                biome_feat, scale_factor=scale_factor, mode="trilinear", align_corners=False
            )  # [B,4,output_size,output_size,output_size]

        # Process coordinates + LOD
        lod_emb = self.lod_embedding(x_lod.squeeze(-1))  # [B,lod_embed_dim]
        coord_feat = self.coord_processor(torch.cat([x_chunk_pos, lod_emb], dim=1))  # [B,4]

        # Broadcast 2D features to 3D at output size
        if output_size != 16:
            # Resize 2D features to match output size
            scale_factor = output_size / 16
            height_feat = nn.functional.interpolate(
                height_feat, scale_factor=scale_factor, mode="bilinear", align_corners=False
            )
            router6_feat = nn.functional.interpolate(
                router6_feat, scale_factor=scale_factor, mode="bilinear", align_corners=False
            )

        # Broadcast to 3D
        height_3d = height_feat.unsqueeze(-1).expand(-1, -1, -1, -1, output_size)
        router6_3d = router6_feat.unsqueeze(-1).expand(-1, -1, -1, -1, output_size)
        coord_3d = coord_feat.view(B, -1, 1, 1, 1).expand(
            -1, -1, output_size, output_size, output_size
        )

        # Combine required features
        features = [height_3d, router6_3d, biome_feat, coord_3d]  # 8+16+4+4 = 32 channels

        # Add optional features if provided
        if x_barrier is not None:
            barrier_2d = x_barrier.squeeze(2)  # [B,1,16,16]
            barrier_feat = self.barrier_processor(barrier_2d)  # [B,2,16,16]
            if output_size != 16:
                scale_factor = output_size / 16
                barrier_feat = nn.functional.interpolate(
                    barrier_feat, scale_factor=scale_factor, mode="bilinear", align_corners=False
                )
            barrier_3d = barrier_feat.unsqueeze(-1).expand(-1, -1, -1, -1, output_size)
            features.append(barrier_3d)  # +2 channels

        if x_aquifer3 is not None:
            aquifer_2d = x_aquifer3.squeeze(2)  # [B,3,16,16]
            aquifer_feat = self.aquifer_processor(aquifer_2d)  # [B,4,16,16]
            if output_size != 16:
                scale_factor = output_size / 16
                aquifer_feat = nn.functional.interpolate(
                    aquifer_feat, scale_factor=scale_factor, mode="bilinear", align_corners=False
                )
            aquifer_3d = aquifer_feat.unsqueeze(-1).expand(-1, -1, -1, -1, output_size)
            features.append(aquifer_3d)  # +4 channels

        if x_cave_prior4 is not None:
            cave_feat = self.cave_processor(x_cave_prior4)  # [B,2,4,4,4]
            if output_size != 4:
                scale_factor = output_size / 4
                cave_feat = nn.functional.interpolate(
                    cave_feat, scale_factor=scale_factor, mode="trilinear", align_corners=False
                )
            features.append(cave_feat)  # +2 channels

        # Combine all features
        combined = torch.cat(
            features, dim=1
        )  # [B, 32+optional, output_size, output_size, output_size]

        # Use appropriate fusion layer based on number of channels
        if combined.shape[1] == self.hidden_dim:  # Only base features (32 channels)
            return self.base_fusion(combined)
        else:  # Has optional features (40 channels)
            return self.full_fusion(combined)


class ProgressiveLODModel0_Initial(nn.Module):
    """
    Model 0: Initial terrain generation (no parent input)
    Generates the very first LOD from conditioning inputs only.

    Outputs: [1, N_blocks, 1, 1, 1] or [1, N_blocks, 2, 2, 2]
    """

    def __init__(self, config: SimpleFlexibleConfig, output_size: int = 2):
        super().__init__()
        self.config = config
        self.output_size = output_size  # 1 or 2

        # Enhanced conditioning fusion
        self.conditioning_fusion = VanillaConditioningFusion()

        # No parent encoder - start from conditioning only
        # Main processing layers
        self.main_conv = nn.Sequential(
            self._make_layer(config.base_channels, config.base_channels * 2),
            self._make_layer(config.base_channels * 2, config.base_channels * 4),
            self._make_layer(config.base_channels * 4, config.base_channels * 2),
            self._make_layer(config.base_channels * 2, config.base_channels),
        )

        # Output heads
        self.air_head = nn.Conv3d(config.base_channels, 1, kernel_size=1)
        self.block_head = nn.Conv3d(config.base_channels, config.block_vocab_size, kernel_size=1)

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create conv layer with normalization and activation."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x_height_planes: torch.Tensor,  # [1,5,1,16,16]
        x_biome_quart: torch.Tensor,  # [1,6,4,4,4]
        x_router6: torch.Tensor,  # [1,6,1,16,16]
        x_chunk_pos: torch.Tensor,  # [1,2]
        x_lod: torch.Tensor,  # [1,1]
        x_barrier: Optional[torch.Tensor] = None,  # [1,1,1,16,16]
        x_aquifer3: Optional[torch.Tensor] = None,  # [1,3,1,16,16]
        x_cave_prior4: Optional[torch.Tensor] = None,  # [1,1,4,4,4]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate initial terrain from conditioning inputs only.
        No parent input since this is the first/coarsest level.
        """
        # Fuse conditioning inputs into spatial features
        conditioning_features = self.conditioning_fusion(
            x_height_planes,
            x_biome_quart,
            x_router6,
            x_chunk_pos,
            x_lod,
            self.output_size,
            x_barrier,
            x_aquifer3,
            x_cave_prior4,
        )  # [1, base_channels, output_size, output_size, output_size]

        # Main processing (no downsampling needed since we generated at target size)
        features = self.main_conv(conditioning_features)

        # Generate outputs
        air_mask_logits = self.air_head(features)
        block_type_logits = self.block_head(features)

        return {
            "air_mask_logits": air_mask_logits,
            "block_type_logits": block_type_logits,
        }


class ProgressiveLODModel(nn.Module):
    """
    Single progressive LOD model that can handle any stage.
    i.e. LOD4→LOD3, LOD3→LOD2, LOD2→LOD1, LOD1→LOD0

    The output size is determined by the LOD level:
    - LOD4→LOD3: 2×2×2
    - LOD3→LOD2: 4×4×4
    - LOD2→LOD1: 8×8×8
    - LOD1→LOD0: 16×16×16
    """

    def __init__(self, config: SimpleFlexibleConfig, output_size: int):
        super().__init__()
        self.config = config
        self.output_size = output_size

        assert output_size in [1, 2, 4, 8, 16], "output_size must be one of [1, 2, 4, 8, 16]"

        # Conditioning fusion
        self.conditioning_fusion = VanillaConditioningFusion(hidden_dim=config.base_channels)

        # Parent processing (if parent exists)
        if output_size > 2:  # Has a parent from previous LOD
            self.parent_encoder = nn.Sequential(
                nn.Conv3d(1, config.base_channels, 3, padding=1),
                nn.BatchNorm3d(config.base_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),  # 2× upsampling
                nn.Conv3d(config.base_channels, config.base_channels, 3, padding=1),
                nn.BatchNorm3d(config.base_channels),
                nn.ReLU(inplace=True),
            )
            combined_channels = config.base_channels * 2  # parent + conditioning
        else:
            self.parent_encoder = None
            combined_channels = config.base_channels  # conditioning only

        # Main processing layers
        self.main_conv = nn.Sequential(
            self._make_layer(combined_channels, config.base_channels * 2),
            self._make_layer(config.base_channels * 2, config.base_channels),
        )

        # Output heads
        self.air_head = nn.Conv3d(config.base_channels, 1, kernel_size=1)
        self.block_head = nn.Conv3d(config.base_channels, config.block_vocab_size, kernel_size=1)

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create conv layer with normalization and activation."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        # Required inputs (all models)
        x_height_planes: torch.Tensor,  # [1,5,1,16,16]
        x_biome_quart: torch.Tensor,  # [1,6,4,4,4]
        x_router6: torch.Tensor,  # [1,6,1,16,16]
        x_chunk_pos: torch.Tensor,  # [1,2]
        x_lod: torch.Tensor,  # [1,1]
        # Parent input (None for LOD4→LOD3)
        x_parent_prev: Optional[torch.Tensor] = None,  # [1,1,parent_size,parent_size,parent_size]
        # Optional inputs
        x_barrier: Optional[torch.Tensor] = None,  # [1,1,1,16,16]
        x_aquifer3: Optional[torch.Tensor] = None,  # [1,3,1,16,16]
        x_cave_prior4: Optional[torch.Tensor] = None,  # [1,1,4,4,4]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for progressive LOD model.

        Handles all 4 stages of the LOD progression based on output_size.
        """
        # Fuse conditioning inputs
        conditioning_features = self.conditioning_fusion(
            x_height_planes,
            x_biome_quart,
            x_router6,
            x_chunk_pos,
            x_lod,
            self.output_size,
            x_barrier,
            x_aquifer3,
            x_cave_prior4,
        )  # [B, base_channels, output_size, output_size, output_size]

        # Process parent if available
        if self.parent_encoder is not None and x_parent_prev is not None:
            parent_features = self.parent_encoder(
                x_parent_prev
            )  # [B, base_channels, output_size, output_size, output_size]
            combined = torch.cat([parent_features, conditioning_features], dim=1)
        else:
            combined = conditioning_features

        # Main processing
        features = self.main_conv(
            combined
        )  # [B, base_channels, output_size, output_size, output_size]

        # Generate outputs
        air_mask = self.air_head(features)  # [B, 1, output_size, output_size, output_size]
        block_logits = self.block_head(
            features
        )  # [B, block_vocab_size, output_size, output_size, output_size]

        return {
            "air_mask": air_mask,
            "block_logits": block_logits,
        }


# Create the 4 progressive models
def create_lod4_to_lod3_model(config: SimpleFlexibleConfig) -> ProgressiveLODModel:
    """LOD4 → LOD3: outputs 2×2×2"""
    return ProgressiveLODModel(config, output_size=2)


def create_lod3_to_lod2_model(config: SimpleFlexibleConfig) -> ProgressiveLODModel:
    """LOD3 → LOD2: outputs 4×4×4"""
    return ProgressiveLODModel(config, output_size=4)


def create_lod2_to_lod1_model(config: SimpleFlexibleConfig) -> ProgressiveLODModel:
    """LOD2 → LOD1: outputs 8×8×8"""
    return ProgressiveLODModel(config, output_size=8)


def create_lod1_to_lod0_model(config: SimpleFlexibleConfig) -> ProgressiveLODModel:
    """LOD1 → LOD0: outputs 16×16×16"""
    return ProgressiveLODModel(config, output_size=16)
