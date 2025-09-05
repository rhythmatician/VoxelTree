"""
Vanilla-parity UNet model using Pack S (surface-centric) inputs.

This model matches vanilla's noise sampling strategy with inputs that directly
correspond to Minecraft's internal noise router, height maps, and biome systems.
"""

from typing import Dict

import torch
import torch.nn as nn

from .unet3d import SimpleFlexibleConfig


class PackSConditioningFusion(nn.Module):
    """
    Conditioning fusion for Pack S (surface-centric) inputs.
    Combines height planes, router-6 slices, biome quart data, and coordinates.
    """

    def __init__(
        self,
        height_channels: int,
        router6_channels: int,
        biome_quart_channels: int,
        coord_channels: int,
        lod_embed_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        # Process each input type separately
        self.height_processor = nn.Sequential(
            nn.Conv2d(height_channels, hidden_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.router6_processor = nn.Sequential(
            nn.Conv2d(router6_channels, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Biome processing at quart resolution (4×4×4) → 16×16×16
        self.biome_processor = nn.Sequential(
            nn.Conv3d(biome_quart_channels, hidden_dim // 4, 3, padding=1),
            nn.Upsample(scale_factor=4, mode="trilinear", align_corners=False),  # 4×4×4 → 16×16×16
            nn.Conv3d(hidden_dim // 4, hidden_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # LOD embedding
        self.lod_embedding = nn.Embedding(8, lod_embed_dim)  # Support LOD 0-7

        # Coordinate processing
        self.coord_processor = nn.Linear(coord_channels + lod_embed_dim, hidden_dim // 4)

        # Final fusion (input: hidden//4 + hidden//2 + hidden//4 + hidden//4 = 5*hidden//4)
        fusion_input_channels = hidden_dim + hidden_dim // 4  # 40 channels for hidden=32
        self.fusion = nn.Sequential(
            nn.Conv3d(fusion_input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        height_planes: torch.Tensor,  # [B,2,16,16]
        router6_slices: torch.Tensor,  # [B,18,16,16]
        biome_quart: torch.Tensor,  # [B,C,4,4,4]
        chunk_coords: torch.Tensor,  # [B,2]
        lod: torch.Tensor,  # [B,1]
    ) -> torch.Tensor:
        """Fuse Pack S inputs into 3D conditioning features."""
        B, _, H, W = height_planes.shape

        # Process 2D features
        height_feat = self.height_processor(height_planes)  # [B, hidden//4, 16, 16]
        router6_feat = self.router6_processor(router6_slices)  # [B, hidden//2, 16, 16]

        # Process 3D biome features
        biome_feat = self.biome_processor(biome_quart)  # [B, hidden//4, 16, 16, 16]

        # Process coordinates + LOD
        lod_emb = self.lod_embedding(lod.squeeze(-1))  # [B, lod_embed_dim]
        coord_feat = self.coord_processor(
            torch.cat([chunk_coords, lod_emb], dim=1)
        )  # [B, hidden//4]

        # Broadcast 2D features to 3D
        height_3d = height_feat.unsqueeze(-1).expand(
            -1, -1, -1, -1, 16
        )  # [B, hidden//4, 16, 16, 16]
        router6_3d = router6_feat.unsqueeze(-1).expand(
            -1, -1, -1, -1, 16
        )  # [B, hidden//2, 16, 16, 16]

        # Broadcast coordinate features to 3D
        coord_3d = coord_feat.view(B, -1, 1, 1, 1).expand(
            -1, -1, 16, 16, 16
        )  # [B, hidden//4, 16, 16, 16]

        # Combine all features (8 + 16 + 8 + 8 = 40 channels total)
        combined = torch.cat(
            [height_3d, router6_3d, biome_feat, coord_3d], dim=1
        )  # [B, 40, 16, 16, 16]

        return self.fusion(combined)


class VanillaParity_UNet3D(nn.Module):
    """
    Vanilla-parity model using Pack S (surface-centric) inputs.

    This model matches vanilla's noise sampling strategy:
    - Height planes at full resolution (16×16)
    - Router-6 slices at key Y levels
    - Biomes at native quart resolution (4×4×4)
    - Progressive LOD: 8×8×8 → 16×16×16 super-resolution

    Pack S Inputs:
    - x_parent: [B,1,8,8,8] parent voxels from previous LOD
    - x_height_planes: [B,2,16,16] WORLD_SURFACE_WG + OCEAN_FLOOR_WG
    - x_router6_slices: [B,18,16,16] temperature,vegetation,continents,
      erosion,depth,ridges at 3 Y levels
    - x_biome_quart: [B,C,4,4,4] biomes at native quart resolution (4×4×4)
    - x_chunk_coords: [B,2] chunk X,Z coordinates
    - x_lod: [B,1] LOD level
    """

    def __init__(self, config: SimpleFlexibleConfig):
        super().__init__()
        self.config = config

        # Pack S input dimensions
        height_channels = 2  # WORLD_SURFACE_WG + OCEAN_FLOOR_WG
        router6_channels = 6 * 3  # 6 router fields × 3 Y slices
        biome_quart_channels = 64  # Max biome count, at 4×4×4 resolution
        coord_channels = 2  # chunk X,Z

        # Conditioning fusion for Pack S inputs
        self.conditioning_fusion = PackSConditioningFusion(
            height_channels=height_channels,
            router6_channels=router6_channels,
            biome_quart_channels=biome_quart_channels,
            coord_channels=coord_channels,
            lod_embed_dim=config.lod_embed_dim,
            hidden_dim=config.base_channels,
        )

        # Parent voxel processing: 8×8×8 → 16×16×16
        self.parent_encoder = nn.Sequential(
            nn.Conv3d(1, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),  # 8×8×8 → 16×16×16
            nn.Conv3d(config.base_channels, config.base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(inplace=True),
        )

        # Main processing layers (no deep downsampling to avoid 1×1×1)
        self.main_conv = nn.Sequential(
            self._make_layer(
                config.base_channels * 3, config.base_channels * 4
            ),  # Combined features
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
        x_parent: torch.Tensor,  # [B,1,8,8,8] parent voxels
        x_height_planes: torch.Tensor,  # [B,2,16,16] height planes
        x_router6_slices: torch.Tensor,  # [B,18,16,16] router-6 at 3 Y levels
        x_biome_quart: torch.Tensor,  # [B,C,4,4,4] biomes at quart resolution
        x_chunk_coords: torch.Tensor,  # [B,2] chunk coordinates
        x_lod: torch.Tensor,  # [B,1] LOD level
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using Pack S (surface-centric) inputs.

        This matches vanilla's noise router + height + biome sampling
        at their native resolutions without upsampling in the mod.
        """
        # Process parent voxels: 8×8×8 → 16×16×16
        parent_features = self.parent_encoder(x_parent)  # [B, base_channels*2, 16, 16, 16]

        # Fuse conditioning inputs into spatial features
        conditioning_features = self.conditioning_fusion(
            x_height_planes, x_router6_slices, x_biome_quart, x_chunk_coords, x_lod
        )  # [B, base_channels, 16, 16, 16]

        # Combine parent and conditioning features
        combined = torch.cat(
            [parent_features, conditioning_features], dim=1
        )  # [B, base_channels*3, 16, 16, 16]

        # Main processing
        features = self.main_conv(combined)  # [B, base_channels, 16, 16, 16]

        # Generate outputs
        air_mask_logits = self.air_head(features)  # [B, 1, 16, 16, 16]
        block_type_logits = self.block_head(features)  # [B, block_vocab_size, 16, 16, 16]

        return {
            "air_mask_logits": air_mask_logits,
            "block_type_logits": block_type_logits,
        }
