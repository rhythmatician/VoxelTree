"""
Progressive LOD Models — 4 separate models for LOD4→LOD1 refinement.

Architecture: Each LOD transition gets its own model with fixed tensor shapes.
LOD0 is NOT generated — vanilla terrain handles full-resolution.

Model family:
  - Init:     Noise → LOD4  (1×1×1 output, tiny MLP)
  - LOD4→3:   1³ → 2³       (small Conv3D)
  - LOD3→2:   2³ → 4³       (medium Conv3D)
  - LOD2→1:   4³ → 8³       (medium-large Conv3D)

Conditioning channels (shared, cheap):
  - x_height_planes [B,5,16,16]   — surface, ocean_floor, slope_x, slope_z, curvature
  - x_router6       [B,6,16,16]   — temperature, vegetation, continents, erosion, depth, ridges
  - x_biome         [B,16,16]     — int64 biome index per column (→ learned embedding)
  - x_y_index       [B]           — int64 y-slab position 0..23 (→ learned embedding)

Dropped channels (too expensive for distant LOD):
  - x_cave_prior4, x_aquifer3, x_barrier, x_biome_quart, x_chunk_pos, x_lod
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .unet3d import SimpleFlexibleConfig


class AnchorConditioningFusion3D(nn.Module):
    """
    Conditioning fusion for anchor channels, adapted for varying 3D output sizes.

    Processes 2D anchor signals (height_planes, router6, biome, y_index) and
    broadcasts them to the target 3D volume size for each progressive model.

    Architecture:
      height_planes → Conv2d(5, out//4)  → GroupNorm → ReLU  ──┐
      router6       → Conv2d(6, out//2)  → GroupNorm → ReLU  ──┤
      biome embed   → Conv2d(E, out//4)  → GroupNorm → ReLU  ──┤
                                                                ├─ cat → Conv2d → 3D
      y_index embed → broadcast [B, y_dim, H, W]              ──┘
    """

    def __init__(
        self,
        height_channels: int = 5,
        router6_channels: int = 6,
        biome_vocab_size: int = 256,
        biome_embed_dim: int = 32,
        y_embed_dim: int = 16,
        y_slabs: int = 24,
        out_channels: int = 32,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.y_slabs = y_slabs
        self.y_embed_dim = y_embed_dim

        quarter = max(out_channels // 4, 4)
        half = max(out_channels // 2, 8)

        # Height planes stream
        self.height_conv = nn.Sequential(
            nn.Conv2d(height_channels, quarter, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, quarter), quarter),
            nn.ReLU(inplace=True),
        )

        # Router-6 stream (dominant: macro terrain shape)
        self.router6_conv = nn.Sequential(
            nn.Conv2d(router6_channels, half, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, half), half),
            nn.ReLU(inplace=True),
        )

        # Biome stream
        self.biome_embedding = nn.Embedding(biome_vocab_size, biome_embed_dim)
        self.biome_conv = nn.Sequential(
            nn.Conv2d(biome_embed_dim, quarter, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, quarter), quarter),
            nn.ReLU(inplace=True),
        )

        # Y-slab embedding
        self.y_embedding = nn.Embedding(y_slabs, y_embed_dim)

        # Fusion (quarter + half + quarter + y_dim → out_channels)
        fusion_in = quarter + half + quarter + y_embed_dim
        norm_groups = min(4, out_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        height_planes: torch.Tensor,  # [B, 5, 16, 16]
        router6: torch.Tensor,  # [B, 6, 16, 16]
        biome_indices: torch.Tensor,  # [B, 16, 16] int64
        y_index: torch.Tensor,  # [B] int64
        output_size: int,  # Target 3D size (1, 2, 4, or 8)
    ) -> torch.Tensor:  # [B, out_channels, D, D, D]
        """Fuse anchor channels into 3D conditioning features at target resolution."""
        B = height_planes.shape[0]

        # Process 2D features
        h_feat = self.height_conv(height_planes)  # [B, quarter, 16, 16]
        r_feat = self.router6_conv(router6)  # [B, half, 16, 16]

        # Biome embedding → Conv2d
        bx = biome_indices.long().clamp(0, self.biome_embedding.num_embeddings - 1)
        biome_emb = self.biome_embedding(bx).permute(0, 3, 1, 2)  # [B, embed_dim, 16, 16]
        b_feat = self.biome_conv(biome_emb)  # [B, quarter, 16, 16]

        # Y-slab embedding → broadcast to spatial dims
        yi = y_index.long().clamp(0, self.y_slabs - 1)
        y_emb = self.y_embedding(yi)  # [B, y_embed_dim]
        y_feat = y_emb.view(B, self.y_embed_dim, 1, 1).expand(-1, -1, 16, 16)

        # Fuse 2D features
        combined_2d = torch.cat([h_feat, r_feat, b_feat, y_feat], dim=1)
        fused_2d = self.fusion(combined_2d)  # [B, out_channels, 16, 16]

        # Resize to output spatial size
        if output_size < 16:
            fused_2d = nn.functional.interpolate(
                fused_2d,
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )  # [B, out_channels, D, D]

        # Broadcast 2D → 3D (replicate across depth dimension)
        fused_3d = fused_2d.unsqueeze(2).expand(
            -1, -1, output_size, -1, -1
        )  # [B, out_channels, D, D, D]

        return fused_3d


class ProgressiveLODModel0_Initial(nn.Module):
    """
    Model 0: Initial terrain generation (Nothing→LOD4: 1×1×1).
    Generates the very first LOD from conditioning inputs only.

    Tiny MLP since it only predicts 1 voxel.
    Inputs: height_planes, router6, biome, y_index (all pooled to scalars)
    Output: [B, N_blocks, 1, 1, 1], [B, 1, 1, 1, 1]
    """

    def __init__(self, config: SimpleFlexibleConfig, output_size: int = 1):
        super().__init__()
        self.config = config
        self.output_size = output_size  # Should be 1 for LOD4

        hidden_dim = 32  # Compact but sufficient

        # Pool 2D inputs to scalars
        self.height_processor = nn.Linear(5, hidden_dim // 4)  # 5→8
        self.router6_processor = nn.Linear(6, hidden_dim // 4)  # 6→8
        self.biome_embedding = nn.Embedding(256, 8)  # mode per column → 8
        self.y_embedding = nn.Embedding(24, 4)  # slab → 4

        # Total fused: 8+8+8+4 = 28 → project to hidden_dim
        self.fuse = nn.Linear(28, hidden_dim)

        # Compact processing network
        self.processor = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Output heads — predict single voxel directly
        self.air_head = nn.Linear(hidden_dim, 1)
        self.block_head = nn.Linear(hidden_dim, config.block_vocab_size)

    def forward(
        self,
        height_planes: torch.Tensor,  # [B, 5, 16, 16]
        router6: torch.Tensor,  # [B, 6, 16, 16]
        biome_indices: torch.Tensor,  # [B, 16, 16] int64
        y_index: torch.Tensor,  # [B] int64
        x_parent: Optional[torch.Tensor] = None,  # unused for init model
    ) -> Dict[str, torch.Tensor]:
        """Predict a single voxel at LOD4 from conditioning inputs."""
        B = height_planes.shape[0]

        # Pool 2D features to scalars
        h_vec = height_planes.mean(dim=(2, 3))  # [B, 5]
        r_vec = router6.mean(dim=(2, 3))  # [B, 6]

        h_feat = self.height_processor(h_vec)  # [B, 8]
        r_feat = self.router6_processor(r_vec)  # [B, 8]

        # Biome: mode of the 16×16 grid → single embedding
        biome_mode = biome_indices[:, 8, 8].long().clamp(0, 255)  # center column
        b_feat = self.biome_embedding(biome_mode)  # [B, 8]

        # Y-slab embedding
        yi = y_index.long().clamp(0, 23)
        y_feat = self.y_embedding(yi)  # [B, 4]

        # Fuse and process
        fused = self.fuse(torch.cat([h_feat, r_feat, b_feat, y_feat], dim=1))  # [B, hidden]
        z = self.processor(fused)  # [B, hidden]

        # Output heads → reshape to 1×1×1
        air_logit = self.air_head(z).view(B, 1, 1, 1, 1)
        block_logits = self.block_head(z).view(B, self.config.block_vocab_size, 1, 1, 1)

        return {"air_mask_logits": air_logit, "block_type_logits": block_logits}


class ProgressiveLODModel(nn.Module):
    """
    Progressive refinement model for a single LOD transition.

    Each instance handles exactly ONE transition (e.g., LOD4→3, LOD3→2, LOD2→1).
    LOD1→LOD0 is NOT needed — vanilla terrain handles LOD0.

    Supported output sizes:
      - 2×2×2  (LOD4→LOD3)
      - 4×4×4  (LOD3→LOD2)
      - 8×8×8  (LOD2→LOD1)

    Inputs:
      - x_parent:       [B, 1, P, P, P] parent occupancy (P = output_size // 2)
      - height_planes:  [B, 5, 16, 16]
      - router6:        [B, 6, 16, 16]
      - biome_indices:  [B, 16, 16] int64
      - y_index:        [B] int64
    """

    def __init__(self, config: SimpleFlexibleConfig, output_size: int):
        super().__init__()
        self.config = config
        self.output_size = output_size

        assert output_size in [
            2,
            4,
            8,
        ], f"output_size must be 2, 4, or 8 (no LOD0); got {output_size}"

        # Conditioning fusion (2D anchors → 3D features)
        self.conditioning = AnchorConditioningFusion3D(
            out_channels=config.base_channels,
            biome_vocab_size=getattr(config, "biome_vocab_size", 256),
        )

        # Parent processing: encode [B, 1, P, P, P] occupancy → features,
        # then upsample to output_size
        parent_channels = config.base_channels
        self.parent_encoder = nn.Sequential(
            nn.Conv3d(1, parent_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, parent_channels), parent_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(parent_channels, parent_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, parent_channels), parent_channels),
            nn.ReLU(inplace=True),
        )

        # Combined: conditioning (base_channels) + parent (base_channels)
        combined_channels = config.base_channels * 2

        # Main processing (scale depth with output_size for capacity scaling)
        if output_size <= 2:
            # Small model for coarse LOD
            self.main_conv = nn.Sequential(
                self._make_layer(combined_channels, config.base_channels),
            )
        elif output_size <= 4:
            # Medium model
            self.main_conv = nn.Sequential(
                self._make_layer(combined_channels, config.base_channels * 2),
                self._make_layer(config.base_channels * 2, config.base_channels),
            )
        else:
            # Medium-large model for LOD2→1
            self.main_conv = nn.Sequential(
                self._make_layer(combined_channels, config.base_channels * 2),
                self._make_layer(config.base_channels * 2, config.base_channels * 2),
                self._make_layer(config.base_channels * 2, config.base_channels),
            )

        # Output heads
        self.air_head = nn.Conv3d(config.base_channels, 1, kernel_size=1)
        self.block_head = nn.Conv3d(config.base_channels, config.block_vocab_size, kernel_size=1)

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """Conv3D block with GroupNorm and ReLU."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        height_planes: torch.Tensor,  # [B, 5, 16, 16]
        router6: torch.Tensor,  # [B, 6, 16, 16]
        biome_indices: torch.Tensor,  # [B, 16, 16] int64
        y_index: torch.Tensor,  # [B] int64
        x_parent: torch.Tensor,  # [B, 1, P, P, P] parent occupancy
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: refine parent voxels into finer output using anchor conditioning.
        """
        # Anchor conditioning → 3D features at output resolution
        cond_features = self.conditioning(
            height_planes, router6, biome_indices, y_index, self.output_size
        )  # [B, base_channels, D, D, D]

        # Encode parent and upsample to output resolution
        parent_features = self.parent_encoder(x_parent)
        if parent_features.shape[-1] != self.output_size:
            parent_features = nn.functional.interpolate(
                parent_features,
                size=(self.output_size, self.output_size, self.output_size),
                mode="trilinear",
                align_corners=False,
            )
        # [B, base_channels, D, D, D]

        # Combine conditioning + parent
        combined = torch.cat([cond_features, parent_features], dim=1)

        # Main processing
        features = self.main_conv(combined)

        # Output heads
        air_mask = self.air_head(features)
        block_logits = self.block_head(features)

        return {"air_mask_logits": air_mask, "block_type_logits": block_logits}


# ── Factory functions for the 4-model family ──────────────────────────────
# (LOD1→LOD0 deliberately absent — vanilla handles LOD0)


def create_init_model(config: SimpleFlexibleConfig) -> ProgressiveLODModel0_Initial:
    """Init: Noise → LOD4 (1×1×1 output, tiny MLP)"""
    return ProgressiveLODModel0_Initial(config, output_size=1)


def create_lod4_to_lod3_model(config: SimpleFlexibleConfig) -> ProgressiveLODModel:
    """LOD4 → LOD3: outputs 2×2×2 (small capacity)"""
    return ProgressiveLODModel(config, output_size=2)


def create_lod3_to_lod2_model(config: SimpleFlexibleConfig) -> ProgressiveLODModel:
    """LOD3 → LOD2: outputs 4×4×4 (medium capacity)"""
    return ProgressiveLODModel(config, output_size=4)


def create_lod2_to_lod1_model(config: SimpleFlexibleConfig) -> ProgressiveLODModel:
    """LOD2 → LOD1: outputs 8×8×8 (medium-large capacity)"""
    return ProgressiveLODModel(config, output_size=8)
