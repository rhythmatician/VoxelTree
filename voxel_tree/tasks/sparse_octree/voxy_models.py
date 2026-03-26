"""Voxy-native per-level models for LODiffusion.

Five independent models (one per Voxy LOD level L4→L0), each producing a 32³
Voxy ``WorldSection`` directly.  Model inputs match vanilla Minecraft's native
noise sampling resolution exactly — no made-up grids, no average pooling.

**Golden Rule — Input Resolution Invariant**:
Never sample any input at higher resolution than vanilla Minecraft.
L0 = exactly vanilla's native grid (all channels, full resolution).
L1–L4 ⊆ vanilla inputs (strict subset of channels, equal or lower resolution).

Noise inputs
------------
All 15 ``RouterField`` channels are sampled on vanilla's native cell grid:
``cellWidth=4`` (4-block spacing XZ), ``cellHeight=8`` (8-block spacing Y).
A Voxy WorldSection at level L covers ``32 × 2^L`` blocks per axis, so the
noise tensor is tiled from ``2^L × 2^L × 2^L`` noise sections:

=====  ==========  ==========================  =========
Level  Footprint   noise_3d shape              Tractable
=====  ==========  ==========================  =========
L0     32³ blk     ``[15, 8, 4, 8]``           Yes
L1     64³ blk     ``[15, 16, 8, 16]``         Yes
L2     128³ blk    ``[15, 32, 16, 32]``        Borderline
L3     256³ blk    ``[15, 64, 32, 64]``        Feature select
L4     512³ blk    ``[15, 128, 64, 128]``      Feature select
=====  ==========  ==========================  =========

L0 and L1 models are fully specified.  L2–L4 models are stubs pending
feature selection analysis.

Architecture
------------
- ``NoiseEncoder3D``: processes noise at its native cell resolution with
  3D convolutions, then trilinearly interpolates to 32³ — mirroring
  vanilla's trilinear interpolation of density cell values.
- ``UNet3D32``: 3-level encoder-decoder with skip connections (from old
  octree/models.py).
- ``OccupancyHead``: spatial octant pooling of the 8³ bottleneck → 8 logits.
- ``ParentEncoder``: Embedding(vocab, dim) → ``[B, dim, 32, 32, 32]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  Feature selection constants (from feature_selection_analysis.py)
# ══════════════════════════════════════════════════════════════════════
#
# At L2+ the 3D noise tensor becomes impractically large.  Data analysis
# shows that (a) nearly all channels are effectively 2D, (b) climate
# channels are extremely spatially smooth, and (c) aquifer/ore channels
# are pure high-frequency noise with no large-scale structure.
#
# See docs/FEATURE_SELECTION_RESULTS.md for the full analysis.

# Channel indices kept at each level (RouterField ordinals)
L2_NOISE_CHANNELS: List[int] = [0, 1, 2, 3, 4, 5, 7]  # 6 climate + final_density
L3_NOISE_CHANNELS: List[int] = [0, 1, 2, 3, 4, 5]      # 6 climate
L4_NOISE_CHANNELS: List[int] = [0, 1, 2, 3, 5]          # 5 climate (drop depth)


# ══════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════


@dataclass
class VoxyModelConfig:
    """Shared configuration for all Voxy-native models.

    Per-level model classes read only the fields they need.
    """

    block_vocab_size: int = 513
    biome_vocab_size: int = 256
    y_vocab_size: int = 24  # section-Y indices (overworld: 0–23)

    # Number of RouterField channels (always 15 for vanilla)
    noise_channels: int = 15

    # Biome embedding dimension (shared across levels)
    biome_embed_dim: int = 8
    # Y-position embedding dimension
    y_embed_dim: int = 8
    # Parent block-ID embedding dimension
    parent_embed_dim: int = 16

    # Heightmap: 5 channels (surface_norm, ocean_floor_approx, slope_x, slope_z, curvature)
    height_channels: int = 5

    # Noise encoder: output channel count after 3D conv processing
    noise_encoder_out: int = 32

    # Climate encoder: output channel count for 2D climate encoder (L2-L4)
    climate_encoder_out: int = 24

    # Per-level U-Net channel widths (C₀, C₁, C₂)
    l0_channels: Tuple[int, int, int] = (48, 96, 192)
    l1_channels: Tuple[int, int, int] = (48, 96, 192)
    l2_channels: Tuple[int, int, int] = (32, 64, 128)
    l3_channels: Tuple[int, int, int] = (24, 48, 96)
    l4_channels: Tuple[int, int, int] = (24, 48, 96)

    # Feature-selected noise channel counts per level
    l2_noise_ch: int = 7   # 6 climate + final_density
    l3_noise_ch: int = 6   # 6 climate
    l4_noise_ch: int = 5   # 5 climate (no depth)

    # Bottleneck extra depth (extra DoubleConv3d blocks at 8³)
    l0_bottleneck_extra: int = 1
    l1_bottleneck_extra: int = 0
    l2_bottleneck_extra: int = 0
    l3_bottleneck_extra: int = 0
    l4_bottleneck_extra: int = 0

    # Biome surface-rule prior (Phase 4)
    surface_prior_dim: int = 8


# ══════════════════════════════════════════════════════════════════════
#  Building blocks
# ══════════════════════════════════════════════════════════════════════


class Conv3dBlock(nn.Module):
    """Conv3d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DoubleConv3d(nn.Module):
    """Two consecutive Conv3d → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Conv3dBlock(in_ch, out_ch),
            Conv3dBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════
#  Noise encoder — processes noise at native cell resolution
# ══════════════════════════════════════════════════════════════════════


class NoiseEncoder3D(nn.Module):
    """Encode vanilla noise at native cell resolution → 32³ feature volume.

    The noise tensor has different spatial dimensions per level (e.g.
    ``[15, 8, 4, 8]`` for L0, ``[15, 16, 8, 16]`` for L1).  This module:

    1. Applies a small 3D CNN at the noise's native resolution to extract
       features (fully convolutional, so any spatial size works).
    2. Trilinearly interpolates the feature map to 32×32×32 — mirroring
       what vanilla does when it interpolates density cell corner values
       to per-block positions.

    The biome tensor (same spatial dims as noise, at quart resolution)
    is embedded and concatenated before processing.
    """

    def __init__(self, cfg: VoxyModelConfig) -> None:
        super().__init__()
        # Biome embedding: map discrete IDs to dense features
        self.biome_embed = nn.Embedding(cfg.biome_vocab_size, cfg.biome_embed_dim)

        # Surface-rule prior (Phase 4): biome → surface type → embedding
        from .biome_priors import NUM_SURFACE_TYPES, biome_to_surface_type_table
        self.register_buffer(
            "_biome_to_surface_type",
            biome_to_surface_type_table(cfg.biome_vocab_size),
            persistent=False,
        )
        self.surface_type_embed = nn.Embedding(NUM_SURFACE_TYPES, cfg.surface_prior_dim)

        in_ch = cfg.noise_channels + cfg.biome_embed_dim + cfg.surface_prior_dim
        mid_ch = cfg.noise_encoder_out
        self.conv = nn.Sequential(
            Conv3dBlock(in_ch, mid_ch),
            Conv3dBlock(mid_ch, mid_ch),
        )

    def forward(
        self,
        noise_3d: torch.Tensor,
        biome_3d: torch.Tensor,
    ) -> torch.Tensor:
        """Encode noise and biome to a 32³ feature volume.

        Args:
            noise_3d: ``[B, 15, Nx, Ny, Nz]`` — all RouterField channels
                at native cell resolution.
            biome_3d: ``[B, Nx, Ny, Nz]`` — biome IDs at same resolution.

        Returns:
            ``[B, noise_encoder_out, 32, 32, 32]`` feature volume.
        """
        # Embed biome IDs: [B, Nx, Ny, Nz] → [B, Nx, Ny, Nz, E] → [B, E, Nx, Ny, Nz]
        ids = biome_3d.long().clamp(0, self.biome_embed.num_embeddings - 1)
        biome_feat = self.biome_embed(ids).permute(0, 4, 1, 2, 3)

        # Surface-rule prior
        _lut = self.get_buffer("_biome_to_surface_type")
        assert _lut is not None
        surface_types = _lut[ids]
        surface_feat = self.surface_type_embed(surface_types).permute(0, 4, 1, 2, 3)

        # Concatenate: [B, 15+E+S, Nx, Ny, Nz]
        x = torch.cat([noise_3d, biome_feat, surface_feat], dim=1)

        # Process at native resolution
        x = self.conv(x)  # [B, out_ch, Nx, Ny, Nz]

        # Trilinear interpolation to 32³ — matches vanilla's interpolation scheme
        x = F.interpolate(x, size=(32, 32, 32), mode="trilinear", align_corners=False)
        return x


# ══════════════════════════════════════════════════════════════════════
#  Heightmap encoder — 2D features broadcast to 3D
# ══════════════════════════════════════════════════════════════════════


class ClimateEncoder2D(nn.Module):
    """Encode 2D climate noise + 2D biome at subsampled resolution → 32³.

    For L2–L4 where feature selection has determined that:
    1. Only a subset of noise channels are useful (climate ± final_density).
    2. All useful channels are effectively 2D (negligible Y variation).
    3. Channels can be spatially subsampled to 8×8 without information loss.

    Pipeline: 2D CNN at 8×8 → bilinear interpolate to 32×32 → broadcast along Y.
    """

    def __init__(
        self,
        noise_channels: int,
        cfg: VoxyModelConfig,
        out_channels: int = 24,
    ) -> None:
        super().__init__()
        self.biome_embed = nn.Embedding(cfg.biome_vocab_size, cfg.biome_embed_dim)

        # Surface-rule prior (Phase 4)
        from .biome_priors import NUM_SURFACE_TYPES, biome_to_surface_type_table
        self.register_buffer(
            "_biome_to_surface_type",
            biome_to_surface_type_table(cfg.biome_vocab_size),
            persistent=False,
        )
        self.surface_type_embed = nn.Embedding(NUM_SURFACE_TYPES, cfg.surface_prior_dim)

        in_ch = noise_channels + cfg.biome_embed_dim + cfg.surface_prior_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        climate_2d: torch.Tensor,
        biome_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Encode 2D climate inputs to 3D volume.

        Args:
            climate_2d: ``[B, C, H, W]`` — selected noise channels, 2D.
            biome_2d:   ``[B, H, W]`` — biome IDs, 2D.

        Returns:
            ``[B, out_channels, 32, 32, 32]`` — broadcast along Y.
        """
        ids = biome_2d.long().clamp(0, self.biome_embed.num_embeddings - 1)
        biome_feat = self.biome_embed(ids).permute(0, 3, 1, 2)  # [B, E, H, W]

        _lut = self.get_buffer("_biome_to_surface_type")
        assert _lut is not None
        surface_types = _lut[ids]
        surface_feat = self.surface_type_embed(surface_types).permute(0, 3, 1, 2)

        x = torch.cat([climate_2d, biome_feat, surface_feat], dim=1)
        x = self.conv(x)  # [B, out_ch, H, W]
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        return x.unsqueeze(2).expand(-1, -1, 32, -1, -1)  # [B, out_ch, 32, 32, 32]


class HeightmapEncoder2D(nn.Module):
    """Encode 2D heightmap features and broadcast to 3D volume.

    The heightmap has varying resolution per level (e.g. ``[5, 8, 8]`` for
    L0, ``[5, 16, 16]`` for L1).  This module:

    1. Processes the 2D heightmap with a small 2D CNN at native resolution.
    2. Bilinearly interpolates to 32×32.
    3. Broadcasts along Y to form ``[B, C, 32, 32, 32]``.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 8) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, heightmap: torch.Tensor) -> torch.Tensor:
        """Encode heightmap to 3D volume.

        Args:
            heightmap: ``[B, 5, Hx, Hz]`` — at level-native resolution.

        Returns:
            ``[B, out_channels, 32, 32, 32]`` — broadcast along Y.
        """
        x = self.conv(heightmap.float())  # [B, C, Hx, Hz]
        # Bilinear to 32×32
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        # Broadcast along Y
        return x.unsqueeze(2).expand(-1, -1, 32, -1, -1)  # [B, C, 32, 32, 32]


# ══════════════════════════════════════════════════════════════════════
#  3D U-Net backbone (adapted from octree/models.py)
# ══════════════════════════════════════════════════════════════════════


class UNet3D32(nn.Module):
    """3-level 3D U-Net on 32³ grids with skip connections.

    ::

        Encoder:
          32³ × in  → DoubleConv → enc1 (C₀) → MaxPool → 16³
          16³ × C₀  → DoubleConv → enc2 (C₁) → MaxPool →  8³
           8³ × C₁  → DoubleConv → bottleneck (C₂)

        Decoder:
           8³ × C₂  → Upsample(2) → cat(enc2) → DoubleConv → dec1 (C₁) → 16³
          16³ × C₁  → Upsample(2) → cat(enc1) → DoubleConv → dec2 (C₀) → 32³

    Returns ``(features_32, bottleneck_8)`` for block head and occ head.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Tuple[int, int, int],
        bottleneck_extra_depth: int = 0,
    ) -> None:
        super().__init__()
        c0, c1, c2 = channels

        # Encoder
        self.enc1 = DoubleConv3d(in_channels, c0)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3d(c0, c1)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3d(c1, c2)

        if bottleneck_extra_depth > 0:
            self.bottleneck_extra: nn.Module | None = nn.Sequential(
                *[DoubleConv3d(c2, c2) for _ in range(bottleneck_extra_depth)]
            )
        else:
            self.bottleneck_extra = None

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = DoubleConv3d(c2 + c1, c1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = DoubleConv3d(c1 + c0, c0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: input ``[B, in, 32³]`` → ``(features [B, C₀, 32³], bottleneck [B, C₂, 8³])``."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))
        if self.bottleneck_extra is not None:
            bn = self.bottleneck_extra(bn)

        d1 = self.dec1(torch.cat([self.up1(bn), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return d2, bn


# ══════════════════════════════════════════════════════════════════════
#  Heads
# ══════════════════════════════════════════════════════════════════════


class OccupancyHead(nn.Module):
    """Spatial octant pooling → shared MLP → 8 child-octant logits.

    Splits the 8³ bottleneck into 2×2×2 octants (4³ sub-volumes each),
    pools each independently, then a shared MLP predicts occupancy.
    Bit convention: ``bit0=X, bit1=Z, bit2=Y`` (Voxy standard).
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """``[B, C, 8, 8, 8]`` → ``[B, 8]``."""
        B, C = bottleneck.shape[:2]
        # Split: [B, C, 2, 4, 2, 4, 2, 4] — (Yh, Yl, Zh, Zl, Xh, Xl)
        x = bottleneck.reshape(B, C, 2, 4, 2, 4, 2, 4)
        x = x.mean(dim=(3, 5, 7))          # pool local → [B, C, 2, 2, 2]
        x = x.reshape(B, C, 8).permute(0, 2, 1)  # [B, 8, C]
        return self.mlp(x).squeeze(-1)      # [B, 8]


# ══════════════════════════════════════════════════════════════════════
#  Parent context encoder
# ══════════════════════════════════════════════════════════════════════


class ParentEncoder(nn.Module):
    """Embed parent block IDs → dense 3D feature volume.

    Input:  ``[B, 32, 32, 32]`` integer block IDs.
    Output: ``[B, embed_dim, 32, 32, 32]``.

    At ONNX export, the embedding stays inside the graph — the model
    accepts raw int64 block IDs.
    """

    def __init__(self, block_vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(block_vocab_size, embed_dim)

    def forward(self, parent_blocks: torch.Tensor) -> torch.Tensor:
        ids = parent_blocks.long().clamp(0, self.embedding.num_embeddings - 1)
        emb = self.embedding(ids)                 # [B, 32, 32, 32, E]
        return emb.permute(0, 4, 1, 2, 3)        # [B, E, 32, 32, 32]


# ══════════════════════════════════════════════════════════════════════
#  Y-position encoder
# ══════════════════════════════════════════════════════════════════════


class YPositionEncoder(nn.Module):
    """Embed scalar y_position and broadcast to 3D volume."""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, y_position: torch.Tensor) -> torch.Tensor:
        """``[B]`` int → ``[B, embed_dim, 32, 32, 32]``."""
        ids = y_position.long().clamp(0, self.embedding.num_embeddings - 1)
        emb = self.embedding(ids)  # [B, E]
        return emb.view(emb.shape[0], -1, 1, 1, 1).expand(-1, -1, 32, 32, 32)


# ══════════════════════════════════════════════════════════════════════
#  VoxyL0Model — finest level, no occupancy head
# ══════════════════════════════════════════════════════════════════════


class VoxyL0Model(nn.Module):
    """Leaf model for Voxy L0 (1 block/voxel, 32³ blocks).

    Inputs:
        - ``noise_3d  [B, 15, 8, 4, 8]``  — 15 RouterField channels at native
          cell resolution (8 cells XZ, 4 cells Y for a 32-block footprint)
        - ``biome_3d  [B, 8, 4, 8]``      — biome IDs at same resolution
        - ``heightmap [B, 5, 8, 8]``       — 5-plane heightmap at 4-block XZ res
        - ``y_position [B]``               — section Y index
        - ``parent_blocks [B, 32, 32, 32]`` — from L1, octant-extracted + upsampled

    Output:
        - ``block_logits [B, V, 32, 32, 32]``
    """

    def __init__(self, cfg: VoxyModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c0, c1, c2 = cfg.l0_channels

        self.noise_encoder = NoiseEncoder3D(cfg)
        self.heightmap_encoder = HeightmapEncoder2D(cfg.height_channels, out_channels=8)
        self.parent_encoder = ParentEncoder(cfg.block_vocab_size, cfg.parent_embed_dim)
        self.y_encoder = YPositionEncoder(cfg.y_vocab_size, cfg.y_embed_dim)

        # U-Net input: noise_enc + heightmap_enc + parent_enc + y_enc
        in_channels = (
            cfg.noise_encoder_out      # 32
            + 8                        # heightmap encoder out
            + cfg.parent_embed_dim     # 16
            + cfg.y_embed_dim          # 8
        )
        self.unet = UNet3D32(in_channels, cfg.l0_channels, cfg.l0_bottleneck_extra)
        self.block_head = nn.Conv3d(c0, cfg.block_vocab_size, kernel_size=1)

    def forward(
        self,
        noise_3d: torch.Tensor,
        biome_3d: torch.Tensor,
        heightmap: torch.Tensor,
        y_position: torch.Tensor,
        parent_blocks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            noise_3d:      ``[B, 15, 8, 4, 8]``
            biome_3d:      ``[B, 8, 4, 8]``
            heightmap:     ``[B, 5, 8, 8]``
            y_position:    ``[B]``
            parent_blocks: ``[B, 32, 32, 32]``

        Returns:
            ``{'block_logits': [B, V, 32, 32, 32]}``
        """
        noise_feat = self.noise_encoder(noise_3d, biome_3d)   # [B, 32, 32, 32, 32]
        hm_feat = self.heightmap_encoder(heightmap)            # [B, 8, 32, 32, 32]
        parent_feat = self.parent_encoder(parent_blocks)       # [B, 16, 32, 32, 32]
        y_feat = self.y_encoder(y_position)                    # [B, 8, 32, 32, 32]

        x = torch.cat([noise_feat, hm_feat, parent_feat, y_feat], dim=1)
        features, _bn = self.unet(x)

        return {"block_logits": self.block_head(features)}


# ══════════════════════════════════════════════════════════════════════
#  VoxyL1Model — with occupancy head
# ══════════════════════════════════════════════════════════════════════


class VoxyL1Model(nn.Module):
    """Refinement model for Voxy L1 (2 blocks/voxel, 64³ blocks).

    Inputs:
        - ``noise_3d  [B, 15, 16, 8, 16]``
        - ``biome_3d  [B, 16, 8, 16]``
        - ``heightmap [B, 5, 16, 16]``
        - ``y_position [B]``
        - ``parent_blocks [B, 32, 32, 32]`` — from L2

    Outputs:
        - ``block_logits [B, V, 32, 32, 32]``
        - ``occ_logits   [B, 8]``
    """

    def __init__(self, cfg: VoxyModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c0, c1, c2 = cfg.l1_channels

        self.noise_encoder = NoiseEncoder3D(cfg)
        self.heightmap_encoder = HeightmapEncoder2D(cfg.height_channels, out_channels=8)
        self.parent_encoder = ParentEncoder(cfg.block_vocab_size, cfg.parent_embed_dim)
        self.y_encoder = YPositionEncoder(cfg.y_vocab_size, cfg.y_embed_dim)

        in_channels = (
            cfg.noise_encoder_out
            + 8
            + cfg.parent_embed_dim
            + cfg.y_embed_dim
        )
        self.unet = UNet3D32(in_channels, cfg.l1_channels, cfg.l1_bottleneck_extra)
        self.block_head = nn.Conv3d(c0, cfg.block_vocab_size, kernel_size=1)
        self.occ_head = OccupancyHead(c2)

    def forward(
        self,
        noise_3d: torch.Tensor,
        biome_3d: torch.Tensor,
        heightmap: torch.Tensor,
        y_position: torch.Tensor,
        parent_blocks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        noise_feat = self.noise_encoder(noise_3d, biome_3d)
        hm_feat = self.heightmap_encoder(heightmap)
        parent_feat = self.parent_encoder(parent_blocks)
        y_feat = self.y_encoder(y_position)

        x = torch.cat([noise_feat, hm_feat, parent_feat, y_feat], dim=1)
        features, bn = self.unet(x)

        return {
            "block_logits": self.block_head(features),
            "occ_logits": self.occ_head(bn),
        }


# ══════════════════════════════════════════════════════════════════════
#  VoxyL2Model — 2D climate inputs (feature-selected)
# ══════════════════════════════════════════════════════════════════════


class VoxyL2Model(nn.Module):
    """Refinement model for Voxy L2 (4 blocks/voxel, 128³ blocks).

    Feature selection analysis determined that at L2 scale:
    - Only 6 climate channels + final_density matter (7 total).
    - All useful channels are effectively 2D (Y-var ratio < 0.02).
    - Spatial subsampling from 32×16×32 → 8×8 loses negligible info.
    - Aquifer and ore channels are pure noise at this scale.

    This reduces the noise input from 245,760 floats to 448 floats (549×).

    Inputs:
        - ``climate_2d    [B, 7, 8, 8]``    — 6 climate + final_density, 2D
        - ``biome_2d      [B, 8, 8]``        — biome IDs at subsampled XZ
        - ``heightmap     [B, 5, 8, 8]``     — 5-plane heightmap
        - ``y_position    [B]``              — section Y index
        - ``parent_blocks [B, 32, 32, 32]``  — from L3

    Outputs:
        - ``block_logits [B, V, 32, 32, 32]``
        - ``occ_logits   [B, 8]``
    """

    def __init__(self, cfg: VoxyModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c0, c1, c2 = cfg.l2_channels

        self.climate_encoder = ClimateEncoder2D(
            cfg.l2_noise_ch, cfg, out_channels=cfg.climate_encoder_out,
        )
        self.heightmap_encoder = HeightmapEncoder2D(cfg.height_channels, out_channels=8)
        self.parent_encoder = ParentEncoder(cfg.block_vocab_size, cfg.parent_embed_dim)
        self.y_encoder = YPositionEncoder(cfg.y_vocab_size, cfg.y_embed_dim)

        in_channels = (
            cfg.climate_encoder_out
            + 8
            + cfg.parent_embed_dim
            + cfg.y_embed_dim
        )
        self.unet = UNet3D32(in_channels, cfg.l2_channels, cfg.l2_bottleneck_extra)
        self.block_head = nn.Conv3d(c0, cfg.block_vocab_size, kernel_size=1)
        self.occ_head = OccupancyHead(c2)

    def forward(
        self,
        climate_2d: torch.Tensor,
        biome_2d: torch.Tensor,
        heightmap: torch.Tensor,
        y_position: torch.Tensor,
        parent_blocks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        climate_feat = self.climate_encoder(climate_2d, biome_2d)
        hm_feat = self.heightmap_encoder(heightmap)
        parent_feat = self.parent_encoder(parent_blocks)
        y_feat = self.y_encoder(y_position)

        x = torch.cat([climate_feat, hm_feat, parent_feat, y_feat], dim=1)
        features, bn = self.unet(x)

        return {
            "block_logits": self.block_head(features),
            "occ_logits": self.occ_head(bn),
        }


# ══════════════════════════════════════════════════════════════════════
#  VoxyL3Model — 2D climate inputs (feature-selected)
# ══════════════════════════════════════════════════════════════════════


class VoxyL3Model(nn.Module):
    """Refinement model for Voxy L3 (8 blocks/voxel, 256³ blocks).

    Feature selection analysis determined that at L3 scale:
    - Only 6 climate channels matter (temperature through ridges).
    - Density fields become noise (between/within ratio = 1.5).
    - All useful channels are 2D, subsampled to 8×8.

    This reduces the noise input from 1,966,080 floats to 384 (5,120×).

    Inputs:
        - ``climate_2d    [B, 6, 8, 8]``    — 6 climate channels, 2D
        - ``biome_2d      [B, 8, 8]``        — biome IDs at subsampled XZ
        - ``heightmap     [B, 5, 8, 8]``     — 5-plane heightmap
        - ``y_position    [B]``              — section Y index
        - ``parent_blocks [B, 32, 32, 32]``  — from L4

    Outputs:
        - ``block_logits [B, V, 32, 32, 32]``
        - ``occ_logits   [B, 8]``
    """

    def __init__(self, cfg: VoxyModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c0, c1, c2 = cfg.l3_channels

        self.climate_encoder = ClimateEncoder2D(
            cfg.l3_noise_ch, cfg, out_channels=cfg.climate_encoder_out,
        )
        self.heightmap_encoder = HeightmapEncoder2D(cfg.height_channels, out_channels=8)
        self.parent_encoder = ParentEncoder(cfg.block_vocab_size, cfg.parent_embed_dim)
        self.y_encoder = YPositionEncoder(cfg.y_vocab_size, cfg.y_embed_dim)

        in_channels = (
            cfg.climate_encoder_out
            + 8
            + cfg.parent_embed_dim
            + cfg.y_embed_dim
        )
        self.unet = UNet3D32(in_channels, cfg.l3_channels, cfg.l3_bottleneck_extra)
        self.block_head = nn.Conv3d(c0, cfg.block_vocab_size, kernel_size=1)
        self.occ_head = OccupancyHead(c2)

    def forward(
        self,
        climate_2d: torch.Tensor,
        biome_2d: torch.Tensor,
        heightmap: torch.Tensor,
        y_position: torch.Tensor,
        parent_blocks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        climate_feat = self.climate_encoder(climate_2d, biome_2d)
        hm_feat = self.heightmap_encoder(heightmap)
        parent_feat = self.parent_encoder(parent_blocks)
        y_feat = self.y_encoder(y_position)

        x = torch.cat([climate_feat, hm_feat, parent_feat, y_feat], dim=1)
        features, bn = self.unet(x)

        return {
            "block_logits": self.block_head(features),
            "occ_logits": self.occ_head(bn),
        }


# ══════════════════════════════════════════════════════════════════════
#  VoxyL4Model — root level, 2D climate, no parent, 24 Y voxels
# ══════════════════════════════════════════════════════════════════════


class VoxyL4Model(nn.Module):
    """Root model for Voxy L4 (16 blocks/voxel, 512³ blocks).

    No parent context — generates from conditioning only.  Output Y
    dimension is 24 (not 32) because at 16m/voxel the MC world height
    (384 blocks = 24 voxels) does not fill the full 32-voxel section.

    Feature selection analysis determined that at L4 scale:
    - Only 5 climate channels matter (drop depth — redundant with
      heightmap + y_position at 16-block resolution).
    - All channels are 2D, subsampled to 8×8.

    This reduces the noise input from 15,728,640 floats to 320 (49,152×).

    Inputs:
        - ``climate_2d [B, 5, 8, 8]``  — 5 climate channels, 2D
        - ``biome_2d   [B, 8, 8]``      — biome IDs at subsampled XZ
        - ``heightmap  [B, 5, 8, 8]``   — 5-plane heightmap
        - ``y_position [B]``            — section Y index

    Outputs:
        - ``block_logits [B, V, 32, 32, 24]``
        - ``occ_logits   [B, 8]``
    """

    def __init__(self, cfg: VoxyModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c0, c1, c2 = cfg.l4_channels

        self.climate_encoder = ClimateEncoder2D(
            cfg.l4_noise_ch, cfg, out_channels=cfg.climate_encoder_out,
        )
        self.heightmap_encoder = HeightmapEncoder2D(cfg.height_channels, out_channels=8)
        self.y_encoder = YPositionEncoder(cfg.y_vocab_size, cfg.y_embed_dim)

        # No parent encoder — L4 is root
        in_channels = (
            cfg.climate_encoder_out
            + 8
            + cfg.y_embed_dim
        )
        self.unet = UNet3D32(in_channels, cfg.l4_channels, cfg.l4_bottleneck_extra)
        self.block_head = nn.Conv3d(c0, cfg.block_vocab_size, kernel_size=1)
        self.occ_head = OccupancyHead(c2)

    def forward(
        self,
        climate_2d: torch.Tensor,
        biome_2d: torch.Tensor,
        heightmap: torch.Tensor,
        y_position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        climate_feat = self.climate_encoder(climate_2d, biome_2d)
        hm_feat = self.heightmap_encoder(heightmap)
        y_feat = self.y_encoder(y_position)

        x = torch.cat([climate_feat, hm_feat, y_feat], dim=1)
        features, bn = self.unet(x)

        # Trim Y to 24 — MC world height at 16m/voxel
        # Volume convention: [B, V, Y, Z, X] — Y is dim 2
        block_logits = self.block_head(features)[:, :, :24, :, :]

        return {
            "block_logits": block_logits,
            "occ_logits": self.occ_head(bn),
        }


# ══════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════

# Level → model class
LEVEL_MODEL_CLASSES = {
    0: VoxyL0Model,
    1: VoxyL1Model,
    2: VoxyL2Model,
    3: VoxyL3Model,
    4: VoxyL4Model,
}


def create_model(level: int, cfg: Optional[VoxyModelConfig] = None) -> nn.Module:
    """Create the Voxy model for a given LOD level.

    Args:
        level: Voxy LOD level (0–4).
        cfg: Model configuration; uses defaults if ``None``.

    Returns:
        The model instance (not initialized with weights).
    """
    if cfg is None:
        cfg = VoxyModelConfig()
    cls = LEVEL_MODEL_CLASSES[level]
    return cls(cfg)


# ══════════════════════════════════════════════════════════════════════
#  Input tensor shapes per level
# ══════════════════════════════════════════════════════════════════════

# L0/L1: 3D noise at full native cell resolution
# L2-L4: 2D feature-selected climate at subsampled 8×8  (see FEATURE_SELECTION_RESULTS.md)

NOISE_SHAPES = {
    0: (15, 8, 4, 8),    # 3D: full native, 32 blk footprint
    1: (15, 16, 8, 16),  # 3D: full native, 64 blk footprint
    2: (7, 8, 8),        # 2D: 6 climate + final_density, subsampled
    3: (6, 8, 8),        # 2D: 6 climate, subsampled
    4: (5, 8, 8),        # 2D: 5 climate (no depth), subsampled
}

BIOME_SHAPES = {
    0: (8, 4, 8),     # 3D: quart-cell resolution
    1: (16, 8, 16),   # 3D: quart-cell resolution
    2: (8, 8),        # 2D: subsampled XZ
    3: (8, 8),        # 2D: subsampled XZ
    4: (8, 8),        # 2D: subsampled XZ
}

HEIGHTMAP_SHAPES = {
    0: (5, 8, 8),
    1: (5, 16, 16),
    2: (5, 8, 8),     # subsampled to match climate grid
    3: (5, 8, 8),
    4: (5, 8, 8),
}


__all__ = [
    "VoxyModelConfig",
    "VoxyL0Model",
    "VoxyL1Model",
    "VoxyL2Model",
    "VoxyL3Model",
    "VoxyL4Model",
    "NoiseEncoder3D",
    "ClimateEncoder2D",
    "HeightmapEncoder2D",
    "UNet3D32",
    "OccupancyHead",
    "ParentEncoder",
    "YPositionEncoder",
    "create_model",
    "LEVEL_MODEL_CLASSES",
    "NOISE_SHAPES",
    "BIOME_SHAPES",
    "HEIGHTMAP_SHAPES",
    "L2_NOISE_CHANNELS",
    "L3_NOISE_CHANNELS",
    "L4_NOISE_CHANNELS",
]
