"""Octree generation models — 3-model architecture for 32³ WorldSections.

Three 3D U-Net models operating on 32³ spatial grids, aligned with Voxy's
WorldSection hierarchy:

- **Model A** ``OctreeInitModel``  (L4 only):  No parent context, root of octree.
- **Model B** ``OctreeRefineModel`` (L3, L2, L1 shared):  Parent context + level embedding.
- **Model C** ``OctreeLeafModel``  (L0 only):  Parent context, no occupancy head.

All models produce ``block_type_logits`` float[B, V, 32, 32, 32].  Models A and B
additionally produce ``occ_logits`` float[B, 8] predicting which child octants
are non-empty (for octree pruning).

Channel widths are designed as configurable defaults pending shootout experiments::

    Init:   C₀=24,  C₁=48,  C₂=96   (~400K params)
    Refine: C₀=32,  C₁=64,  C₂=128  (~800K params)
    Leaf:   C₀=48,  C₁=96,  C₂=192  (~1.8M params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class OctreeConfig:
    """Configuration for the 3-model octree architecture.

    Serialised into checkpoints so export can reconstruct models without
    an external config file.
    """

    block_vocab_size: int = 1104
    biome_vocab_size: int = 256
    y_vocab_size: int = 24
    level_vocab_size: int = 5  # levels 0-4

    # Per-model channel widths  (C₀, C₁, C₂)
    init_channels: Tuple[int, int, int] = (24, 48, 96)
    refine_channels: Tuple[int, int, int] = (32, 64, 128)
    leaf_channels: Tuple[int, int, int] = (48, 96, 192)

    # Conditioning embedding dimensions
    parent_embed_dim: int = 16
    biome_embed_dim: int = 8
    y_embed_dim: int = 8
    level_embed_dim: int = 4

    # Heightmap channels (surface, ocean_floor, slope_x, slope_z, curvature)
    height_channels: int = 5


# ── Building blocks ───────────────────────────────────────────────────


class Conv3dBlock(nn.Module):
    """Conv3d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> None:
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


# ── 3D U-Net backbone ─────────────────────────────────────────────────


class UNet3D32(nn.Module):
    """3-level 3D U-Net on 32³ grids with skip connections.

    Architecture::

        Encoder:
          32³ × in  → DoubleConv → enc1 (C₀) → MaxPool → 16³
          16³ × C₀  → DoubleConv → enc2 (C₁) → MaxPool →  8³
           8³ × C₁  → DoubleConv → bottleneck (C₂)

        Decoder:
           8³ × C₂  → Upsample(2) → cat(enc2) → DoubleConv → dec1 (C₁) → 16³
          16³ × C₁  → Upsample(2) → cat(enc1) → DoubleConv → dec2 (C₀) → 32³

    Returns ``(features_32, bottleneck_8)`` so callers can attach both a
    block head (on features_32) and an occupancy head (on bottleneck_8).

    Args:
        in_channels: Number of input channels (conditioning + parent).
        channels: ``(C₀, C₁, C₂)`` — widths at each encoder level.
    """

    def __init__(self, in_channels: int, channels: Tuple[int, int, int]) -> None:
        super().__init__()
        c0, c1, c2 = channels

        # Encoder
        self.enc1 = DoubleConv3d(in_channels, c0)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3d(c0, c1)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3d(c1, c2)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = DoubleConv3d(c2 + c1, c1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = DoubleConv3d(c1 + c0, c0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor ``[B, in_channels, 32, 32, 32]``.

        Returns:
            ``(features, bottleneck)`` where features is ``[B, C₀, 32, 32, 32]``
            and bottleneck is ``[B, C₂, 8, 8, 8]``.
        """
        # Encoder
        e1 = self.enc1(x)  # [B, C₀, 32, 32, 32]
        e2 = self.enc2(self.pool1(e1))  # [B, C₁, 16, 16, 16]
        bn = self.bottleneck(self.pool2(e2))  # [B, C₂, 8, 8, 8]

        # Decoder
        d1 = self.dec1(torch.cat([self.up1(bn), e2], dim=1))  # [B, C₁, 16, 16, 16]
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))  # [B, C₀, 32, 32, 32]

        return d2, bn


# ── Heads ──────────────────────────────────────────────────────────────


class OccupancyHead(nn.Module):
    """Spatial octant pooling → shared MLP → 8 child-octant logits.

    Splits the 8³ bottleneck into eight 4³ sub-volumes (one per child
    octant, using the Voxy bit convention: ``bit0=X, bit1=Z, bit2=Y``)
    and average-pools each independently.  A shared MLP then predicts
    a single occupancy logit per octant.

    This is inspired by OGN's per-node classification: each octant
    prediction is made from its own spatial features rather than from
    a global average pool that discards spatial locality.

    Architecture::

        bottleneck [B, C₂, 8, 8, 8]
          → reshape [B, C₂, 2,4, 2,4, 2,4]  # split Y,Z,X into half+local
          → mean over local dims → [B, C₂, 2, 2, 2]
          → reshape [B, C₂, 8] → permute [B, 8, C₂]
          → shared Linear(C₂, 32) → ReLU → Linear(32, 1)
          → squeeze → [B, 8]

    Used by init and refine models to predict which child octants are
    non-empty, enabling octree pruning at inference.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Predict 8 child-octant occupancy logits.

        Args:
            bottleneck: ``[B, C₂, 8, 8, 8]`` bottleneck features.

        Returns:
            ``[B, 8]`` occupancy logits (apply sigmoid/BCE at loss time).
        """
        B, C = bottleneck.shape[:2]
        # Split each spatial axis into (half=2, local=4):
        #   [B, C, Y=8, Z=8, X=8] → [B, C, Yh, Yl, Zh, Zl, Xh, Xl]
        x = bottleneck.reshape(B, C, 2, 4, 2, 4, 2, 4)
        # Pool over local dims (Yl=3, Zl=5, Xl=7) → [B, C, 2, 2, 2]
        x = x.mean(dim=(3, 5, 7))
        # Flatten octant grid → [B, C, 8]  (Y-major order matches
        # bit0=X, bit1=Z, bit2=Y convention automatically)
        x = x.reshape(B, C, 8).permute(0, 2, 1)  # [B, 8, C]
        logits = self.mlp(x)  # [B, 8, 1]
        return logits.squeeze(-1)  # [B, 8]


# ── Parent context encoder ────────────────────────────────────────────


class ParentEncoder(nn.Module):
    """Embed parent block IDs → dense channel features.

    Takes integer block-ID volumes ``[B, 32, 32, 32]`` (from the parent
    WorldSection's argmax predictions, octant-extracted and upsampled)
    and produces ``[B, embed_dim, 32, 32, 32]`` via a learned
    ``Embedding(block_vocab_size, embed_dim)`` lookup.

    At ONNX export time the embedding weights are exported separately so
    the Java runtime can pre-compute the lookup; the ONNX graph receives
    pre-embedded ``float32[B, embed_dim, 32, 32, 32]`` instead.
    """

    def __init__(self, block_vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(block_vocab_size, embed_dim)

    def forward(self, parent_blocks: torch.Tensor) -> torch.Tensor:
        """Embed parent block IDs.

        Args:
            parent_blocks: ``[B, 32, 32, 32]`` integer block IDs.

        Returns:
            ``[B, embed_dim, 32, 32, 32]`` dense features.
        """
        # clamp to valid range for safety
        ids = parent_blocks.long().clamp(0, self.embedding.num_embeddings - 1)
        emb = self.embedding(ids)  # [B, 32, 32, 32, embed_dim]
        return emb.permute(0, 4, 1, 2, 3)  # [B, embed_dim, 32, 32, 32]


# ── Conditioning helpers ──────────────────────────────────────────────


def _build_height_3d(heightmap: torch.Tensor) -> torch.Tensor:
    """Expand 2D heightmap to 3D by broadcasting along Y.

    Args:
        heightmap: ``[B, 5, 32, 32]`` float.

    Returns:
        ``[B, 5, 32, 32, 32]`` — repeated along dim 2 (Y axis).
    """
    return heightmap.unsqueeze(2).expand(-1, -1, 32, -1, -1)


def _build_biome_3d(biome: torch.Tensor, embed: nn.Embedding) -> torch.Tensor:
    """Embed biome indices and expand to 3D.

    Args:
        biome: ``[B, 32, 32]`` integer biome IDs.
        embed: ``Embedding(biome_vocab, biome_dim)``.

    Returns:
        ``[B, biome_dim, 32, 32, 32]``.
    """
    ids = biome.long().clamp(0, embed.num_embeddings - 1)
    emb = embed(ids)  # [B, 32, 32, biome_dim]
    emb = emb.permute(0, 3, 1, 2)  # [B, biome_dim, 32, 32]
    return emb.unsqueeze(2).expand(-1, -1, 32, -1, -1)


def _build_scalar_3d(value: torch.Tensor, embed: nn.Embedding) -> torch.Tensor:
    """Embed a scalar index and broadcast to 3D.

    Args:
        value: ``[B]`` integer index.
        embed: ``Embedding(vocab, dim)``.

    Returns:
        ``[B, dim, 32, 32, 32]``.
    """
    ids = value.long().clamp(0, embed.num_embeddings - 1)
    emb = embed(ids)  # [B, dim]
    return emb.view(emb.shape[0], -1, 1, 1, 1).expand(-1, -1, 32, 32, 32)


# ── Model A: OctreeInitModel (L4 — root, no parent) ──────────────────


class OctreeInitModel(nn.Module):
    """Root octree model for L4 WorldSections.

    No parent context — generates from conditioning inputs only
    (heightmap, biome, y_position).

    Returns ``block_type_logits`` and ``occ_logits``.

    Args:
        config: :class:`OctreeConfig` with vocabulary sizes and channel widths.
    """

    def __init__(self, config: OctreeConfig) -> None:
        super().__init__()
        self.config = config
        c0, c1, c2 = config.init_channels

        # Conditioning embeddings
        self.biome_embed = nn.Embedding(config.biome_vocab_size, config.biome_embed_dim)
        self.y_embed = nn.Embedding(config.y_vocab_size, config.y_embed_dim)

        # U-Net
        in_channels = config.height_channels + config.biome_embed_dim + config.y_embed_dim
        self.unet = UNet3D32(in_channels, config.init_channels)

        # Heads
        self.block_head = nn.Conv3d(c0, config.block_vocab_size, kernel_size=1)
        self.occ_head = OccupancyHead(c2)

    def forward(
        self,
        heightmap: torch.Tensor,
        biome: torch.Tensor,
        y_position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            heightmap: ``[B, 5, 32, 32]`` float — surface, ocean, slopes, curvature.
            biome:     ``[B, 32, 32]`` int — biome IDs.
            y_position: ``[B]`` int — section Y index.

        Returns:
            Dict with ``block_type_logits`` ``[B, V, 32, 32, 32]``
            and ``occ_logits`` ``[B, 8]``.
        """
        # Build 3D conditioning
        h3d = _build_height_3d(heightmap.float())
        b3d = _build_biome_3d(biome, self.biome_embed)
        y3d = _build_scalar_3d(y_position, self.y_embed)

        x = torch.cat([h3d, b3d, y3d], dim=1)  # [B, in_ch, 32, 32, 32]

        features, bottleneck = self.unet(x)

        return {
            "block_type_logits": self.block_head(features),
            "occ_logits": self.occ_head(bottleneck),
        }


# ── Model B: OctreeRefineModel (shared L3, L2, L1) ───────────────────


class OctreeRefineModel(nn.Module):
    """Shared refinement model for L3/L2/L1 WorldSections.

    Takes parent context (block-ID embedding or pre-embedded float) plus
    a level embedding telling the model which LOD scale it operates at.

    Returns ``block_type_logits`` and ``occ_logits``.

    Args:
        config: :class:`OctreeConfig` with vocabulary sizes and channel widths.
    """

    def __init__(self, config: OctreeConfig) -> None:
        super().__init__()
        self.config = config
        c0, c1, c2 = config.refine_channels

        # Parent embedding (used during training; bypassed in ONNX)
        self.parent_encoder = ParentEncoder(config.block_vocab_size, config.parent_embed_dim)

        # Conditioning embeddings
        self.biome_embed = nn.Embedding(config.biome_vocab_size, config.biome_embed_dim)
        self.y_embed = nn.Embedding(config.y_vocab_size, config.y_embed_dim)
        self.level_embed = nn.Embedding(config.level_vocab_size, config.level_embed_dim)

        # U-Net
        in_channels = (
            config.parent_embed_dim
            + config.height_channels
            + config.biome_embed_dim
            + config.y_embed_dim
            + config.level_embed_dim
        )
        self.unet = UNet3D32(in_channels, config.refine_channels)

        # Heads
        self.block_head = nn.Conv3d(c0, config.block_vocab_size, kernel_size=1)
        self.occ_head = OccupancyHead(c2)

    def forward(
        self,
        heightmap: torch.Tensor,
        biome: torch.Tensor,
        y_position: torch.Tensor,
        level: torch.Tensor,
        parent_blocks: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Supply **either** ``parent_blocks`` (int, for training) or
        ``parent_context`` (float, for ONNX inference).  If both are
        ``None``, raises ``ValueError``.

        Args:
            heightmap:      ``[B, 5, 32, 32]`` float.
            biome:          ``[B, 32, 32]`` int.
            y_position:     ``[B]`` int.
            level:          ``[B]`` int — LOD level (1, 2, or 3).
            parent_blocks:  ``[B, 32, 32, 32]`` int block IDs (training path).
            parent_context: ``[B, C_parent, 32, 32, 32]`` float (ONNX path).

        Returns:
            Dict with ``block_type_logits`` ``[B, V, 32, 32, 32]``
            and ``occ_logits`` ``[B, 8]``.
        """
        # Resolve parent context
        if parent_context is None:
            if parent_blocks is None:
                raise ValueError("Must supply either parent_blocks or parent_context")
            parent_context = self.parent_encoder(parent_blocks)

        # Build 3D conditioning
        h3d = _build_height_3d(heightmap.float())
        b3d = _build_biome_3d(biome, self.biome_embed)
        y3d = _build_scalar_3d(y_position, self.y_embed)
        l3d = _build_scalar_3d(level, self.level_embed)

        assert parent_context is not None  # guaranteed by check above
        x = torch.cat([parent_context, h3d, b3d, y3d, l3d], dim=1)

        features, bottleneck = self.unet(x)

        return {
            "block_type_logits": self.block_head(features),
            "occ_logits": self.occ_head(bottleneck),
        }


# ── Model C: OctreeLeafModel (L0 — full resolution, no occ head) ─────


class OctreeLeafModel(nn.Module):
    """Leaf model for L0 (block-level) WorldSections.

    Takes parent context but does **not** produce occupancy logits
    (L0 sections have no children to prune).

    Args:
        config: :class:`OctreeConfig` with vocabulary sizes and channel widths.
    """

    def __init__(self, config: OctreeConfig) -> None:
        super().__init__()
        self.config = config
        c0, c1, c2 = config.leaf_channels

        # Parent embedding
        self.parent_encoder = ParentEncoder(config.block_vocab_size, config.parent_embed_dim)

        # Conditioning embeddings
        self.biome_embed = nn.Embedding(config.biome_vocab_size, config.biome_embed_dim)
        self.y_embed = nn.Embedding(config.y_vocab_size, config.y_embed_dim)

        # U-Net
        in_channels = (
            config.parent_embed_dim
            + config.height_channels
            + config.biome_embed_dim
            + config.y_embed_dim
        )
        self.unet = UNet3D32(in_channels, config.leaf_channels)

        # Block head only — no occupancy head for L0
        self.block_head = nn.Conv3d(c0, config.block_vocab_size, kernel_size=1)

    def forward(
        self,
        heightmap: torch.Tensor,
        biome: torch.Tensor,
        y_position: torch.Tensor,
        parent_blocks: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            heightmap:      ``[B, 5, 32, 32]`` float.
            biome:          ``[B, 32, 32]`` int.
            y_position:     ``[B]`` int.
            parent_blocks:  ``[B, 32, 32, 32]`` int block IDs (training path).
            parent_context: ``[B, C_parent, 32, 32, 32]`` float (ONNX path).

        Returns:
            Dict with ``block_type_logits`` ``[B, V, 32, 32, 32]`` only.
        """
        if parent_context is None:
            if parent_blocks is None:
                raise ValueError("Must supply either parent_blocks or parent_context")
            parent_context = self.parent_encoder(parent_blocks)

        h3d = _build_height_3d(heightmap.float())
        b3d = _build_biome_3d(biome, self.biome_embed)
        y3d = _build_scalar_3d(y_position, self.y_embed)

        assert parent_context is not None  # guaranteed by check above
        x = torch.cat([parent_context, h3d, b3d, y3d], dim=1)

        features, _ = self.unet(x)

        return {"block_type_logits": self.block_head(features)}


# ── Factory functions ─────────────────────────────────────────────────


def create_init_model(config: Optional[OctreeConfig] = None) -> OctreeInitModel:
    """Create Model A — octree root (L4).

    Args:
        config: Model configuration.  Uses defaults if ``None``.
    """
    if config is None:
        config = OctreeConfig()
    return OctreeInitModel(config)


def create_refine_model(config: Optional[OctreeConfig] = None) -> OctreeRefineModel:
    """Create Model B — shared refinement (L3/L2/L1).

    Args:
        config: Model configuration.  Uses defaults if ``None``.
    """
    if config is None:
        config = OctreeConfig()
    return OctreeRefineModel(config)


def create_leaf_model(config: Optional[OctreeConfig] = None) -> OctreeLeafModel:
    """Create Model C — leaf (L0, block-level resolution).

    Args:
        config: Model configuration.  Uses defaults if ``None``.
    """
    if config is None:
        config = OctreeConfig()
    return OctreeLeafModel(config)
