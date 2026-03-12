"""Octree generation models — 3-model architecture for 32³ WorldSections.

Model choices are now driven by the notebook shootouts:

- **Model A** ``OctreeInitModel``  (L4 only): defaults to a 2D encoder → 3D
    decoder backbone, which outperformed the old all-3D baseline in the init
    shootout while staying much faster on CPU.
- **Model B** ``OctreeRefineModel`` (L3, L2, L1 shared): parent context +
    level embedding, still using the canonical 3D U-Net path while the refine
    shootout continues.
- **Model C** ``OctreeLeafModel``  (L0 only): parent context, no occupancy
    head, with a configurable deeper bottleneck for the leaf-level path.

All models produce ``block_type_logits`` float[B, V, 32, 32, 32]. Models A and B
additionally produce ``occ_logits`` float[B, 8] predicting which child octants
are non-empty (for octree pruning).
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

    # Architecture selections from notebook shootouts.
    # Old checkpoints won't have these fields, so model code uses getattr()
    # fallbacks when loading historical runs.
    init_architecture: str = "encoder2d_decoder3d"

    # Conditioning embedding dimensions
    parent_embed_dim: int = 16
    biome_embed_dim: int = 8
    y_embed_dim: int = 8
    level_embed_dim: int = 4

    # Heightmap channels (surface, ocean_floor, slope_x, slope_z, curvature)
    height_channels: int = 5

    # Parent-context ablation mode:
    #   "embed"    — learned embedding lookup (default, production path)
    #   "zeros"    — replace parent features with zeros (same channel count)
    #   "disabled" — remove parent channels entirely (smaller U-Net input)
    parent_context_mode: str = "embed"

    # ── Step 8: OGN-inspired targeted improvements ────────────────────
    # All default to off (0 / False) so existing checkpoints are unaffected.

    # A. Stronger refine bottleneck: extra DoubleConv3d blocks at 8³.
    #    OGN applies multiple convolutions per octree level — deeper
    #    bottleneck gives the occupancy head richer features to work with.
    bottleneck_extra_depth: int = 0

    # Per-model bottleneck overrides. ``None`` means "use the legacy global
    # bottleneck_extra_depth value" so older checkpoints remain compatible.
    init_bottleneck_extra_depth: Optional[int] = None
    refine_bottleneck_extra_depth: Optional[int] = None
    leaf_bottleneck_extra_depth: Optional[int] = 1

    # B. Parent feature refiner: a Conv3dBlock after the embedding lookup
    #    so the model can learn local spatial correlations from parent
    #    block IDs before concatenation.
    parent_refine_conv: bool = False

    # D. Occupancy-gated feature modulation: sigmoid(occ_logits) produces
    #    per-octant attention weights broadcast to 8³ and multiplicatively
    #    gate the bottleneck before decoding.  OGN-inspired: the propagation
    #    decision feeds back into how features are expanded to children.
    use_occ_gate: bool = False


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


class Conv2dBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DoubleConv2d(nn.Module):
    """Two consecutive Conv2d → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Conv2dBlock(in_ch, out_ch),
            Conv2dBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _get_model_bottleneck_extra_depth(config: OctreeConfig, model_name: str) -> int:
    """Resolve per-model bottleneck depth with legacy fallback support."""

    specific = getattr(config, f"{model_name}_bottleneck_extra_depth", None)
    if specific is not None:
        return int(specific)
    return int(getattr(config, "bottleneck_extra_depth", 0))


class InitEncoder2DDecoder3D(nn.Module):
    """Init-model backbone from the A3 notebook winner.

    This architecture treats L4 conditioning as fundamentally 2D:
    heightmap + biome are encoded on the XZ plane, then expanded into a 3D
    bottleneck and decoded back to 32³ block features.

    Returns ``(features_32, bottleneck_8)`` so the caller can attach the
    block head and optional occupancy-gating path just like the 3D U-Net.
    """

    def __init__(
        self,
        height_channels: int,
        biome_vocab_size: int,
        biome_embed_dim: int,
        y_vocab_size: int,
        y_embed_dim: int,
        channels: Tuple[int, int, int],
        bottleneck_extra_depth: int = 0,
    ) -> None:
        super().__init__()
        c0, c1, c2 = channels

        self.biome_embed = nn.Embedding(biome_vocab_size, biome_embed_dim)
        self.y_embed = nn.Embedding(y_vocab_size, y_embed_dim)

        in_2d = height_channels + biome_embed_dim
        self.enc2d = nn.Sequential(
            DoubleConv2d(in_2d, c0),
            nn.MaxPool2d(2),
            DoubleConv2d(c0, c1),
            nn.MaxPool2d(2),
            DoubleConv2d(c1, c2),
            nn.MaxPool2d(2),
        )
        self.y_proj = nn.Linear(y_embed_dim, c2)

        self.expand = nn.Sequential(
            nn.ConvTranspose3d(c2, c2, kernel_size=2, stride=2),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = DoubleConv3d(c2, c2)
        if bottleneck_extra_depth > 0:
            self.bottleneck_extra = nn.Sequential(
                *[DoubleConv3d(c2, c2) for _ in range(bottleneck_extra_depth)]
            )
        else:
            self.bottleneck_extra = None  # type: ignore[assignment]

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = DoubleConv3d(c2, c1)
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = DoubleConv3d(c1, c0)

    def encode(
        self,
        heightmap: torch.Tensor,
        biome: torch.Tensor,
        y_position: torch.Tensor,
    ) -> torch.Tensor:
        """Encode 2D conditioning into an 8³ bottleneck volume."""
        batch_size = heightmap.shape[0]
        biome_ids = biome.long().clamp(0, self.biome_embed.num_embeddings - 1)
        biome_feat = self.biome_embed(biome_ids).permute(0, 3, 1, 2)
        feat_2d = self.enc2d(torch.cat([heightmap.float(), biome_feat], dim=1))

        y_ids = y_position.long().clamp(0, self.y_embed.num_embeddings - 1)
        y_feat = self.y_proj(self.y_embed(y_ids)).view(batch_size, -1, 1, 1, 1)

        bottleneck = feat_2d.unsqueeze(2).expand(-1, -1, 4, -1, -1)
        bottleneck = self.expand(bottleneck + y_feat)
        bottleneck = self.bottleneck(bottleneck)
        if self.bottleneck_extra is not None:
            bottleneck = self.bottleneck_extra(bottleneck)
        return bottleneck

    def decode(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Decode an 8³ bottleneck volume back to 32³ features."""
        d2 = self.dec2(self.up2(bottleneck))
        d1 = self.dec1(self.up1(d2))
        return d1

    def forward(
        self,
        heightmap: torch.Tensor,
        biome: torch.Tensor,
        y_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.encode(heightmap, biome, y_position)
        return self.decode(bottleneck), bottleneck


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

        # Extra bottleneck depth (OGN-inspired: multiple convolutions per level)
        if bottleneck_extra_depth > 0:
            self.bottleneck_extra = nn.Sequential(
                *[DoubleConv3d(c2, c2) for _ in range(bottleneck_extra_depth)]
            )
        else:
            self.bottleneck_extra = None  # type: ignore[assignment]

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = DoubleConv3d(c2 + c1, c1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = DoubleConv3d(c1 + c0, c0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder path only.

        Args:
            x: ``[B, in_channels, 32, 32, 32]``

        Returns:
            ``(e1, e2, bottleneck)`` — skip features and bottleneck.
        """
        e1 = self.enc1(x)  # [B, C₀, 32, 32, 32]
        e2 = self.enc2(self.pool1(e1))  # [B, C₁, 16, 16, 16]
        bn = self.bottleneck(self.pool2(e2))  # [B, C₂, 8, 8, 8]
        if self.bottleneck_extra is not None:
            bn = self.bottleneck_extra(bn)
        return e1, e2, bn

    def decode(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        bn: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder path only, using pre-computed skip features.

        Args:
            e1: ``[B, C₀, 32, 32, 32]`` — level-1 skip.
            e2: ``[B, C₁, 16, 16, 16]`` — level-2 skip.
            bn: ``[B, C₂, 8, 8, 8]``   — (possibly gated) bottleneck.

        Returns:
            ``[B, C₀, 32, 32, 32]`` decoded features.
        """
        d1 = self.dec1(torch.cat([self.up1(bn), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return d2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor ``[B, in_channels, 32, 32, 32]``.

        Returns:
            ``(features, bottleneck)`` where features is ``[B, C₀, 32, 32, 32]``
            and bottleneck is ``[B, C₂, 8, 8, 8]``.
        """
        e1, e2, bn = self.encode(x)
        d2 = self.decode(e1, e2, bn)
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


class OccGateModule(nn.Module):
    """Occupancy-gated bottleneck modulation (OGN-inspired propagation block).

    Computes occupancy logits from the 8³ bottleneck, converts them to
    per-octant attention weights via sigmoid, broadcasts to 8³ spatial
    grid, and multiplicatively gates the bottleneck features before they
    reach the decoder.

    This mimics OGN's propagation layer: the occupancy decision (which
    octants are non-empty) directly influences how features are expanded
    to finer levels.  Unlike OGN's hard gating (only MIXED nodes get
    children), this uses soft gating so the gradient flows everywhere
    and ONNX export remains straightforward.

    Architecture::

        bottleneck [B, C₂, 8, 8, 8]
          ├── OccupancyHead → occ_logits [B, 8]
          │                 → sigmoid → [B, 8]
          │                 → reshape [B, 1, 2, 2, 2]
          │                 → F.interpolate(4) → [B, 1, 8, 8, 8]
          └── * gate → gated_bottleneck [B, C₂, 8, 8, 8]

    The gated bottleneck replaces the raw bottleneck in the U-Net
    decoder, while occ_logits are returned unchanged for the loss.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.occ_head = OccupancyHead(in_channels)

    def forward(self, bottleneck: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply occupancy gate to bottleneck features.

        Args:
            bottleneck: ``[B, C₂, 8, 8, 8]``

        Returns:
            ``(gated_bottleneck, occ_logits)`` where gated_bottleneck has
            the same shape as the input and ``occ_logits`` is ``[B, 8]``.
        """
        occ_logits = self.occ_head(bottleneck)  # [B, 8]
        # Soft gate: per-octant attention from occupancy prediction
        gate = torch.sigmoid(occ_logits)  # [B, 8]
        # Reshape to 2×2×2 octant grid, upsample to 8×8×8
        gate_3d = gate.reshape(gate.shape[0], 1, 2, 2, 2)
        gate_3d = torch.nn.functional.interpolate(
            gate_3d, size=(8, 8, 8), mode="nearest"
        )  # [B, 1, 8, 8, 8]
        return bottleneck * gate_3d, occ_logits


# ── Parent context encoder ────────────────────────────────────────────


class ParentEncoder(nn.Module):
    """Embed parent block IDs → dense channel features.

    Takes integer block-ID volumes ``[B, 32, 32, 32]`` (from the parent
    WorldSection's argmax predictions, octant-extracted and upsampled)
    and produces ``[B, embed_dim, 32, 32, 32]`` via a learned
    ``Embedding(block_vocab_size, embed_dim)`` lookup.

    At ONNX export time, the embedding is retained **inside** the ONNX
    graph — models accept raw int64 block IDs directly and perform the
    lookup internally.
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
        self.init_architecture = getattr(config, "init_architecture", "full_3d_unet")
        init_extra_depth = _get_model_bottleneck_extra_depth(config, "init")

        if self.init_architecture == "encoder2d_decoder3d":
            self.backbone_2d3d = InitEncoder2DDecoder3D(
                height_channels=config.height_channels,
                biome_vocab_size=config.biome_vocab_size,
                biome_embed_dim=config.biome_embed_dim,
                y_vocab_size=config.y_vocab_size,
                y_embed_dim=config.y_embed_dim,
                channels=config.init_channels,
                bottleneck_extra_depth=init_extra_depth,
            )
        elif self.init_architecture == "full_3d_unet":
            # Conditioning embeddings
            self.biome_embed = nn.Embedding(config.biome_vocab_size, config.biome_embed_dim)
            self.y_embed = nn.Embedding(config.y_vocab_size, config.y_embed_dim)

            # Legacy all-3D backbone retained for backward compatibility.
            in_channels = config.height_channels + config.biome_embed_dim + config.y_embed_dim
            self.unet = UNet3D32(in_channels, config.init_channels, init_extra_depth)
        else:
            raise ValueError(
                f"Unknown init_architecture={self.init_architecture!r}. "
                "Expected 'encoder2d_decoder3d' or 'full_3d_unet'."
            )

        # Heads
        self.block_head = nn.Conv3d(c0, config.block_vocab_size, kernel_size=1)
        self._use_occ_gate = config.use_occ_gate
        if config.use_occ_gate:
            self.occ_gate = OccGateModule(c2)
            self.occ_head = self.occ_gate.occ_head  # expose for weight sharing
        else:
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
        # Input validation
        if heightmap.shape != torch.Size([heightmap.shape[0], 5, 32, 32]):
            raise ValueError(f"heightmap must be [B, 5, 32, 32], got {list(heightmap.shape)}")
        if biome.shape != torch.Size([biome.shape[0], 32, 32]):
            raise ValueError(f"biome must be [B, 32, 32], got {list(biome.shape)}")
        if y_position.shape != torch.Size([y_position.shape[0]]):
            raise ValueError(f"y_position must be [B], got {list(y_position.shape)}")
        if not (0 <= y_position.min() and y_position.max() < self.config.y_vocab_size):
            raise ValueError(
                f"y_position values must be in [0, {self.config.y_vocab_size - 1}], "
                f"got min={y_position.min().item()} max={y_position.max().item()}"
            )
        if self.init_architecture == "encoder2d_decoder3d":
            if self._use_occ_gate:
                bn = self.backbone_2d3d.encode(heightmap, biome, y_position)
                gated_bn, occ_logits = self.occ_gate(bn)
                features = self.backbone_2d3d.decode(gated_bn)
            else:
                features, bn = self.backbone_2d3d(heightmap, biome, y_position)
                occ_logits = self.occ_head(bn)
        else:
            # Build 3D conditioning for the legacy all-3D path.
            h3d = _build_height_3d(heightmap.float())
            b3d = _build_biome_3d(biome, self.biome_embed)
            y3d = _build_scalar_3d(y_position, self.y_embed)

            x = torch.cat([h3d, b3d, y3d], dim=1)  # [B, in_ch, 32, 32, 32]

            if self._use_occ_gate:
                e1, e2, bn = self.unet.encode(x)
                gated_bn, occ_logits = self.occ_gate(bn)
                features = self.unet.decode(e1, e2, gated_bn)
            else:
                features, bn = self.unet(x)
                occ_logits = self.occ_head(bn)

        return {
            "block_type_logits": self.block_head(features),
            "occ_logits": occ_logits,
        }


# ── Shared base for conditioned models (Refine + Leaf) ───────────────


class _OctreeConditionedBase(nn.Module):
    """Abstract base for refine and leaf models that share parent/biome/y conditioning.

    Provides shared ``__init__`` fields and the ``_resolve_parent()`` helper
    so the parent-resolution guard is defined once rather than copy-pasted.
    """

    parent_encoder: Optional[ParentEncoder]
    parent_refine_conv_layer: Optional[Conv3dBlock]
    biome_embed: nn.Embedding
    y_embed: nn.Embedding
    config: OctreeConfig
    _parent_mode: str

    def _resolve_parent(
        self,
        parent_blocks: Optional[torch.Tensor],
        parent_context: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Resolve parent context from either raw block IDs or pre-embedded float.

        If ``config.parent_refine_conv`` is True, a learned Conv3dBlock is
        applied after the embedding lookup to capture spatial correlations
        in the parent context.  This mirrors how OGN convolves features at
        each level before propagating them downward.

        Args:
            parent_blocks:  ``[B, 32, 32, 32]`` int64 — training path.
            parent_context: ``[B, C_parent, 32, 32, 32]`` float32 — ONNX path.
            batch_size: B, used for zeros mode.
            device: target device.

        Returns:
            Resolved ``[B, C_parent, 32, 32, 32]`` float32, or ``None`` if
            mode is ``"disabled"``.

        Raises:
            ValueError: if mode is ``"embed"`` and both inputs are ``None``.
        """
        if self._parent_mode == "embed":
            if parent_context is None:
                if parent_blocks is None:
                    raise ValueError(
                        "Must supply either parent_blocks (int64) or " "parent_context (float32)"
                    )
                parent_context = self.parent_encoder(parent_blocks)  # type: ignore[misc]
        elif self._parent_mode == "zeros":
            parent_context = torch.zeros(
                batch_size,
                self.config.parent_embed_dim,
                32,
                32,
                32,
                device=device,
            )
        # else "disabled": returns None

        # Apply optional parent feature refiner (OGN-inspired spatial conv)
        if parent_context is not None and self.parent_refine_conv_layer is not None:
            parent_context = self.parent_refine_conv_layer(parent_context)

        return parent_context


# ── Model B: OctreeRefineModel (shared L3, L2, L1) ───────────────────


class OctreeRefineModel(_OctreeConditionedBase):
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
        self._parent_mode = config.parent_context_mode

        # Parent embedding (used during training; bypassed in ONNX)
        if self._parent_mode == "embed":
            self.parent_encoder = ParentEncoder(config.block_vocab_size, config.parent_embed_dim)
        else:
            self.parent_encoder = None  # type: ignore[assignment]

        # Optional parent feature refiner (OGN-inspired spatial conv)
        if config.parent_refine_conv and self._parent_mode != "disabled":
            self.parent_refine_conv_layer = Conv3dBlock(
                config.parent_embed_dim, config.parent_embed_dim
            )
        else:
            self.parent_refine_conv_layer = None  # type: ignore[assignment]

        # Conditioning embeddings
        self.biome_embed = nn.Embedding(config.biome_vocab_size, config.biome_embed_dim)
        self.y_embed = nn.Embedding(config.y_vocab_size, config.y_embed_dim)
        self.level_embed = nn.Embedding(config.level_vocab_size, config.level_embed_dim)

        # U-Net
        parent_ch = config.parent_embed_dim if self._parent_mode != "disabled" else 0
        in_channels = (
            parent_ch
            + config.height_channels
            + config.biome_embed_dim
            + config.y_embed_dim
            + config.level_embed_dim
        )
        self.unet = UNet3D32(
            in_channels,
            config.refine_channels,
            _get_model_bottleneck_extra_depth(config, "refine"),
        )

        # Heads
        self.block_head = nn.Conv3d(c0, config.block_vocab_size, kernel_size=1)
        self._use_occ_gate = config.use_occ_gate
        if config.use_occ_gate:
            self.occ_gate = OccGateModule(c2)
            self.occ_head = self.occ_gate.occ_head
        else:
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
        ``None`` and mode is ``"embed"``, raises ``ValueError``.

        In ``"zeros"`` mode, parent channels are replaced with all-zeros.
        In ``"disabled"`` mode, parent channels are omitted entirely.

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
        # Input validation
        if (
            heightmap.dim() != 4
            or heightmap.shape[1] != 5
            or heightmap.shape[2] != 32
            or heightmap.shape[3] != 32
        ):
            raise ValueError(f"heightmap must be [B, 5, 32, 32], got {list(heightmap.shape)}")
        if biome.dim() != 3 or biome.shape[1] != 32 or biome.shape[2] != 32:
            raise ValueError(f"biome must be [B, 32, 32], got {list(biome.shape)}")
        if y_position.dim() != 1:
            raise ValueError(f"y_position must be [B], got {list(y_position.shape)}")
        if level.dim() != 1:
            raise ValueError(f"level must be [B], got {list(level.shape)}")
        if parent_blocks is not None and (
            parent_blocks.dim() != 4
            or parent_blocks.shape[1] != 32
            or parent_blocks.shape[2] != 32
            or parent_blocks.shape[3] != 32
        ):
            raise ValueError(
                f"parent_blocks must be [B, 32, 32, 32], got {list(parent_blocks.shape)}"
            )
        B = heightmap.shape[0]
        device = heightmap.device

        # Resolve parent context based on ablation mode
        parent_context = self._resolve_parent(parent_blocks, parent_context, B, device)

        # Build 3D conditioning
        h3d = _build_height_3d(heightmap.float())
        b3d = _build_biome_3d(biome, self.biome_embed)
        y3d = _build_scalar_3d(y_position, self.y_embed)
        l3d = _build_scalar_3d(level, self.level_embed)

        parts = [h3d, b3d, y3d, l3d]
        if parent_context is not None:
            parts.insert(0, parent_context)
        x = torch.cat(parts, dim=1)

        if self._use_occ_gate:
            e1, e2, bn = self.unet.encode(x)
            gated_bn, occ_logits = self.occ_gate(bn)
            features = self.unet.decode(e1, e2, gated_bn)
        else:
            features, bn = self.unet(x)
            occ_logits = self.occ_head(bn)

        return {
            "block_type_logits": self.block_head(features),
            "occ_logits": occ_logits,
        }


# ── Model C: OctreeLeafModel (L0 — full resolution, no occ head) ─────


class OctreeLeafModel(_OctreeConditionedBase):
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
        self._parent_mode = config.parent_context_mode

        # Parent embedding
        if self._parent_mode == "embed":
            self.parent_encoder = ParentEncoder(config.block_vocab_size, config.parent_embed_dim)
        else:
            self.parent_encoder = None  # type: ignore[assignment]

        # Optional parent feature refiner (OGN-inspired spatial conv)
        if config.parent_refine_conv and self._parent_mode != "disabled":
            self.parent_refine_conv_layer = Conv3dBlock(
                config.parent_embed_dim, config.parent_embed_dim
            )
        else:
            self.parent_refine_conv_layer = None  # type: ignore[assignment]

        # Conditioning embeddings
        self.biome_embed = nn.Embedding(config.biome_vocab_size, config.biome_embed_dim)
        self.y_embed = nn.Embedding(config.y_vocab_size, config.y_embed_dim)

        # U-Net
        parent_ch = config.parent_embed_dim if self._parent_mode != "disabled" else 0
        in_channels = (
            parent_ch + config.height_channels + config.biome_embed_dim + config.y_embed_dim
        )
        self.unet = UNet3D32(
            in_channels,
            config.leaf_channels,
            _get_model_bottleneck_extra_depth(config, "leaf"),
        )

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
        # Input validation
        if (
            heightmap.dim() != 4
            or heightmap.shape[1] != 5
            or heightmap.shape[2] != 32
            or heightmap.shape[3] != 32
        ):
            raise ValueError(f"heightmap must be [B, 5, 32, 32], got {list(heightmap.shape)}")
        if biome.dim() != 3 or biome.shape[1] != 32 or biome.shape[2] != 32:
            raise ValueError(f"biome must be [B, 32, 32], got {list(biome.shape)}")
        if y_position.dim() != 1:
            raise ValueError(f"y_position must be [B], got {list(y_position.shape)}")
        if parent_blocks is not None and (
            parent_blocks.dim() != 4
            or parent_blocks.shape[1] != 32
            or parent_blocks.shape[2] != 32
            or parent_blocks.shape[3] != 32
        ):
            raise ValueError(
                f"parent_blocks must be [B, 32, 32, 32], got {list(parent_blocks.shape)}"
            )
        B = heightmap.shape[0]
        device = heightmap.device

        # Resolve parent context based on ablation mode
        parent_context = self._resolve_parent(parent_blocks, parent_context, B, device)

        h3d = _build_height_3d(heightmap.float())
        b3d = _build_biome_3d(biome, self.biome_embed)
        y3d = _build_scalar_3d(y_position, self.y_embed)

        parts = [h3d, b3d, y3d]
        if parent_context is not None:
            parts.insert(0, parent_context)
        x = torch.cat(parts, dim=1)

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
