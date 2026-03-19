"""Noise-conditioned top-down sparse octree generator.

TODO (deferred): Aquifer and ore vein noise channels
  - Aquifer: flood-fill + atmospheric barometric noise for water table conditioning
  - Ore veins: vein_toggle, vein_ridged, vein_gap for vein placement (low priority)

Architecture
------------
1. **NoiseEncoder** — flattens vanilla noise fields + biome embedding into a
   per-subchunk context vector.

2. **OctreeDecoder** — top-down, teacher-forced during training.
   For levels L4 → L0 it maintains a feature tensor with one row per node,
   applies split and label heads, then projects every node feature to 8 child
   features ready for the next level.

   Training: all nodes at every level are always expanded (teacher forcing).
   Inference: expand nodes by applying a sigmoid to ``split_logit`` and
   comparing against the runtime split threshold (commonly 0.43).

Output keys are **integers** (4, 3, 2, 1, 0) matching
``build_sparse_octree_targets``.

Shapes (B = batch, D = hidden)
------------------------------
v7 pipeline (default):
  noise_2d              : [B, n2d, 4, 4]       climate fields at 4×4 cell res per chunk (0 or 7)
  noise_3d              : [B, n3d, 4, 2, 4]    15 RouterField channels at 4×2×4 quart resolution
  biome_ids             : [B, 4, 2, 4]         discrete biome IDs at 4×2×4 quart resolution
  heightmap_surface     : [B, 16, 16]          WORLD_SURFACE_WG heightmap in block Y
  heightmap_ocean_floor : [B, 16, 16]          OCEAN_FLOOR_WG heightmap in block Y
  ctx       : [B, D]
  node_feat : [B * N_nodes, D]      N_nodes = 1 / 8 / 64 / 512 / 4096 at L4..L0
  split     : [B, N_nodes]          binary logits
  label     : [B, N_nodes, C]       class logits

Vanilla uses cellWidth=4 (4-block quart spacing on X/Z) and cellHeight=8
(8-block cell spacing on Y), yielding 4×2×4 cells per 16-block section.
"""

from __future__ import annotations

import typing as t

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Chunk-location binary encoding
# ---------------------------------------------------------------------------

POSITION_BITS = 17  # 4 (chunk_x % 16) + 4 (chunk_z % 16) + 9 (block_y + 64)


def encode_chunk_position_bits(
    chunk_x: int,
    chunk_z: int,
    section_y: int,
) -> np.ndarray:
    """Encode chunk location as a 17-element binary float32 vector.

    Layout (LSB-first within each group)::

        [x_bit0 .. x_bit3,  z_bit0 .. z_bit3,  y_bit0 .. y_bit8]

    * chunk_x mod 16 → 4 bits  (wraps — nearby chunks share high bits)
    * chunk_z mod 16 → 4 bits
    * block_y + 64   → 9 bits  (encodes subchunk base Y in block coordinates)

    ``section_y`` is the vertical section index. We convert it to the subchunk's
    absolute block-space base Y via ``block_y = section_y * 16`` so the model
    sees real world height rather than only a compressed section ordinal.
    """
    x_mod = chunk_x % 16       # Python % is always non-negative for positive divisor
    z_mod = chunk_z % 16
    block_y = int(section_y) * 16
    y_val = max(0, min(block_y + 64, 511))  # clamp to 9-bit range

    bits = np.zeros(POSITION_BITS, dtype=np.float32)
    for i in range(4):
        bits[i] = float((x_mod >> i) & 1)
        bits[4 + i] = float((z_mod >> i) & 1)
    for i in range(9):
        bits[8 + i] = float((y_val >> i) & 1)
    return bits


# ---------------------------------------------------------------------------
# Positional embedding helpers
# ---------------------------------------------------------------------------

_MAX_LEVEL = 4  # L4 is coarsest (1³), L0 is finest (16³)
_MAX_GRID = 16  # maximum grid size at L0


class _OctreePosEmb(nn.Module):
    """Learnable positional embedding:  level_emb + y_emb + z_emb + x_emb."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.level_emb = nn.Embedding(_MAX_LEVEL + 1, hidden)
        self.y_emb = nn.Embedding(_MAX_GRID, hidden)
        self.z_emb = nn.Embedding(_MAX_GRID, hidden)
        self.x_emb = nn.Embedding(_MAX_GRID, hidden)

    def forward(self, level: int, device: torch.device) -> torch.Tensor:
        """Return pos-emb tensor of shape [N_nodes, hidden] for this level."""
        side = 2 ** (_MAX_LEVEL - level)  # e.g. L4→1, L3→2, L0→16
        n = side**3
        lvl_t = torch.full((n,), level, dtype=torch.long, device=device)
        # enumerate y, z, x in C-order (matches build_sparse_octree_targets)
        ys, zs, xs = torch.meshgrid(
            torch.arange(side, device=device),
            torch.arange(side, device=device),
            torch.arange(side, device=device),
            indexing="ij",
        )
        ys = ys.reshape(-1)
        zs = zs.reshape(-1)
        xs = xs.reshape(-1)
        return (
            self.level_emb(lvl_t) + self.y_emb(ys) + self.z_emb(zs) + self.x_emb(xs)
        )  # [N_nodes, hidden]


# ---------------------------------------------------------------------------
# Noise encoder
# ---------------------------------------------------------------------------


class _NoiseEncoder(nn.Module):
    """MLP encoder over flattened 2-D and 3-D vanilla noise fields + biome embedding + heightmaps.

    Parameters
    ----------
    n2d:                number of 2-D climate channels (default 0)
    n3d:                number of 3-D volumetric channels (default 15 for v7)
    hidden:             output context dimensionality
    biome_vocab_size:   vocabulary size for biome IDs (default 256)
    biome_embed_dim:    embedding dimensionality for biome (default 8)
    spatial_y:          Y-axis quart cells per section (2 for v7 — vanilla cellHeight=8)
    position_bits:      number of binary chunk-location features (0 = disabled, 17 = full)
    """

    def __init__(
        self,
        n2d: int,
        n3d: int,
        hidden: int,
        biome_vocab_size: int = 256,
        biome_embed_dim: int = 8,
        spatial_y: int = 2,
        position_bits: int = 0,
    ) -> None:
        super().__init__()
        self.spatial_y = spatial_y
        self.position_bits = position_bits
        flat_2d = n2d * 4 * 4
        flat_3d = n3d * 4 * spatial_y * 4
        flat_biome = 4 * spatial_y * 4 * biome_embed_dim
        flat_heightmaps = 16 * 16 * 2  # surface + ocean_floor, each [16,16]
        in_dim = flat_2d + flat_3d + flat_biome + flat_heightmaps + position_bits

        self.biome_embed = nn.Embedding(biome_vocab_size, biome_embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

    def forward(
        self,
        noise_2d: torch.Tensor,
        noise_3d: torch.Tensor,
        biome_ids: torch.Tensor,
        heightmap_surface: torch.Tensor,
        heightmap_ocean_floor: torch.Tensor,
        position_bits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            noise_2d:              [B, n2d, 4, 4]
            noise_3d:              [B, n3d, 4, spatial_y, 4]
            biome_ids:             [B, 4, spatial_y, 4] (integer biome indices)
            heightmap_surface:     [B, 16, 16] (WORLD_SURFACE_WG block Y)
            heightmap_ocean_floor: [B, 16, 16] (OCEAN_FLOOR_WG block Y)
            position_bits:         [B, position_bits] (binary chunk-location bools, or None)
        Returns:
            ctx: [B, hidden]
        """
        B = noise_2d.shape[0]
        if noise_2d.numel() == 0:
            # ONNX runtimes may not support 0-length tensors; accept a dummy
            # input shape but ignore it for encoding.
            noise_2d_flat = noise_2d.new_zeros((B, 0))
        else:
            noise_2d_flat = noise_2d.reshape(B, -1)

        biome_ids_long = biome_ids.long().clamp(0, self.biome_embed.num_embeddings - 1)
        biome_feat = self.biome_embed(biome_ids_long)  # [B, 4, spatial_y, 4, embed_dim]

        parts = [
            noise_2d_flat,
            noise_3d.reshape(B, -1),
            biome_feat.reshape(B, -1),
            heightmap_surface.reshape(B, -1),
            heightmap_ocean_floor.reshape(B, -1),
        ]
        if self.position_bits > 0 and position_bits is not None:
            parts.append(position_bits.to(noise_2d.dtype))

        flat = torch.cat(parts, dim=1)
        return self.mlp(flat)


class _FactorizedHead(nn.Module):
    """Low-rank projection head used to cut params and improve regularization."""

    def __init__(self, in_dim: int, out_dim: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.act(self.in_proj(x)))


class _LevelFiLM(nn.Module):
    """Per-level feature modulation derived from the global noise context."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden * 2)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.proj(ctx).chunk(2, dim=-1)
        scale = 1.0 + 0.1 * torch.tanh(scale)
        return scale, shift


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SparseOctreeModel(nn.Module):
    """Noise-conditioned, teacher-forced sparse octree generator.

    Parameters
    ----------
    n2d:            2-D noise channel count (default 0; can be > 0 for extra climate)
    n3d:            3-D noise channel count (default 15 for v7 RouterFields)
    hidden:         internal feature dimensionality
    num_classes:    block vocabulary size
    levels:         number of octree levels (5 → L4..L0 for a 16³ subchunk)
    biome_vocab_size: vocabulary size for biome IDs (default 256)
    biome_embed_dim: embedding dimensionality for biome (default 8)
    spatial_y:      Y-axis quart cells per section (2 for v7 — vanilla cellHeight=8)
    """

    def __init__(
        self,
        n2d: int = 0,
        n3d: int = 15,
        hidden: int = 128,
        num_classes: int = 256,
        levels: int = 5,
        biome_vocab_size: int = 256,
        biome_embed_dim: int = 8,
        spatial_y: int = 2,
        position_bits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_classes = num_classes
        self.levels = levels  # coarsest level index = levels - 1
        self.max_level = levels - 1  # e.g. 4 for a 5-level tree
        self.position_bits = position_bits

        self.noise_enc = _NoiseEncoder(
            n2d,
            n3d,
            hidden,
            biome_vocab_size=biome_vocab_size,
            biome_embed_dim=biome_embed_dim,
            spatial_y=spatial_y,
            position_bits=position_bits,
        )
        self.pos_emb = _OctreePosEmb(hidden)

        # ctx → initial root node feature
        self.root_proj = nn.Linear(hidden, hidden)

        # Per-node heads
        self.split_head = nn.Linear(hidden, 1)
        self.label_head = nn.Linear(hidden, num_classes)

        # Expand one node feature → 8 child features
        self.child_proj = nn.Linear(hidden, hidden * 8)

    # ------------------------------------------------------------------

    def forward(
        self,
        noise_2d: torch.Tensor,
        noise_3d: torch.Tensor,
        biome_ids: torch.Tensor,
        heightmap_surface: torch.Tensor,
        heightmap_ocean_floor: torch.Tensor,
        position_bits: torch.Tensor | None = None,
    ) -> t.Dict[int, t.Dict[str, torch.Tensor]]:
        """Teacher-forced forward pass; expands ALL nodes at every level.

        Args:
            noise_2d: [B, n2d, 4, 4]
            noise_3d: [B, n3d, 4, spatial_y, 4]
            biome_ids: [B, 4, spatial_y, 4]
            heightmap_surface: [B, 16, 16]
            heightmap_ocean_floor: [B, 16, 16]

        Returns:
            Dict[level → {'split': [B, N], 'label': [B, N, C]}]
            where N = (2^(max_level - level))^3 and levels are integers.
        """
        B = noise_2d.shape[0]
        device = noise_2d.device

        ctx = self.noise_enc(
            noise_2d, noise_3d, biome_ids, heightmap_surface, heightmap_ocean_floor,
            position_bits=position_bits,
        )  # [B, D]

        # Initialise root as a single node per sample
        # root_feat: [B * 1, D]
        root_feat = self.root_proj(ctx)  # [B, D]
        cur_feat = root_feat.unsqueeze(1)  # [B, 1, D]

        outputs: t.Dict[int, t.Dict[str, torch.Tensor]] = {}

        for lvl in range(self.max_level, -1, -1):
            N = cur_feat.shape[1]  # number of nodes at this level

            # Add positional embedding [N, D] → broadcast over batch
            pe = self.pos_emb(lvl, device)  # [N, D]
            feat = cur_feat + pe.unsqueeze(0)  # [B, N, D]  (broadcast)

            # Per-node predictions
            flat = feat.reshape(B * N, self.hidden)
            split_logits = self.split_head(flat).reshape(B, N)  # [B, N]
            label_logits = self.label_head(flat).reshape(B, N, self.num_classes)  # [B, N, C]

            outputs[lvl] = {"split": split_logits, "label": label_logits}

            if lvl == 0:
                break

            # Project each node to 8 children: [B*N, D] → [B*N, D*8]
            child_raw = self.child_proj(flat)  # [B*N, D*8]
            child_raw = child_raw.reshape(B, N * 8, self.hidden)  # [B, N*8, D]
            cur_feat = child_raw

        return outputs


class SparseOctreeFastModel(nn.Module):
    """Faster sparse-root variant with factorized heads and level conditioning.

    Design goals
    ------------
    - reduce the dominant `label_head` and `child_proj` parameter blocks;
    - preserve the same external forward contract as `SparseOctreeModel`;
    - add cheap per-level conditioning to improve fine-level accuracy.
    """

    def __init__(
        self,
        n2d: int = 0,
        n3d: int = 15,
        hidden: int = 72,
        num_classes: int = 256,
        levels: int = 5,
        biome_vocab_size: int = 256,
        biome_embed_dim: int = 8,
        label_rank: int | None = None,
        child_rank: int | None = None,
        split_rank: int | None = None,
        spatial_y: int = 2,
        position_bits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_classes = num_classes
        self.levels = levels
        self.max_level = levels - 1
        self.position_bits = position_bits

        label_rank = label_rank or max(48, min(hidden, (hidden * 2) // 3))
        child_rank = child_rank or max(32, hidden // 2)
        split_rank = split_rank or max(24, hidden // 3)

        self.noise_enc = _NoiseEncoder(
            n2d,
            n3d,
            hidden,
            biome_vocab_size=biome_vocab_size,
            biome_embed_dim=biome_embed_dim,
            spatial_y=spatial_y,
            position_bits=position_bits,
        )
        self.pos_emb = _OctreePosEmb(hidden)

        self.root_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.level_mod = nn.ModuleList(_LevelFiLM(hidden) for _ in range(levels))

        self.split_head = _FactorizedHead(hidden, 1, split_rank)
        self.label_head = _FactorizedHead(hidden, num_classes, label_rank)
        self.child_proj = _FactorizedHead(hidden, hidden * 8, child_rank)

    def forward(
        self,
        noise_2d: torch.Tensor,
        noise_3d: torch.Tensor,
        biome_ids: torch.Tensor,
        heightmap_surface: torch.Tensor,
        heightmap_ocean_floor: torch.Tensor,
        position_bits: torch.Tensor | None = None,
    ) -> t.Dict[int, t.Dict[str, torch.Tensor]]:
        B = noise_2d.shape[0]
        device = noise_2d.device

        ctx = self.noise_enc(
            noise_2d, noise_3d, biome_ids, heightmap_surface, heightmap_ocean_floor,
            position_bits=position_bits,
        )
        cur_feat = self.root_proj(ctx).unsqueeze(1)

        outputs: t.Dict[int, t.Dict[str, torch.Tensor]] = {}

        for lvl in range(self.max_level, -1, -1):
            N = cur_feat.shape[1]
            pe = self.pos_emb(lvl, device)
            scale, shift = self.level_mod[self.max_level - lvl](ctx)
            feat = cur_feat + pe.unsqueeze(0)
            feat = feat * scale.unsqueeze(1) + shift.unsqueeze(1)

            flat = feat.reshape(B * N, self.hidden)
            split_logits = self.split_head(flat).reshape(B, N)
            label_logits = self.label_head(flat).reshape(B, N, self.num_classes)
            outputs[lvl] = {"split": split_logits, "label": label_logits}

            if lvl == 0:
                break

            cur_feat = self.child_proj(flat).reshape(B, N * 8, self.hidden)

        return outputs


__all__ = [
    "POSITION_BITS",
    "encode_chunk_position_bits",
    "SparseOctreeModel",
    "SparseOctreeFastModel",
]
