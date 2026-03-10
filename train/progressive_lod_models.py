"""Progressive LOD models — notebook-winner integration.

Winning architectures per transition (verified in shootout notebooks):

- Init→LOD4 : ``TabularResidualMLP``      (1×1×1)
- LOD4→LOD3 : ``TinyAttentionUpscaler``   (2×2×2)
- LOD3→LOD2 : ``BaselineConv3D4Cube``     (4×4×4)
- LOD2→LOD1 : ``TinyUNet8Cube``           (8×8×8)

All models keep a production forward signature::

    forward(height_planes, biome_indices, y_index, [x_parent],
            tabular=None, biome_center=None, biome_mode=None)

so the ONNX export adapters (``InitModelAdapter``, ``RefinementModelAdapter``)
continue to work without change.  Derived features (tabular, biome_center,
biome_mode) are computed *inside* the model when not supplied externally,
preserving the existing ONNX contract.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_utils import TAB_DIM, build_tabular_features_torch
from .unet3d import SimpleFlexibleConfig


# ── Base type for all progressive LOD models ───────────────────────────────


class ProgressiveLODModel(nn.Module):
    """Base class for all refinement LOD models.

    All progressive models must implement a meaningful block_head attribute
    that can be initialized with class priors (see prior_init.py).
    """

    pass


# ── Shared helpers ─────────────────────────────────────────────────────


def _safe_biome_center(biome_indices: torch.Tensor) -> torch.Tensor:
    """Extract center biome from a [B, 16, 16] grid."""
    return biome_indices[:, 8, 8].long().clamp(0, 255)


# ── Conditioning blocks (notebook-faithful) ──────────────────────────────


class AnchorCond3D4(nn.Module):
    """2D anchor → 3D conditioning at 4³ resolution.

    Used by ``BaselineConv3D4Cube`` (LOD3→LOD2 winner).

    Architecture (from lod3_to_lod2 shootout)::

        height_planes → Conv2d(5, base) → GELU → Conv2d → GELU → Pool(4,4)  ──┐
        biome_center  → Embedding(256, base) → broadcast                      ├─ add → 3D
        y_index       → Embedding(y_vocab, base) → broadcast                  ──┘
        biome_idx     → Embedding(256, base) → permute → subsample(::4, ::4)
        cat([cond3d, biome_plane]) → Conv3d fuse
    """

    def __init__(self, base: int = 48, biome_vocab: int = 256, y_vocab: int = 64):
        super().__init__()
        self.biome_vocab = biome_vocab
        self.height_conv = nn.Sequential(
            nn.Conv2d(5, base, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.biome_emb = nn.Embedding(biome_vocab, base)
        self.y_emb = nn.Embedding(y_vocab, base)
        self.fuse = nn.Sequential(
            nn.Conv3d(base * 2, base, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(base, base, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(
        self,
        height_planes: torch.Tensor,  # [B, 5, 16, 16]
        biome_indices: torch.Tensor,  # [B, 16, 16]
        y_index: torch.Tensor,  # [B]
        biome_center: Optional[torch.Tensor] = None,  # [B]
    ) -> torch.Tensor:
        hp = self.height_conv(height_planes.float())  # [B, base, 4, 4]

        if biome_center is None:
            biome_center = _safe_biome_center(biome_indices)
        bc = self.biome_emb(biome_center.long().clamp(0, self.biome_vocab - 1))
        bc = bc[:, :, None, None]  # [B, base, 1, 1]

        y = self.y_emb(y_index.long().clamp(0, self.y_emb.num_embeddings - 1))
        y = y[:, :, None, None]  # [B, base, 1, 1]

        cond2d = hp + bc + y  # [B, base, 4, 4]
        cond3d = cond2d[:, :, None].expand(-1, -1, 4, -1, -1)  # [B, base, 4, 4, 4]

        # Full biome embedding at 16×16, subsampled to 4×4
        biome_plane = self.biome_emb(
            biome_indices.long().clamp(0, self.biome_vocab - 1)
        )  # [B, 16, 16, base]
        biome_plane = biome_plane.permute(0, 3, 1, 2)  # [B, base, 16, 16]
        biome_plane = biome_plane[:, :, None].expand(-1, -1, 4, -1, -1)  # [B, base, 4, 16, 16]
        biome_plane = biome_plane[:, :, :, ::4, ::4]  # [B, base, 4, 4, 4]

        return self.fuse(torch.cat([cond3d, biome_plane], dim=1))


class AnchorCond3D8(nn.Module):
    """2D anchor → 3D conditioning at 8³ resolution.

    Used by ``TinyUNet8Cube`` (LOD2→LOD1 winner).
    Same pattern as ``AnchorCond3D4`` but pools to 8×8 and subsamples ``::2``.
    """

    def __init__(self, base: int = 32, biome_vocab: int = 256, y_vocab: int = 64):
        super().__init__()
        self.biome_vocab = biome_vocab
        self.height_conv = nn.Sequential(
            nn.Conv2d(5, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.biome_emb = nn.Embedding(biome_vocab, base)
        self.y_emb = nn.Embedding(y_vocab, base)
        self.fuse = nn.Sequential(
            nn.Conv3d(base * 2, base, 1),
            nn.GELU(),
            nn.Conv3d(base, base, 3, padding=1),
            nn.GELU(),
        )

    def forward(
        self,
        height_planes: torch.Tensor,  # [B, 5, 16, 16]
        biome_indices: torch.Tensor,  # [B, 16, 16]
        y_index: torch.Tensor,  # [B]
        biome_center: Optional[torch.Tensor] = None,  # [B]
    ) -> torch.Tensor:
        hp = self.height_conv(height_planes.float())  # [B, base, 8, 8]

        if biome_center is None:
            biome_center = _safe_biome_center(biome_indices)
        bc = self.biome_emb(biome_center.long().clamp(0, self.biome_vocab - 1))
        bc = bc[:, :, None, None]

        y = self.y_emb(y_index.long().clamp(0, self.y_emb.num_embeddings - 1))
        y = y[:, :, None, None]

        cond2d = hp + bc + y  # [B, base, 8, 8]
        cond3d = cond2d[:, :, None].expand(-1, -1, 8, -1, -1)  # [B, base, 8, 8, 8]

        biome_plane = self.biome_emb(biome_indices.long().clamp(0, self.biome_vocab - 1))
        biome_plane = biome_plane.permute(0, 3, 1, 2)  # [B, base, 16, 16]
        biome_plane = biome_plane[:, :, None].expand(-1, -1, 8, -1, -1)
        biome_plane = biome_plane[:, :, :, ::2, ::2]  # [B, base, 8, 8, 8]

        return self.fuse(torch.cat([cond3d, biome_plane], dim=1))


# ── Model 0: Init → LOD4 (1×1×1) — TabularResidualMLP ────────────────


class ProgressiveLODModel0_Initial(nn.Module):
    """Notebook winner: TabularResidualMLP.

    Generates 1×1×1 from conditioning inputs only (no parent).

    Input:  29-dim tabular + biome_center embed (12) + y_index embed (8) → 49
    Hidden: residual MLP with 2 skip-connected blocks (hidden=96)
    Output: [B, block_vocab_size, 1, 1, 1]
    """

    def __init__(self, config: SimpleFlexibleConfig, output_size: int = 1):
        super().__init__()
        self.config = config
        self.output_size = output_size
        hidden = 96

        self.biome_embedding = nn.Embedding(256, 12)
        self.y_embedding = nn.Embedding(24, 8)
        self.input_proj = nn.Linear(TAB_DIM + 12 + 8, hidden)
        self.block1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.block_head = nn.Linear(hidden, config.block_vocab_size)

    def forward(
        self,
        height_planes: torch.Tensor,  # [B, 5, 16, 16]
        biome_indices: torch.Tensor,  # [B, 16, 16] int64
        y_index: torch.Tensor,  # [B] int64
        tabular: Optional[torch.Tensor] = None,
        biome_center: Optional[torch.Tensor] = None,
        biome_mode: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = height_planes.shape[0]

        # Derive features (or use pre-computed from batch)
        if tabular is None or biome_center is None:
            tab, bc, _ = build_tabular_features_torch(height_planes, biome_indices, y_index)
            tabular = tabular if tabular is not None else tab
            biome_center = biome_center if biome_center is not None else bc
        _ = biome_mode  # not used by this model

        bc_emb = self.biome_embedding(biome_center.long().clamp(0, 255))
        y_emb = self.y_embedding(y_index.long().clamp(0, 23))

        z = self.input_proj(torch.cat([tabular.float(), bc_emb, y_emb], dim=1))
        z = z + self.block1(z)
        z = z + self.block2(z)
        block_logits = self.block_head(z).view(B, self.config.block_vocab_size, 1, 1, 1)

        return {"block_type_logits": block_logits}


# ── Model 1: LOD4 → LOD3 (2×2×2) — TinyAttentionUpscaler ────────────


class TinyAttentionUpscaler(ProgressiveLODModel):
    """Notebook winner: 8 learnable position queries cross-attend over
    (parent_token, cond_token).

    Uses parent occupancy as a scalar mean, not spatial 3D.
    DIM=64, N_HEADS=4, single cross-attention + FFN block.
    Output: [B, block_vocab_size, 2, 2, 2]
    """

    def __init__(self, config: SimpleFlexibleConfig):
        super().__init__()
        self.output_size = 2
        self.num_classes = config.block_vocab_size
        dim = 64

        self.pos_queries = nn.Parameter(torch.randn(8, dim) * 0.02)
        self.biome_emb = nn.Embedding(256, 12)
        self.y_emb = nn.Embedding(24, 8)

        # Parent occ scalar → token
        self.parent_tok_proj = nn.Sequential(nn.Linear(1, dim), nn.ReLU(inplace=True))
        # Conditioning → token
        self.cond_proj = nn.Sequential(
            nn.Linear(TAB_DIM + 12 + 8, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.block_head = nn.Linear(dim, self.num_classes)

    def forward(
        self,
        height_planes: torch.Tensor,
        biome_indices: torch.Tensor,
        y_index: torch.Tensor,
        x_parent: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        biome_center: Optional[torch.Tensor] = None,
        biome_mode: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = x_parent.shape[0]
        _ = biome_mode  # not used

        # Derive features (or use pre-computed from batch)
        if tabular is None or biome_center is None:
            tab, bc, _ = build_tabular_features_torch(height_planes, biome_indices, y_index)
            tabular = tabular if tabular is not None else tab
            biome_center = biome_center if biome_center is not None else bc

        bc_emb = self.biome_emb(biome_center.long().clamp(0, 255))
        y_emb = self.y_emb(y_index.long().clamp(0, 23))

        # Parent occupancy as a single scalar
        p_occ = x_parent.float().reshape(B, -1).mean(dim=1, keepdim=True)
        parent_tok = self.parent_tok_proj(p_occ).unsqueeze(1)  # [B, 1, dim]
        cond_tok = self.cond_proj(torch.cat([tabular.float(), bc_emb, y_emb], dim=1)).unsqueeze(
            1
        )  # [B, 1, dim]
        context = torch.cat([parent_tok, cond_tok], dim=1)  # [B, 2, dim]

        queries = self.pos_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 8, dim]
        attn_out, _ = self.attn(queries, context, context)
        z = self.norm(queries + attn_out)
        z = self.norm2(z + self.ff(z))  # [B, 8, dim]

        logits = self.block_head(z)  # [B, 8, num_classes]
        logits = logits.permute(0, 2, 1).reshape(B, self.num_classes, 2, 2, 2)
        return {"block_type_logits": logits}


# ── Model 2: LOD3 → LOD2 (4×4×4) — BaselineConv3D4Cube ──────────────


class BaselineConv3D4Cube(ProgressiveLODModel):
    """Notebook winner: AnchorCond3D4 + parent upsample + Conv3D refinement.

    Uses ``AnchorCond3D4`` for rich 3D conditioning (height + biome spatial
    + y_index), concatenated with upsampled parent occupancy.
    Output: [B, block_vocab_size, 4, 4, 4]
    """

    def __init__(self, config: SimpleFlexibleConfig, base: int = 48):
        super().__init__()
        self.config = config
        self.output_size = 4
        self.cond = AnchorCond3D4(base=base, biome_vocab=config.biome_vocab_size)
        self.parent_up = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.main = nn.Sequential(
            nn.Conv3d(base * 2, base, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(base, base, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.block_head = nn.Conv3d(base, config.block_vocab_size, kernel_size=1)

    def forward(
        self,
        height_planes: torch.Tensor,
        biome_indices: torch.Tensor,
        y_index: torch.Tensor,
        x_parent: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        biome_center: Optional[torch.Tensor] = None,
        biome_mode: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _ = (tabular, biome_mode)  # not used; conditioning is spatial

        if biome_center is None:
            biome_center = _safe_biome_center(biome_indices)

        # Parent: interpolate any input size (1³, 2³, 8³) → 4³
        parent = F.interpolate(
            x_parent.float(), size=(4, 4, 4), mode="trilinear", align_corners=False
        )
        parent = self.parent_up(parent)

        # Anchor conditioning → 3D features at 4³
        cond = self.cond(height_planes, biome_indices, y_index, biome_center)

        logits = self.block_head(self.main(torch.cat([parent, cond], dim=1)))
        return {"block_type_logits": logits}


# ── Model 3: LOD2 → LOD1 (8×8×8) — TinyUNet8Cube ────────────────────


class TinyUNet8Cube(ProgressiveLODModel):
    """Notebook winner: AnchorCond3D8 + encoder-decoder with skip connection.

    Architecture::

        parent (→ 8³) + cond (8³) → in_conv → x0
        x0 → down (stride-2) → x1 → mid → x2
        x2 → up (deconv) → x3
        cat(x3, x0) → out → logits

    Output: [B, block_vocab_size, 8, 8, 8]
    """

    def __init__(self, config: SimpleFlexibleConfig, base: int = 32):
        super().__init__()
        self.config = config
        self.output_size = 8
        self.cond = AnchorCond3D8(base=base, biome_vocab=config.biome_vocab_size)
        self.in_conv = nn.Sequential(nn.Conv3d(1 + base, base, 3, padding=1), nn.GELU())
        self.down = nn.Sequential(nn.Conv3d(base, base * 2, 3, stride=2, padding=1), nn.GELU())
        self.mid = nn.Sequential(nn.Conv3d(base * 2, base * 2, 3, padding=1), nn.GELU())
        self.up = nn.Sequential(nn.ConvTranspose3d(base * 2, base, 2, 2), nn.GELU())
        self.refine = nn.Sequential(
            nn.Conv3d(base * 2, base, 3, padding=1),
            nn.GELU(),
        )
        # Separate block_head for prior initialization
        self.block_head = nn.Conv3d(base, config.block_vocab_size, 1)

    def forward(
        self,
        height_planes: torch.Tensor,
        biome_indices: torch.Tensor,
        y_index: torch.Tensor,
        x_parent: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        biome_center: Optional[torch.Tensor] = None,
        biome_mode: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _ = (tabular, biome_mode)  # not used; conditioning is spatial

        if biome_center is None:
            biome_center = _safe_biome_center(biome_indices)

        # Parent: interpolate any input size → 8³
        parent = F.interpolate(
            x_parent.float(), size=(8, 8, 8), mode="trilinear", align_corners=False
        )

        cond = self.cond(height_planes, biome_indices, y_index, biome_center)

        x0 = self.in_conv(torch.cat([parent, cond], dim=1))
        x1 = self.down(x0)
        x2 = self.mid(x1)
        x3 = self.up(x2)
        x = self.refine(torch.cat([x3, x0], dim=1))
        logits = self.block_head(x)

        return {"block_type_logits": logits}


# ── Factory functions for the 4-model family ──────────────────────────────
# (LOD1→LOD0 deliberately absent — vanilla handles LOD0)


def create_init_model(config: SimpleFlexibleConfig) -> ProgressiveLODModel0_Initial:
    """Init: Noise → LOD4 (1×1×1 output, TabularResidualMLP)"""
    return ProgressiveLODModel0_Initial(config, output_size=1)


def create_lod4_to_lod3_model(config: SimpleFlexibleConfig) -> nn.Module:
    """LOD4 → LOD3 winner: TinyAttentionUpscaler (2×2×2 output)."""
    return TinyAttentionUpscaler(config)


def create_lod3_to_lod2_model(config: SimpleFlexibleConfig) -> nn.Module:
    """LOD3 → LOD2 winner: BaselineConv3D4Cube (4×4×4 output)."""
    return BaselineConv3D4Cube(config)


def create_lod2_to_lod1_model(config: SimpleFlexibleConfig) -> nn.Module:
    """LOD2 → LOD1 winner: TinyUNet8Cube (8×8×8 output)."""
    return TinyUNet8Cube(config)
