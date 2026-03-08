"""
AnchorConditioningFusion — Vanilla-parity spatial conditioning.

Replaces SimpleConditioningFusion with a richer module that accepts all anchor
channels defined in §2.3 of the project outline:

  - height_planes  [B, 5, H, W]  surface, ocean_floor, slope_x, slope_z, curvature
  - router6        [B, 6, H, W]  temperature, vegetation, continentalness, erosion, depth, ridges
  - biome_indices  [B, H, W]     integer biome IDs (0..255)
  - y_index        [B]           integer Y-slab index (0..23 covers -64..320 at 16-block pitch)

Output: [B, out_channels, H, W]  — fused spatial features that broadcast to 3D in the UNet.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of Y-slab slots covering -64..320 at 16-block pitch (384 / 16 = 24).
Y_SLABS: int = 24

#: Channel indices into height_planes tensor.
HEIGHT_SURFACE: int = 0
HEIGHT_OCEAN_FLOOR: int = 1
HEIGHT_SLOPE_X: int = 2
HEIGHT_SLOPE_Z: int = 3
HEIGHT_CURVATURE: int = 4

# Router6 channels removed — biome already encodes the outcome of
# temperature/vegetation/continentalness/erosion/ridges, and heightmap
# encodes the outcome of depth.  For LOD4→LOD1 (no LOD0), the marginal
# information from raw noise values is negligible.


def compute_height_planes(heightmap: torch.Tensor) -> torch.Tensor:
    """
    Compute the 5-channel height_planes tensor from a single heightmap.

    Used when only a raw heightmap is available (existing training data,
    approximate client-side inputs).  The ocean_floor channel is zeroed.

    Args:
        heightmap: [B, 1, H, W] or [B, H, W] normalised (0..1) heightmap

    Returns:
        height_planes: [B, 5, H, W]
    """
    if heightmap.dim() == 3:
        h = heightmap.unsqueeze(1)  # [B,1,H,W]
    elif heightmap.dim() == 5:
        h = heightmap.squeeze(-1)  # drop trailing 1
    else:
        h = heightmap  # already [B,1,H,W]

    B, _, H, W = h.shape

    surface = h  # [B,1,H,W]
    ocean_floor = torch.zeros_like(h)  # [B,1,H,W] — unknown

    # Central-difference gradients on the spatial surface
    # F.pad by 1 on each side; use reflect to avoid edge artefacts
    h_pad = nn.functional.pad(h, (1, 1, 1, 1), mode="reflect")  # [B,1,H+2,W+2]
    slope_x = (h_pad[:, :, 1:-1, 2:] - h_pad[:, :, 1:-1, :-2]) / 2.0  # dh/dx
    slope_z = (h_pad[:, :, 2:, 1:-1] - h_pad[:, :, :-2, 1:-1]) / 2.0  # dh/dz

    # Laplacian curvature (d²h/dx² + d²h/dz²)
    d2x = h_pad[:, :, 1:-1, 2:] - 2 * h + h_pad[:, :, 1:-1, :-2]
    d2z = h_pad[:, :, 2:, 1:-1] - 2 * h + h_pad[:, :, :-2, 1:-1]
    curvature = d2x + d2z

    return torch.cat([surface, ocean_floor, slope_x, slope_z, curvature], dim=1)  # [B,5,H,W]


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class AnchorConditioningFusion(nn.Module):
    """
    Spatial conditioning fusion for vanilla anchor channels.

    Architecture (router6 removed — biome already encodes its information):
      height_planes → Conv2d(5, out//3) → GroupNorm → ReLU  ──┐
      biome embed   → Conv2d(E, out//3) → GroupNorm → ReLU  ──┼─ cat ─→ Conv2d fusion ─→ out
      y_index embed → broadcast [B, y_dim, H, W]             ──┘

    Total input to fusion: out//3 + out//3 + y_dim channels.
    """

    def __init__(
        self,
        height_channels: int = 5,
        biome_vocab_size: int = 256,
        biome_embed_dim: int = 32,
        y_embed_dim: int = 16,
        y_slabs: int = Y_SLABS,
        out_channels: int = 64,
        # Kept for backward-compat deserialization; ignored.
        router6_channels: int = 0,
    ):
        super().__init__()

        self.height_channels = height_channels
        self.out_channels = out_channels

        third = max(out_channels // 3, 8)

        # Height planes stream
        self.height_conv = nn.Sequential(
            nn.Conv2d(height_channels, third, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, third), third),
            nn.ReLU(inplace=True),
        )

        # Biome stream (larger now that router6 is gone)
        self.biome_embedding = nn.Embedding(biome_vocab_size, biome_embed_dim)
        self.biome_conv = nn.Sequential(
            nn.Conv2d(biome_embed_dim, third, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, third), third),
            nn.ReLU(inplace=True),
        )

        # Y-slab embedding
        self.y_slabs = y_slabs
        self.y_embed_dim = y_embed_dim
        self.y_embedding = nn.Embedding(y_slabs, y_embed_dim)

        # Fusion (third + third + y_dim channels → out_channels)
        fusion_in = third + third + y_embed_dim

        norm_groups = min(8, out_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        height_planes: torch.Tensor,  # [B, height_ch, H, W]
        biome_indices: torch.Tensor,  # [B, H, W]  int64
        y_index: torch.Tensor,  # [B]        int64
    ) -> torch.Tensor:  # [B, out_channels, H, W]

        B, _, H, W = height_planes.shape

        # --- height ---
        h_feat = self.height_conv(height_planes)  # [B, third, H, W]

        # --- biome ---
        bx = biome_indices.long().clamp(0, self.biome_embedding.num_embeddings - 1)
        biome_emb = self.biome_embedding(bx)  # [B, H, W, embed_dim]
        biome_emb = biome_emb.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]
        b_feat = self.biome_conv(biome_emb)  # [B, third, H, W]

        # --- y_index ---
        yi = y_index.long().clamp(0, self.y_slabs - 1)
        y_emb = self.y_embedding(yi)  # [B, y_embed_dim]
        y_feat = y_emb.view(B, self.y_embed_dim, 1, 1).expand(-1, -1, H, W)

        # --- fuse ---
        combined = torch.cat([h_feat, b_feat, y_feat], dim=1)
        return self.fusion(combined)  # [B, out_channels, H, W]
