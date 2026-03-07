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

#: Channel indices into router6 tensor.
ROUTER_TEMPERATURE: int = 0
ROUTER_VEGETATION: int = 1
ROUTER_CONTINENTALNESS: int = 2
ROUTER_EROSION: int = 3
ROUTER_DEPTH: int = 4
ROUTER_RIDGES: int = 5


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


def approximate_router6_from_biome(
    biome_indices: torch.Tensor, heightmap: torch.Tensor
) -> torch.Tensor:
    """
    Approximate router6 channels from biome indices and heightmap.

    Used when real noise-router values are unavailable (client-side runtime,
    old training data).  The approximation is deliberately coarse — it signals
    the *type* of terrain (ocean vs land, warm vs cold) without the fine spatial
    detail that comes from actual density functions.

    The 6 output channels follow the training convention:
      [temperature, vegetation, continentalness, erosion, depth, ridges]

    All output values are in the z-scored range expected by the model (mean≈0 std≈1).
    When dataset_stats.json is unavailable we assume mean=0, std=1 for all channels.

    Args:
        biome_indices: [B, H, W] integer biome IDs  (0..255)
        heightmap:     [B, 1, H, W] or [B, H, W] normalised (0..1)

    Returns:
        router6: [B, 6, H, W]  float32, approximate
    """
    if heightmap.dim() == 3:
        h = heightmap.unsqueeze(1).float()  # [B,1,H,W]
    elif heightmap.dim() == 5:
        h = heightmap.squeeze(-1).float()
    else:
        h = heightmap.float()

    B, _, H, W = h.shape
    device = h.device

    # Rough biome-ID–based climate approximations.
    # MC biome IDs 0..255; we categorise by known ranges:
    #   0-5:   Ocean (cold/cold/deep/warm/lukewarm/frozen)  → low continentalness
    #   6-9:   Plains / Meadow  → medium everything
    #   10-13: Desert / Savanna → high temperature
    #   14-27: Forest / jungle  → high vegetation
    #   28-32: Snowy            → low temperature
    # (This is approximate — actual mappings vary by MC version.)

    b = biome_indices.float()  # [B,H,W]

    # Temperature: high for desert(10-13)/savanna(15-16), low for frozen(28-32/0)
    temp = (b / 50.0 - 1.5).clamp(-1.5, 1.5)  # rough linear proxy

    # Vegetation: high for jungle(21-24)/forest(18-20), low for desert
    veg = torch.sin(b * 0.1).clamp(-1.5, 1.5)

    # Continentalness: oceans (0-5) get negative = "far from land"
    ocean_mask = (b < 6).float()
    cont = (h.squeeze(1) * 2.0 - 1.0) * (1.0 - ocean_mask) - 1.5 * ocean_mask

    # Erosion: inversely correlated with terrain height (low heights → more erosion)
    erosion = (1.0 - h.squeeze(1)) * 1.5 - 0.75

    # Depth: vertical positioning — use y_index proxy (filled downstream when available)
    depth = torch.zeros(B, H, W, device=device)

    # Ridges: peakiness — approximate with curvature of heightmap
    h_pad = nn.functional.pad(h, (1, 1, 1, 1), mode="reflect")
    d2x = h_pad[:, :, 1:-1, 2:] - 2 * h + h_pad[:, :, 1:-1, :-2]
    d2z = h_pad[:, :, 2:, 1:-1] - 2 * h + h_pad[:, :, :-2, 1:-1]
    ridges = (d2x + d2z).squeeze(1) * 5.0  # amplify curvature

    return torch.stack([temp, veg, cont, erosion, depth, ridges], dim=1)  # [B,6,H,W]


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class AnchorConditioningFusion(nn.Module):
    """
    Spatial conditioning fusion for all vanilla anchor channels.

    Architecture:
      height_planes → Conv2d(5, out//4) → GroupNorm → ReLU  ──┐
      router6       → Conv2d(6, out//2) → GroupNorm → ReLU  ──┤
      biome embed   → Conv2d(E, out//4) → GroupNorm → ReLU  ──┼─ cat ─→ Conv2d fusion ─→ out
      y_index embed → broadcast [B, y_dim, H, W]            ──┘

    Total input to fusion: out//4 + out//2 + out//4 + y_dim = out + y_dim channels.
    """

    def __init__(
        self,
        height_channels: int = 5,
        router6_channels: int = 6,
        biome_vocab_size: int = 256,
        biome_embed_dim: int = 32,
        y_embed_dim: int = 16,
        y_slabs: int = Y_SLABS,
        out_channels: int = 64,
    ):
        super().__init__()

        self.height_channels = height_channels
        self.router6_channels = router6_channels
        self.out_channels = out_channels

        quarter = max(out_channels // 4, 8)
        half = max(out_channels // 2, 16)

        # Height planes stream
        self.height_conv = nn.Sequential(
            nn.Conv2d(height_channels, quarter, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, quarter), quarter),
            nn.ReLU(inplace=True),
        )

        # Router-6 stream (dominant: temperature/erosion/continentalness drive macro shape)
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
        self.y_slabs = y_slabs
        self.y_embed_dim = y_embed_dim
        self.y_embedding = nn.Embedding(y_slabs, y_embed_dim)

        # Fusion (quarter + half + quarter + y_dim channels → out_channels)
        fusion_in = quarter + half + quarter + y_embed_dim

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
        router6: torch.Tensor,  # [B, 6, H, W]
        biome_indices: torch.Tensor,  # [B, H, W]  int64
        y_index: torch.Tensor,  # [B]        int64
    ) -> torch.Tensor:  # [B, out_channels, H, W]

        B, _, H, W = height_planes.shape

        # --- height ---
        h_feat = self.height_conv(height_planes)  # [B, quarter, H, W]

        # --- router6 ---
        r_feat = self.router6_conv(router6)  # [B, half, H, W]

        # --- biome ---
        bx = biome_indices.long().clamp(0, self.biome_embedding.num_embeddings - 1)
        biome_emb = self.biome_embedding(bx)  # [B, H, W, embed_dim]
        biome_emb = biome_emb.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]
        b_feat = self.biome_conv(biome_emb)  # [B, quarter, H, W]

        # --- y_index ---
        yi = y_index.long().clamp(0, self.y_slabs - 1)
        y_emb = self.y_embedding(yi)  # [B, y_embed_dim]
        y_feat = y_emb.view(B, self.y_embed_dim, 1, 1).expand(-1, -1, H, W)

        # --- fuse ---
        combined = torch.cat([h_feat, r_feat, b_feat, y_feat], dim=1)
        return self.fusion(combined)  # [B, out_channels, H, W]
