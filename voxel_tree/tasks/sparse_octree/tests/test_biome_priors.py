"""Unit tests for biome surface-rule priors (Phase 4).

Validates that ``biome_priors.py`` produces a correct biome → SurfaceType
lookup table matching ``HeightmapFallbackGenerator.surfaceTypeForBiome()``
in Java, and that the prior injection in ``_NoiseEncoder`` works end-to-end.
"""

from __future__ import annotations

import torch

from voxel_tree.tasks.sparse_octree.biome_priors import (
    NUM_SURFACE_TYPES,
    SurfaceType,
    biome_to_surface_type_table,
)


class TestSurfaceType:
    """SurfaceType enum correctness."""

    def test_count(self):
        assert NUM_SURFACE_TYPES == 8

    def test_ordinals(self):
        assert SurfaceType.GRASS == 0
        assert SurfaceType.SAND == 1
        assert SurfaceType.RED_SAND == 2
        assert SurfaceType.GRAVEL == 3
        assert SurfaceType.STONE == 4
        assert SurfaceType.SNOW == 5
        assert SurfaceType.PODZOL == 6
        assert SurfaceType.MYCELIUM == 7


class TestBiomeToSurfaceTypeTable:
    """Lookup table shape and correctness."""

    def test_shape(self):
        table = biome_to_surface_type_table(256)
        assert table.shape == (256,)
        assert table.dtype == torch.long

    def test_default_is_grass(self):
        table = biome_to_surface_type_table(256)
        # Unknown biome 255 should default to GRASS
        assert table[255].item() == SurfaceType.GRASS
        # Unassigned indices (e.g. 100) should also be GRASS
        assert table[100].item() == SurfaceType.GRASS

    def test_sand_biomes(self):
        """beach(2), deep_lukewarm_ocean(10), desert(12), lukewarm_ocean(24),
        snowy_beach(38), warm_ocean(48) → SAND."""
        table = biome_to_surface_type_table(256)
        for biome_id in [2, 10, 12, 24, 38, 48]:
            assert table[biome_id].item() == SurfaceType.SAND, f"biome {biome_id}"

    def test_red_sand_biomes(self):
        """badlands(0), eroded_badlands(14), wooded_badlands(53) → RED_SAND."""
        table = biome_to_surface_type_table(256)
        for biome_id in [0, 14, 53]:
            assert table[biome_id].item() == SurfaceType.RED_SAND, f"biome {biome_id}"

    def test_gravel_biomes(self):
        """cold_ocean(5), deep_cold_ocean(7), deep_frozen_ocean(9), deep_ocean(11),
        frozen_ocean(17), ocean(29), stony_peaks(43), stony_shore(44),
        windswept_gravelly_hills(50) → GRAVEL."""
        table = biome_to_surface_type_table(256)
        for biome_id in [5, 7, 9, 11, 17, 29, 43, 44, 50]:
            assert table[biome_id].item() == SurfaceType.GRAVEL, f"biome {biome_id}"

    def test_stone_biomes(self):
        """windswept_hills(51) → STONE."""
        table = biome_to_surface_type_table(256)
        assert table[51].item() == SurfaceType.STONE

    def test_snow_biomes(self):
        """frozen_peaks(18), frozen_river(19), grove(20), ice_spikes(21),
        jagged_peaks(22), snowy_plains(39), snowy_slopes(40), snowy_taiga(41) → SNOW."""
        table = biome_to_surface_type_table(256)
        for biome_id in [18, 19, 20, 21, 22, 39, 40, 41]:
            assert table[biome_id].item() == SurfaceType.SNOW, f"biome {biome_id}"

    def test_podzol_biomes(self):
        """old_growth_pine_taiga(31), old_growth_spruce_taiga(32) → PODZOL."""
        table = biome_to_surface_type_table(256)
        for biome_id in [31, 32]:
            assert table[biome_id].item() == SurfaceType.PODZOL, f"biome {biome_id}"

    def test_mycelium_biomes(self):
        """mushroom_fields(28) → MYCELIUM."""
        table = biome_to_surface_type_table(256)
        assert table[28].item() == SurfaceType.MYCELIUM

    def test_grass_biomes(self):
        """Spot-check that typical grassland biomes map to GRASS."""
        table = biome_to_surface_type_table(256)
        # plains(34), forest(16), jungle(23), meadow(27)
        for biome_id in [34, 16, 23, 27]:
            assert table[biome_id].item() == SurfaceType.GRASS, f"biome {biome_id}"

    def test_all_values_valid(self):
        table = biome_to_surface_type_table(256)
        assert (table >= 0).all()
        assert (table < NUM_SURFACE_TYPES).all()


class TestBiomePriorInModel:
    """End-to-end test that the prior injection doesn't break forward pass."""

    def test_noise_encoder_forward(self):
        """_NoiseEncoder with prior produces correct output shape."""
        from voxel_tree.tasks.sparse_octree.sparse_octree import _NoiseEncoder

        enc = _NoiseEncoder(n2d=0, n3d=15, hidden=64)
        B = 2
        noise_2d = torch.empty(B, 0, 4, 4)
        noise_3d = torch.randn(B, 15, 4, 2, 4)
        biome_ids = torch.randint(0, 54, (B, 4, 2, 4))
        heightmap5 = torch.randn(B, 5, 16, 16)

        ctx = enc(noise_2d, noise_3d, biome_ids, heightmap5)
        assert ctx.shape == (B, 64)

    def test_prior_differentiates_biomes(self):
        """Different biome surface types produce different conditioning vectors."""
        from voxel_tree.tasks.sparse_octree.sparse_octree import _NoiseEncoder

        enc = _NoiseEncoder(n2d=0, n3d=15, hidden=64)
        enc.eval()

        noise_2d = torch.empty(2, 0, 4, 4)
        noise_3d = torch.zeros(2, 15, 4, 2, 4)
        heightmap5 = torch.zeros(2, 5, 16, 16)

        # Batch 0: all desert (SAND=12), Batch 1: all snowy_plains (SNOW=39)
        biome_desert = torch.full((1, 4, 2, 4), 12, dtype=torch.long)
        biome_snow = torch.full((1, 4, 2, 4), 39, dtype=torch.long)
        biome_ids = torch.cat([biome_desert, biome_snow], dim=0)

        with torch.no_grad():
            ctx = enc(noise_2d, noise_3d, biome_ids, heightmap5)

        # The two samples should differ (prior + biome embed both differ)
        assert not torch.allclose(
            ctx[0], ctx[1]
        ), "Different biome types should produce different ctx"

    def test_full_model_forward_with_prior(self):
        """Full SparseOctreeFastModel forward works with biome priors."""
        from voxel_tree.tasks.sparse_octree.sparse_octree import SparseOctreeFastModel

        model = SparseOctreeFastModel(n2d=0, n3d=15, hidden=48, num_classes=32)
        model.eval()

        B = 2
        noise_2d = torch.empty(B, 0, 4, 4)
        noise_3d = torch.randn(B, 15, 4, 2, 4)
        biome_ids = torch.randint(0, 54, (B, 4, 2, 4))
        heightmap5 = torch.randn(B, 5, 16, 16)

        with torch.no_grad():
            outputs = model(noise_2d, noise_3d, biome_ids, heightmap5)

        assert set(outputs.keys()) == {0, 1, 2, 3, 4}
        # L4 has 1 node
        assert outputs[4]["occ"].shape == (B, 1, 8)
        assert outputs[4]["label"].shape == (B, 1, 32)
        # L0 has 8^4 = 4096 nodes
        assert outputs[0]["occ"].shape == (B, 4096, 8)
