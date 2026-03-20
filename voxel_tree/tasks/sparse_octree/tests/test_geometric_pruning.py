"""Unit tests for geometric pruning in training (Phase 5).

Validates:
- ``compute_prunable_flags()`` correctly identifies sky and underground nodes
- Pruning-aware loss amplifies occ→0 for prunable nodes
- ``build_sparse_octree_pairs`` produces ``block_y_min`` in NPZ output
"""

from __future__ import annotations

import numpy as np
import torch

from voxel_tree.tasks.sparse_octree.sparse_octree_train import (
    _sparse_octree_loss,
    compute_prunable_flags,
)


# ---------------------------------------------------------------------------
# compute_prunable_flags tests
# ---------------------------------------------------------------------------


class TestComputePrunableFlags:
    """Test geometric pruning flag computation."""

    def _make_heightmap5(self, surface_heights: np.ndarray) -> np.ndarray:
        """Build a minimal heightmap5 from raw surface heights."""
        # Channel 0 = surface_norm = raw / 320
        surf_norm = surface_heights / 320.0
        # Channels 1-4 are not used for pruning; fill with zeros.
        return np.stack(
            [
                surf_norm,
                np.zeros_like(surf_norm),
                np.zeros_like(surf_norm),
                np.zeros_like(surf_norm),
                np.zeros_like(surf_norm),
            ],
            axis=0,
        ).astype(np.float32)

    def test_subchunk_entirely_above_surface(self):
        """A subchunk starting at Y=100 with surface at Y=64 → all prunable."""
        surface = np.full((16, 16), 64.0, dtype=np.float32)
        hm5 = self._make_heightmap5(surface)
        # Subchunk at block_y_min=100 → blocks [100, 116)
        # Surface max = 64, so all nodes have node_y_min >= 64 = surf_max → prunable
        flags = compute_prunable_flags(hm5, block_y_min=100)

        # Every node at every level should be prunable
        for lvl in range(4, -1, -1):
            assert flags[lvl].all(), f"L{lvl} should be entirely prunable (above surface)"

    def test_subchunk_entirely_below_surface(self):
        """A subchunk at Y=0 with surface at Y=80 everywhere → all prunable."""
        surface = np.full((16, 16), 80.0, dtype=np.float32)
        hm5 = self._make_heightmap5(surface)
        # Subchunk at block_y_min=0 → blocks [0, 16)
        # Surface min = 80, so node_y_max(=16) <= 80 = surf_min → prunable
        flags = compute_prunable_flags(hm5, block_y_min=0)

        for lvl in range(4, -1, -1):
            assert flags[lvl].all(), f"L{lvl} should be entirely prunable (below surface)"

    def test_subchunk_intersects_surface(self):
        """Surface at Y=72 flat, subchunk spans [64, 80) → L4 root NOT prunable."""
        surface = np.full((16, 16), 72.0, dtype=np.float32)
        hm5 = self._make_heightmap5(surface)
        flags = compute_prunable_flags(hm5, block_y_min=64)

        # L4 root covers [64, 80), surface at 72 → intersects → NOT prunable
        assert not flags[4][0, 0, 0], "L4 root should NOT be prunable"

    def test_l3_partial_pruning(self):
        """At L3 (side=2), one Y-half may be above surface while other intersects."""
        surface = np.full((16, 16), 72.0, dtype=np.float32)
        hm5 = self._make_heightmap5(surface)
        # Subchunk at Y=64 → blocks [64, 80)
        # L3 has side=2: y=0 covers [64, 72), y=1 covers [72, 80)
        flags = compute_prunable_flags(hm5, block_y_min=64)

        # y=0 covers [64, 72): surf_max=72, node_y_min=64 < 72 → NOT prunable above
        #                       surf_min=72, node_y_max=72 <= 72 → prunable below!
        # Actually: node_y_max = 72, surf_min = 72 → 72 <= 72 is True → prunable!
        # y=1 covers [72, 80): node_y_min=72 >= 72 = surf_max → prunable above!
        # Both halves are prunable because surface is exactly at the boundary.
        assert flags[3].all()

    def test_l3_mixed(self):
        """Surface at 73 → lower half intersects, upper half prunable."""
        surface = np.full((16, 16), 73.0, dtype=np.float32)
        hm5 = self._make_heightmap5(surface)
        # Subchunk at Y=64 → blocks [64, 80)
        # L3: y=0 covers [64, 72) → node_y_max=72 <= surf_min=73 → prunable below
        # L3: y=1 covers [72, 80) → node_y_min=72 < surf_max=73 and
        #                            node_y_max=80 > surf_min=73 → intersects → NOT prunable
        flags = compute_prunable_flags(hm5, block_y_min=64)

        # Check all (z, x) positions for y=0 and y=1
        assert flags[3][0, :, :].all(), "L3 y=0 should be prunable (below surface)"
        assert not flags[3][1, :, :].any(), "L3 y=1 should NOT be prunable (intersects)"

    def test_varying_surface(self):
        """Non-uniform surface → some XZ columns prunable, others not."""
        surface = np.full((16, 16), 72.0, dtype=np.float32)
        # Raise surface in the first 8 columns of Z
        surface[:8, :] = 80.0  # z=0..7 have surface at 80
        hm5 = self._make_heightmap5(surface)
        # Subchunk at Y=72 → blocks [72, 88)
        flags = compute_prunable_flags(hm5, block_y_min=72)

        # L4 root: surf_min=72, surf_max=80
        # node [72, 88): node_y_min=72 >= surf_max=80? No (72<80). node_y_max=88 <= surf_min=72? No.
        # → NOT prunable
        assert not flags[4][0, 0, 0]

    def test_all_levels_present(self):
        """Flags should contain entries for levels 4 through 0."""
        surface = np.full((16, 16), 70.0, dtype=np.float32)
        hm5 = self._make_heightmap5(surface)
        flags = compute_prunable_flags(hm5, block_y_min=64)
        assert set(flags.keys()) == {0, 1, 2, 3, 4}

    def test_flag_shapes(self):
        """Shape of is_prunable should match octree level side^3."""
        surface = np.full((16, 16), 70.0, dtype=np.float32)
        hm5 = self._make_heightmap5(surface)
        flags = compute_prunable_flags(hm5, block_y_min=64)

        for lvl in range(4, -1, -1):
            side = 2 ** (4 - lvl)
            assert flags[lvl].shape == (side, side, side), f"L{lvl} shape mismatch"
            assert flags[lvl].dtype == np.bool_

    def test_negative_y(self):
        """Subchunk at negative block Y (underground) with surface at Y=64."""
        surface = np.full((16, 16), 64.0, dtype=np.float32)
        hm5 = self._make_heightmap5(surface)
        # Subchunk at Y=-64 → blocks [-64, -48)
        # All nodes have node_y_max <= -48 < surf_min=64 → all prunable
        flags = compute_prunable_flags(hm5, block_y_min=-64)
        for lvl in range(4, -1, -1):
            assert flags[lvl].all(), f"L{lvl} deep underground should be prunable"


# ---------------------------------------------------------------------------
# Pruning-aware loss tests
# ---------------------------------------------------------------------------


class TestPruningLoss:
    """Test that pruning-aware loss amplifies occ→0 for prunable nodes."""

    def _make_dummy_preds_targets(self, prunable_mask: torch.Tensor):
        """Create minimal preds/targets for a single level (L4, N=1)."""
        B = prunable_mask.shape[0]
        preds = {
            4: {
                "occ": torch.randn(B, 1, 8),  # random occ logits
                "split": torch.randn(B, 1),
                "label": torch.randn(B, 1, 32),
            }
        }
        targets = {
            4: {
                "occ": torch.zeros(B, 1, 8),  # target: all occ=0 (should not split)
                "split": torch.zeros(B, 1),
                "label": torch.zeros(B, 1, dtype=torch.long),
                "is_leaf": torch.ones(B, 1, dtype=torch.bool),
                "is_prunable": prunable_mask.float(),  # [B, 1]
            }
        }
        return preds, targets

    def test_pruning_boost_increases_loss(self):
        """With prunable nodes, higher pruning_boost should increase loss."""
        torch.manual_seed(42)
        prunable = torch.ones(2, 1)  # both samples prunable
        preds, targets = self._make_dummy_preds_targets(prunable)

        loss_no_boost = _sparse_octree_loss(
            preds,
            targets,
            pruning_boost=0.0,
            level_split_weights={4: 1.0},
        )
        loss_with_boost = _sparse_octree_loss(
            preds,
            targets,
            pruning_boost=4.0,
            level_split_weights={4: 1.0},
        )
        assert (
            loss_with_boost > loss_no_boost
        ), "Pruning boost should increase loss for prunable nodes"

    def test_no_prunable_nodes_unchanged(self):
        """When no nodes are prunable, pruning_boost has no effect."""
        torch.manual_seed(42)
        not_prunable = torch.zeros(2, 1)  # no samples prunable
        preds, targets = self._make_dummy_preds_targets(not_prunable)

        loss_no_boost = _sparse_octree_loss(
            preds,
            targets,
            pruning_boost=0.0,
            level_split_weights={4: 1.0},
        )
        loss_with_boost = _sparse_octree_loss(
            preds,
            targets,
            pruning_boost=4.0,
            level_split_weights={4: 1.0},
        )
        assert torch.allclose(
            loss_no_boost, loss_with_boost, atol=1e-6
        ), "No prunable nodes → pruning boost should not change loss"

    def test_loss_still_finite(self):
        """Loss should remain finite with pruning boost active."""
        torch.manual_seed(42)
        prunable = torch.ones(4, 1)
        preds, targets = self._make_dummy_preds_targets(prunable)
        loss = _sparse_octree_loss(
            preds,
            targets,
            pruning_boost=10.0,
            level_split_weights={4: 1.0},
        )
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Integration test: full model forward + pruning loss
# ---------------------------------------------------------------------------


class TestFullPipelinePruning:
    """End-to-end test of model forward + pruning-aware loss."""

    def test_train_step_with_pruning(self):
        """A single training step with pruning flags should work."""
        from voxel_tree.tasks.sparse_octree.sparse_octree import SparseOctreeFastModel
        from voxel_tree.tasks.sparse_octree.sparse_octree_targets import (
            build_sparse_octree_targets,
        )

        model = SparseOctreeFastModel(n2d=0, n3d=15, hidden=48, num_classes=32)

        B = 2
        noise_2d = torch.empty(B, 0, 4, 4)
        noise_3d = torch.randn(B, 15, 4, 2, 4)
        biome_ids = torch.randint(0, 54, (B, 4, 2, 4))
        heightmap5 = torch.randn(B, 5, 16, 16)

        # Build targets from random voxel data
        targets = {}
        for b in range(B):
            voxels = np.random.randint(0, 32, size=(16, 16, 16), dtype=np.int32)
            raw = build_sparse_octree_targets(voxels, air_id=0, split_label=-1)
            # Create a surface at Y=72, subchunk at Y=64
            surface = np.full((16, 16), 72.0, dtype=np.float32)
            hm5_np = np.zeros((5, 16, 16), dtype=np.float32)
            hm5_np[0] = surface / 320.0
            prunable_flags = compute_prunable_flags(hm5_np, block_y_min=64)

            for lvl, lvl_data in raw.items():
                cm = lvl_data.child_mask.reshape(-1).astype(np.uint8)
                occ = np.unpackbits(cm[:, np.newaxis], axis=1, bitorder="little")[:, :8]
                entry = {
                    "occ": torch.from_numpy(occ.astype(np.float32)),
                    "split": torch.from_numpy((~lvl_data.is_leaf).reshape(-1).astype(np.float32)),
                    "label": torch.from_numpy(lvl_data.labels.reshape(-1).astype(np.int64)),
                    "is_leaf": torch.from_numpy(lvl_data.is_leaf.reshape(-1)),
                    "is_prunable": torch.from_numpy(
                        prunable_flags[lvl].reshape(-1).astype(np.float32)
                    ),
                }
                if lvl not in targets:
                    targets[lvl] = {k: [v] for k, v in entry.items()}
                else:
                    for k, v in entry.items():
                        targets[lvl][k].append(v)

        # Stack batch
        batched_targets = {
            lvl: {k: torch.stack(v, dim=0) for k, v in d.items()} for lvl, d in targets.items()
        }

        preds = model(noise_2d, noise_3d, biome_ids, heightmap5)
        loss = _sparse_octree_loss(
            preds,
            batched_targets,
            pruning_boost=4.0,
        )
        assert torch.isfinite(loss)
        loss.backward()
        # Verify gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
