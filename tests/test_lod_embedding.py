"""
RED Phase 4.4: LOD Timestep Embedding Tests

This module tests that the model's output varies meaningfully with different LOD values.
The core requirement is that changing the LOD input should produce measurably different
outputs, enabling the model to learn different refinement strategies at different scales.

Test Focus: "Fails if output doesn't vary by timestep"
"""

import pytest
import torch
import torch.nn as nn

from train.unet3d import UNet3DConfig, VoxelUNet3D


class TestLODTimestepEmbedding:
    """Test suite for LOD timestep embedding functionality."""

    @pytest.fixture
    def basic_config(self) -> UNet3DConfig:
        """Basic configuration for LOD embedding tests."""
        return UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=32,
            depth=3,
            biome_vocab_size=50,
            biome_embed_dim=16,
            heightmap_channels=1,
            river_channels=1,
            y_embed_dim=8,
            lod_embed_dim=32,  # Updated to match new enhanced LOD embedding
            dropout_rate=0.0,  # Disable dropout for deterministic testing
            use_batch_norm=True,
            activation="relu",
        )

    @pytest.fixture
    def model_and_inputs(self, basic_config):
        """Create model and consistent test inputs for LOD embedding tests."""
        model = VoxelUNet3D(basic_config)
        model.eval()  # Ensure deterministic behavior

        batch_size = 2

        # Create identical inputs (except LOD will vary)
        inputs = {
            "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8),
            "biome_patch": torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long),
            "heightmap_patch": torch.randn(batch_size, 1, 16, 16),
            "river_patch": torch.randn(batch_size, 1, 16, 16),
            "y_index": torch.randint(0, 24, (batch_size,), dtype=torch.long),
        }

        return model, inputs

    def test_output_varies_with_lod(self, model_and_inputs):
        """
        RED TEST: Verify that model outputs change meaningfully when LOD changes.

        This test should FAIL until LOD embedding is properly integrated and learned.
        """
        model, inputs = model_and_inputs

        # Create two different LOD values
        lod_low = torch.tensor([1, 1], dtype=torch.long)
        lod_high = torch.tensor([4, 4], dtype=torch.long)

        with torch.no_grad():
            # Forward pass with low LOD
            outputs_low = model(**inputs, lod=lod_low)

            # Forward pass with high LOD (same inputs, different LOD)
            outputs_high = model(**inputs, lod=lod_high)

        # Extract logits for comparison
        air_logits_low = outputs_low["air_mask_logits"]
        air_logits_high = outputs_high["air_mask_logits"]
        block_logits_low = outputs_low["block_type_logits"]
        block_logits_high = outputs_high["block_type_logits"]

        # Calculate differences
        air_diff = (air_logits_low - air_logits_high).abs().mean()
        block_diff = (block_logits_low - block_logits_high).abs().mean()

        # The core requirement: outputs MUST vary with LOD change
        min_expected_diff = 0.01  # Minimum meaningful difference

        assert air_diff > min_expected_diff, (
            f"Air mask logits should vary with LOD change. "
            f"Got mean difference: {air_diff.item():.6f}, expected > {min_expected_diff}"
        )

        assert block_diff > min_expected_diff, (
            f"Block type logits should vary with LOD change. "
            f"Got mean difference: {block_diff.item():.6f}, expected > {min_expected_diff}"
        )

    def test_output_deterministic_with_same_lod(self, model_and_inputs):
        """
        Verify that identical inputs with same LOD produce identical outputs.

        This ensures the LOD variation test is meaningful - if outputs are always
        different, the LOD test would be meaningless.
        """
        model, inputs = model_and_inputs

        # Same LOD for both runs
        lod = torch.tensor([2, 2], dtype=torch.long)

        with torch.no_grad():
            # First forward pass
            outputs1 = model(**inputs, lod=lod)

            # Second forward pass with identical inputs
            outputs2 = model(**inputs, lod=lod)

        # Outputs should be identical for deterministic model
        air_diff = (outputs1["air_mask_logits"] - outputs2["air_mask_logits"]).abs().max()
        block_diff = (outputs1["block_type_logits"] - outputs2["block_type_logits"]).abs().max()

        tolerance = 1e-6  # Very small tolerance for floating point errors

        assert air_diff < tolerance, (
            f"Air mask outputs should be deterministic with same inputs. "
            f"Got max difference: {air_diff.item():.8f}"
        )

        assert block_diff < tolerance, (
            f"Block type outputs should be deterministic with same inputs. "
            f"Got max difference: {block_diff.item():.8f}"
        )

    def test_lod_embedding_gradients_flow(self, basic_config):
        """
        Verify that LOD embeddings receive gradients during backpropagation.

        This ensures the LOD embedding is actually being learned during training.
        """
        model = VoxelUNet3D(basic_config)
        model.train()  # Enable training mode for gradient computation

        batch_size = 2

        # Create inputs
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.tensor([1, 3], dtype=torch.long)  # Different LOD values

        # Forward pass
        outputs = model(
            parent_voxel=parent_voxel,
            biome_patch=biome_patch,
            heightmap_patch=heightmap_patch,
            river_patch=river_patch,
            y_index=y_index,
            lod=lod,
        )

        # Compute loss (sum of all outputs for gradient flow)
        loss = outputs["air_mask_logits"].sum() + outputs["block_type_logits"].sum()

        # Backpropagate
        loss.backward()

        # Check that LOD embedding received gradients
        lod_embedding_grad = model.lod_embedding.weight.grad

        assert (
            lod_embedding_grad is not None
        ), "LOD embedding should receive gradients during backpropagation"

        assert (
            lod_embedding_grad.abs().sum() > 0
        ), "LOD embedding gradients should be non-zero, indicating learning"

    def test_different_lod_levels_produce_different_embeddings(self, basic_config):
        """
        Verify that different LOD values produce different embeddings.

        This tests the embedding layer itself, not the full model.
        """
        model = VoxelUNet3D(basic_config)

        # Test all valid LOD levels
        lod_levels = torch.tensor([1, 2, 3, 4], dtype=torch.long)

        # Get embeddings for each LOD level
        embeddings = model.lod_embedding(lod_levels)  # Shape: (4, lod_embed_dim)

        # Each embedding should be different
        for i in range(len(lod_levels)):
            for j in range(i + 1, len(lod_levels)):
                diff = (embeddings[i] - embeddings[j]).abs().sum()
                assert diff > 0, (
                    f"LOD embeddings for levels {lod_levels[i]} and {lod_levels[j]} "
                    f"should be different. Got difference: {diff.item()}"
                )

    def test_lod_embedding_bypass_fails_variation_test(self, model_and_inputs):
        """
        Create a model that ignores LOD and verify it FAILS the variation test.

        This is a negative test to ensure our main test is meaningful.
        A complete bypass of LOD conditioning requires disabling:
        1. The LOD embedding in ConditioningFusion
        2. The sinusoidal timestep embeddings via lod_projection
        3. The FiLM layer conditioning at each level
        """
        model, inputs = model_and_inputs

        # Manually zero out LOD embedding weights and disable other LOD conditioning mechanisms
        with torch.no_grad():
            # 1. Zero out the embedding vectors
            model.lod_embedding.weight.fill_(0.0)

            # 2. Zero out the sinusoidal projection layers
            if hasattr(model, "lod_projection"):
                # Access each layer in the Sequential module
                for name, module in model.lod_projection.named_modules():
                    if isinstance(module, nn.Linear):
                        module.weight.fill_(0.0)
                        module.bias.fill_(0.0)
            if hasattr(model, "encoder_film_layers"):
                for layer in model.encoder_film_layers:
                    if hasattr(layer, "scale_net"):
                        layer.scale_net.weight.fill_(0.0)
                        layer.scale_net.bias.fill_(0.0)
                    if hasattr(layer, "shift_net"):
                        layer.shift_net.weight.fill_(0.0)
                        layer.shift_net.bias.fill_(0.0)
            if hasattr(model, "decoder_film_layers"):
                for layer in model.decoder_film_layers:
                    if hasattr(layer, "scale_net"):
                        layer.scale_net.weight.fill_(0.0)
                        layer.scale_net.bias.fill_(0.0)
                    if hasattr(layer, "shift_net"):
                        layer.shift_net.weight.fill_(0.0)
                        layer.shift_net.bias.fill_(0.0)

        # Same test as test_output_varies_with_lod
        lod_low = torch.tensor([1, 1], dtype=torch.long)
        lod_high = torch.tensor([4, 4], dtype=torch.long)

        with torch.no_grad():
            outputs_low = model(**inputs, lod=lod_low)
            outputs_high = model(**inputs, lod=lod_high)

        # Calculate differences
        air_diff = (outputs_low["air_mask_logits"] - outputs_high["air_mask_logits"]).abs().mean()
        block_diff = (
            (outputs_low["block_type_logits"] - outputs_high["block_type_logits"]).abs().mean()
        )

        # With zeroed LOD embeddings, differences should be minimal
        # This test verifies that our main test would fail if LOD was truly ignored
        max_expected_diff = 1e-6  # Essentially zero

        # These assertions should pass, proving that zeroing LOD makes no difference
        assert air_diff < max_expected_diff, (
            f"With zeroed LOD embeddings, air outputs should be nearly identical. "
            f"Got difference: {air_diff.item():.8f}"
        )

        assert block_diff < max_expected_diff, (
            f"With zeroed LOD embeddings, block outputs should be nearly identical. "
            f"Got difference: {block_diff.item():.8f}"
        )

    @pytest.mark.parametrize("lod_level", [1, 2, 3, 4])
    def test_all_lod_levels_produce_valid_outputs(self, model_and_inputs, lod_level):
        """
        Verify that all LOD levels produce valid model outputs.

        Ensures no LOD level causes crashes or invalid tensor shapes.
        """
        model, inputs = model_and_inputs

        batch_size = 2
        lod = torch.tensor([lod_level, lod_level], dtype=torch.long)

        with torch.no_grad():
            outputs = model(**inputs, lod=lod)

        # Check output shapes
        assert outputs["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert outputs["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

        # Check outputs are finite
        assert torch.isfinite(outputs["air_mask_logits"]).all()
        assert torch.isfinite(outputs["block_type_logits"]).all()
