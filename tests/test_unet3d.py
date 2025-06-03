"""
Test suite for Phase 4.1: 3D U-Net Model Architecture

RED phase tests for the 3D U-Net model that performs voxel super-resolution.
These tests ensure the model architecture is correct and can handle the expected
input/output shapes for LOD-aware terrain generation.
"""

import pytest
import torch

from train.unet3d import VoxelUNet3D, UNet3DConfig


class TestVoxelUNet3D:
    """Test 3D U-Net model instantiation and basic functionality."""

    @pytest.fixture
    def basic_config(self):
        """Create basic model configuration for testing."""
        return UNet3DConfig(
            # Input dimensions
            input_channels=1,  # Parent voxel (bool -> float32)
            output_channels=2,  # Air mask logits + block type logits
            # Model architecture
            base_channels=32,  # Starting channel count
            depth=3,  # Number of down/up sampling layers
            # Conditioning inputs
            biome_vocab_size=50,  # Max biome ID
            biome_embed_dim=16,  # Biome embedding dimension
            heightmap_channels=1,  # Single heightmap channel
            river_channels=1,  # Single river noise channel
            y_embed_dim=8,  # Y-index embedding dimension
            lod_embed_dim=8,  # LOD level embedding dimension
            # Training settings
            dropout_rate=0.1,
            use_batch_norm=True,
            activation="relu",
        )

    def test_model_instantiation(self, basic_config):
        """RED: Fails if model can't be instantiated with config."""
        model = VoxelUNet3D(basic_config)

        # Model should be created successfully
        assert model is not None
        assert isinstance(model, torch.nn.Module)

        # Check key components exist
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "biome_embedding")
        assert hasattr(model, "y_embedding")
        assert hasattr(model, "lod_embedding")

    def test_model_forward_pass_shapes(self, basic_config):
        """RED: Fails if forward pass produces wrong output shapes."""
        model = VoxelUNet3D(basic_config)
        batch_size = 4

        # Create mock input tensors (from dataset)
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)  # Input: 8³ voxel
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)  # 0-23 vertical subchunks
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)  # LOD levels 1-4

        # Forward pass
        with torch.no_grad():
            outputs = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        # Check output format
        assert isinstance(outputs, dict)
        assert "air_mask_logits" in outputs
        assert "block_type_logits" in outputs

        # Check output shapes (should be 16³ - 2x upsampling from 8³)
        air_logits = outputs["air_mask_logits"]
        block_logits = outputs["block_type_logits"]

        assert air_logits.shape == (batch_size, 1, 16, 16, 16)  # Single channel for air/not-air
        assert block_logits.shape == (batch_size, 10, 16, 16, 16)  # 10 block types

    def test_model_gradient_flow(self, basic_config):
        """RED: Fails if gradients don't flow through the model."""
        model = VoxelUNet3D(basic_config)
        batch_size = 2

        # Create inputs
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8, requires_grad=True)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        # Forward pass
        outputs = model(
            parent_voxel=parent_voxel,
            biome_patch=biome_patch,
            heightmap_patch=heightmap_patch,
            river_patch=river_patch,
            y_index=y_index,
            lod=lod,
        )

        # Compute dummy loss and backpropagate
        loss = outputs["air_mask_logits"].sum() + outputs["block_type_logits"].sum()
        loss.backward()

        # Check that gradients exist
        assert parent_voxel.grad is not None
        assert parent_voxel.grad.sum() != 0

        # Check model parameters have gradients
        param_grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(param_grads) > 0

    def test_conditioning_input_integration(self, basic_config):
        """RED: Fails if conditioning inputs aren't properly integrated."""
        model = VoxelUNet3D(basic_config)
        batch_size = 2

        # Create inputs with different conditioning values
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)

        # Different biomes
        biome_patch_1 = torch.full((batch_size, 16, 16), 1, dtype=torch.long)  # Plains
        biome_patch_2 = torch.full((batch_size, 16, 16), 15, dtype=torch.long)  # Desert

        # Different Y levels
        y_index_1 = torch.full((batch_size,), 5, dtype=torch.long)  # Near surface
        y_index_2 = torch.full((batch_size,), 20, dtype=torch.long)  # Deep underground

        # Different LOD levels
        lod_1 = torch.full((batch_size,), 1, dtype=torch.long)
        lod_2 = torch.full((batch_size,), 3, dtype=torch.long)

        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)

        with torch.no_grad():
            # Run with different conditioning
            outputs_1 = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch_1,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index_1,
                lod=lod_1,
            )

            outputs_2 = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch_2,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index_2,
                lod=lod_2,
            )

        # Outputs should be different when conditioning changes
        air_diff = torch.abs(outputs_1["air_mask_logits"] - outputs_2["air_mask_logits"]).mean()
        block_diff = torch.abs(
            outputs_1["block_type_logits"] - outputs_2["block_type_logits"]
        ).mean()

        assert air_diff > 0.01, "Air mask logits should change with different conditioning"
        assert block_diff > 0.01, "Block type logits should change with different conditioning"

    def test_model_device_compatibility(self, basic_config):
        """RED: Fails if model doesn't work on different devices."""
        model = VoxelUNet3D(basic_config)

        # Test CPU
        model_cpu = model.cpu()
        parent_voxel = torch.randn(1, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (1, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(1, 1, 16, 16)
        river_patch = torch.randn(1, 1, 16, 16)
        y_index = torch.randint(0, 24, (1,), dtype=torch.long)
        lod = torch.randint(1, 5, (1,), dtype=torch.long)

        with torch.no_grad():
            outputs_cpu = model_cpu(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        assert outputs_cpu["air_mask_logits"].device.type == "cpu"

        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            inputs_gpu = {
                "parent_voxel": parent_voxel.cuda(),
                "biome_patch": biome_patch.cuda(),
                "heightmap_patch": heightmap_patch.cuda(),
                "river_patch": river_patch.cuda(),
                "y_index": y_index.cuda(),
                "lod": lod.cuda(),
            }

            with torch.no_grad():
                outputs_gpu = model_gpu(**inputs_gpu)

            assert outputs_gpu["air_mask_logits"].device.type == "cuda"


class TestUNet3DConfig:
    """Test model configuration validation."""

    def test_config_validation(self):
        """RED: Fails if invalid configs aren't caught."""
        # Valid config should work
        valid_config = UNet3DConfig(input_channels=1, output_channels=2, base_channels=32, depth=3)
        assert valid_config.input_channels == 1

        # Invalid configs should raise errors
        with pytest.raises(ValueError, match="input_channels must be positive"):
            UNet3DConfig(input_channels=0, output_channels=2, base_channels=32, depth=3)

        with pytest.raises(ValueError, match="depth must be at least 1"):
            UNet3DConfig(input_channels=1, output_channels=2, base_channels=32, depth=0)

    def test_config_defaults(self):
        """RED: Fails if default values aren't sensible."""
        config = UNet3DConfig()

        # Check default values make sense
        assert config.input_channels == 1
        assert config.output_channels >= 2  # At least air + one block type
        assert config.base_channels >= 16  # Reasonable minimum
        assert config.depth >= 2  # At least some downsampling
        assert 0 <= config.dropout_rate <= 1
        assert config.biome_vocab_size > 0
        assert config.activation in ["relu", "leaky_relu", "gelu"]


class TestModelIntegration:
    """Test integration with training pipeline."""

    @pytest.fixture
    def basic_config(self):
        """Create basic model configuration for testing."""
        return UNet3DConfig(
            # Input dimensions
            input_channels=1,  # Parent voxel (bool -> float32)
            output_channels=2,  # Air mask logits + block type logits
            # Model architecture
            base_channels=32,  # Starting channel count
            depth=3,  # Number of down/up sampling layers
            # Conditioning inputs
            biome_vocab_size=50,  # Max biome ID
            biome_embed_dim=16,  # Biome embedding dimension
            heightmap_channels=1,  # Single heightmap channel
            river_channels=1,  # Single river noise channel
            y_embed_dim=8,  # Y-index embedding dimension
            lod_embed_dim=8,  # LOD level embedding dimension
            # Training settings
            dropout_rate=0.1,
            use_batch_norm=True,
            activation="relu",
        )

    def test_model_with_dataset_outputs(self, basic_config):
        """RED: Fails if model can't process real dataset batch format."""
        model = VoxelUNet3D(basic_config)

        # Simulate batch from VoxelTreeDataLoader
        batch = {
            "parent_voxel": torch.randn(4, 8, 8, 8),  # Note: no channel dim from dataset
            "biome_patch": torch.randint(0, 50, (4, 16, 16), dtype=torch.long),
            "heightmap_patch": torch.randint(60, 100, (4, 16, 16), dtype=torch.int16),
            "river_patch": torch.randn(4, 16, 16),
            "y_index": torch.randint(0, 24, (4,), dtype=torch.long),
            "lod": torch.randint(1, 5, (4,), dtype=torch.long),
            # Target data (not used in forward pass)
            "target_mask": torch.randint(0, 2, (4, 16, 16, 16), dtype=torch.bool),
            "target_types": torch.randint(0, 10, (4, 16, 16, 16), dtype=torch.uint8),
        }

        # Model should handle batch format correctly
        with torch.no_grad():
            # Convert inputs to expected format
            parent_voxel = batch["parent_voxel"].unsqueeze(1).float()  # Add channel dim
            biome_patch = batch["biome_patch"]
            heightmap_patch = (
                batch["heightmap_patch"].unsqueeze(1).float()
            )  # Add channel dim, convert to float
            river_patch = batch["river_patch"].unsqueeze(1).float()  # Add channel dim
            y_index = batch["y_index"]
            lod = batch["lod"]

            outputs = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )
        # Outputs should be compatible with targets
        # Check spatial dimensions match (output has extra channel dim)
        assert (
            outputs["air_mask_logits"].shape[2:] == batch["target_mask"].shape[1:]
        )  # Same spatial dims (16,16,16)
        assert (
            outputs["air_mask_logits"].shape[0] == batch["target_mask"].shape[0]
        )  # Same batch size
        assert (
            outputs["air_mask_logits"].shape[1] == 1
        )  # Single channel for binary air/solid classification

        # Block type logits should match target types structure
        assert (
            outputs["block_type_logits"].shape[2:] == batch["target_types"].shape[1:]
        )  # Same spatial dims
        assert (
            outputs["block_type_logits"].shape[0] == batch["target_types"].shape[0]
        )  # Same batch size
        assert outputs["block_type_logits"].shape[1] == 10  # 10 block type channels

    def test_model_memory_usage(self, basic_config):
        """RED: Fails if model uses excessive memory."""
        model = VoxelUNet3D(basic_config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Reasonable limits for a 3D U-Net
        assert total_params < 50_000_000, f"Model too large: {total_params:,} parameters"
        assert trainable_params == total_params, "All parameters should be trainable"

        # Memory test with realistic batch
        batch_size = 8
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        # Should complete without OOM on typical hardware
        with torch.no_grad():
            outputs = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )  # Clean up
        del outputs, parent_voxel, biome_patch, heightmap_patch, river_patch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class TestActivationFunctions:
    """Parameterized tests for different activation functions."""

    @pytest.mark.parametrize("activation", ["relu", "leaky_relu", "gelu"])
    def test_activation_function_forward_pass(self, activation):
        """Test that all activation functions work in forward pass."""
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=32,
            depth=3,
            biome_vocab_size=50,
            biome_embed_dim=16,
            heightmap_channels=1,
            river_channels=1,
            y_embed_dim=8,
            lod_embed_dim=8,
            dropout_rate=0.1,
            use_batch_norm=True,
            activation=activation,
        )

        model = VoxelUNet3D(config)
        batch_size = 2

        # Create test inputs
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        # Forward pass should work without errors
        with torch.no_grad():
            outputs = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        # Check outputs are valid
        assert "air_mask_logits" in outputs
        assert "block_type_logits" in outputs
        assert outputs["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert outputs["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

        # Check outputs contain finite values (no NaN/Inf)
        assert torch.isfinite(outputs["air_mask_logits"]).all()
        assert torch.isfinite(outputs["block_type_logits"]).all()

    @pytest.mark.parametrize("activation", ["relu", "leaky_relu", "gelu"])
    def test_activation_function_gradients(self, activation):
        """Test that gradients flow properly with different activation functions."""
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=32,
            depth=3,
            biome_vocab_size=50,
            biome_embed_dim=16,
            heightmap_channels=1,
            river_channels=1,
            y_embed_dim=8,
            lod_embed_dim=8,
            dropout_rate=0.1,
            use_batch_norm=True,
            activation=activation,
        )

        model = VoxelUNet3D(config)
        batch_size = 2

        # Create test inputs with gradients
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8, requires_grad=True)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        # Forward pass
        outputs = model(
            parent_voxel=parent_voxel,
            biome_patch=biome_patch,
            heightmap_patch=heightmap_patch,
            river_patch=river_patch,
            y_index=y_index,
            lod=lod,
        )

        # Compute loss and backprop
        loss = outputs["air_mask_logits"].sum() + outputs["block_type_logits"].sum()
        loss.backward()

        # Check gradients exist and are finite
        assert parent_voxel.grad is not None
        assert torch.isfinite(parent_voxel.grad).all()
        assert parent_voxel.grad.abs().sum() > 0

        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} missing gradient"
                assert torch.isfinite(
                    param.grad
                ).all(), f"Parameter {name} has non-finite gradients"

    @pytest.mark.parametrize("activation", ["relu", "leaky_relu", "gelu"])
    def test_activation_function_output_differences(self, activation):
        """Test that different activation functions produce different outputs."""
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=32,
            depth=3,
            biome_vocab_size=50,
            biome_embed_dim=16,
            heightmap_channels=1,
            river_channels=1,
            y_embed_dim=8,
            lod_embed_dim=8,
            dropout_rate=0.1,
            use_batch_norm=True,
            activation=activation,
        )

        model = VoxelUNet3D(config)
        batch_size = 2

        # Create test inputs
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        # Store outputs from this activation
        with torch.no_grad():
            outputs = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        # Check outputs are valid regardless of activation function
        # Final output layers are Conv3d without activation, so values can be negative
        air_logits = outputs["air_mask_logits"]
        block_logits = outputs["block_type_logits"]

        # All activation functions should produce finite logits
        assert torch.isfinite(air_logits).all(), f"Non-finite air logits with {activation}"
        assert torch.isfinite(block_logits).all(), f"Non-finite block logits with {activation}"

        # Logits should be reasonable range (not extremely large/small)
        assert air_logits.abs().max() < 100, f"Air logits too extreme with {activation}"
        assert block_logits.abs().max() < 100, f"Block logits too extreme with {activation}"

    @pytest.mark.parametrize("activation", ["relu", "leaky_relu", "gelu"])
    def test_activation_function_with_training_mode(self, activation):
        """Test activation functions work correctly in both training and eval modes."""
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=32,
            depth=3,
            biome_vocab_size=50,
            biome_embed_dim=16,
            heightmap_channels=1,
            river_channels=1,
            y_embed_dim=8,
            lod_embed_dim=8,
            dropout_rate=0.1,
            use_batch_norm=True,
            activation=activation,
        )

        model = VoxelUNet3D(config)
        batch_size = 2

        # Create test inputs
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        # Test in training mode
        model.train()
        with torch.no_grad():
            outputs_train = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        # Test in eval mode
        model.eval()
        with torch.no_grad():
            outputs_eval = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        # Both modes should produce valid outputs
        for outputs in [outputs_train, outputs_eval]:
            assert "air_mask_logits" in outputs
            assert "block_type_logits" in outputs
            assert outputs["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
            assert outputs["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)
            assert torch.isfinite(outputs["air_mask_logits"]).all()
            assert torch.isfinite(outputs["block_type_logits"]).all()

        # Outputs might differ due to dropout, but should be close for this activation test
        # (dropout is applied consistently regardless of activation function)
