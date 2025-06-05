"""
Tests for structure-aware fine-tuning functionality.
Phase 4: Structure-Aware Fine-Tuning - RED phase

These tests check that the model properly supports transfer learning
from a baseline terrain model to a structure-aware model.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from train.fine_tuning import (
    create_structure_aware_config,
    freeze_encoder_layers,
    load_baseline_weights,
)
from train.unet3d import UNet3DConfig, VoxelUNet3D


class TestStructureFineTuning:
    """Test suite for structure-aware fine-tuning functionality."""

    @pytest.fixture
    def baseline_model(self):
        """Create a baseline model for testing."""
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=16,
            depth=2,
            block_type_channels=4,
        )
        return VoxelUNet3D(config)

    @pytest.fixture
    def structure_aware_model(self):
        """Create a structure-aware model for testing."""
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=16,
            depth=2,
            block_type_channels=4,
            structure_enabled=True,
            structure_mask_channels=1,
            structure_type_count=10,
            structure_embed_dim=16,
            structure_pos_channels=2,
        )
        return VoxelUNet3D(config)

    @pytest.fixture
    def temp_checkpoint(self, baseline_model):
        """Create a temporary checkpoint file."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            # Save model state to temp file
            torch.save({"model_state": baseline_model.state_dict()}, f.name)
            yield Path(f.name)
        # Clean up
        try:
            Path(f.name).unlink()
        except FileNotFoundError:
            pass  # File may already be deleted

    def test_create_structure_aware_config(self):
        """Test creating structure-aware config from baseline config."""
        # Create baseline config
        base_config = UNet3DConfig()

        # Convert to structure-aware config
        structure_config = create_structure_aware_config(
            base_config,
            structure_type_count=8,
        )

        # Check that structure parameters are set
        assert structure_config.structure_enabled is True
        assert structure_config.structure_mask_channels == 1
        assert structure_config.structure_type_count == 8
        assert structure_config.structure_embed_dim == 16
        assert structure_config.structure_pos_channels == 2

    def test_freeze_encoder_layers(self, structure_aware_model):
        """Test freezing encoder layers for fine-tuning."""
        # Initially all parameters require grad
        assert all(param.requires_grad for param in structure_aware_model.parameters())

        # Freeze encoder layers
        freeze_encoder_layers(structure_aware_model)

        # Check that encoder parameters are frozen
        for name, param in structure_aware_model.named_parameters():
            if "encoder" in name:
                assert not param.requires_grad, f"Parameter {name} should be frozen"

            # Structure branches should remain unfrozen
            if "structure" in name:
                assert param.requires_grad, f"Parameter {name} should be unfrozen"

            # Decoder layers should remain unfrozen
            if "decoder" in name:
                assert param.requires_grad, f"Parameter {name} should be unfrozen"

    def test_load_baseline_weights(self, structure_aware_model, temp_checkpoint):
        """Test loading baseline weights for fine-tuning."""
        # Load baseline weights
        result = load_baseline_weights(structure_aware_model, temp_checkpoint, strict=False)

        # Should have both missing and unexpected keys
        assert len(result["missing_keys"]) > 0
        assert "structure" in " ".join(result["missing_keys"])

        # Try loading with strict=True (should fail)
        with pytest.raises(RuntimeError):
            load_baseline_weights(structure_aware_model, temp_checkpoint, strict=True)

    def test_structure_aware_forward_pass(self, structure_aware_model):
        """Test forward pass with structure-aware inputs."""
        batch_size = 2

        # Create test inputs
        parent_voxel = torch.randint(0, 2, (batch_size, 1, 8, 8, 8), dtype=torch.float32)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randint(60, 100, (batch_size, 1, 16, 16), dtype=torch.float32)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(0, 5, (batch_size,), dtype=torch.long)

        # Create structure inputs
        structure_mask = torch.randint(0, 2, (batch_size, 1, 8, 8), dtype=torch.float32)
        structure_types = torch.zeros((batch_size, 10), dtype=torch.float32)
        structure_types[:, 0] = 1.0  # First type active
        structure_positions = torch.randn(batch_size, 2)

        # Forward pass with structure inputs
        outputs = structure_aware_model(
            parent_voxel,
            biome_patch,
            heightmap_patch,
            river_patch,
            y_index,
            lod,
            structure_mask,
            structure_types,
            structure_positions,
        )

        # Check outputs
        assert "air_mask_logits" in outputs
        assert "block_type_logits" in outputs

        # Check output shapes
        assert outputs["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert outputs["block_type_logits"].shape == (batch_size, 4, 16, 16, 16)
