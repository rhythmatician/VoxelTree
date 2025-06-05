"""
Phase 4.3 RED - Test Conditioning Input Influence

This test module verifies that the model's output meaningfully varies
in response to different conditioning inputs (biome, heightmap, y_index).

RED Phase: These tests MUST FAIL initially if the model ignores conditioning inputs.
"""

import pytest
import torch

from train.unet3d import UNet3DConfig, VoxelUNet3D


class TestConditioning:
    """Test that conditioning inputs actually influence model outputs."""

    @pytest.fixture
    def model(self):
        """Create model with deterministic behavior for testing."""
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=16,  # Smaller for faster tests
            depth=2,  # Shallower for faster tests
            biome_vocab_size=50,
            biome_embed_dim=8,
            heightmap_channels=1,
            river_channels=1,
            y_embed_dim=4,
            lod_embed_dim=4,
            block_type_channels=10,
            dropout_rate=0.0,  # No dropout for deterministic testing
        )
        model = VoxelUNet3D(config)
        model.eval()  # Set to eval mode for deterministic behavior
        return model

    @pytest.fixture
    def base_inputs(self):
        """Base input tensors for testing."""
        batch_size = 2
        return {
            "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8),
            "biome_patch": torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long),
            "heightmap_patch": torch.randn(batch_size, 1, 16, 16),
            "river_patch": torch.randn(batch_size, 1, 16, 16),
            "y_index": torch.randint(0, 24, (batch_size,), dtype=torch.long),
            "lod": torch.randint(1, 5, (batch_size,), dtype=torch.long),
        }

    def test_identical_conditioning_produces_identical_output(self, model, base_inputs):
        """
        RED TEST: Identical conditioning inputs should produce identical outputs.

        This test ensures deterministic behavior and will pass once model
        properly processes conditioning inputs.
        """
        with torch.no_grad():
            # Run model twice with identical inputs
            output1 = model(**base_inputs)
            output2 = model(**base_inputs)

            # Outputs should be identical for identical inputs
            assert torch.allclose(
                output1["air_mask_logits"], output2["air_mask_logits"], atol=1e-6
            ), "Model should be deterministic for identical inputs"

            assert torch.allclose(
                output1["block_type_logits"], output2["block_type_logits"], atol=1e-6
            ), "Model should be deterministic for identical inputs"

    def test_different_biomes_produce_different_outputs(self, model, base_inputs):
        """
        RED TEST: Different biome patches should produce different outputs.

        This test will FAIL if the model ignores biome conditioning.
        """
        inputs_biome_A = base_inputs.copy()
        inputs_biome_B = base_inputs.copy()

        # Create distinctly different biome patches
        batch_size = base_inputs["biome_patch"].shape[0]
        inputs_biome_A["biome_patch"] = torch.full(
            (batch_size, 16, 16), 1, dtype=torch.long
        )  # Forest
        inputs_biome_B["biome_patch"] = torch.full(
            (batch_size, 16, 16), 2, dtype=torch.long
        )  # Desert

        with torch.no_grad():
            output_A = model(**inputs_biome_A)
            output_B = model(**inputs_biome_B)

            # Outputs should be different for different biomes
            assert not torch.allclose(
                output_A["air_mask_logits"], output_B["air_mask_logits"], atol=1e-3
            ), "Model output should change when biome changes"

            assert not torch.allclose(
                output_A["block_type_logits"], output_B["block_type_logits"], atol=1e-3
            ), "Block type logits should change when biome changes"

    def test_different_heightmaps_produce_different_outputs(self, model, base_inputs):
        """
        RED TEST: Different heightmap patches should produce different outputs.

        This test will FAIL if the model ignores heightmap conditioning.
        """
        inputs_height_A = base_inputs.copy()
        inputs_height_B = base_inputs.copy()

        # Create distinctly different heightmap patches
        batch_size = base_inputs["heightmap_patch"].shape[0]
        inputs_height_A["heightmap_patch"] = torch.full((batch_size, 1, 16, 16), 64.0)  # Sea level
        inputs_height_B["heightmap_patch"] = torch.full(
            (batch_size, 1, 16, 16), 128.0
        )  # Mountain level

        with torch.no_grad():
            output_A = model(**inputs_height_A)
            output_B = model(**inputs_height_B)

            # Outputs should be different for different heightmaps
            assert not torch.allclose(
                output_A["air_mask_logits"], output_B["air_mask_logits"], atol=1e-3
            ), "Model output should change when heightmap changes"

            assert not torch.allclose(
                output_A["block_type_logits"], output_B["block_type_logits"], atol=1e-3
            ), "Block type logits should change when heightmap changes"

    def test_different_y_index_produces_different_outputs(self, model, base_inputs):
        """
        RED TEST: Different Y indices should produce different outputs.

        This test will FAIL if the model ignores Y-level conditioning.
        """
        inputs_y_low = base_inputs.copy()
        inputs_y_high = base_inputs.copy()

        # Create distinctly different Y indices
        batch_size = base_inputs["y_index"].shape[0]
        inputs_y_low["y_index"] = torch.full((batch_size,), 0, dtype=torch.long)  # Bedrock level
        inputs_y_high["y_index"] = torch.full((batch_size,), 20, dtype=torch.long)  # Sky level

        with torch.no_grad():
            output_low = model(**inputs_y_low)
            output_high = model(**inputs_y_high)

            # Outputs should be different for different Y levels
            assert not torch.allclose(
                output_low["air_mask_logits"], output_high["air_mask_logits"], atol=1e-3
            ), "Model output should change when Y index changes"

            assert not torch.allclose(
                output_low["block_type_logits"], output_high["block_type_logits"], atol=1e-3
            ), "Block type logits should change when Y index changes"

    def test_different_river_patches_produce_different_outputs(self, model, base_inputs):
        """
        RED TEST: Different river patches should produce different outputs.

        This test will FAIL if the model ignores river conditioning.
        """
        inputs_no_river = base_inputs.copy()
        inputs_with_river = base_inputs.copy()

        # Create distinctly different river patches
        batch_size = base_inputs["river_patch"].shape[0]
        inputs_no_river["river_patch"] = torch.full((batch_size, 1, 16, 16), 0.0)  # No river
        inputs_with_river["river_patch"] = torch.full(
            (batch_size, 1, 16, 16), 1.0
        )  # Strong river signal

        with torch.no_grad():
            output_no_river = model(**inputs_no_river)
            output_with_river = model(**inputs_with_river)

            # Outputs should be different for different river signals
            assert not torch.allclose(
                output_no_river["air_mask_logits"], output_with_river["air_mask_logits"], atol=1e-3
            ), "Model output should change when river patch changes"

            assert not torch.allclose(
                output_no_river["block_type_logits"],
                output_with_river["block_type_logits"],
                atol=1e-3,
            ), "Block type logits should change when river patch changes"

    def test_different_lod_produces_different_outputs(self, model, base_inputs):
        """
        RED TEST: Different LOD levels should produce different outputs.

        This test will FAIL if the model ignores LOD conditioning.
        """
        inputs_lod_1 = base_inputs.copy()
        inputs_lod_4 = base_inputs.copy()

        # Create distinctly different LOD levels
        batch_size = base_inputs["lod"].shape[0]
        inputs_lod_1["lod"] = torch.full((batch_size,), 1, dtype=torch.long)  # Fine detail
        inputs_lod_4["lod"] = torch.full((batch_size,), 4, dtype=torch.long)  # Coarse detail

        with torch.no_grad():
            output_lod_1 = model(**inputs_lod_1)
            output_lod_4 = model(**inputs_lod_4)

            # Outputs should be different for different LOD levels
            assert not torch.allclose(
                output_lod_1["air_mask_logits"], output_lod_4["air_mask_logits"], atol=1e-3
            ), "Model output should change when LOD changes"

            assert not torch.allclose(
                output_lod_1["block_type_logits"], output_lod_4["block_type_logits"], atol=1e-3
            ), "Block type logits should change when LOD changes"

    def test_conditioning_gradients_flow_properly(self, model, base_inputs):
        """
        RED TEST: Gradients should flow through conditioning embeddings.

        This test ensures conditioning inputs actually affect the loss.
        """
        model.train()  # Enable gradients

        # Create a simple loss that should depend on conditioning
        output = model(**base_inputs)
        loss = output["air_mask_logits"].sum() + output["block_type_logits"].sum()
        loss.backward()

        # Check that conditioning embeddings have gradients
        assert (
            model.biome_embedding.weight.grad is not None
        ), "Biome embedding should receive gradients"
        assert torch.any(
            model.biome_embedding.weight.grad != 0
        ), "Biome embedding gradients should be non-zero"

        assert model.y_embedding.weight.grad is not None, "Y embedding should receive gradients"
        assert torch.any(
            model.y_embedding.weight.grad != 0
        ), "Y embedding gradients should be non-zero"

        assert model.lod_embedding.weight.grad is not None, "LOD embedding should receive gradients"
        assert torch.any(
            model.lod_embedding.weight.grad != 0
        ), "LOD embedding gradients should be non-zero"

    def test_embedding_outputs_vary_by_input(self, model):
        """
        RED TEST: Individual embedding layers should produce different outputs.

        This test verifies the embedding layers themselves work correctly.
        """
        # Test biome embedding
        biome_1 = torch.tensor([1])
        biome_2 = torch.tensor([2])

        embed_1 = model.biome_embedding(biome_1)
        embed_2 = model.biome_embedding(biome_2)

        assert not torch.allclose(
            embed_1, embed_2
        ), "Different biome IDs should produce different embeddings"

        # Test Y embedding
        y_1 = torch.tensor([5])
        y_2 = torch.tensor([15])

        y_embed_1 = model.y_embedding(y_1)
        y_embed_2 = model.y_embedding(y_2)

        assert not torch.allclose(
            y_embed_1, y_embed_2
        ), "Different Y indices should produce different embeddings"

        # Test LOD embedding
        lod_1 = torch.tensor([1])
        lod_2 = torch.tensor([4])

        lod_embed_1 = model.lod_embedding(lod_1)
        lod_embed_2 = model.lod_embedding(lod_2)

        assert not torch.allclose(
            lod_embed_1, lod_embed_2
        ), "Different LOD levels should produce different embeddings"

    def test_conditioning_robustness_with_extreme_differences(self, model, base_inputs):
        """
        RED TEST: Extreme differences in conditioning should produce very different outputs.

        This test uses maximum contrast in conditioning inputs to ensure
        the model is truly sensitive to these inputs.
        """
        # Create maximally different conditioning scenarios
        scenario_A = base_inputs.copy()
        scenario_B = base_inputs.copy()

        batch_size = base_inputs["biome_patch"].shape[0]

        # Scenario A: Low everything
        scenario_A["biome_patch"] = torch.zeros((batch_size, 16, 16), dtype=torch.long)  # Biome 0
        scenario_A["heightmap_patch"] = torch.full((batch_size, 1, 16, 16), -64.0)  # Lowest height
        scenario_A["river_patch"] = torch.zeros((batch_size, 1, 16, 16))  # No river
        scenario_A["y_index"] = torch.zeros((batch_size,), dtype=torch.long)  # Bottom Y
        scenario_A["lod"] = torch.ones((batch_size,), dtype=torch.long)  # Finest LOD

        # Scenario B: High everything
        scenario_B["biome_patch"] = torch.full(
            (batch_size, 16, 16), 49, dtype=torch.long
        )  # Max biome
        scenario_B["heightmap_patch"] = torch.full((batch_size, 1, 16, 16), 320.0)  # Highest height
        scenario_B["river_patch"] = torch.ones((batch_size, 1, 16, 16))  # Max river
        scenario_B["y_index"] = torch.full((batch_size,), 23, dtype=torch.long)  # Top Y
        scenario_B["lod"] = torch.full((batch_size,), 4, dtype=torch.long)  # Coarsest LOD

        with torch.no_grad():
            output_A = model(**scenario_A)
            output_B = model(**scenario_B)

            # Calculate difference magnitude
            air_diff = torch.abs(output_A["air_mask_logits"] - output_B["air_mask_logits"]).mean()
            block_diff = torch.abs(
                output_A["block_type_logits"] - output_B["block_type_logits"]
            ).mean()

            # Outputs should be VERY different for extreme conditioning differences
            assert (
                air_diff > 0.1
            ), f"Air mask logits difference {air_diff:.6f} too small for extreme conditioning changes"
            assert (
                block_diff > 0.1
            ), f"Block type logits difference {block_diff:.6f} too small for extreme conditioning changes"

    def test_single_conditioning_input_isolation(self, model, base_inputs):
        """
        RED TEST: Changing only ONE conditioning input should affect output.

        This test isolates each conditioning input to verify it has independent influence.
        """
        base_output = None
        with torch.no_grad():
            base_output = model(**base_inputs)

        # Test each conditioning input in isolation
        conditioning_tests = [
            ("biome_patch", lambda x: torch.full_like(x, 42 if x.dtype == torch.long else 42.0)),
            ("heightmap_patch", lambda x: torch.full_like(x, 200.0)),
            ("river_patch", lambda x: torch.full_like(x, 0.8)),
            ("y_index", lambda x: torch.full_like(x, 10)),
            ("lod", lambda x: torch.full_like(x, 3)),
        ]

        for input_name, modifier_func in conditioning_tests:
            modified_inputs = base_inputs.copy()
            modified_inputs[input_name] = modifier_func(base_inputs[input_name])

            with torch.no_grad():
                modified_output = model(**modified_inputs)

            # Each conditioning input should independently affect the output
            air_diff = torch.abs(
                base_output["air_mask_logits"] - modified_output["air_mask_logits"]
            ).mean()
            block_diff = torch.abs(
                base_output["block_type_logits"] - modified_output["block_type_logits"]
            ).mean()

            assert (
                air_diff > 1e-4
            ), f"Changing {input_name} should affect air mask logits (diff: {air_diff:.8f})"
            assert (
                block_diff > 1e-4
            ), f"Changing {input_name} should affect block type logits (diff: {block_diff:.8f})"

    def test_debug_conditioning_sensitivity(self, model, base_inputs):
        """
        DEBUG TEST: Investigate actual magnitude of differences to calibrate assertions.

        This test helps us understand if our assertions are too weak or too strong.
        """
        print("\n🔍 DEBUGGING CONDITIONING SENSITIVITY:")

        # Baseline output
        with torch.no_grad():
            baseline_output = model(**base_inputs)

        # Test each conditioning input and measure actual differences
        conditioning_changes = [
            ("identical", base_inputs),  # Should be zero difference
            (
                "biome_change",
                {**base_inputs, "biome_patch": torch.full_like(base_inputs["biome_patch"], 42)},
            ),
            (
                "heightmap_change",
                {
                    **base_inputs,
                    "heightmap_patch": torch.full_like(base_inputs["heightmap_patch"], 200.0),
                },
            ),
            (
                "river_change",
                {**base_inputs, "river_patch": torch.full_like(base_inputs["river_patch"], 0.9)},
            ),
            (
                "y_index_change",
                {**base_inputs, "y_index": torch.full_like(base_inputs["y_index"], 15)},
            ),
            ("lod_change", {**base_inputs, "lod": torch.full_like(base_inputs["lod"], 3)}),
        ]

        for change_name, modified_inputs in conditioning_changes:
            with torch.no_grad():
                modified_output = model(**modified_inputs)

            # Calculate multiple difference metrics
            air_mean_diff = (
                torch.abs(baseline_output["air_mask_logits"] - modified_output["air_mask_logits"])
                .mean()
                .item()
            )
            air_max_diff = (
                torch.abs(baseline_output["air_mask_logits"] - modified_output["air_mask_logits"])
                .max()
                .item()
            )

            block_mean_diff = (
                torch.abs(
                    baseline_output["block_type_logits"] - modified_output["block_type_logits"]
                )
                .mean()
                .item()
            )
            block_max_diff = (
                torch.abs(
                    baseline_output["block_type_logits"] - modified_output["block_type_logits"]
                )
                .max()
                .item()
            )

            # Check if torch.allclose would pass (this reveals if our assertions are too weak)
            air_allclose = torch.allclose(
                baseline_output["air_mask_logits"], modified_output["air_mask_logits"], atol=1e-3
            )
            block_allclose = torch.allclose(
                baseline_output["block_type_logits"],
                modified_output["block_type_logits"],
                atol=1e-3,
            )

            print(
                f"  {change_name:15} | Air: mean={air_mean_diff:.8f}, max={air_max_diff:.6f}, allclose={air_allclose}"
            )
            print(
                f"  {'':<15} | Block: mean={block_mean_diff:.8f}, max={block_max_diff:.6f}, allclose={block_allclose}"
            )

        # The identical case should have zero difference
        identical_air_diff = (
            torch.abs(baseline_output["air_mask_logits"] - baseline_output["air_mask_logits"])
            .mean()
            .item()
        )
        assert (
            identical_air_diff == 0.0
        ), f"Identical inputs should have zero difference, got {identical_air_diff}"

    def test_conditioning_with_stricter_assertions(self, model, base_inputs):
        """
        RED TEST: Use stricter difference measurements instead of torch.allclose.

        This test should FAIL if conditioning inputs are being ignored.
        """

        def calculate_output_difference(output1, output2):
            """Calculate meaningful difference between two model outputs."""
            air_diff = (
                torch.abs(output1["air_mask_logits"] - output2["air_mask_logits"]).mean().item()
            )
            block_diff = (
                torch.abs(output1["block_type_logits"] - output2["block_type_logits"]).mean().item()
            )
            return air_diff, block_diff

        # Baseline output
        with torch.no_grad():
            baseline_output = model(**base_inputs)

        # Test biome conditioning with extreme change
        biome_inputs = base_inputs.copy()
        biome_inputs["biome_patch"] = torch.zeros_like(
            base_inputs["biome_patch"]
        )  # Change to biome 0

        with torch.no_grad():
            biome_output = model(**biome_inputs)

        air_diff, block_diff = calculate_output_difference(baseline_output, biome_output)

        # These should be meaningful differences if conditioning works
        MIN_EXPECTED_DIFF = 1e-6  # Very conservative threshold
        assert (
            air_diff > MIN_EXPECTED_DIFF
        ), f"Biome change should affect air mask (diff: {air_diff:.10f})"
        assert (
            block_diff > MIN_EXPECTED_DIFF
        ), f"Biome change should affect block types (diff: {block_diff:.10f})"

        # Test Y-index conditioning
        y_inputs = base_inputs.copy()
        y_inputs["y_index"] = torch.zeros_like(base_inputs["y_index"])  # Change to Y=0

        with torch.no_grad():
            y_output = model(**y_inputs)

        air_diff_y, block_diff_y = calculate_output_difference(baseline_output, y_output)

        assert (
            air_diff_y > MIN_EXPECTED_DIFF
        ), f"Y-index change should affect air mask (diff: {air_diff_y:.10f})"
        assert (
            block_diff_y > MIN_EXPECTED_DIFF
        ), f"Y-index change should affect block types (diff: {block_diff_y:.10f})"

    def test_conditioning_bypass_detection(self, model, base_inputs):
        """
        CRITICAL RED TEST: Detect if model accidentally bypasses conditioning.

        This test creates a model that we KNOW ignores conditioning and verifies
        our tests would catch it.
        """

        class ConditioningBypassModel(torch.nn.Module):
            """A model that ignores all conditioning inputs - should FAIL our tests."""

            def __init__(self, config):
                super().__init__()
                # Only use the parent voxel, ignore all conditioning
                self.conv = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1)
                self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear")

            def forward(
                self, parent_voxel, biome_patch, heightmap_patch, river_patch, y_index, lod
            ):
                # Deliberately ignore all conditioning inputs
                x = self.conv(parent_voxel)
                x = self.upsample(x)
                return {
                    "air_mask_logits": x[:, :1],
                    "block_type_logits": x[:, 1:2].expand(
                        -1, 10, -1, -1, -1
                    ),  # Fake 10 block types
                }

        # Create bypass model
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=16,
            depth=2,
            biome_vocab_size=50,
            biome_embed_dim=8,
            heightmap_channels=1,
            river_channels=1,
            y_embed_dim=4,
            lod_embed_dim=4,
            block_type_channels=10,
            dropout_rate=0.0,
        )
        bypass_model = ConditioningBypassModel(config)
        bypass_model.eval()

        # Test if our assertions would catch the bypass
        with torch.no_grad():
            baseline_output = bypass_model(**base_inputs)

            # Change biome
            biome_inputs = base_inputs.copy()
            biome_inputs["biome_patch"] = torch.zeros_like(base_inputs["biome_patch"])
            biome_output = bypass_model(**biome_inputs)

            # Calculate difference
            air_diff = (
                torch.abs(baseline_output["air_mask_logits"] - biome_output["air_mask_logits"])
                .mean()
                .item()
            )
            block_diff = (
                torch.abs(baseline_output["block_type_logits"] - biome_output["block_type_logits"])
                .mean()
                .item()
            )

            # These should be ZERO for the bypass model
            assert air_diff == 0.0, f"Bypass model should have zero air difference, got {air_diff}"
            assert (
                block_diff == 0.0
            ), f"Bypass model should have zero block difference, got {block_diff}"

        print("✅ Bypass detection working: bypass model shows zero difference as expected")
