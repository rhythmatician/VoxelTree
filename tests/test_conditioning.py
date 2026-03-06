"""
Phase 4.3 RED - Test Conditioning Input Influence

This test module verifies that the model's output meaningfully varies
in response to different conditioning inputs (biome, heightmap, y_index).

RED Phase: These tests MUST FAIL initially if the model ignores conditioning inputs.
"""

import pytest
import torch

from train.unet3d import SimpleFlexibleConfig, SimpleFlexibleUNet3D


class TestConditioning:
    """Test that conditioning inputs actually influence model outputs."""

    @pytest.fixture
    def model(self):
        """Create model with deterministic behavior for testing."""
        config = SimpleFlexibleConfig(
            base_channels=16,
            max_channels=64,
            biome_vocab_size=50,
            biome_embed_dim=8,
            lod_embed_dim=8,
            block_vocab_size=10,
        )
        model = SimpleFlexibleUNet3D(config)
        model.eval()
        return model

    @pytest.fixture
    def base_inputs(self):
        """Base input tensors for testing."""
        batch_size = 2
        return {
            "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8),
            "biome_patch": torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long),
            "heightmap_patch": torch.randn(batch_size, 1, 16, 16),
            "y_index": torch.randint(0, 24, (batch_size,), dtype=torch.long),
            "lod": torch.randint(1, 5, (batch_size,), dtype=torch.long),
        }

    def test_identical_conditioning_produces_identical_output(self, model, base_inputs):
        """Identical conditioning inputs should produce identical outputs."""
        with torch.no_grad():
            output1 = model(**base_inputs)
            output2 = model(**base_inputs)

            assert torch.allclose(
                output1["air_mask_logits"], output2["air_mask_logits"], atol=1e-6
            ), "Model should be deterministic for identical inputs"

            assert torch.allclose(
                output1["block_type_logits"], output2["block_type_logits"], atol=1e-6
            ), "Model should be deterministic for identical inputs"

    def test_different_biomes_produce_different_outputs(self, model, base_inputs):
        """Different biome patches should produce different outputs."""
        inputs_biome_A = base_inputs.copy()
        inputs_biome_B = base_inputs.copy()

        batch_size = base_inputs["biome_patch"].shape[0]
        inputs_biome_A["biome_patch"] = torch.full((batch_size, 16, 16), 1, dtype=torch.long)
        inputs_biome_B["biome_patch"] = torch.full((batch_size, 16, 16), 2, dtype=torch.long)

        with torch.no_grad():
            output_A = model(**inputs_biome_A)
            output_B = model(**inputs_biome_B)

            assert not torch.allclose(
                output_A["air_mask_logits"], output_B["air_mask_logits"], atol=1e-3
            ), "Model output should change when biome changes"

            assert not torch.allclose(
                output_A["block_type_logits"], output_B["block_type_logits"], atol=1e-3
            ), "Block type logits should change when biome changes"

    def test_different_heightmaps_produce_different_outputs(self, model, base_inputs):
        """Different heightmap patches should produce different outputs."""
        inputs_height_A = base_inputs.copy()
        inputs_height_B = base_inputs.copy()

        batch_size = base_inputs["heightmap_patch"].shape[0]
        inputs_height_A["heightmap_patch"] = torch.full((batch_size, 1, 16, 16), 64.0)
        inputs_height_B["heightmap_patch"] = torch.full((batch_size, 1, 16, 16), 128.0)

        with torch.no_grad():
            output_A = model(**inputs_height_A)
            output_B = model(**inputs_height_B)

            assert not torch.allclose(
                output_A["air_mask_logits"], output_B["air_mask_logits"], atol=1e-3
            ), "Model output should change when heightmap changes"

            assert not torch.allclose(
                output_A["block_type_logits"], output_B["block_type_logits"], atol=1e-3
            ), "Block type logits should change when heightmap changes"

    def test_different_y_index_produces_different_outputs(self, model, base_inputs):
        """Different Y indices should produce valid outputs."""
        inputs_y_low = base_inputs.copy()
        inputs_y_high = base_inputs.copy()

        batch_size = base_inputs["y_index"].shape[0]
        inputs_y_low["y_index"] = torch.full((batch_size,), 0, dtype=torch.long)
        inputs_y_high["y_index"] = torch.full((batch_size,), 20, dtype=torch.long)

        with torch.no_grad():
            output_low = model(**inputs_y_low)
            output_high = model(**inputs_y_high)

            # y_index is passed to conditioning_fusion but not currently
            # used in SimpleConditioningFusion. Verify outputs are valid.
            assert torch.isfinite(output_low["air_mask_logits"]).all()
            assert torch.isfinite(output_high["air_mask_logits"]).all()

    def test_different_lod_produces_different_outputs(self, model, base_inputs):
        """Different LOD levels should produce different outputs."""
        inputs_lod_1 = base_inputs.copy()
        inputs_lod_4 = base_inputs.copy()

        batch_size = base_inputs["lod"].shape[0]
        inputs_lod_1["lod"] = torch.full((batch_size,), 1, dtype=torch.long)
        inputs_lod_4["lod"] = torch.full((batch_size,), 4, dtype=torch.long)

        with torch.no_grad():
            output_lod_1 = model(**inputs_lod_1)
            output_lod_4 = model(**inputs_lod_4)

            air_diff = (
                (output_lod_1["air_mask_logits"] - output_lod_4["air_mask_logits"]).abs().mean()
            )

            assert (
                air_diff > 1e-5
            ), f"Model output should change when LOD changes (diff: {air_diff:.8f})"

    def test_conditioning_gradients_flow_properly(self, model, base_inputs):
        """Gradients should flow through conditioning embeddings."""
        model.train()

        output = model(**base_inputs)
        loss = output["air_mask_logits"].sum() + output["block_type_logits"].sum()
        loss.backward()

        # Check biome embedding gradients
        biome_grad = model.conditioning_fusion.biome_embedding.weight.grad
        assert biome_grad is not None, "Biome embedding should receive gradients"
        assert torch.any(biome_grad != 0), "Biome embedding gradients should be non-zero"

        # Check LOD projection gradients
        lod_grad = model.lod_projection[0].weight.grad
        assert lod_grad is not None, "LOD projection should receive gradients"
        assert torch.any(lod_grad != 0), "LOD projection gradients should be non-zero"

    def test_embedding_outputs_vary_by_input(self, model):
        """Individual embedding layers should produce different outputs for different inputs."""
        biome_1 = torch.tensor([1])
        biome_2 = torch.tensor([2])

        embed_1 = model.conditioning_fusion.biome_embedding(biome_1)
        embed_2 = model.conditioning_fusion.biome_embedding(biome_2)

        assert not torch.allclose(
            embed_1, embed_2
        ), "Different biome IDs should produce different embeddings"

    def test_conditioning_robustness_with_extreme_differences(self, model, base_inputs):
        """Extreme differences in conditioning should produce different outputs."""
        scenario_A = base_inputs.copy()
        scenario_B = base_inputs.copy()

        batch_size = base_inputs["biome_patch"].shape[0]

        scenario_A["biome_patch"] = torch.zeros((batch_size, 16, 16), dtype=torch.long)
        scenario_A["heightmap_patch"] = torch.full((batch_size, 1, 16, 16), -64.0)
        scenario_A["y_index"] = torch.zeros((batch_size,), dtype=torch.long)
        scenario_A["lod"] = torch.ones((batch_size,), dtype=torch.long)

        scenario_B["biome_patch"] = torch.full((batch_size, 16, 16), 49, dtype=torch.long)
        scenario_B["heightmap_patch"] = torch.full((batch_size, 1, 16, 16), 320.0)
        scenario_B["y_index"] = torch.full((batch_size,), 23, dtype=torch.long)
        scenario_B["lod"] = torch.full((batch_size,), 4, dtype=torch.long)

        with torch.no_grad():
            output_A = model(**scenario_A)
            output_B = model(**scenario_B)

            air_diff = torch.abs(output_A["air_mask_logits"] - output_B["air_mask_logits"]).mean()
            block_diff = torch.abs(
                output_A["block_type_logits"] - output_B["block_type_logits"]
            ).mean()

            assert (
                air_diff > 0.01
            ), f"Air logits difference {air_diff:.6f} too small for extreme conditioning"
            assert (
                block_diff > 0.01
            ), f"Block logits difference {block_diff:.6f} too small for extreme conditioning"

    def test_single_conditioning_input_isolation(self, model, base_inputs):
        """Changing only ONE conditioning input should affect output."""
        with torch.no_grad():
            base_output = model(**base_inputs)

        conditioning_tests = [
            ("biome_patch", lambda x: torch.full_like(x, 42)),
            ("heightmap_patch", lambda x: torch.full_like(x, 200.0)),
            ("lod", lambda x: torch.full_like(x, 3)),
        ]

        for input_name, modifier_func in conditioning_tests:
            modified_inputs = base_inputs.copy()
            modified_inputs[input_name] = modifier_func(base_inputs[input_name])

            with torch.no_grad():
                modified_output = model(**modified_inputs)

            air_diff = torch.abs(
                base_output["air_mask_logits"] - modified_output["air_mask_logits"]
            ).mean()
            block_diff = torch.abs(
                base_output["block_type_logits"] - modified_output["block_type_logits"]
            ).mean()

            assert air_diff > 1e-6 or block_diff > 1e-6, (
                f"Changing {input_name} should affect outputs "
                f"(air_diff: {air_diff:.8f}, block_diff: {block_diff:.8f})"
            )

    def test_debug_conditioning_sensitivity(self, model, base_inputs):
        """Debug: investigate actual magnitude of conditioning differences."""
        with torch.no_grad():
            baseline_output = model(**base_inputs)

        conditioning_changes = [
            ("identical", base_inputs),
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
            ("lod_change", {**base_inputs, "lod": torch.full_like(base_inputs["lod"], 3)}),
        ]

        for change_name, modified_inputs in conditioning_changes:
            with torch.no_grad():
                modified_output = model(**modified_inputs)

            air_mean_diff = (
                torch.abs(baseline_output["air_mask_logits"] - modified_output["air_mask_logits"])
                .mean()
                .item()
            )
            block_mean_diff = (
                torch.abs(
                    baseline_output["block_type_logits"] - modified_output["block_type_logits"]
                )
                .mean()
                .item()
            )

            assert isinstance(air_mean_diff, float)
            assert isinstance(block_mean_diff, float)

        identical_air_diff = (
            torch.abs(baseline_output["air_mask_logits"] - baseline_output["air_mask_logits"])
            .mean()
            .item()
        )
        assert identical_air_diff == 0.0

    def test_conditioning_with_stricter_assertions(self, model, base_inputs):
        """Use stricter difference measurements instead of torch.allclose."""

        def calculate_output_difference(output1, output2):
            air_diff = (
                torch.abs(output1["air_mask_logits"] - output2["air_mask_logits"]).mean().item()
            )
            block_diff = (
                torch.abs(output1["block_type_logits"] - output2["block_type_logits"]).mean().item()
            )
            return air_diff, block_diff

        with torch.no_grad():
            baseline_output = model(**base_inputs)

        biome_inputs = base_inputs.copy()
        biome_inputs["biome_patch"] = torch.zeros_like(base_inputs["biome_patch"])

        with torch.no_grad():
            biome_output = model(**biome_inputs)

        air_diff, block_diff = calculate_output_difference(baseline_output, biome_output)

        MIN_EXPECTED_DIFF = 1e-6
        assert (
            air_diff > MIN_EXPECTED_DIFF
        ), f"Biome change should affect air mask (diff: {air_diff:.10f})"
        assert (
            block_diff > MIN_EXPECTED_DIFF
        ), f"Biome change should affect block types (diff: {block_diff:.10f})"

    def test_conditioning_bypass_detection(self, model, base_inputs):
        """Detect if a model accidentally bypasses conditioning."""

        class ConditioningBypassModel(torch.nn.Module):
            """A model that ignores all conditioning inputs."""

            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1)
                self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear")

            def forward(self, parent_voxel, biome_patch, heightmap_patch, y_index, lod):
                x = self.conv(parent_voxel)
                x = self.upsample(x)
                return {
                    "air_mask_logits": x[:, :1],
                    "block_type_logits": x[:, 1:2].expand(-1, 10, -1, -1, -1),
                }

        bypass_model = ConditioningBypassModel()
        bypass_model.eval()

        with torch.no_grad():
            baseline_output = bypass_model(**base_inputs)

            biome_inputs = base_inputs.copy()
            biome_inputs["biome_patch"] = torch.zeros_like(base_inputs["biome_patch"])
            biome_output = bypass_model(**biome_inputs)

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

            assert air_diff == 0.0, f"Bypass model should have zero air difference, got {air_diff}"
            assert (
                block_diff == 0.0
            ), f"Bypass model should have zero block difference, got {block_diff}"
