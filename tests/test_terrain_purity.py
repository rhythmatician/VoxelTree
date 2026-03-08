"""
Tests for terrain purity — ensuring generated/extracted terrain is pristine.

These tests verify that our worldgen configuration prevents terrain from
evolving beyond its initial generation state. Training data must be exactly
as the Minecraft worldgen produced it, with no:

- Water flow (scheduled block ticks)
- Leaf decay (random ticks)
- Fire spread (random ticks)
- Mob terrain modification (mob griefing)
- Structure generation (villages, temples, etc.)
- Weather effects (lightning fires, snow, ice)

The primary safeguards are:
1. server.properties: generate-structures=false, peaceful, no spawning
2. Gamerules: randomTickSpeed=0, doFireTick=false, etc.
3. /tick freeze (MC 1.20.3+): halts ALL game ticking
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.worldgen.bootstrap import (
    STRUCTURE_ONLY_BLOCKS,
    TERRAIN_PURITY_GAMERULES,
    TICK_FREEZE_COMMAND,
    WorldGenBootstrap,
)


# ---------------------------------------------------------------
# 1. Constants are correct and comprehensive
# ---------------------------------------------------------------


class TestTerrainPurityConstants:
    """Verify that purity constants are comprehensive and correct."""

    def test_tick_freeze_command(self):
        """MC 1.20.3+ tick freeze must be the exact command string."""
        assert TICK_FREEZE_COMMAND == "tick freeze"

    def test_random_tick_speed_disabled(self):
        """randomTickSpeed=0 prevents leaf decay, crop growth, ice formation."""
        assert any("randomTickSpeed 0" in cmd for cmd in TERRAIN_PURITY_GAMERULES)

    def test_fire_tick_disabled(self):
        """doFireTick=false prevents fire spread."""
        assert any("doFireTick false" in cmd for cmd in TERRAIN_PURITY_GAMERULES)

    def test_mob_griefing_disabled(self):
        """mobGriefing=false prevents mobs from modifying terrain."""
        assert any("mobGriefing false" in cmd for cmd in TERRAIN_PURITY_GAMERULES)

    def test_weather_cycle_disabled(self):
        """doWeatherCycle=false prevents lightning fires, snow layers, ice."""
        assert any("doWeatherCycle false" in cmd for cmd in TERRAIN_PURITY_GAMERULES)

    def test_mob_spawning_disabled(self):
        """doMobSpawning=false prevents all mob spawning (belt-and-suspenders)."""
        assert any("doMobSpawning false" in cmd for cmd in TERRAIN_PURITY_GAMERULES)

    def test_all_gamerules_are_valid_format(self):
        """Each gamerule must start with 'gamerule ' (server console syntax)."""
        for cmd in TERRAIN_PURITY_GAMERULES:
            assert cmd.startswith("gamerule "), f"Invalid gamerule format: {cmd}"

    def test_no_duplicate_gamerules(self):
        """Each gamerule should appear exactly once."""
        assert len(TERRAIN_PURITY_GAMERULES) == len(set(TERRAIN_PURITY_GAMERULES))

    def test_structure_only_blocks_is_frozen(self):
        """STRUCTURE_ONLY_BLOCKS must be immutable."""
        assert isinstance(STRUCTURE_ONLY_BLOCKS, frozenset)

    def test_structure_only_blocks_not_empty(self):
        """Must have a meaningful set of structure-only blocks."""
        assert len(STRUCTURE_ONLY_BLOCKS) >= 10


# ---------------------------------------------------------------
# 2. server.properties has all required purity settings
# ---------------------------------------------------------------


class TestServerPropertiesPurity:
    """Verify server.properties written by bootstrap contains purity settings."""

    def setup_method(self):
        self.test_dir = Path(tempfile.mkdtemp())
        # Patch tool-path validation — Chunky JAR may not be on disk
        with patch.object(WorldGenBootstrap, "_validate_tool_paths"):
            self.bootstrap = WorldGenBootstrap(
                seed="PurityTest",
                java_heap="2G",
                temp_world_dir=self.test_dir / "temp",
                test_mode=True,
            )

    def teardown_method(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _write_server_properties(self) -> str:
        """Trigger server.properties creation and return its content."""
        world_path = self.test_dir / "test_world"
        world_path.mkdir(parents=True)

        # Mock subprocess to prevent actually starting a server
        with patch("subprocess.Popen"), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            try:
                self.bootstrap._start_fabric_server(world_path)
            except Exception:
                pass  # May fail on mocked subprocess

        props_file = world_path / "server.properties"
        assert props_file.exists(), "server.properties should have been created"
        return props_file.read_text()

    REQUIRED_PROPERTIES = {
        "generate-structures": "false",
        "difficulty": "peaceful",
        "spawn-npcs": "false",
        "spawn-animals": "false",
        "spawn-monsters": "false",
    }

    def test_generate_structures_false(self):
        """generate-structures=false is non-negotiable for training data."""
        content = self._write_server_properties()
        assert "generate-structures=false" in content

    def test_difficulty_peaceful(self):
        """Peaceful difficulty prevents hostile mob spawning."""
        content = self._write_server_properties()
        assert "difficulty=peaceful" in content

    def test_spawn_npcs_false(self):
        """No NPC spawning."""
        content = self._write_server_properties()
        assert "spawn-npcs=false" in content

    def test_spawn_animals_false(self):
        """No animal spawning."""
        content = self._write_server_properties()
        assert "spawn-animals=false" in content

    def test_spawn_monsters_false(self):
        """No monster spawning."""
        content = self._write_server_properties()
        assert "spawn-monsters=false" in content

    def test_all_required_properties_present(self):
        """Every required property must be present with the correct value."""
        content = self._write_server_properties()
        for key, value in self.REQUIRED_PROPERTIES.items():
            assert f"{key}={value}" in content, f"Missing required property: {key}={value}"


# ---------------------------------------------------------------
# 3. Terrain purity settings are applied correctly
# ---------------------------------------------------------------


class TestTerrainPurityApplication:
    """Verify purity settings method sends commands in correct order."""

    def setup_method(self):
        self.test_dir = Path(tempfile.mkdtemp())
        with patch.object(WorldGenBootstrap, "_validate_tool_paths"):
            self.bootstrap = WorldGenBootstrap(
                seed="PurityTest",
                java_heap="2G",
                temp_world_dir=self.test_dir / "temp",
                test_mode=False,
            )

    def teardown_method(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _mock_running_server(self):
        """Create a mock server process with writable stdin."""
        mock_stdin = MagicMock()
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.stdin = mock_stdin
        self.bootstrap.server_process = mock_process
        return mock_stdin

    def test_method_exists(self):
        """The _apply_terrain_purity_settings method must exist."""
        assert hasattr(self.bootstrap, "_apply_terrain_purity_settings")
        assert callable(self.bootstrap._apply_terrain_purity_settings)

    def test_sends_all_gamerules(self):
        """All gamerule commands must be written to server stdin."""
        mock_stdin = self._mock_running_server()

        with patch("time.sleep"):
            result = self.bootstrap._apply_terrain_purity_settings()

        assert result is True
        written = [c.args[0] for c in mock_stdin.write.call_args_list]
        for gamerule in TERRAIN_PURITY_GAMERULES:
            assert f"{gamerule}\n" in written, f"Missing gamerule: {gamerule}"

    def test_sends_tick_freeze(self):
        """tick freeze command must be sent."""
        mock_stdin = self._mock_running_server()

        with patch("time.sleep"):
            self.bootstrap._apply_terrain_purity_settings()

        written = [c.args[0] for c in mock_stdin.write.call_args_list]
        assert f"{TICK_FREEZE_COMMAND}\n" in written

    def test_tick_freeze_sent_after_gamerules(self):
        """tick freeze must be sent AFTER all gamerules.

        Gamerules take effect immediately when processed by the server.
        tick freeze halts the tick loop, so we want gamerules applied
        first as belt-and-suspenders, then freeze to stop everything.
        """
        mock_stdin = self._mock_running_server()

        with patch("time.sleep"):
            self.bootstrap._apply_terrain_purity_settings()

        written = [c.args[0] for c in mock_stdin.write.call_args_list]

        gamerule_indices = [i for i, w in enumerate(written) if w.startswith("gamerule")]
        tick_freeze_idx = written.index(f"{TICK_FREEZE_COMMAND}\n")

        assert all(
            gi < tick_freeze_idx for gi in gamerule_indices
        ), "All gamerules must be sent before tick freeze"

    def test_flush_called_after_each_command(self):
        """Each command write must be followed by a flush."""
        mock_stdin = self._mock_running_server()

        with patch("time.sleep"):
            self.bootstrap._apply_terrain_purity_settings()

        # Number of flushes should equal number of writes
        n_writes = mock_stdin.write.call_count
        n_flushes = mock_stdin.flush.call_count
        assert (
            n_flushes == n_writes
        ), f"Each write must be flushed: {n_writes} writes vs {n_flushes} flushes"

    def test_returns_false_without_server(self):
        """Must return False if no server process is running."""
        self.bootstrap.server_process = None
        result = self.bootstrap._apply_terrain_purity_settings()
        assert result is False

    def test_returns_false_with_dead_server(self):
        """Must return False if server process has exited."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process exited
        self.bootstrap.server_process = mock_process
        result = self.bootstrap._apply_terrain_purity_settings()
        assert result is False

    def test_returns_false_without_stdin(self):
        """Must return False if server stdin is not available."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdin = None
        self.bootstrap.server_process = mock_process
        result = self.bootstrap._apply_terrain_purity_settings()
        assert result is False


# ---------------------------------------------------------------
# 4. Structure block detection (extraction-time validation)
# ---------------------------------------------------------------


class TestStructureBlockDetection:
    """Verify we can detect blocks that indicate structure generation."""

    @pytest.fixture()
    def vocab(self):
        """Load the canonical Voxy vocabulary."""
        vocab_path = Path("config/voxy_vocab.json")
        if not vocab_path.exists():
            pytest.skip("Voxy vocab not available")
        with open(vocab_path) as f:
            return json.load(f)

    def test_structure_blocks_are_in_vocabulary(self, vocab):
        """Structure-only blocks must be in the Voxy vocab so we can
        detect them during extraction."""
        found = STRUCTURE_ONLY_BLOCKS & set(vocab.keys())
        assert len(found) > 0, "Expected some structure blocks in vocabulary"

    def test_structure_block_ids_resolvable(self, vocab):
        """Enough structure blocks should be detectable via vocab IDs."""
        detectable = {b: vocab[b] for b in STRUCTURE_ONLY_BLOCKS if b in vocab}
        assert len(detectable) >= 5, (
            f"Should detect ≥5 structure blocks, found "
            f"{len(detectable)}: {list(detectable.keys())}"
        )

    def test_air_is_not_a_structure_block(self):
        """Sanity: air should never be flagged as a structure block."""
        assert "minecraft:air" not in STRUCTURE_ONLY_BLOCKS

    def test_common_terrain_blocks_not_flagged(self):
        """Common terrain blocks must not be in the structure-only list."""
        terrain_blocks = {
            "minecraft:stone",
            "minecraft:dirt",
            "minecraft:grass_block",
            "minecraft:water",
            "minecraft:sand",
            "minecraft:gravel",
            "minecraft:oak_log",
            "minecraft:oak_leaves",
            "minecraft:deepslate",
            "minecraft:bedrock",
        }
        overlap = STRUCTURE_ONLY_BLOCKS & terrain_blocks
        assert len(overlap) == 0, (
            f"Terrain blocks incorrectly flagged as structure-only: " f"{overlap}"
        )


# ---------------------------------------------------------------
# 5. Extraction-time validation utility
# ---------------------------------------------------------------


def check_training_data_purity(
    data_dir: Path,
    vocab_path: Path = Path("config/voxy_vocab.json"),
    *,
    max_files: int = 0,
) -> dict:
    """Scan extracted NPZ training data for signs of terrain evolution.

    Returns a dict with:
      - files_checked: int
      - structure_blocks_found: dict[str, int]  (block_name -> count)
      - flagged_files: list[str]

    This is NOT a test — it's a utility that tests and the pipeline
    can call to validate extracted data.
    """
    import numpy as np

    with open(vocab_path) as f:
        vocab = json.load(f)

    # Build reverse map: id -> name
    id_to_name = {v: k for k, v in vocab.items()}

    # Build set of structure-only block IDs
    structure_ids = {vocab[b] for b in STRUCTURE_ONLY_BLOCKS if b in vocab}

    result: dict[str, Any] = {
        "files_checked": 0,
        "structure_blocks_found": {},
        "flagged_files": [],
    }

    npz_files = sorted(data_dir.glob("*.npz"))
    if max_files > 0:
        npz_files = npz_files[:max_files]

    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            labels = data["labels16"]
            result["files_checked"] += 1

            unique_ids = set(np.unique(labels).tolist())
            bad_ids = unique_ids & structure_ids

            if bad_ids:
                result["flagged_files"].append(str(npz_path.name))
                for bid in bad_ids:
                    name = id_to_name.get(bid, f"unknown_{bid}")
                    count = int((labels == bid).sum())
                    result["structure_blocks_found"][name] = (
                        result["structure_blocks_found"].get(name, 0) + count
                    )
        except Exception:
            pass  # Skip unreadable files

    return result


class TestPurityValidationUtility:
    """Test the check_training_data_purity utility itself."""

    def setup_method(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_clean_data_returns_no_flags(self):
        """Clean training data (all stone/air) should produce no flags."""
        import numpy as np

        vocab_path = Path("config/voxy_vocab.json")
        if not vocab_path.exists():
            pytest.skip("Voxy vocab not available")

        # Create a clean NPZ with only air (0) and stone
        with open(vocab_path) as f:
            vocab = json.load(f)
        stone_id = vocab.get("minecraft:stone", 1)

        labels = np.full((16, 16, 16), stone_id, dtype=np.int32)
        labels[:, 8:, :] = 0  # Air above y=8
        np.savez_compressed(
            self.test_dir / "clean_chunk.npz",
            labels16=labels,
            biome_patch=np.zeros((16, 16), dtype=np.int32),
            heightmap_patch=np.zeros((16, 16), dtype=np.float32),
            y_index=np.array(4, dtype=np.int64),
        )

        result = check_training_data_purity(self.test_dir, vocab_path)
        assert result["files_checked"] == 1
        assert len(result["structure_blocks_found"]) == 0
        assert len(result["flagged_files"]) == 0

    def test_contaminated_data_is_flagged(self):
        """Data containing structure blocks should be flagged."""
        import numpy as np

        vocab_path = Path("config/voxy_vocab.json")
        if not vocab_path.exists():
            pytest.skip("Voxy vocab not available")

        with open(vocab_path) as f:
            vocab = json.load(f)

        # Find a structure block that's in the vocab
        chest_id = vocab.get("minecraft:chest")
        if chest_id is None:
            pytest.skip("minecraft:chest not in vocab")

        labels = np.zeros((16, 16, 16), dtype=np.int32)
        labels[5, 5, 5] = chest_id  # Plant a chest

        np.savez_compressed(
            self.test_dir / "contaminated_chunk.npz",
            labels16=labels,
            biome_patch=np.zeros((16, 16), dtype=np.int32),
            heightmap_patch=np.zeros((16, 16), dtype=np.float32),
            y_index=np.array(4, dtype=np.int64),
        )

        result = check_training_data_purity(self.test_dir, vocab_path)
        assert result["files_checked"] == 1
        assert "minecraft:chest" in result["structure_blocks_found"]
        assert len(result["flagged_files"]) == 1
