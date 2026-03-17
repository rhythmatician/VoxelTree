"""
Unit tests for VoxelTree.preprocessing.harvest module.

Tests the orchestration logic for automated Voxy training data harvesting:
  - Spiral position generation
  - RocksDB discovery and size calculation
  - Database stabilisation monitoring
  - RCON command handling
  - Player connection waiting
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

from VoxelTree.preprocessing.harvest import (
    find_voxy_databases,
    get_db_size,
    spiral_positions,
    wait_for_player,
    wait_for_voxy_db,
)


class TestSpiralPositions:
    """Tests for the spiral position generator."""

    def test_spiral_center(self):
        """Spiral should start at the center."""
        positions = spiral_positions(radius_blocks=512, step_blocks=256)
        assert positions[0] == (0, 0), "First position should be the center"

    def test_spiral_covers_radius(self):
        """All positions should be within the specified radius."""
        radius = 512
        positions = spiral_positions(radius_blocks=radius, step_blocks=256)
        for x, z in positions:
            assert (
                abs(x) <= radius and abs(z) <= radius
            ), f"Position ({x}, {z}) outside radius {radius}"

    def test_spiral_respects_step(self):
        """Successive spiral positions should be step_blocks apart (Manhattan distance)."""
        step = 256
        positions = spiral_positions(radius_blocks=1024, step_blocks=step)
        # Check at least some consecutive jumps match the step (allowing for diagonal moves)
        for i in range(1, min(5, len(positions))):
            dx = abs(positions[i][0] - positions[i - 1][0])
            dz = abs(positions[i][1] - positions[i - 1][1])
            dist = (dx**2 + dz**2) ** 0.5
            # Allow small tolerance
            assert dist <= step * 1.5, f"Jump from {positions[i-1]} to {positions[i]} is too large"

    def test_spiral_custom_center(self):
        """Spiral should be offset by custom center coordinates."""
        center_x, center_z = 100, 200
        positions = spiral_positions(
            radius_blocks=256, step_blocks=128, center_x=center_x, center_z=center_z
        )
        assert positions[0] == (center_x, center_z), "First position should match custom center"

    def test_spiral_large_radius(self):
        """Spiral should handle large radius values."""
        positions = spiral_positions(radius_blocks=2048, step_blocks=256)
        assert len(positions) > 10, "Large radius should generate many positions"
        # Verify we have coverage far from center
        max_dist = max(abs(x) + abs(z) for x, z in positions)
        assert max_dist >= 1024, "Should reach far into the specified radius"

    def test_spiral_small_radius(self):
        """Spiral should work even with small radius."""
        positions = spiral_positions(radius_blocks=128, step_blocks=256)
        assert len(positions) >= 1, "Should have at least the center position"
        assert positions[0] == (0, 0)


class TestVoxyDatabaseDiscovery:
    """Tests for Voxy RocksDB discovery and monitoring."""

    def test_find_voxy_databases_empty(self):
        """Should return empty list for non-existent directory."""
        dbs = find_voxy_databases(Path("/nonexistent/path"))
        assert dbs == []

    def test_find_voxy_databases_no_matches(self):
        """Should return empty list if no RocksDB directories exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "save1").mkdir()
            (base / "save2").mkdir()
            dbs = find_voxy_databases(base)
            assert dbs == [], "Should not find RocksDB without CURRENT or .sst files"

    def test_find_voxy_databases_with_current_marker(self):
        """Should find RocksDB with CURRENT marker file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            storage = base / "save1" / "storage"
            storage.mkdir(parents=True)
            (storage / "CURRENT").touch()
            dbs = find_voxy_databases(base)
            assert len(dbs) == 1
            assert dbs[0] == storage

    def test_find_voxy_databases_with_sst_files(self):
        """Should find RocksDB with .sst files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            storage = base / "save1" / "storage"
            storage.mkdir(parents=True)
            (storage / "000001.sst").touch()
            (storage / "000002.sst").touch()
            dbs = find_voxy_databases(base)
            assert len(dbs) == 1

    def test_find_voxy_databases_multiple(self):
        """Should find multiple RocksDB directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            for i in range(3):
                storage = base / f"save{i}" / "storage"
                storage.mkdir(parents=True)
                (storage / "CURRENT").touch()
            dbs = find_voxy_databases(base)
            assert len(dbs) == 3


class TestDbSize:
    """Tests for database size calculation."""

    def test_get_db_size_empty(self):
        """Should return 0 for empty database list."""
        size = get_db_size([])
        assert size == 0

    def test_get_db_size_single_file(self):
        """Should calculate size of single .sst file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir)
            sst = db / "000001.sst"
            sst.write_bytes(b"x" * 1024)  # 1 KB file
            size = get_db_size([db])
            assert size == 1024

    def test_get_db_size_multiple_files(self):
        """Should sum sizes of all .sst files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir)
            (db / "000001.sst").write_bytes(b"x" * 1024)
            (db / "000002.sst").write_bytes(b"y" * 2048)
            (db / "000003.sst").write_bytes(b"z" * 512)
            size = get_db_size([db])
            assert size == 1024 + 2048 + 512

    def test_get_db_size_ignores_non_sst(self):
        """Should ignore non-.sst files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir)
            (db / "000001.sst").write_bytes(b"x" * 1024)
            (db / "CURRENT").write_bytes(b"000001\n")
            (db / "manifest.txt").write_bytes(b"y" * 10000)
            size = get_db_size([db])
            assert size == 1024, "Should only count .sst files"

    def test_get_db_size_multiple_databases(self):
        """Should sum sizes across multiple databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            db1 = base / "db1"
            db2 = base / "db2"
            db1.mkdir()
            db2.mkdir()
            (db1 / "000001.sst").write_bytes(b"a" * 1024)
            (db2 / "000001.sst").write_bytes(b"b" * 2048)
            size = get_db_size([db1, db2])
            assert size == 3072


class TestWaitForVoxyDb:
    """Tests for database stabilisation monitoring."""

    def test_wait_for_voxy_db_not_exists_timeout(self):
        """Should timeout if RocksDB never appears."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = wait_for_voxy_db(
                Path(tmpdir), stable_seconds=1, poll_interval=0.05, timeout=0.2
            )
            assert result is False

    def test_wait_for_voxy_db_appears_and_stabilises(self):
        """Should return True when DB appears and stabilises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            def create_db_after_delay():
                """Simulate database appearing after a delay."""
                time.sleep(0.1)
                storage = base / "save1" / "storage"
                storage.mkdir(parents=True)
                (storage / "000001.sst").write_bytes(b"x" * 1024)

            # Start a thread to create the DB
            import threading

            thread = threading.Thread(target=create_db_after_delay, daemon=True)
            thread.start()

            result = wait_for_voxy_db(base, stable_seconds=0.2, poll_interval=0.05, timeout=2)
            assert result is True

    def test_wait_for_voxy_db_detects_growth(self):
        """Should detect database growth and reset stability timer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            storage = base / "save1" / "storage"
            storage.mkdir(parents=True)

            # Simulate growing database
            def grow_db():
                for _ in range(3):
                    time.sleep(0.1)
                    # Append to a file to simulate growth
                    sst = storage / "000001.sst"
                    sst.write_bytes(sst.read_bytes() + b"x" * 512)

            import threading

            thread = threading.Thread(target=grow_db, daemon=True)
            thread.start()

            # Initial DB
            (storage / "000001.sst").write_bytes(b"x" * 1024)

            result = wait_for_voxy_db(base, stable_seconds=0.15, poll_interval=0.05, timeout=2)
            assert result is True


class TestRconHelpers:
    """Tests for RCON command and player waiting."""

    def test_wait_for_player_success(self):
        """Should return True when player connects."""
        mock_rcon = Mock()
        # First calls: no player, then player connects
        mock_rcon.command.side_effect = [
            "There are 0 of a max of 1 players online",
            "There are 0 of a max of 1 players online",
            "There are 1 of a max of 1 players online: Steve",
        ]

        result = wait_for_player(mock_rcon, timeout=10, poll=0.01)
        assert result is True

    def test_wait_for_player_timeout(self):
        """Should return False if no player connects in time."""
        mock_rcon = Mock()
        mock_rcon.command.return_value = "There are 0 of a max of 1 players online"

        result = wait_for_player(mock_rcon, timeout=0.1, poll=0.05)
        assert result is False

    def test_wait_for_player_handles_malformed_response(self):
        """Should handle unexpected response formats gracefully."""
        mock_rcon = Mock()
        mock_rcon.command.side_effect = [
            "Invalid response",
            "There are 1 of a max of 1 players online: Alice",
        ]

        result = wait_for_player(mock_rcon, timeout=10, poll=0.01)
        assert result is True

    def test_wait_for_player_case_insensitive(self):
        """Should be case-insensitive when checking response."""
        mock_rcon = Mock()
        mock_rcon.command.return_value = "THERE ARE 1 OF A MAX OF 1 PLAYERS ONLINE: BOB"

        result = wait_for_player(mock_rcon, timeout=1, poll=0.01)
        assert result is True
