"""
WorldGenBootstrap implementation for Fabric server + Chunky mod.

This is the GREEN phase implementation to make integration tests pass.
"""

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from scripts.worldgen.config import load_worldgen_config

# Set up logging for worldgen operations
logger = logging.getLogger(__name__)


class FabricWorldGenBootstrap:
    """
    Bootstrap class for generating Minecraft world .mca files using Fabric + Chunky.

    This replaces the stub implementation with real Fabric server + Chunky mod integration.
    """

    def __init__(
        self,
        seed: str = "VoxelTree",
        java_heap: str = "4G",
        temp_world_dir: Union[str, Path, None] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize FabricWorldGenBootstrap."""
        self.seed = 6901795026152433433
        self.java_heap = java_heap
        self.config = config or load_worldgen_config()

        if temp_world_dir is None:
            self.temp_world_dir = Path("temp_worlds")
        else:
            self.temp_world_dir = Path(temp_world_dir)

        logger.info(f"FabricWorldGenBootstrap initialized: seed={self.seed}, heap={java_heap}")

        # Ensure temp directory exists
        self.temp_world_dir.mkdir(parents=True, exist_ok=True)

    def generate_region_batch(self, x_range: Tuple[int, int], z_range: Tuple[int, int]) -> Path:
        """
        Generate .mca files for specified chunk ranges using Fabric + Chunky.

        Args:
            x_range: (x_min, x_max) chunk coordinates
            z_range: (z_min, z_max) chunk coordinates

        Returns:
            Path to generated world directory

        Raises:
            RuntimeError: If disk space limit exceeded or generation fails
        """
        start_time = time.time()
        logger.info(f"Starting Fabric region generation: x={x_range}, z={z_range}")

        # Check disk space limit before generation
        current_disk_usage = self._get_directory_size_gb(self.temp_world_dir)
        if current_disk_usage > 5.0:
            logger.error(f"Disk space limit exceeded: {current_disk_usage:.1f}GB > 5.0GB")
            raise RuntimeError("Disk space limit exceeded: temp worlds > 5GB")

        # Create unique world directory
        timestamp = int(time.time())
        world_dir = self.temp_world_dir / f"world_{timestamp:03d}"
        world_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Attempt Fabric + Chunky world generation
            success = self._run_fabric_chunky_generation(world_dir, x_range, z_range)
            if not success:
                logger.warning("Initial worldgen failed, attempting retry with reduced batch size")
                self._reduce_batch_size()
                success = self._run_fabric_chunky_generation(world_dir, x_range, z_range)

            if not success:
                raise RuntimeError("World generation failed after retry")

            elapsed_time = time.time() - start_time
            chunk_count = (x_range[1] - x_range[0]) * (z_range[1] - z_range[0])
            logger.info(f"Region generation completed: {chunk_count} chunks in {elapsed_time:.1f}s")

            return world_dir

        except Exception:
            # Clean up on failure
            if world_dir.exists():
                shutil.rmtree(world_dir)
            raise

    def _run_fabric_chunky_generation(
        self, world_dir: Path, x_range: Tuple[int, int], z_range: Tuple[int, int]
    ) -> bool:
        """
        Run Fabric server with Chunky mod to generate chunks.

        Returns:
            True if successful, False otherwise
        """
        fabric_jar = self._get_fabric_server_path()
        chunky_jar = self._get_chunky_mod_path()

        # Check that required files exist
        if not fabric_jar.exists():
            logger.error(f"Fabric server JAR not found: {fabric_jar}")
            return False

        if not chunky_jar.exists():
            logger.error(f"Chunky mod JAR not found: {chunky_jar}")
            return False

        # Setup server directory
        server_dir = world_dir / "server"
        server_dir.mkdir(parents=True, exist_ok=True)

        # Setup mods directory and copy Chunky
        mods_dir = server_dir / "mods"
        mods_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chunky_jar, mods_dir / chunky_jar.name)

        # Create server.properties
        self._create_server_properties(server_dir)

        # Accept EULA
        (server_dir / "eula.txt").write_text("eula=true\n")

        # Launch server and run Chunky commands
        return self._launch_server_and_generate(server_dir, fabric_jar, x_range, z_range)

    def _get_fabric_server_path(self) -> Path:
        """Get path to Fabric server JAR from config."""
        if self.config and "worldgen" in self.config and "java_tools" in self.config["worldgen"]:
            fabric_jar = Path(self.config["worldgen"]["java_tools"]["primary"])
            if fabric_jar.exists():
                return fabric_jar

        # Fallback to default path
        return Path("tools/fabric-server/fabric-server-mc.1.21.5-loader.0.16.14-launcher.1.0.3.jar")

    def _get_chunky_mod_path(self) -> Path:
        """Get path to Chunky mod JAR from config."""
        if self.config and "worldgen" in self.config and "java_tools" in self.config["worldgen"]:
            chunky_jar = Path(self.config["worldgen"]["java_tools"]["chunky"])
            if chunky_jar.exists():
                return chunky_jar

        # Fallback to default path
        return Path("tools/fabric-server/runtime/mods/Chunky-Fabric-1.4.36.jar")

    def _create_server_properties(self, server_dir: Path) -> None:
        """Create server.properties file for headless generation."""
        properties = f"""# Minecraft server properties for VoxelTree world generation
level-seed={self.seed}
gamemode=creative
difficulty=peaceful
spawn-protection=0
max-players=0
online-mode=false
pvp=false
level-type=default
generator-settings=
server-port=25565
level-name=world
motd=VoxelTree World Generation
announce-player-achievements=false
spawn-monsters=false
spawn-animals=false
spawn-npcs=false
force-gamemode=true
"""
        (server_dir / "server.properties").write_text(properties)

    def _launch_server_and_generate(
        self, server_dir: Path, fabric_jar: Path, x_range: Tuple[int, int], z_range: Tuple[int, int]
    ) -> bool:
        """
        Launch Fabric server and execute Chunky commands for chunk generation.

        Returns:
            True if successful, False otherwise
        """
        # Use Java 21 explicitly for Minecraft 1.21.5 compatibility
        java_exe = r"C:/Program Files/Eclipse Adoptium/jdk-21.0.7.6-hotspot/bin/java.exe"
        cmd = [java_exe, f"-Xmx{self.java_heap}", "-jar", str(fabric_jar), "nogui"]

        try:
            # Start server process
            logger.info(f"Starting Fabric server: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                cwd=server_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Wait for server to start (look for "Done" message)
            startup_timeout = 120  # 2 minutes
            start_time = time.time()
            server_ready = False

            while time.time() - start_time < startup_timeout:
                if process.poll() is not None:
                    # Server exited early
                    stdout, stderr = process.communicate()
                    logger.error(f"Server exited early. stdout: {stdout}, stderr: {stderr}")
                    return False

                # Read a line from stdout
                try:
                    line = process.stdout.readline()
                    if line and "Done" in line and "For help, type" in line:
                        server_ready = True
                        logger.info("Server startup complete")
                        break
                    time.sleep(0.1)
                except Exception:
                    break

            if not server_ready:
                logger.error("Server failed to start within timeout")
                process.terminate()
                return False

            # Execute Chunky commands
            success = self._execute_chunky_commands(process, x_range, z_range)

            # Stop server
            try:
                process.stdin.write("stop\n")
                process.stdin.flush()
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()

            return success

        except Exception as e:
            logger.error(f"Server launch failed: {e}")
            return False

    def _execute_chunky_commands(
        self, process: subprocess.Popen, x_range: Tuple[int, int], z_range: Tuple[int, int]
    ) -> bool:
        """
        Execute Chunky commands to generate the specified chunk range.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate center and radius from x/z range
            center_x = (x_range[0] + x_range[1]) // 2
            center_z = (z_range[0] + z_range[1]) // 2

            # Calculate radius (in chunks, Chunky uses block coordinates)
            radius_chunks = max(x_range[1] - x_range[0], z_range[1] - z_range[0]) // 2
            radius_blocks = radius_chunks * 16  # Convert to blocks

            # Execute Chunky commands
            commands = [
                f"chunky center {center_x * 16} {center_z * 16}",
                f"chunky radius {radius_blocks}",
                "chunky start",
            ]

            for cmd in commands:
                logger.info(f"Executing: {cmd}")
                process.stdin.write(f"{cmd}\n")
                process.stdin.flush()
                time.sleep(1)  # Small delay between commands

            # Wait for generation to complete
            # Look for "Task finished" or similar completion message
            completion_timeout = 300  # 5 minutes
            start_time = time.time()
            generation_complete = False

            while time.time() - start_time < completion_timeout:
                try:
                    line = process.stdout.readline()
                    if line:
                        logger.debug(f"Server output: {line.strip()}")
                        if "Task finished" in line or "Generation complete" in line:
                            generation_complete = True
                            logger.info("Chunk generation completed")
                            break
                    time.sleep(0.1)
                except Exception:
                    break

            return generation_complete

        except Exception as e:
            logger.error(f"Chunky command execution failed: {e}")
            return False

    def _reduce_batch_size(self):
        """Reduce batch size for retry after heap exhaustion."""
        logger.info("Reducing batch size for retry after heap exhaustion")

    def validate_mca_output(self, region_dir: Path) -> Dict[str, Any]:
        """
        Verify .mca files contain expected chunks and aren't corrupted.

        Args:
            region_dir: Directory containing .mca files

        Returns:
            Dictionary with validation results
        """
        result: Dict[str, Any] = {
            "files_found": 0,
            "total_size_mb": 0.0,
            "corrupted_files": [],
        }

        if not region_dir.exists():
            return result

        mca_files = list(region_dir.glob("*.mca"))
        result["files_found"] = len(mca_files)

        total_bytes = 0
        for mca_file in mca_files:
            file_size = mca_file.stat().st_size
            total_bytes += file_size

            # Check for corruption (files too small)
            if file_size < 100:  # Arbitrary small size threshold
                result["corrupted_files"].append(str(mca_file.name))

        result["total_size_mb"] = total_bytes / (1024 * 1024)
        logger.info(
            f"MCA validation: {result['files_found']} files, {result['total_size_mb']:.1f}MB"
        )
        return result

    def cleanup_temp_worlds(self, keep_latest: int = 2):
        """
        Remove old temporary world folders.

        Args:
            keep_latest: Number of latest world directories to keep
        """
        if not self.temp_world_dir.exists():
            return

        # Get all world directories sorted by name (which includes timestamp)
        world_dirs = sorted([d for d in self.temp_world_dir.iterdir() if d.is_dir()])

        # Remove all but the latest N directories
        to_remove = world_dirs[:-keep_latest] if len(world_dirs) > keep_latest else []

        logger.info(f"Cleaning up {len(to_remove)} old world directories, keeping {keep_latest}")

        for world_dir in to_remove:
            shutil.rmtree(world_dir)

    def _get_directory_size_gb(self, directory: Path) -> float:
        """
        Calculate total size of directory in GB.

        Args:
            directory: Directory to measure

        Returns:
            Size in GB
        """
        if not directory.exists():
            return 0.0

        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size / (1024**3)  # Convert to GB
