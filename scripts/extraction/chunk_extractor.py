"""
Chunk extraction module for converting Minecraft .mca files to .npz training data.

This module provides the ChunkExtractor class that handles parsing Minecraft
region files and converting them into compressed numpy arrays.
"""

import logging
import multiprocessing
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anvil  # type: ignore
import numpy as np

from scripts.extraction.palette_decode import decode_palette_indices, map_palette_to_block_ids
from scripts.worldgen.config import load_config

logger = logging.getLogger(__name__)


class ChunkExtractor:
    """
    Extracts Minecraft chunk data from .mca files and converts to .npz format.

    This class handles the conversion pipeline from raw Minecraft region files
    to compressed numpy arrays suitable for machine learning training.
    """

    # Load complete block vocabulary mapping
    import json

    # Load complete block mapping we just created
    complete_mapping_path = "scripts/extraction/complete_block_mapping.json"
    with open(complete_mapping_path) as f:
        VANILLA_BLOCK_MAPPING = json.load(f)

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ChunkExtractor with configuration.

        Args:
            config_path: Path to config.yaml file, defaults to "config.yaml"
        """
        # Load configuration
        if config_path is None:
            config_path = Path("config.yaml")

        self.config = load_config(config_path)
        extraction_config = self.config.get("extraction", {})

        # Set up paths
        self.output_dir = Path(extraction_config.get("output_dir", "data/chunks"))
        self.temp_dir = Path(extraction_config.get("temp_dir", "temp_extraction"))

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Extraction parameters
        self.batch_size = extraction_config.get("batch_size", 64)
        self.num_workers = extraction_config.get("num_workers", 4)
        self.compression_level = extraction_config.get("compression_level", 6)
        self.max_disk_usage_gb = extraction_config.get("max_disk_usage_gb", 10)

        # Block mappings
        block_mapping = extraction_config.get("block_mapping", {})
        self.air_blocks = set(block_mapping.get("air_blocks", [0]))
        self.solid_blocks = set(block_mapping.get("solid_blocks", [1, 2, 3]))

        # Heightmap parameters
        heightmap_config = extraction_config.get("heightmap", {})
        self.surface_blocks = set(heightmap_config.get("surface_blocks", [2, 3, 4]))
        self.min_height = heightmap_config.get("min_height", -64)
        self.max_height = heightmap_config.get("max_height", 320)

        # Validation settings
        validation_config = extraction_config.get("validation", {})
        self.verify_checksums = validation_config.get("verify_checksums", True)
        self.detect_corruption = validation_config.get("detect_corruption", True)

        logger.info(f"ChunkExtractor initialized with output_dir={self.output_dir}")

    def extract_chunk_data(
        self, region_file: Path, chunk_x: int, chunk_z: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract single chunk data from .mca file.

        Args:
            region_file: Path to .mca region file
            chunk_x: Chunk X coordinate (within the region)
            chunk_z: Chunk Z coordinate (within the region)

        Returns:
            Dictionary containing chunk data arrays

        Raises:
            FileNotFoundError: If region file doesn't exist
            ValueError: If chunk coordinates are invalid
            RuntimeError: If chunk data is corrupted or invalid
        """
        if not region_file.exists():
            raise FileNotFoundError(f"Region file not found: {region_file}")

        if not (0 <= chunk_x < 32 and 0 <= chunk_z < 32):
            raise ValueError(
                f"Invalid chunk coordinates ({chunk_x}, {chunk_z}). Must be in range 0-31"  # noqa: E501
            )

        try:
            # Open region file with anvil-parser
            region = anvil.Region.from_file(str(region_file))

            # Determine region origin from filename r.X.Z.mca
            try:
                stem = region_file.stem  # e.g., 'r.-1.0'
                parts = stem.split(".")
                rx, rz = int(parts[1]), int(parts[2])
            except Exception:
                rx, rz = 0, 0

            # Attempt to fetch the chunk: try local (0..31) first, then global fallback
            def _is_chunk_not_found(exc: Exception) -> bool:
                name = exc.__class__.__name__
                msg = str(exc)
                return (
                    name == "ChunkNotFound"
                    or "Chunk does not exist" in msg
                    or "Could not find chunk" in msg
                    or "chunk not found" in msg.lower()
                )

            try:
                chunk = region.get_chunk(chunk_x, chunk_z)
            except Exception as e_local:
                # Fallback to global chunk coordinates if library expects absolutes
                gx, gz = rx * 32 + chunk_x, rz * 32 + chunk_z
                try:
                    chunk = region.get_chunk(gx, gz)
                except Exception as e_global:
                    # Known benign case: chunk slot is absent in this region file
                    if _is_chunk_not_found(e_local) or _is_chunk_not_found(e_global):
                        logger.debug(
                            "Skipping missing chunk (%d,%d) in %s: %s",
                            chunk_x,
                            chunk_z,
                            region_file.name,
                            e_global,
                        )
                        return None
                    # If both attempts fail for another reason, surface the original error
                    raise e_global from e_local

            if chunk is None:
                # Chunk slot not generated in this region; benign skip
                logger.debug(
                    "Skipping absent chunk (%d,%d) in %s",
                    chunk_x,
                    chunk_z,
                    region_file.name,
                )
                return None

            # Process real chunk data
            block_types, air_mask = self.process_block_data(chunk)
            biomes = self.extract_biome_data(chunk)
            heightmap = self.compute_heightmap(block_types)

            # Pack data into dictionary for .npz storage
            chunk_data = {
                "block_types": block_types,
                "air_mask": air_mask,
                "biomes": biomes,
                "heightmap": heightmap,
                "chunk_x": chunk_x,
                "chunk_z": chunk_z,
                "region_file": str(region_file.name),
            }

            logger.debug(
                f"Successfully extracted chunk ({chunk_x}, {chunk_z}) from {region_file.name}"  # noqa: E501
            )
            return chunk_data
        except Exception as e:
            # Treat known 'missing chunk' conditions as benign skips
            cls_name = e.__class__.__name__
            msg = str(e)
            if (
                cls_name in {"ChunkNotFound", "MissingChunk"}
                or "Chunk does not exist" in msg
                or "Could not find chunk" in msg
                or "chunk not found" in msg.lower()
            ):
                logger.debug(
                    "Skipping missing chunk (%d,%d) in %s: %s",
                    chunk_x,
                    chunk_z,
                    region_file.name,
                    e,
                )
                return None
            elif isinstance(e, (IndexError, struct.error, ValueError, EOFError)):
                # Low-level parse errors → corrupt / truncated region bytes; skip chunk
                logger.debug(
                    "Skipping corrupt chunk (%d,%d) in %s: %s",
                    chunk_x,
                    chunk_z,
                    region_file.name,
                    e,
                )
                return None
            elif "Region does not exist" in str(e):
                logger.error(f"Invalid region file {region_file}: {e}")
                raise RuntimeError(f"Failed to read region: {e}")
            # If not handled above, always raise to satisfy return type
            logger.error(f"Failed to extract chunk {chunk_x},{chunk_z} from {region_file}: {e}")
            raise RuntimeError(f"Unexpected error during chunk extraction: {e}")

    def process_block_data(self, chunk) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert chunk block data to numpy arrays.

        Args:
            chunk: anvil.Chunk object

        Returns:
            Tuple of (block_types, air_mask) arrays
        """
        # Initialize arrays
        block_types = np.zeros((16, 16, 384), dtype=np.uint8)
        air_mask = np.ones((16, 16, 384), dtype=bool)  # Start with all air

        # Process chunk sections (16x16x16 blocks each)
        try:
            if hasattr(chunk, "sections") and chunk.sections:
                for section in chunk.sections:
                    if section is None:
                        continue

                    # Get section Y level (each section is 16 blocks tall)
                    section_y = section.get("Y", 0) if isinstance(section, dict) else 0
                    y_start = section_y * 16
                    y_end = min(y_start + 16, 384)

                    if y_start >= 384 or y_end <= 0:
                        continue

                    # Real block extraction using palette decoding
                    try:
                        # Get section block states - this varies by anvil-parser2 API
                        section_blocks = None
                        if hasattr(section, "block_states"):
                            section_blocks = section.block_states
                        elif hasattr(section, "blocks"):
                            section_blocks = section.blocks
                        elif isinstance(section, dict):
                            section_blocks = section.get("block_states") or section.get("blocks")

                        if section_blocks and hasattr(section_blocks, "palette"):
                            # Use palette decoder for real block extraction
                            palette = section_blocks.palette
                            data = getattr(section_blocks, "data", None)

                            if data is not None:
                                # Decode using our palette decoder
                                decoded_indices = decode_palette_indices(data, len(palette))
                                block_ids = map_palette_to_block_ids(
                                    palette, decoded_indices, self.VANILLA_BLOCK_MAPPING
                                )

                                # block_ids is now (16,16,16) array in (x,z,y) order
                                # Copy to our global arrays
                                for x in range(16):
                                    for z in range(16):
                                        for y_local in range(16):
                                            y_global = y_start + y_local
                                            if 0 <= y_global < 384:
                                                block_id = block_ids[x, z, y_local]
                                                block_types[x, z, y_global] = block_id
                                                air_mask[x, z, y_global] = block_id == 0
                                continue

                        # Fallback: Try direct block access if palette approach fails
                        logger.warning(
                            f"Palette extraction failed for section Y={section_y}, using fallback"
                        )
                        for x in range(16):
                            for z in range(16):
                                for y_local in range(16):
                                    y_global = y_start + y_local
                                    if 0 <= y_global < 384:
                                        try:
                                            # Try to get block directly
                                            block = chunk.get_block(x, y_global, z)
                                            if block:
                                                block_name = str(block)
                                                block_id = self.get_block_id(block_name)
                                                block_types[x, z, y_global] = block_id
                                                air_mask[x, z, y_global] = block_id == 0
                                        except Exception:
                                            # If all else fails, default to air
                                            block_types[x, z, y_global] = 0
                                            air_mask[x, z, y_global] = True

                    except Exception as e:
                        logger.warning(
                            f"Real block extraction failed for section Y={section_y}: {e}"
                        )
                        # Fallback to simple terrain for this section only
                        for x in range(16):
                            for z in range(16):
                                for y in range(max(0, y_start), min(384, y_end)):
                                    if y < 60:  # Below sea level
                                        block_types[x, z, y] = self.get_block_id("minecraft:stone")
                                        air_mask[x, z, y] = False
                                    elif y < 62:  # Surface
                                        block_types[x, z, y] = self.get_block_id("minecraft:dirt")
                                        air_mask[x, z, y] = False
                                    elif y == 62:  # Top
                                        grass_id = self.get_block_id("minecraft:grass_block")
                                        block_types[x, z, y] = grass_id
                                        air_mask[x, z, y] = False
        except Exception as e:
            logger.warning(f"Failed to process chunk sections: {e}")
            # Keep default air-filled arrays

        return block_types, air_mask

    def get_block_id(self, block_name: str) -> int:
        """
        Maps Minecraft block names to integer IDs for neural network training.

        Args:
            block_name: Minecraft block identifier (e.g., "minecraft:stone")

        Returns:
            Integer block ID for training (0-1023 range)
        """
        # Handle None or empty block names
        if not block_name:
            return 0  # Air

        # Direct mapping from our comprehensive vanilla block mapping
        if block_name in self.VANILLA_BLOCK_MAPPING:
            return self.VANILLA_BLOCK_MAPPING[block_name]

        # For completely unknown blocks, log and use air as fallback
        logger.warning(f"Unknown block type encountered: {block_name}, mapping to air")
        return 0  # Air is always safe fallback

    def extract_biome_data(self, chunk) -> np.ndarray:
        """
        Extract biome IDs for chunk.

        Args:
            chunk: anvil.Chunk object

        Returns:
            Biome array with shape (16, 16)
        """
        biomes = np.zeros((16, 16), dtype=np.uint8)

        # Extract biomes from chunk
        try:
            # anvil-parser2 has a biomes property that should give us biome data
            if hasattr(chunk, "biomes") and chunk.biomes is not None:
                # Modern chunk format has 3D biomes, we want surface biomes
                chunk_biomes = chunk.biomes
                if isinstance(chunk_biomes, (list, np.ndarray)):
                    # Take surface biomes (y=64 level or highest available)
                    if len(chunk_biomes) >= 16 * 16:
                        biomes = (
                            np.array(chunk_biomes[: 16 * 16]).reshape((16, 16)).astype(np.uint8)
                        )
                    else:
                        # Fallback: use available data and pad
                        biomes.fill(1)  # Plains biome as fallback
        except Exception:
            # Fallback to plains biome if extraction fails
            biomes.fill(1)  # Plains biome ID

        # Ensure valid biome IDs (0-255)
        np.clip(biomes, 0, 255, out=biomes)

        return biomes

    def _block_to_id(self, block) -> int:
        """
        Convert anvil block object to full Minecraft block ID.

        Args:
            block: Block object from anvil-parser2

        Returns:
            Full Minecraft block ID (preserves all vanilla terrain blocks)
        """
        if block is None:
            return 0  # Air

        # Extract the actual numeric block ID from anvil-parser2
        # anvil-parser2 should provide the raw block ID
        if hasattr(block, "id"):
            return int(block.id)
        elif hasattr(block, "block_id"):
            return int(block.block_id)
        elif hasattr(block, "numeric_id"):
            return int(block.numeric_id)
        else:
            # Fallback: try to parse from string representation
            block_str = str(block)
            # Look for numeric ID in the string representation
            import re

            id_match = re.search(r"id[:\s]*(\d+)", block_str)
            if id_match:
                return int(id_match.group(1))

            # Final fallback: return 0 (air) for unknown blocks
            logger.warning(f"Could not extract block ID from {block}, defaulting to air")
            return 0

    def compute_heightmap(self, block_types: np.ndarray) -> np.ndarray:
        """
        Compute surface heightmap from block data.

        Args:
            block_types: Block type array with shape (16, 16, 384)

        Returns:
            Heightmap array with shape (16, 16)
        """
        # Identify surface blocks (non-air or specific surface blocks) for the grid
        surface_mask = np.logical_or(
            ~np.isin(block_types, list(self.air_blocks)),
            np.isin(block_types, list(self.surface_blocks)),
        )

        # Find the highest surface block along the vertical axis (axis=2)
        surface_positions = np.argmax(surface_mask[:, :, ::-1], axis=2)

        # Reverse the indices to match the original orientation
        heightmap = 384 - surface_positions

        # Handle columns with no surface blocks (all air)
        no_surface_mask = ~np.any(surface_mask, axis=2)
        heightmap[no_surface_mask] = 0

        return heightmap.astype(np.uint16)

    def save_chunk_npz(
        self, chunk_data: Dict[str, Any], chunk_x: int, chunk_z: int, region_id: str = ""
    ) -> Path:
        """
        Save chunk data as compressed .npz file.

        Args:
            chunk_data: Dictionary containing chunk arrays
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate
            region_id: Optional region identifier to make filename unique

        Returns:
            Path to saved .npz file
        """
        # Include region_id in filename to avoid conflicts when processing multiple
        # regions in parallel
        if region_id:
            output_path = self.output_dir / f"chunk_{region_id}_{chunk_x}_{chunk_z}.npz"
        else:
            output_path = self.output_dir / f"chunk_{chunk_x}_{chunk_z}.npz"

        # Save with compression
        np.savez_compressed(output_path, **chunk_data)

        logger.debug(f"Saved chunk {chunk_x},{chunk_z} to {output_path}")
        return output_path

    def extract_region_batch(self, region_file: Path) -> List[Path]:
        """
        Extract all chunks from region file to .npz files.

        Args:
            region_file: Path to .mca region file

        Returns:
            List of paths to created .npz files
        """
        output_files = []

        try:
            # Iterate over full 32x32 region coordinate space
            region_id = region_file.stem
            attempted = 0
            last_log = 0
            logger.info(f"Extracting region {region_file.name} -> {self.output_dir} (1024 chunks)")
            for chunk_x in range(32):
                for chunk_z in range(32):
                    attempted += 1
                    try:
                        chunk_data = self.extract_chunk_data(region_file, chunk_x, chunk_z)
                        if chunk_data is not None:  # Skip missing/corrupted chunks
                            output_path = self.save_chunk_npz(
                                chunk_data, chunk_x, chunk_z, region_id
                            )
                            output_files.append(output_path)
                    except Exception as e:
                        # Individual chunk failures shouldn't abort region processing
                        logger.debug(
                            "Chunk extract failed in region %s at %d,%d: %s",
                            region_file.name,
                            chunk_x,
                            chunk_z,
                            e,
                        )
                        continue
                    # Periodic progress log every 128 chunk slots
                    if attempted - last_log >= 128 or attempted == 1024:
                        last_log = attempted
                        logger.info(
                            "Region %s: processed %d/1024 slots, written %d files",
                            region_file.name,
                            attempted,
                            len(output_files),
                        )

            logger.info(
                "Region %s: extracted %d / %d chunk slots (skipped missing/corrupted)",
                region_file.name,
                len(output_files),
                attempted,
            )
            return output_files

        except Exception as e:
            logger.error(f"Failed to process region file {region_file}: {e}")
            raise

    def extract_regions_parallel(
        self, region_files: List[Path], num_workers: Optional[int] = None
    ) -> List[str]:
        """
        Extract multiple regions in parallel using multiprocessing.

        Args:
            region_files: List of .mca region file paths
            num_workers: Number of worker processes, defaults to self.num_workers

        Returns:
            List of output .npz file paths
        """
        if num_workers is None:
            num_workers = self.num_workers

        logger.info(f"Processing {len(region_files)} regions with {num_workers} workers")

        # Use multiprocessing Pool
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(self.extract_region_batch, region_files)

        # Flatten results
        all_output_files = []
        for result in results:
            if isinstance(result, list):
                all_output_files.extend([str(p) for p in result])
            else:
                all_output_files.append(str(result))

        logger.info(f"Parallel extraction complete: {len(all_output_files)} chunks processed")
        return all_output_files

    def validate_extraction_results(self, output_dir: Path) -> Dict[str, Any]:
        """
        Validate extracted .npz files for corruption and completeness.

        Args:
            output_dir: Directory containing .npz files

        Returns:
            Dictionary with validation results
        """
        valid_files = []
        corrupted_files = []
        invalid_files = []

        npz_files = list(output_dir.glob("*.npz"))

        for npz_file in npz_files:
            try:
                # Try to load the file
                data = np.load(npz_file)

                # Check required fields
                required_fields = [
                    "block_types",
                    "air_mask",
                    "biomes",
                    "heightmap",
                    "chunk_x",
                    "chunk_z",
                    "region_file",
                ]

                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    corrupted_files.append(f"{npz_file.name} (missing: {missing_fields})")
                    continue

                # Check array shapes
                if data["block_types"].shape != (16, 16, 384):
                    invalid_files.append(f"{npz_file.name} (invalid block_types shape)")
                    continue

                if data["air_mask"].shape != (16, 16, 384):
                    invalid_files.append(f"{npz_file.name} (invalid air_mask shape)")
                    continue

                if data["biomes"].shape != (16, 16):
                    invalid_files.append(f"{npz_file.name} (invalid biomes shape)")
                    continue

                if data["heightmap"].shape != (16, 16):
                    invalid_files.append(f"{npz_file.name} (invalid heightmap shape)")
                    continue

                # If we get here, file is valid
                valid_files.append(str(npz_file))

            except Exception as e:
                corrupted_files.append(f"{npz_file.name} (error: {e})")

        validation_result = {
            "valid_files": valid_files,
            "corrupted_files": corrupted_files,
            "invalid_files": invalid_files,
            "total_chunks": len(npz_files),
        }

        logger.info(f"Validation complete: {len(valid_files)}/{len(npz_files)} files valid")
        return validation_result
