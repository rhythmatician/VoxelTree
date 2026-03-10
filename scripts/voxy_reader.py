"""voxy_reader.py — Read Voxy world data from its RocksDB storage.

Mirrors the encoding in voxy_writer.py and matches Voxy's SaveLoadSystem3
serialization format (palette-based, ZSTD-compressed).

Voxy stores each WorldSection as a 32×32×32 voxel grid at a single LOD level.
The RocksDB database has three column families:
  - ``default`` (unused by us)
  - ``world_sections`` — section key → ZSTD(palette-packed voxel data)
  - ``id_mappings`` — Voxy-internal block/biome IDs → gzipped NBT names

Voxel long format (64-bit):
    bits 63-56  light           (8 bits: upper nibble = block light, lower = sky light)
    bits 55-47  biomeId         (9 bits)
    bits 46-27  blockId         (20 bits; 0 = air)
    bits 26- 0  reserved        (27 bits, zero)

Section key (RocksDB key, 8 bytes big-endian):
    bits 63-60  level  (4 bits)
    bits 59-52  y      (8 bits, signed)
    bits 51-28  z      (24 bits, signed)
    bits 27- 4  x      (24 bits, signed)
    bits  3- 0  spare  (4 bits, zero)

SaveLoadSystem3 decompressed layout:
    offset 0:     8 bytes — section key (little-endian long)
    offset 8:     8 bytes — metadata (LE long; bits 0-15 = palette size, 16-23 = nonEmptyChildren)
    offset 16:    65536 bytes — 32768 × uint16 LE palette indices (index = y<<10 | z<<5 | x)
    offset 65552: N × 8 bytes — palette table (N LE int64s, each a packed voxel long)
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import rocksdict  # type: ignore  # external package has no stub

import zstandard


# ---------------------------------------------------------------------------
# Bit constants (shared with voxy_writer.py)
# ---------------------------------------------------------------------------
_BLOCK_ID_SHIFT = 27
_BLOCK_ID_MASK = (1 << 20) - 1  # 20 bits = 0xFFFFF

_BIOME_ID_SHIFT = 47
_BIOME_ID_MASK = (1 << 9) - 1  # 9 bits = 0x1FF

_LIGHT_SHIFT = 56
_LIGHT_MASK = 0xFF


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def _sign_extend(val: int, bits: int) -> int:
    """Sign-extend an unsigned integer from ``bits`` width."""
    if val >= (1 << (bits - 1)):
        val -= 1 << bits
    return val


def decode_section_key(key_bytes: Union[bytes, bytearray]) -> Tuple[int, int, int, int]:
    """Decode an 8-byte big-endian section key → (level, x, y, z).

    Accept either ``bytes`` or ``bytearray`` since RocksDB keys may come as
    either type.
    """
    key_int = struct.unpack(">q", key_bytes)[0]
    level = (key_int >> 60) & 0xF
    y = _sign_extend((key_int >> 52) & 0xFF, 8)
    z = _sign_extend((key_int >> 28) & 0xFFFFFF, 24)
    x = _sign_extend((key_int >> 4) & 0xFFFFFF, 24)
    return level, x, y, z


def encode_section_key(level: int, x: int, y: int, z: int) -> bytes:
    """Encode (level, x, y, z) into an 8-byte big-endian section key."""
    key_int = (
        ((level & 0xF) << 60) | ((y & 0xFF) << 52) | ((z & 0xFFFFFF) << 28) | ((x & 0xFFFFFF) << 4)
    )
    return struct.pack(">q", key_int)


# ---------------------------------------------------------------------------
# Section decoding (SaveLoadSystem3)
# ---------------------------------------------------------------------------


def decode_section(data: bytes) -> Dict[str, Any]:
    """Decode a ZSTD-decompressed SaveLoadSystem3 section blob.

    Returns a dict with keys:
      - level, x, y, z: section coordinates
      - palette_size: number of unique voxel entries
      - non_empty_children: 8-bit bitmask
      - block_ids: np.ndarray shape (32, 32, 32) int32, index order (y, z, x)
      - biome_ids: np.ndarray shape (32, 32, 32) int32
      - light: np.ndarray shape (32, 32, 32) uint8
      - raw_voxels: np.ndarray shape (32768,) int64 — full packed longs
      - palette: np.ndarray shape (N,) int64 — the palette lookup table
    """
    # Header
    section_key = struct.unpack_from("<q", data, 0)[0]
    metadata = struct.unpack_from("<Q", data, 8)[0]

    palette_size = int(metadata & 0xFFFF)
    non_empty_children = int((metadata >> 16) & 0xFF)

    level = (section_key >> 60) & 0xF
    y = _sign_extend((section_key >> 52) & 0xFF, 8)
    z = _sign_extend((section_key >> 28) & 0xFFFFFF, 24)
    x = _sign_extend((section_key >> 4) & 0xFFFFFF, 24)

    # Palette indices: 32768 uint16 LE at offset 16
    indices = np.frombuffer(data, dtype="<u2", count=32768, offset=16)

    # Palette table: palette_size int64 LE at offset 65552
    lut_offset = 16 + 32768 * 2  # 65552
    palette = np.frombuffer(data, dtype="<i8", count=palette_size, offset=lut_offset)

    # Reconstruct full voxel array
    raw_voxels = palette[indices]  # shape (32768,)

    # Extract fields — use unsigned view for bit operations
    voxels_u = raw_voxels.view(np.uint64)
    block_ids = ((voxels_u >> _BLOCK_ID_SHIFT) & _BLOCK_ID_MASK).astype(np.int32)
    biome_ids = ((voxels_u >> _BIOME_ID_SHIFT) & _BIOME_ID_MASK).astype(np.int32)
    light = ((voxels_u >> _LIGHT_SHIFT) & _LIGHT_MASK).astype(np.uint8)

    # Reshape to (32, 32, 32) in Voxy's indexing: linear index = y<<10 | z<<5 | x
    # So axis 0 = y, axis 1 = z, axis 2 = x
    block_ids = block_ids.reshape(32, 32, 32)
    biome_ids = biome_ids.reshape(32, 32, 32)
    light = light.reshape(32, 32, 32)

    return {
        "level": int(level),
        "x": int(x),
        "y": int(y),
        "z": int(z),
        "palette_size": palette_size,
        "non_empty_children": non_empty_children,
        "block_ids": block_ids,  # (32, 32, 32) — (y, z, x) ordering
        "biome_ids": biome_ids,  # (32, 32, 32)
        "light": light,  # (32, 32, 32)
        "raw_voxels": raw_voxels,  # (32768,) flat
        "palette": palette,  # (N,)
    }


# ---------------------------------------------------------------------------
# ID Mapping decoder
# ---------------------------------------------------------------------------


def _parse_nbt_name(nbt_data: bytes) -> Optional[str]:
    """Extract the 'Name' string from a gzipped NBT block-state entry."""
    idx = 0
    while idx < len(nbt_data) - 10:
        if nbt_data[idx] == 0x08:  # TAG_String
            name_len = struct.unpack(">H", nbt_data[idx + 1 : idx + 3])[0]
            tag_name = nbt_data[idx + 3 : idx + 3 + name_len].decode("utf-8", errors="replace")
            if tag_name == "Name":
                str_start = idx + 3 + name_len
                str_len = struct.unpack(">H", nbt_data[str_start : str_start + 2])[0]
                return nbt_data[str_start + 2 : str_start + 2 + str_len].decode("utf-8")
        idx += 1
    return None


def _parse_nbt_properties(nbt_data: bytes) -> Dict[str, str]:
    """Extract block state properties from NBT.  Returns dict of name→value."""
    props: Dict[str, str] = {}
    # Find TAG_Compound named "Properties"
    target = b"\x0a\x00\x0aProperties"  # TAG_Compound, name len=10, "Properties"
    idx = nbt_data.find(target)
    if idx < 0:
        return props
    cursor = idx + len(target)
    # Read TAG_String entries until TAG_End (0x00)
    while cursor < len(nbt_data) - 3:
        tag_type = nbt_data[cursor]
        if tag_type == 0x00:  # TAG_End
            break
        if tag_type != 0x08:  # only TAG_String expected
            break
        cursor += 1
        name_len = struct.unpack(">H", nbt_data[cursor : cursor + 2])[0]
        cursor += 2
        pname = nbt_data[cursor : cursor + name_len].decode("utf-8", errors="replace")
        cursor += name_len
        val_len = struct.unpack(">H", nbt_data[cursor : cursor + 2])[0]
        cursor += 2
        pval = nbt_data[cursor : cursor + val_len].decode("utf-8", errors="replace")
        cursor += val_len
        props[pname] = pval
    return props


# ---------------------------------------------------------------------------
# Main reader class
# ---------------------------------------------------------------------------


class VoxyReader:
    """Read-only interface to a Voxy RocksDB world database.

    Usage::

        reader = VoxyReader("path/to/voxy/<hash>/storage")
        for section in reader.iter_sections(level=0):
            print(section["x"], section["y"], section["z"], section["block_ids"].shape)
        reader.close()

    Or as a context manager::

        with VoxyReader("path/to/storage") as reader:
            block_names = reader.block_names
            section = reader.get_section(level=0, x=0, y=0, z=0)
    """

    def __init__(self, db_path: str | Path, *, read_only: bool = True) -> None:
        if rocksdict is None:
            raise ImportError("rocksdict is required: pip install rocksdict")
        if zstandard is None:
            raise ImportError("zstandard is required: pip install zstandard")

        self._db_path = str(db_path)
        opts = rocksdict.Options(raw_mode=True)
        opts.set_max_open_files(64)

        self._db = rocksdict.Rdict(
            self._db_path,
            options=opts,
            column_families={
                "default": opts,
                "world_sections": opts,
                "id_mappings": opts,
            },
            # rocksdict doesn't have a read_only kwarg — just open normally
        )
        self._ws_cf = self._db.get_column_family("world_sections")
        self._id_cf = self._db.get_column_family("id_mappings")
        self._dctx = zstandard.ZstdDecompressor()

        # Decode id mappings
        self._block_names: Dict[int, str] = {0: "minecraft:air"}
        self._block_properties: Dict[int, Dict[str, str]] = {}
        self._biome_names: Dict[int, str] = {0: "unknown"}
        self._decode_id_mappings()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def block_names(self) -> Dict[int, str]:
        """Mapping from Voxy block-state ID → Minecraft block name (e.g. ``minecraft:stone``)."""
        return self._block_names

    @property
    def block_properties(self) -> Dict[int, Dict[str, str]]:
        """Mapping from Voxy block-state ID → block state properties dict."""
        return self._block_properties

    @property
    def biome_names(self) -> Dict[int, str]:
        """Mapping from Voxy biome ID → Minecraft biome name."""
        return self._biome_names

    # ------------------------------------------------------------------
    # Reading sections
    # ------------------------------------------------------------------

    def get_section(self, level: int, x: int, y: int, z: int) -> Optional[dict]:
        """Read a single WorldSection by (level, x, y, z).

        Returns decoded dict (see ``decode_section``) or None if not found.
        """
        key = encode_section_key(level, x, y, z)
        try:
            compressed = bytes(self._ws_cf[key])  # type: ignore[arg-type]
        except KeyError:
            return None
        data = self._dctx.decompress(compressed)
        return decode_section(data)

    def iter_sections(
        self,
        level: Optional[int] = None,
        *,
        decode: bool = True,
    ):
        """Iterate over all world sections, optionally filtering by LOD level.

        Yields decoded section dicts if ``decode=True``, else yields
        ``(key_bytes, compressed_bytes)`` tuples.
        """
        for key in self._ws_cf.keys():
            if isinstance(key, (bytes, bytearray)):
                key_bytes = key
            else:
                key_bytes = bytes(key)  # type: ignore[arg-type]
            if level is not None:
                lvl, _, _, _ = decode_section_key(key_bytes)
                if lvl != level:
                    continue

            compressed = bytes(self._ws_cf[key])  # type: ignore[arg-type]
            if decode:
                data = self._dctx.decompress(compressed)
                yield decode_section(data)
            else:
                yield key_bytes, compressed

    def count_sections(self) -> Dict[int, int]:
        """Count sections per LOD level.  Returns ``{level: count}``."""
        counts: Dict[int, int] = {}
        for key in self._ws_cf.keys():
            if isinstance(key, (bytes, bytearray)):
                key_bytes = key
            else:
                key_bytes = bytes(key)  # type: ignore[arg-type]
            lvl, _, _, _ = decode_section_key(key_bytes)
            counts[lvl] = counts.get(lvl, 0) + 1
        return counts

    def summary(self) -> str:
        """Return a human-readable summary of the database."""
        lines = [
            "VoxyReader: %s" % self._db_path,
            "  Block states: %d" % len(self._block_names),
            "  Biomes: %d" % len(self._biome_names),
        ]
        counts = self.count_sections()
        for lvl in sorted(counts.keys()):
            voxel_m = 2**lvl
            lines.append(
                "  LOD %d: %d sections (voxel=%dm, section=%dm)"
                % (lvl, counts[lvl], voxel_m, 32 * voxel_m)
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Section → 16³ sub-blocks (for training data extraction)
    # ------------------------------------------------------------------

    def section_to_subblocks(self, section: dict) -> List[dict]:
        """Split a 32×32×32 WorldSection into eight 16×16×16 sub-blocks.

        Each sub-block corresponds to a VoxelizedSection at this LOD level.
        The block_ids array is in (y, z, x) order matching Voxy's indexing.

        Returns list of 8 dicts with keys:
          - block_ids: (16, 16, 16) int32 — (y, z, x)
          - biome_ids: (16, 16, 16) int32
          - sub_x, sub_y, sub_z: local offset within the WorldSection (0 or 1)
          - world_x, world_y, world_z: absolute section coordinates
          - level: LOD level
          - non_air_count: number of non-air voxels
        """
        block_ids = section["block_ids"]  # (32, 32, 32) — (y, z, x)
        biome_ids = section["biome_ids"]
        level = section["level"]
        sx, sy, sz = section["x"], section["y"], section["z"]

        sub_blocks = []
        for dy in range(2):
            for dz in range(2):
                for dx in range(2):
                    y_slice = slice(dy * 16, (dy + 1) * 16)
                    z_slice = slice(dz * 16, (dz + 1) * 16)
                    x_slice = slice(dx * 16, (dx + 1) * 16)

                    blk = block_ids[y_slice, z_slice, x_slice].copy()
                    bio = biome_ids[y_slice, z_slice, x_slice].copy()
                    non_air = int(np.sum(blk > 0))

                    sub_blocks.append(
                        {
                            "block_ids": blk,  # (16, 16, 16) — (y, z, x)
                            "biome_ids": bio,
                            "sub_x": dx,
                            "sub_y": dy,
                            "sub_z": dz,
                            "world_x": sx * 2 + dx,
                            "world_y": sy * 2 + dy,
                            "world_z": sz * 2 + dz,
                            "level": level,
                            "non_air_count": non_air,
                        }
                    )
        return sub_blocks

    # ------------------------------------------------------------------
    # Block ID mapping: Voxy → VoxelTree
    # ------------------------------------------------------------------

    def build_voxy_to_voxeltree_map(
        self,
        voxeltree_mapping: Dict[str, int],
    ) -> Dict[int, int]:
        """Build a mapping from Voxy block IDs to VoxelTree block IDs.

        Args:
            voxeltree_mapping: Dict like ``{"minecraft:air": 0, "minecraft:stone": 1, ...}``
                               (from ``complete_block_mapping.json`` / ``standard_minecraft_blocks.json``).

        Returns:
            Dict mapping Voxy block-state ID → VoxelTree block ID.
            Unmapped block states map to 0 (air).
        """
        result: Dict[int, int] = {0: 0}  # air → air
        for voxy_id, mc_name in self._block_names.items():
            if mc_name in voxeltree_mapping:
                result[voxy_id] = voxeltree_mapping[mc_name]
            else:
                # Try base name without properties
                base = mc_name.split("[")[0] if "[" in mc_name else mc_name
                result[voxy_id] = voxeltree_mapping.get(base, 0)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decode_id_mappings(self) -> None:
        """Parse id_mappings column family to build name tables."""
        for key in self._id_cf.keys():
            if isinstance(key, (bytes, bytearray)):
                key_bytes = key
            else:
                key_bytes = bytes(key)  # type: ignore[arg-type]
            val = bytes(self._id_cf[key])  # type: ignore[arg-type]

            if len(key_bytes) != 4:
                continue

            key_int = struct.unpack(">I", key_bytes)[0]
            type_nibble = (key_int >> 28) & 0xF
            entry_id = key_int & 0x0FFFFFFF

            try:
                nbt_data = gzip.decompress(val)
            except Exception:
                continue

            if type_nibble == 4:  # block state
                name = _parse_nbt_name(nbt_data)
                if name:
                    self._block_names[entry_id] = name
                props = _parse_nbt_properties(nbt_data)
                if props:
                    self._block_properties[entry_id] = props

            elif type_nibble == 2:  # biome
                # Biome entries have a simpler NBT with just a string ID
                name = _parse_nbt_name(nbt_data)
                if name:
                    self._biome_names[entry_id] = name

    def close(self) -> None:
        """Close the database."""
        if self._db is not None:
            self._db.close()
            self._db = None  # type: ignore[assignment]

    def __enter__(self) -> "VoxyReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI — quick inspection tool
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect a Voxy RocksDB world database.")
    parser.add_argument(
        "db_path",
        help="Path to the Voxy storage directory (contains .sst files).",
    )
    parser.add_argument(
        "--level",
        "-l",
        type=int,
        default=None,
        help="Filter to a specific LOD level.",
    )
    parser.add_argument(
        "--sample",
        "-n",
        type=int,
        default=5,
        help="Number of sections to inspect in detail.",
    )
    parser.add_argument(
        "--blocks",
        action="store_true",
        help="Print block distribution for sampled sections.",
    )
    parser.add_argument(
        "--list-blocks",
        action="store_true",
        help="Print all block state mappings.",
    )
    parser.add_argument(
        "--export-mapping",
        type=str,
        default=None,
        help="Export Voxy block name mapping to a JSON file.",
    )
    args = parser.parse_args()

    with VoxyReader(args.db_path) as reader:
        print(reader.summary())
        print()

        if args.list_blocks:
            print("Block state mappings:")
            for bid in sorted(reader.block_names.keys()):
                name = reader.block_names[bid]
                props = reader.block_properties.get(bid, {})
                if props:
                    print("  %4d: %s %s" % (bid, name, props))
                else:
                    print("  %4d: %s" % (bid, name))
            print()

        if args.export_mapping:
            with open(args.export_mapping, "w") as f:
                json.dump(reader.block_names, f, indent=2)
            print("Exported block mapping to %s" % args.export_mapping)

        # Sample sections
        count = 0
        for section in reader.iter_sections(level=args.level):
            if count >= args.sample:
                break

            blk = section["block_ids"]
            non_air = int(np.sum(blk > 0))
            total = blk.size
            print(
                "Section LOD=%d x=%d y=%d z=%d  palette=%d  solid=%d/%d (%.1f%%)"
                % (
                    section["level"],
                    section["x"],
                    section["y"],
                    section["z"],
                    section["palette_size"],
                    non_air,
                    total,
                    100.0 * non_air / total,
                )
            )

            if args.blocks:
                unique, counts = np.unique(blk, return_counts=True)
                for bid, cnt in sorted(
                    zip(unique.tolist(), counts.tolist()),
                    key=lambda x: -x[1],
                )[:10]:
                    name = reader.block_names.get(bid, "voxy_id:%d" % bid)
                    print("    %s: %d" % (name, cnt))

            count += 1


if __name__ == "__main__":
    main()
