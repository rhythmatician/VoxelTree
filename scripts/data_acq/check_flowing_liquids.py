#!/usr/bin/env python3
"""Scan .mca region files for flowing water / lava blocks.

Flowing liquids are water or lava blocks with ``level`` > 0.
Level 0 = source block (placed intentionally or natural).
Level 1–7 = flowing (physics-created, means /tick freeze didn't run in time).

Usage:
    python scripts/data_acq/check_flowing_liquids.py <world_dir>
    python scripts/data_acq/check_flowing_liquids.py tools/fabric-server/runtime/world

Reads all ``region/*.mca`` files, extracts chunk NBT, and reports any
palette entries containing ``minecraft:water`` or ``minecraft:lava``
with ``level`` != ``"0"``.

Pure Python — no external NBT libraries required.
"""

from __future__ import annotations

import io
import struct
import sys
import zlib
from pathlib import Path


# ── Minimal NBT reader (just enough to walk chunk data) ─────────────────

TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


def read_nbt_value(buf: io.BytesIO, tag_type: int):
    """Read a single NBT value of the given type.  Returns Python objects."""
    if tag_type == TAG_BYTE:
        return struct.unpack(">b", buf.read(1))[0]
    if tag_type == TAG_SHORT:
        return struct.unpack(">h", buf.read(2))[0]
    if tag_type == TAG_INT:
        return struct.unpack(">i", buf.read(4))[0]
    if tag_type == TAG_LONG:
        return struct.unpack(">q", buf.read(8))[0]
    if tag_type == TAG_FLOAT:
        return struct.unpack(">f", buf.read(4))[0]
    if tag_type == TAG_DOUBLE:
        return struct.unpack(">d", buf.read(8))[0]
    if tag_type == TAG_BYTE_ARRAY:
        length = struct.unpack(">i", buf.read(4))[0]
        return buf.read(length)
    if tag_type == TAG_STRING:
        length = struct.unpack(">H", buf.read(2))[0]
        return buf.read(length).decode("utf-8", errors="replace")
    if tag_type == TAG_LIST:
        elem_type = struct.unpack(">b", buf.read(1))[0]
        length = struct.unpack(">i", buf.read(4))[0]
        return [read_nbt_value(buf, elem_type) for _ in range(length)]
    if tag_type == TAG_COMPOUND:
        result = {}
        while True:
            child_type = struct.unpack(">b", buf.read(1))[0]
            if child_type == TAG_END:
                break
            name_len = struct.unpack(">H", buf.read(2))[0]
            name = buf.read(name_len).decode("utf-8", errors="replace")
            result[name] = read_nbt_value(buf, child_type)
        return result
    if tag_type == TAG_INT_ARRAY:
        length = struct.unpack(">i", buf.read(4))[0]
        return list(struct.unpack(f">{length}i", buf.read(4 * length)))
    if tag_type == TAG_LONG_ARRAY:
        length = struct.unpack(">i", buf.read(4))[0]
        return list(struct.unpack(f">{length}q", buf.read(8 * length)))
    raise ValueError(f"Unknown NBT tag type: {tag_type}")


def read_nbt_root(data: bytes) -> dict:
    """Parse a root NBT compound from raw (decompressed) bytes."""
    buf = io.BytesIO(data)
    tag_type = struct.unpack(">b", buf.read(1))[0]
    if tag_type != TAG_COMPOUND:
        raise ValueError(f"Expected root compound, got tag type {tag_type}")
    name_len = struct.unpack(">H", buf.read(2))[0]
    buf.read(name_len)  # root name (usually empty)
    return read_nbt_value(buf, TAG_COMPOUND)


# ── Region file reader ──────────────────────────────────────────────────


def iter_chunks(mca_path: Path):
    """Yield (chunk_x, chunk_z, nbt_dict) for each chunk in a region file."""
    data = mca_path.read_bytes()
    if len(data) < 8192:
        return

    # Parse region coords from filename: r.{rx}.{rz}.mca
    parts = mca_path.stem.split(".")
    rx, rz = int(parts[1]), int(parts[2])

    for idx in range(1024):
        offset_bytes = data[idx * 4 : idx * 4 + 4]
        raw = struct.unpack(">I", offset_bytes)[0]
        offset_sectors = raw >> 8
        sector_count = raw & 0xFF
        if offset_sectors == 0 and sector_count == 0:
            continue

        byte_offset = offset_sectors * 4096
        if byte_offset + 5 > len(data):
            continue

        chunk_len = struct.unpack(">I", data[byte_offset : byte_offset + 4])[0]
        compression = data[byte_offset + 4]
        raw_chunk = data[byte_offset + 5 : byte_offset + 4 + chunk_len]

        try:
            if compression == 2:  # zlib
                decompressed = zlib.decompress(raw_chunk)
            elif compression == 1:  # gzip
                import gzip

                decompressed = gzip.decompress(raw_chunk)
            else:
                continue
            nbt = read_nbt_root(decompressed)
        except Exception:
            continue

        local_x = idx % 32
        local_z = idx // 32
        cx = rx * 32 + local_x
        cz = rz * 32 + local_z
        yield cx, cz, nbt


# ── Flowing liquid detector ─────────────────────────────────────────────


def find_flowing_liquids(nbt: dict) -> list[dict]:
    """Return a list of flowing liquid palette entries in a chunk NBT."""
    results: list[dict] = []
    sections = nbt.get("sections", [])
    if not isinstance(sections, list):
        return results

    for section in sections:
        if not isinstance(section, dict):
            continue
        block_states = section.get("block_states", {})
        if not isinstance(block_states, dict):
            continue
        palette = block_states.get("palette", [])
        if not isinstance(palette, list):
            continue

        y = section.get("Y", "?")

        for entry in palette:
            if not isinstance(entry, dict):
                continue
            name = entry.get("Name", "")
            if name not in ("minecraft:water", "minecraft:lava"):
                continue
            props = entry.get("Properties", {})
            level = props.get("level", "0")
            if level != "0":
                results.append(
                    {
                        "block": name,
                        "level": level,
                        "section_y": y,
                    }
                )
    return results


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <world_dir>")
        print(f"  e.g. {sys.argv[0]} tools/fabric-server/runtime/world")
        return 1

    world_dir = Path(sys.argv[1])
    region_dir = world_dir / "region"
    if not region_dir.is_dir():
        print(f"ERROR: No region directory at {region_dir}")
        return 1

    mca_files = sorted(region_dir.glob("*.mca"))
    print(f"Scanning {len(mca_files)} region file(s) in {region_dir}")
    print()

    total_chunks = 0
    total_flowing = 0
    flowing_chunks: list[tuple[int, int, list[dict]]] = []

    for mca in mca_files:
        file_flowing = 0
        file_chunks = 0
        for cx, cz, nbt in iter_chunks(mca):
            file_chunks += 1
            liquids = find_flowing_liquids(nbt)
            if liquids:
                file_flowing += 1
                flowing_chunks.append((cx, cz, liquids))
        total_chunks += file_chunks
        total_flowing += file_flowing
        status = "CLEAN" if file_flowing == 0 else f"⚠ {file_flowing} chunks with flowing liquids"
        print(f"  {mca.name}: {file_chunks} chunks — {status}")

    print()
    print(f"Total: {total_chunks} chunks scanned, {total_flowing} with flowing liquids")

    if flowing_chunks:
        print()
        print("=== FLOWING LIQUID DETAILS ===")
        for cx, cz, liquids in flowing_chunks[:50]:  # cap output
            for liq in liquids:
                print(
                    f"  chunk ({cx:4d}, {cz:4d}) section Y={liq['section_y']:>3}: "
                    f"{liq['block']} level={liq['level']}"
                )
        if len(flowing_chunks) > 50:
            print(f"  ... and {len(flowing_chunks) - 50} more chunks")
        print()
        print("RESULT: FLOWING LIQUIDS DETECTED — /tick freeze may not have run in time")
        return 1
    else:
        print()
        print("RESULT: CLEAN — no flowing liquids found, /tick freeze was effective")
        return 0


if __name__ == "__main__":
    sys.exit(main())
