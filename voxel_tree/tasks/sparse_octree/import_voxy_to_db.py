"""Import Voxy NPZ ground-truth grids into SQLite for fast JOINed training.

Creates a ``voxy_sections`` table inside the **noise-dumps** database so
that training data can be assembled via a single SQL JOIN — no filesystem
NPZ reads required.

Usage (CLI)::

    voxel-tree --step import_voxy --run --profile phase7_multilevel

Or programmatically::

    from voxel_tree.tasks.sparse_octree.import_voxy_to_db import import_voxy
    import_voxy(
        dumps_db_path=Path("v7_dumps.db"),
        voxy_dir=Path("data/voxy_octree"),
    )
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ── Schema ───────────────────────────────────────────────────────────────

_VOXY_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS voxy_sections (
    level   INTEGER NOT NULL,
    ws_x    INTEGER NOT NULL,
    ws_y    INTEGER NOT NULL,
    ws_z    INTEGER NOT NULL,
    labels32 BLOB NOT NULL,
    PRIMARY KEY (level, ws_x, ws_y, ws_z)
);
"""


def _read_and_compress(args: tuple[int, int, int, int, str, int]) -> tuple[int, int, int, int, bytes]:
    """Read one NPZ file and zlib-compress its labels32 array.

    Designed to run in a ThreadPoolExecutor — I/O-bound, releases GIL.
    """
    level, wx, wy, wz, fpath, comp_level = args
    with np.load(fpath) as npz:
        labels32 = npz["labels32"]
    blob = zlib.compress(labels32.astype(np.int32).tobytes(), level=comp_level)
    return (level, wx, wy, wz, blob)


def import_voxy(
    dumps_db_path: Path,
    voxy_dir: Path,
    *,
    batch_size: int = 2000,
    compression_level: int = 1,
    num_workers: int = 0,
) -> int:
    """Import Voxy NPZ files into the ``voxy_sections`` table.

    Scans ``voxy_dir/level_0/`` through ``voxy_dir/level_4/`` for NPZ files,
    reads each ``labels32`` array (int32, 32x32x32), compresses with zlib,
    and inserts into the ``voxy_sections`` table in the dumps database.

    If ``voxy_sections`` already exists it is **dropped and recreated** so
    the import is idempotent.

    Parameters
    ----------
    dumps_db_path : Path
        Path to the v7 noise-dumps SQLite database.
    voxy_dir : Path
        Root directory containing ``level_0/`` ... ``level_4/`` subdirs.
    batch_size : int
        Rows to accumulate between commits (controls memory & WAL size).
    compression_level : int
        zlib compression level (1 = fast, 9 = small).  Level 1 is ~10x
        faster than level 9 with only ~15% larger BLOBs.
    num_workers : int
        Number of threads for parallel NPZ reads.  0 = auto (cpu_count).

    Returns
    -------
    int
        Total number of Voxy sections imported.
    """
    if num_workers <= 0:
        num_workers = min(os.cpu_count() or 4, 12)

    t0 = time.monotonic()
    conn = sqlite3.connect(str(dumps_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-256000")  # 256 MB

    # Drop + recreate for idempotent reimport
    conn.execute("DROP TABLE IF EXISTS voxy_sections")
    conn.executescript(_VOXY_TABLE_SQL)

    # Scan all 5 levels
    total_imported = 0
    for level in range(5):
        level_dir = voxy_dir / f"level_{level}"
        if not level_dir.is_dir():
            print(f"  Warning: {level_dir} not found, skipping")
            continue

        # Match both old (voxy_L4_...) and new (w0_voxy_L4_...) naming.
        pat = re.compile(
            rf"(?:w\d+_)?voxy_L{level}_x(-?\d+)_y(-?\d+)_z(-?\d+)\.npz$"
        )

        files: list[tuple[int, int, int, int, str, int]] = []
        for f in level_dir.iterdir():
            m = pat.search(f.name)
            if m:
                x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
                files.append((level, x, y, z, str(f), compression_level))

        if not files:
            print(f"  level_{level}: no files found")
            continue

        print(
            f"  level_{level}: importing {len(files):,} files "
            f"({num_workers} workers) ...",
            flush=True,
        )

        level_count = 0
        batch: list[tuple[int, int, int, int, bytes]] = []

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for row in pool.map(_read_and_compress, files, chunksize=64):
                batch.append(row)
                if len(batch) >= batch_size:
                    conn.executemany(
                        "INSERT OR REPLACE INTO voxy_sections "
                        "(level, ws_x, ws_y, ws_z, labels32) VALUES (?,?,?,?,?)",
                        batch,
                    )
                    conn.commit()
                    level_count += len(batch)
                    # Progress every 10k rows
                    if level_count % 10_000 < batch_size:
                        elapsed = time.monotonic() - t0
                        rate = level_count / max(elapsed, 0.01)
                        eta = (len(files) - level_count) / max(rate, 1)
                        print(
                            f"    {level_count:>9,}/{len(files):,} "
                            f"({rate:,.0f}/s, ETA {eta:.0f}s)",
                            flush=True,
                        )
                    batch.clear()

        # Flush remainder
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO voxy_sections "
                "(level, ws_x, ws_y, ws_z, labels32) VALUES (?,?,?,?,?)",
                batch,
            )
            conn.commit()
            level_count += len(batch)
            batch.clear()

        total_imported += level_count
        elapsed = time.monotonic() - t0
        print(f"    -> {level_count:,} rows ({elapsed:.0f}s)")

    # Verify
    (stored,) = conn.execute("SELECT COUNT(*) FROM voxy_sections").fetchone()
    elapsed = time.monotonic() - t0
    db_size_mb = Path(dumps_db_path).stat().st_size / (1024 * 1024)
    print(f"\n  Done: {stored:,} Voxy sections imported in {elapsed:.0f}s")
    print(f"  DB size: {db_size_mb:,.0f} MB")
    conn.close()
    return total_imported
