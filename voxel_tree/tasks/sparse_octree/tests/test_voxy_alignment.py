"""Validate subchunk alignment and parent–child consistency across Voxy LOD levels.

Two independent validation checks against real Voxy data:

1. **Subchunk alignment** — for a sampled player-section, verify that
   ``extract_section_subcube`` pulls data from the correct spatial region at
   every LOD level.  Specifically:
     * The WorldSection coord derived from the section coord matches the NPZ file.
     * The voxel slice indices stay within [0, 32) for every level.
     * All 5 levels refer to the same block-coordinate bounding box.
     * Coarser levels are *spatial averages* of finer levels (non-air content
       should be roughly consistent).

2. **Parent–child consistency** — for every Voxy WorldSection at levels 1–4,
   when ``non_empty_children`` says a child octant is populated, the
   corresponding child WorldSection at ``level − 1`` must actually exist.

Usage::

    pytest voxel_tree/tasks/sparse_octree/tests/test_voxy_alignment.py -v

    # Or standalone:
    python -m voxel_tree.tasks.sparse_octree.tests.test_voxy_alignment
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxel_tree.utils.coords import (
    child_x,
    child_y,
    child_z,
    section_to_world_section,
    world_section_to_block_min,
)
from voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs import (
    build_voxy_indices,
    extract_section_subcube,
)

# ── Data directory discovery ──────────────────────────────────────────────

_VT_ROOT = Path(__file__).resolve().parents[4]  # VoxelTree repo root
_VOXY_DIR = _VT_ROOT / "data" / "voxy_octree"


def _have_voxy_data() -> bool:
    """Return True when at least L0 and L4 Voxy data directories exist."""
    return (_VOXY_DIR / "level_0").is_dir() and (_VOXY_DIR / "level_4").is_dir()


# Skip the entire module if Voxy data isn't available (CI environments).
pytestmark = pytest.mark.skipif(
    not _have_voxy_data(),
    reason=f"Voxy data not found at {_VOXY_DIR}",
)


# ── Index helpers ─────────────────────────────────────────────────────────


def _build_indices() -> dict[int, dict[tuple[int, int, int], Path]]:
    """Build Voxy section indices for all 5 levels (cached across tests)."""
    return build_voxy_indices(_VOXY_DIR)


@pytest.fixture(scope="module")
def voxy_indices():
    """Module-scoped fixture: Voxy indices for all levels."""
    return _build_indices()


# ══════════════════════════════════════════════════════════════════════════
#  1. Subchunk alignment validation
# ══════════════════════════════════════════════════════════════════════════


class TestSubchunkAlignment:
    """Verify extract_section_subcube maps the right block region at all LODs."""

    # -- Helper: find player-sections that have Voxy data at all 5 levels ---

    @staticmethod
    def _find_all_level_sections(
        indices: dict[int, dict[tuple[int, int, int], Path]],
        max_samples: int = 50,
    ) -> list[tuple[int, int, int]]:
        """Return player-section (sx, sy, sz) coords with Voxy data at all 5 levels.

        We start from L0 WorldSections and derive the player-sections they
        contain, then verify each coarser level also has the corresponding WS.
        """
        l0_index = indices.get(0, {})
        if not l0_index:
            return []

        candidates: list[tuple[int, int, int]] = []
        for wx, wy, wz in sorted(l0_index.keys()):
            # Each L0 WorldSection covers 2 player-sections per axis.
            for dsx in range(2):
                for dsy in range(2):
                    for dsz in range(2):
                        sx = (wx << 1) + dsx
                        sy = (wy << 1) + dsy
                        sz = (wz << 1) + dsz

                        # Check all 5 levels have the matching WorldSection.
                        all_present = True
                        for level in range(5):
                            ws = (
                                section_to_world_section(sx, level),
                                section_to_world_section(sy, level),
                                section_to_world_section(sz, level),
                            )
                            if ws not in indices.get(level, {}):
                                all_present = False
                                break

                        if all_present:
                            candidates.append((sx, sy, sz))
                            if len(candidates) >= max_samples:
                                return candidates
        return candidates

    def test_coordinates_match_npz_metadata(self, voxy_indices) -> None:
        """WorldSection coords derived from section coords match NPZ metadata."""
        sections = self._find_all_level_sections(voxy_indices, max_samples=20)
        if not sections:
            pytest.skip("No player-sections found with all 5 levels")

        mismatches = []
        for sx, sy, sz in sections:
            for level in range(5):
                ws = (
                    section_to_world_section(sx, level),
                    section_to_world_section(sy, level),
                    section_to_world_section(sz, level),
                )
                npz_path = voxy_indices[level][ws]
                with np.load(npz_path) as data:
                    meta_x = int(data["section_x"])
                    meta_y = int(data["section_y"])
                    meta_z = int(data["section_z"])
                    meta_level = int(data["level"])

                if (meta_x, meta_y, meta_z) != ws or meta_level != level:
                    mismatches.append(
                        f"Section ({sx},{sy},{sz}) L{level}: "
                        f"expected ws={ws} but NPZ has ({meta_x},{meta_y},{meta_z}) L{meta_level}"
                    )

        assert not mismatches, f"{len(mismatches)} coordinate mismatches:\n" + "\n".join(
            mismatches[:10]
        )

    def test_subcube_slice_within_bounds(self, voxy_indices) -> None:
        """Extract indices must stay within [0, 32) for every level and section."""
        sections = self._find_all_level_sections(voxy_indices, max_samples=30)
        if not sections:
            pytest.skip("No player-sections found with all 5 levels")

        errors = []
        for sx, sy, sz in sections:
            for level in range(5):
                n = 16 >> level

                def _voxel_range(s: int):
                    ws = s >> (level + 1)
                    ls = s - (ws << (level + 1))
                    vs = ls * n
                    return vs, vs + n

                for axis_name, s in [("x", sx), ("y", sy), ("z", sz)]:
                    v0, v1 = _voxel_range(s)
                    if v0 < 0 or v1 > 32:
                        errors.append(
                            f"Section ({sx},{sy},{sz}) L{level} {axis_name}: "
                            f"slice [{v0}:{v1}) out of [0, 32)"
                        )

        assert not errors, f"{len(errors)} out-of-bounds slices:\n" + "\n".join(errors[:10])

    def test_all_levels_cover_same_block_range(self, voxy_indices) -> None:
        """All 5 LOD levels for the same section must span the same block range.

        The bounding box of blocks covered by a player-section at level L is
        computed from the WorldSection coords + voxel slice offsets + voxels/axis.
        """
        sections = self._find_all_level_sections(voxy_indices, max_samples=30)
        if not sections:
            pytest.skip("No player-sections found with all 5 levels")

        errors = []
        for sx, sy, sz in sections:
            block_ranges: dict[int, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {}

            for level in range(5):
                n = 16 >> level
                # blocks per voxel at this level
                bpv = 1 << level

                def _block_range_for_axis(s: int):
                    ws = s >> (level + 1)
                    ls = s - (ws << (level + 1))
                    vs = ls * n

                    # The WorldSection's block_min is ws << (5 + level)
                    ws_block_min = world_section_to_block_min(ws, level)
                    # Each voxel covers bpv blocks.
                    axis_block_min = ws_block_min + vs * bpv
                    axis_block_max = ws_block_min + (vs + n) * bpv - 1
                    return axis_block_min, axis_block_max

                bx = _block_range_for_axis(sx)
                by = _block_range_for_axis(sy)
                bz = _block_range_for_axis(sz)
                block_ranges[level] = (bx, by, bz)

            # All levels should produce the same block bounding box.
            ref = block_ranges[0]
            for level in range(1, 5):
                if block_ranges[level] != ref:
                    errors.append(
                        f"Section ({sx},{sy},{sz}): "
                        f"L0 blocks={ref} vs L{level} blocks={block_ranges[level]}"
                    )

        assert not errors, f"{len(errors)} block-range mismatches:\n" + "\n".join(errors[:10])

    def test_subcube_extraction_produces_valid_data(self, voxy_indices) -> None:
        """Extracted subcubes must have correct shape and non-negative IDs."""
        sections = self._find_all_level_sections(voxy_indices, max_samples=20)
        if not sections:
            pytest.skip("No player-sections found with all 5 levels")

        errors = []
        for sx, sy, sz in sections[:10]:  # Limit I/O
            for level in range(5):
                ws = (
                    section_to_world_section(sx, level),
                    section_to_world_section(sy, level),
                    section_to_world_section(sz, level),
                )
                with np.load(voxy_indices[level][ws]) as data:
                    labels32 = data["labels32"]

                sub = extract_section_subcube(labels32, sx, sy, sz, level)
                expected_n = 16 >> level
                if sub.shape != (expected_n, expected_n, expected_n):
                    errors.append(
                        f"Section ({sx},{sy},{sz}) L{level}: "
                        f"expected shape ({expected_n},)*3, got {sub.shape}"
                    )
                if sub.min() < 0:
                    errors.append(
                        f"Section ({sx},{sy},{sz}) L{level}: " f"negative block ID {sub.min()}"
                    )

        assert not errors, f"{len(errors)} extraction errors:\n" + "\n".join(errors[:10])

    def test_coarse_level_content_consistent_with_finer(self, voxy_indices) -> None:
        """Coarser levels should have non-air content where finer levels do.

        At each pair of adjacent levels (L_coarse, L_fine), if a sub-cube at
        the fine level is non-empty (has non-air blocks), then the corresponding
        single voxel at the coarse level should also be non-air.

        This catches spatial misalignment bugs where we'd pull the wrong
        WorldSection data for a given player-section.
        """
        sections = self._find_all_level_sections(voxy_indices, max_samples=15)
        if not sections:
            pytest.skip("No player-sections found with all 5 levels")

        # Counters for reporting
        total_checks = 0
        inconsistent = 0
        details = []

        for sx, sy, sz in sections[:10]:
            # Load sub-cubes at all levels
            subcubes: dict[int, np.ndarray] = {}
            for level in range(5):
                ws = (
                    section_to_world_section(sx, level),
                    section_to_world_section(sy, level),
                    section_to_world_section(sz, level),
                )
                with np.load(voxy_indices[level][ws]) as data:
                    labels32 = data["labels32"]
                subcubes[level] = extract_section_subcube(labels32, sx, sy, sz, level)

            # Compare adjacent level pairs.
            # At level L_coarse, each voxel corresponds to 2³ voxels at L_fine.
            for l_coarse in range(4, 0, -1):
                l_fine = l_coarse - 1
                coarse = subcubes[l_coarse]
                fine = subcubes[l_fine]
                n_coarse = coarse.shape[0]

                for cy in range(n_coarse):
                    for cz in range(n_coarse):
                        for cx in range(n_coarse):
                            total_checks += 1
                            coarse_val = coarse[cy, cz, cx]
                            # Corresponding 2³ region in fine level
                            fine_region = fine[
                                cy * 2 : cy * 2 + 2,
                                cz * 2 : cz * 2 + 2,
                                cx * 2 : cx * 2 + 2,
                            ]
                            fine_has_content = np.any(fine_region != 0)
                            coarse_has_content = coarse_val != 0

                            # If fine level has blocks, coarse should too.
                            # (Reverse is also expected but less strict — coarse
                            #  LOD may represent a "most-common" block.)
                            if fine_has_content and not coarse_has_content:
                                inconsistent += 1
                                if len(details) < 5:
                                    details.append(
                                        f"Section ({sx},{sy},{sz}) "
                                        f"L{l_coarse}[{cy},{cz},{cx}]=air but "
                                        f"L{l_fine} children have content "
                                        f"(unique: {np.unique(fine_region).tolist()})"
                                    )

        # Allow a small tolerance — Voxy LOD quantization can occasionally
        # drop thin features.  Fail if >5% of checks are inconsistent.
        rate = inconsistent / max(total_checks, 1)
        assert rate < 0.05, (
            f"{inconsistent}/{total_checks} ({rate:.1%}) coarse-vs-fine inconsistencies "
            f"exceed 5% threshold:\n" + "\n".join(details)
        )


# ══════════════════════════════════════════════════════════════════════════
#  2. Parent–child consistency validation
# ══════════════════════════════════════════════════════════════════════════


class TestParentChildConsistency:
    """Verify Voxy non_empty_children bitmask matches actual child NPZ files.

    IMPORTANT CONTEXT — ``non_empty_children`` comes directly from Voxy's
    RocksDB and is *not* used by the sparse-octree training pipeline (which
    computes its own child occupancy from actual voxel data via
    ``child_occupancy_mask()``).  Therefore these tests validate **data
    quality** rather than training correctness.

    Known data-quality characteristics (as of 2025-06):
    - ~3–4% of flagged children are missing (mostly upper-Y boundary sections
      skipped by the ``min_solid_fraction`` filter during extraction).
    - ~16% of unflagged slots have children with >5% solid content (Voxy's
      bitmask is stale for sections populated after the parent was written).
    Both are expected for a snapshot extraction from a live Voxy world.
    """

    def test_children_exist_when_flagged(self, voxy_indices) -> None:
        """For every WorldSection at L1–L4, each flagged child octant must
        exist as an NPZ at the next finer level.

        The ``non_empty_children`` byte mask tells us which of the 8 octant
        children (at the next finer LOD level) are populated.  We compute the
        child WorldSection coordinates using the standard octree child
        expansion and verify the file exists.

        Tolerance: <10%.  Most misses are upper-Y boundary sections that were
        below the solid-fraction filter during extraction.  A rate above 10%
        would suggest a coordinate math bug.
        """
        missing_children = []
        total_flagged = 0
        total_found = 0

        for parent_level in range(4, 0, -1):  # L4, L3, L2, L1
            child_level = parent_level - 1
            parent_index = voxy_indices.get(parent_level, {})
            child_index = voxy_indices.get(child_level, {})

            for (px, py, pz), npz_path in parent_index.items():
                with np.load(npz_path) as data:
                    nec = int(data["non_empty_children"])

                for octant in range(8):
                    if not (nec & (1 << octant)):
                        continue  # This child not flagged — skip

                    total_flagged += 1

                    # Compute child WS coords from parent + octant
                    cx = child_x(px, octant)
                    cy = child_y(py, octant)
                    cz = child_z(pz, octant)

                    if (cx, cy, cz) in child_index:
                        total_found += 1
                    else:
                        missing_children.append(
                            f"L{parent_level} parent ({px},{py},{pz}) nec={nec:08b} "
                            f"→ octant {octant} child L{child_level} ({cx},{cy},{cz}) MISSING"
                        )
                        if len(missing_children) >= 200:
                            break
                if len(missing_children) >= 200:
                    break
            if len(missing_children) >= 200:
                break

        print(
            f"\nParent-child check: {total_found}/{total_flagged} flagged children found "
            f"({len(missing_children)} missing)"
        )

        # >10% would indicate a systematic coordinate bug, not just extraction
        # boundary effects.  The known rate is ~3-4%.
        if total_flagged > 0:
            miss_rate = len(missing_children) / total_flagged
            assert miss_rate < 0.10, (
                f"{len(missing_children)}/{total_flagged} flagged children are missing "
                f"({miss_rate:.1%}) — exceeds 10% threshold suggesting coordinate bug.\n\n"
                f"First examples:\n" + "\n".join(missing_children[:20])
            )
            if miss_rate > 0.01:
                import warnings

                warnings.warn(
                    f"{len(missing_children)}/{total_flagged} flagged children missing "
                    f"({miss_rate:.1%}) — within tolerance (extraction boundary effect)"
                )

    def test_unflagged_children_mostly_absent(self, voxy_indices) -> None:
        """When ``non_empty_children`` does NOT flag a child, the child NPZ
        should generally not exist (or if it does, should be mostly-air).

        This validates the bit→octant mapping direction: if the bitmask bits
        were swapped (e.g. x↔z or x↔y) we'd see a *dramatic* mismatch rate
        (>50%), not the ~16% observed from stale bitmasks.

        Tolerance: <30%.  The known rate is ~16% due to Voxy writing child
        sections after the parent's bitmask was last updated.  A rate above
        30% would suggest the octant bit layout is wrong.
        """
        phantom_children = 0
        total_unflagged = 0

        for parent_level in range(4, 0, -1):
            child_level = parent_level - 1
            parent_index = voxy_indices.get(parent_level, {})
            child_index = voxy_indices.get(child_level, {})

            # Sample a subset to keep runtime manageable
            sample = list(parent_index.items())[:200]

            for (px, py, pz), npz_path in sample:
                with np.load(npz_path) as data:
                    nec = int(data["non_empty_children"])

                for octant in range(8):
                    if nec & (1 << octant):
                        continue  # Flagged — skip

                    total_unflagged += 1

                    cx = child_x(px, octant)
                    cy = child_y(py, octant)
                    cz = child_z(pz, octant)

                    if (cx, cy, cz) in child_index:
                        # Child exists but was NOT flagged.  Check if it's
                        # substantive (i.e. has non-air content).
                        with np.load(child_index[(cx, cy, cz)]) as cdata:
                            labels = cdata["labels32"]
                            solid_frac = float(np.mean(labels != 0))
                            if solid_frac > 0.05:
                                phantom_children += 1

        if total_unflagged > 0:
            phantom_rate = phantom_children / total_unflagged
            assert phantom_rate < 0.30, (
                f"{phantom_children}/{total_unflagged} unflagged children have >5% solid content "
                f"({phantom_rate:.1%}) — exceeds 30% threshold suggesting octant bit layout is wrong"
            )
            if phantom_rate > 0.05:
                import warnings

                warnings.warn(
                    f"{phantom_children}/{total_unflagged} unflagged children have >5% solid "
                    f"({phantom_rate:.1%}) — expected for stale Voxy bitmasks"
                )

    def test_child_coord_identity(self) -> None:
        """Pure-math check: child expansion + octant decomposition round-trips.

        For every parent coord and octant, the child coord must be deterministic
        and reversible.
        """
        from voxel_tree.utils.coords import octant_index

        for parent in [-5, -1, 0, 1, 7, 15]:
            for octant in range(8):
                cx_val = child_x(parent, octant)
                cy_val = child_y(parent, octant)
                cz_val = child_z(parent, octant)

                # Verify the child local offset reconstructs the octant
                lx = cx_val & 1
                ly = cy_val & 1
                lz = cz_val & 1
                recovered = octant_index(lx, ly, lz)
                assert recovered == octant, (
                    f"parent={parent} octant={octant}: "
                    f"child=({cx_val},{cy_val},{cz_val}) → "
                    f"recovered octant={recovered}"
                )

                # Verify parent is recoverable
                assert cx_val >> 1 == parent
                assert cy_val >> 1 == parent
                assert cz_val >> 1 == parent


# ══════════════════════════════════════════════════════════════════════════
#  3. Cross-level spatial coverage sanity
# ══════════════════════════════════════════════════════════════════════════


class TestCrossLevelCoverage:
    """Verify that higher LOD levels geometrically contain lower ones."""

    def test_l4_covers_l0_ws_coords(self, voxy_indices) -> None:
        """Every L0 WorldSection's block range must fall within some L4 WS."""
        l0_index = voxy_indices.get(0, {})
        l4_index = voxy_indices.get(4, {})
        if not l0_index or not l4_index:
            pytest.skip("Need both L0 and L4 data")

        uncovered = 0
        for wx, wy, wz in list(l0_index.keys())[:500]:
            # Find expected L4 WorldSection from block coords
            block_x = world_section_to_block_min(wx, 0)
            block_y = world_section_to_block_min(wy, 0)
            block_z = world_section_to_block_min(wz, 0)

            from voxel_tree.utils.coords import block_to_world_section

            l4_x = block_to_world_section(block_x, 4)
            l4_y = block_to_world_section(block_y, 4)
            l4_z = block_to_world_section(block_z, 4)

            if (l4_x, l4_y, l4_z) not in l4_index:
                uncovered += 1

        # Some L0 sections at the edges may not have L4 coverage (exploration
        # boundaries), but the vast majority should.
        total = min(len(l0_index), 500)
        coverage = 1.0 - uncovered / total if total > 0 else 0
        assert coverage > 0.80, (
            f"Only {coverage:.0%} of L0 WorldSections have a corresponding L4 parent "
            f"({uncovered}/{total} uncovered)"
        )


# ══════════════════════════════════════════════════════════════════════════
#  Standalone runner
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run all validations and print a summary."""
    import sys
    import time

    print("=" * 70)
    print("  Voxy Alignment & Parent–Child Consistency Validation")
    print("=" * 70)
    print(f"  Data dir: {_VOXY_DIR}")
    print()

    if not _have_voxy_data():
        print("ERROR: Voxy data not found. Exiting.")
        sys.exit(1)

    t0 = time.time()

    # Build indices
    print("Building Voxy indices...")
    indices = _build_indices()
    for level in range(5):
        n = len(indices.get(level, {}))
        print(f"  L{level}: {n:,} WorldSections indexed")
    print()

    # --- Test 1: Subchunk alignment ----
    print("─" * 70)
    print("  Test 1: Subchunk Alignment")
    print("─" * 70)

    tester = TestSubchunkAlignment()
    sections = tester._find_all_level_sections(indices, max_samples=30)
    print(f"  Found {len(sections)} player-sections with all 5 levels")

    if not sections:
        print("  SKIP: No multi-level sections found")
    else:
        # 1a: Coordinate metadata match
        errors_1a = []
        for sx, sy, sz in sections:
            for level in range(5):
                ws = (
                    section_to_world_section(sx, level),
                    section_to_world_section(sy, level),
                    section_to_world_section(sz, level),
                )
                with np.load(indices[level][ws]) as data:
                    meta = (int(data["section_x"]), int(data["section_y"]), int(data["section_z"]))
                    if meta != ws:
                        errors_1a.append(f"({sx},{sy},{sz}) L{level}: ws={ws} meta={meta}")
        print(
            f"  1a WS coord match: {'PASS' if not errors_1a else 'FAIL'} "
            f"({len(sections) * 5} checks, {len(errors_1a)} errors)"
        )

        # 1b: Slice bounds
        errors_1b = 0
        for sx, sy, sz in sections:
            for level in range(5):
                n = 16 >> level
                for s in [sx, sy, sz]:
                    ws = s >> (level + 1)
                    ls = s - (ws << (level + 1))
                    vs = ls * n
                    if vs < 0 or vs + n > 32:
                        errors_1b += 1
        print(
            f"  1b Slice bounds:   {'PASS' if not errors_1b else 'FAIL'} "
            f"({len(sections) * 5 * 3} checks, {errors_1b} errors)"
        )

        # 1c: Block range consistency
        errors_1c = 0
        for sx, sy, sz in sections:
            ranges = {}
            for level in range(5):
                n = 16 >> level
                bpv = 1 << level

                def _br(s, lvl=level, nn=n, bpv_=bpv):
                    ws = s >> (lvl + 1)
                    ls = s - (ws << (lvl + 1))
                    vs = ls * nn
                    ws_bmin = world_section_to_block_min(ws, lvl)
                    return ws_bmin + vs * bpv_, ws_bmin + (vs + nn) * bpv_ - 1

                ranges[level] = (_br(sx), _br(sy), _br(sz))
            ref = ranges[0]
            for lvl in range(1, 5):
                if ranges[lvl] != ref:
                    errors_1c += 1
                    break
        print(
            f"  1c Block ranges:   {'PASS' if not errors_1c else 'FAIL'} "
            f"({len(sections)} sections, {errors_1c} mismatches)"
        )

        # 1d: Coarse-fine content consistency
        total_checks = 0
        inconsistent = 0
        for sx, sy, sz in sections[:10]:
            subcubes = {}
            for level in range(5):
                ws = (
                    section_to_world_section(sx, level),
                    section_to_world_section(sy, level),
                    section_to_world_section(sz, level),
                )
                with np.load(indices[level][ws]) as data:
                    subcubes[level] = extract_section_subcube(data["labels32"], sx, sy, sz, level)
            for lc in range(4, 0, -1):
                lf = lc - 1
                coarse = subcubes[lc]
                fine = subcubes[lf]
                nc = coarse.shape[0]
                for cy in range(nc):
                    for cz in range(nc):
                        for cx in range(nc):
                            total_checks += 1
                            fine_region = fine[
                                cy * 2 : cy * 2 + 2, cz * 2 : cz * 2 + 2, cx * 2 : cx * 2 + 2
                            ]
                            if np.any(fine_region != 0) and coarse[cy, cz, cx] == 0:
                                inconsistent += 1
        rate = inconsistent / max(total_checks, 1)
        print(
            f"  1d Content check:  {'PASS' if rate < 0.05 else 'FAIL'} "
            f"({total_checks} voxels, {inconsistent} inconsistent, {rate:.1%})"
        )

    # --- Test 2: Parent–child consistency ----
    print()
    print("─" * 70)
    print("  Test 2: Parent–Child Consistency")
    print("─" * 70)

    total_flagged = 0
    total_found = 0
    missing: list[str] = []

    for parent_level in range(4, 0, -1):
        child_level = parent_level - 1
        parent_index = indices.get(parent_level, {})
        child_index = indices.get(child_level, {})

        level_flagged = 0
        level_found = 0
        level_missing = 0

        for (px, py, pz), npz_path in parent_index.items():
            with np.load(npz_path) as data:
                nec = int(data["non_empty_children"])

            for octant in range(8):
                if not (nec & (1 << octant)):
                    continue
                level_flagged += 1
                cx_val = child_x(px, octant)
                cy_val = child_y(py, octant)
                cz_val = child_z(pz, octant)
                if (cx_val, cy_val, cz_val) in child_index:
                    level_found += 1
                else:
                    level_missing += 1
                    if len(missing) < 20:
                        missing.append(
                            f"  L{parent_level}({px},{py},{pz}) oct={octant} → "
                            f"L{child_level}({cx_val},{cy_val},{cz_val}) MISSING"
                        )

        total_flagged += level_flagged
        total_found += level_found
        miss_rate = level_missing / max(level_flagged, 1)
        status = "PASS" if miss_rate < 0.01 else ("WARN" if miss_rate < 0.05 else "FAIL")
        print(
            f"  L{parent_level}→L{child_level}: {level_found}/{level_flagged} found, "
            f"{level_missing} missing ({miss_rate:.1%}) [{status}]"
        )

    total_miss = total_flagged - total_found
    total_rate = total_miss / max(total_flagged, 1)
    print(
        f"\n  Overall: {total_found}/{total_flagged} found, "
        f"{total_miss} missing ({total_rate:.1%})"
    )
    if missing:
        print("\n  First missing children:")
        for m in missing[:10]:
            print(m)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    all_pass = (
        not sections or (not errors_1a and not errors_1b and not errors_1c and rate < 0.05)
    ) and total_rate < 0.01
    print(f"  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'} ({elapsed:.1f}s)")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
