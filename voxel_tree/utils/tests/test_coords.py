"""Tests for voxel_tree.utils.coords — Python port of WorldSectionCoord.java.

Test cases are taken directly from the Java WorldSectionCoordTest.java to
ensure perfect cross-language agreement.
"""

from __future__ import annotations

import pytest

from voxel_tree.utils.coords import (
    block_to_section,
    block_to_world_section,
    child_x,
    child_y,
    child_z,
    describe,
    l0_to_voxy_section,
    l0_to_voxy_section_max,
    l0_to_voxy_section_min,
    octant_index,
    section_to_block_max,
    section_to_block_min,
    section_to_world_section,
    trace_block,
    voxy_section_contains,
    voxy_section_to_block_max,
    voxy_section_to_block_min,
    world_section_contains,
    world_section_to_block_max,
    world_section_to_block_min,
    world_section_to_section_max,
    world_section_to_section_min,
    world_section_width,
)


# ══════════════════════════════════════════════════════════════════════════
#  Block ↔ PlayerSection
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "block, expected",
    [
        (0, 0),
        (1, 0),
        (15, 0),
        (16, 1),
        (17, 1),
        (31, 1),
        (32, 2),
        (255, 15),
        (256, 16),
        (-1, -1),
        (-7, -1),
        (-15, -1),
        (-16, -1),  # -16 >> 4 = -1
        (-17, -2),
        (-32, -2),
        (-33, -3),
    ],
)
def test_block_to_section(block: int, expected: int) -> None:
    assert block_to_section(block) == expected


def test_section_to_block_range_round_trips() -> None:
    for sec in range(-10, 11):
        lo = section_to_block_min(sec)
        hi = section_to_block_max(sec)
        assert hi - lo + 1 == 16, f"section {sec} should span 16 blocks"
        assert block_to_section(lo) == sec
        assert block_to_section(hi) == sec


# ══════════════════════════════════════════════════════════════════════════
#  Block ↔ WorldSection (level-aware)
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "block, level, expected",
    [
        # L0: 32 blocks per ws
        (0, 0, 0),
        (31, 0, 0),
        (32, 0, 1),
        (-1, 0, -1),
        (-32, 0, -1),
        (-33, 0, -2),
        # L1: 64 blocks per ws
        (0, 1, 0),
        (63, 1, 0),
        (64, 1, 1),
        (-1, 1, -1),
        (-64, 1, -1),
        (-65, 1, -2),
        # L2: 128 blocks per ws
        (127, 2, 0),
        (128, 2, 1),
        (-1, 2, -1),
        (-128, 2, -1),
        (-129, 2, -2),
        # L3: 256 blocks per ws
        (255, 3, 0),
        (256, 3, 1),
        (-1, 3, -1),
        (-256, 3, -1),
        (-257, 3, -2),
        # L4: 512 blocks per ws
        (511, 4, 0),
        (512, 4, 1),
        (-1, 4, -1),
        (-512, 4, -1),
        (-513, 4, -2),
    ],
)
def test_block_to_world_section(block: int, level: int, expected: int) -> None:
    assert block_to_world_section(block, level) == expected


@pytest.mark.parametrize(
    "level, expected_width",
    [(0, 32), (1, 64), (2, 128), (3, 256), (4, 512)],
)
def test_world_section_width(level: int, expected_width: int) -> None:
    assert world_section_width(level) == expected_width


def test_world_section_to_block_range_round_trips() -> None:
    for level in range(5):
        width = world_section_width(level)
        for ws in range(-5, 6):
            lo = world_section_to_block_min(ws, level)
            hi = world_section_to_block_max(ws, level)
            assert hi - lo + 1 == width
            assert block_to_world_section(lo, level) == ws
            assert block_to_world_section(hi, level) == ws


# ══════════════════════════════════════════════════════════════════════════
#  PlayerSection ↔ WorldSection
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "section, level, expected",
    [
        # L0: section >> 1
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, -1),
        (-2, 0, -1),
        (6, 0, 3),
        # L4: section >> 5
        (0, 4, 0),
        (6, 4, 0),
        (31, 4, 0),
        (32, 4, 1),
        (-1, 4, -1),
        (-32, 4, -1),
        (-33, 4, -2),
    ],
)
def test_section_to_world_section(section: int, level: int, expected: int) -> None:
    assert section_to_world_section(section, level) == expected


# ══════════════════════════════════════════════════════════════════════════
#  ★ CONSISTENCY IDENTITY ★
#  block_to_world_section(b, L) == section_to_world_section(block_to_section(b), L)
# ══════════════════════════════════════════════════════════════════════════


def test_consistency_sweep() -> None:
    """One-step and two-step conversions must agree for ALL inputs."""
    for block in range(-600, 601):
        for level in range(5):
            direct = block_to_world_section(block, level)
            two_step = section_to_world_section(block_to_section(block), level)
            assert direct == two_step, f"Consistency failed: block={block} level={level}"


@pytest.mark.parametrize(
    "block",
    [
        0,
        1,
        -1,
        15,
        16,
        -16,
        -17,
        31,
        32,
        -32,
        -33,
        100,
        -100,
        255,
        256,
        -256,
        -257,
        511,
        512,
        -512,
        -513,
        1023,
        1024,
        -1024,
        -1025,
    ],
)
def test_consistency_selected(block: int) -> None:
    for level in range(5):
        direct = block_to_world_section(block, level)
        two_step = section_to_world_section(block_to_section(block), level)
        assert direct == two_step


# ══════════════════════════════════════════════════════════════════════════
#  L0 ↔ Voxy Section
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "l0ws, exp_min, exp_max",
    [(0, 0, 1), (1, 2, 3), (-1, -2, -1), (5, 10, 11)],
)
def test_l0_to_voxy_section_range(l0ws: int, exp_min: int, exp_max: int) -> None:
    assert l0_to_voxy_section_min(l0ws) == exp_min
    assert l0_to_voxy_section_max(l0ws) == exp_max


def test_l0_to_voxy_section_offset() -> None:
    for l0ws in range(-5, 6):
        assert l0_to_voxy_section(l0ws, 0) == l0_to_voxy_section_min(l0ws)
        assert l0_to_voxy_section(l0ws, 1) == l0_to_voxy_section_max(l0ws)


def test_voxy_section_block_range() -> None:
    for vs in range(-5, 6):
        lo = voxy_section_to_block_min(vs)
        hi = voxy_section_to_block_max(vs)
        assert hi - lo + 1 == 16
        assert lo == vs * 16


# ══════════════════════════════════════════════════════════════════════════
#  Octree child expansion
# ══════════════════════════════════════════════════════════════════════════


def test_child_octant0_doubles() -> None:
    assert child_x(5, 0) == 10
    assert child_y(5, 0) == 10
    assert child_z(5, 0) == 10


def test_child_octant7_doubles_and_adds() -> None:
    assert child_x(5, 7) == 11  # bit0=1
    assert child_y(5, 7) == 11  # bit2=1
    assert child_z(5, 7) == 11  # bit1=1


def test_octant_index_round_trips() -> None:
    for octant in range(8):
        lx = octant & 1
        lz = (octant >> 1) & 1
        ly = (octant >> 2) & 1
        assert octant_index(lx, ly, lz) == octant


def test_child_expansion_covers_parent_range() -> None:
    """All 8 children, unioned, should exactly cover the parent's block range."""
    for level in range(1, 5):
        child_level = level - 1
        for parent_x in range(-3, 4):
            parent_min = world_section_to_block_min(parent_x, level)
            parent_max = world_section_to_block_max(parent_x, level)
            child_mins = []
            child_maxs = []
            for oct in range(2):  # only X axis bits here
                cx = child_x(parent_x, oct)
                child_mins.append(world_section_to_block_min(cx, child_level))
                child_maxs.append(world_section_to_block_max(cx, child_level))
            assert min(child_mins) == parent_min
            assert max(child_maxs) == parent_max


# ══════════════════════════════════════════════════════════════════════════
#  Containment
# ══════════════════════════════════════════════════════════════════════════


def test_world_section_contains_boundaries() -> None:
    # L4 ws=0 covers [0, 511]
    assert world_section_contains(0, 4, 0)
    assert world_section_contains(0, 4, 511)
    assert not world_section_contains(0, 4, 512)
    assert not world_section_contains(0, 4, -1)
    # L4 ws=-1 covers [-512, -1]
    assert world_section_contains(-1, 4, -1)
    assert world_section_contains(-1, 4, -512)
    assert not world_section_contains(-1, 4, -513)
    assert not world_section_contains(-1, 4, 0)


def test_voxy_section_contains() -> None:
    assert voxy_section_contains(0, 0)
    assert voxy_section_contains(0, 15)
    assert not voxy_section_contains(0, 16)
    assert not voxy_section_contains(0, -1)


# ══════════════════════════════════════════════════════════════════════════
#  WorldSection ↔ PlayerSection ranges
# ══════════════════════════════════════════════════════════════════════════


def test_world_section_section_range() -> None:
    # L4 ws=0: player sections 0..31
    assert world_section_to_section_min(0, 4) == 0
    assert world_section_to_section_max(0, 4) == 31
    # L4 ws=-1: player sections -32..-1
    assert world_section_to_section_min(-1, 4) == -32
    assert world_section_to_section_max(-1, 4) == -1


# ══════════════════════════════════════════════════════════════════════════
#  Diagnostic helpers
# ══════════════════════════════════════════════════════════════════════════


def test_describe_format() -> None:
    desc = describe(-1, 4)
    assert "L4" in desc
    assert "-512" in desc
    assert "-1" in desc
    assert "512 wide" in desc


def test_trace_block_contains_all_levels() -> None:
    trace = trace_block(100, 64, -200)
    assert "Block (100, 64, -200)" in trace
    for level in range(5):
        assert f"L{level}" in trace
    assert "Voxy:" in trace


# ══════════════════════════════════════════════════════════════════════════
#  Cross-validation: noise section → Voxy L4 key mapping
#  This is the exact mapping fix in build_sparse_octree_pairs.py
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "noise_sy, expected_voxy_y",
    [
        # Overworld noise sections: sy=-4..19
        (-4, -1),  # blocks [-64, -49] → L4 ws -1 ([-512, -1])
        (-3, -1),
        (-2, -1),
        (-1, -1),
        (0, 0),  # blocks [0, 15] → L4 ws 0 ([0, 511])
        (1, 0),
        (3, 0),  # surface! blocks [48, 63]
        (4, 0),  # blocks [64, 79]
        (10, 0),  # blocks [160, 175]
        (19, 0),  # blocks [304, 319]
        (31, 0),  # still ws 0
        (32, 1),  # above overworld, ws 1
        (-32, -1),  # still ws -1
        (-33, -2),  # below overworld, ws -2
    ],
)
def test_noise_section_to_voxy_l4(noise_sy: int, expected_voxy_y: int) -> None:
    """Noise dump sy → Voxy L4 WorldSection Y mapping.

    This is the critical mapping that was WRONG (raw key equality) in
    build_sparse_octree_pairs.py before the fix.
    """
    assert section_to_world_section(noise_sy, 4) == expected_voxy_y
