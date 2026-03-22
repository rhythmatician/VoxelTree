"""Minecraft → Octree → Voxy coordinate conversions (Python port).

This is a direct port of ``WorldSectionCoord.java`` from LODiffusion so that
every coordinate system is defined in **one** tested place for each language.

Coordinate systems
------------------

=======================  ==============  ===========================
Name                     Width (blocks)  Conversion from block
=======================  ==============  ===========================
Block                    1               identity
PlayerSection (Voxy L0)  16              ``block >> 4``
WorldSection at level L  ``32 × 2^L``   ``block >> (5 + L)``
Voxy section (16-block)  16              ``L0_ws * 2 + offset``
=======================  ==============  ===========================

Key identities::

    player_section(block)        = block >> 4
    world_section(block, L)      = block >> (5 + L)
    world_section(pSec, L)       = pSec  >> (L + 1)

    # Two-step equals one-step:
    block >> (5 + L)  ≡  (block >> 4) >> (L + 1)

Octree child expansion::

    child_x = (parent_x << 1) + (octant & 1)
    child_y = (parent_y << 1) + ((octant >> 2) & 1)
    child_z = (parent_z << 1) + ((octant >> 1) & 1)
    octant  = (lx & 1) | ((lz & 1) << 1) | ((ly & 1) << 2)

All functions are pure (no shared state) and safe for use from any thread.
"""

from __future__ import annotations

# ── Constants ──────────────────────────────────────────────────────────────

BLOCKS_PER_SECTION: int = 16  # 2^4  (PlayerSection / Voxy-level-0 section)
BLOCKS_PER_L0_WS: int = 32  # 2^5  (L0 WorldSection)
VOXY_PER_L0: int = 2  # 16-block Voxy sections per L0 WorldSection axis
SECTION_SHIFT: int = 4  # block >> 4 → PlayerSection
L0_SHIFT: int = 5  # block >> 5 → L0 WorldSection


# ── Block ↔ PlayerSection (16-block) ──────────────────────────────────────


def block_to_section(block: int) -> int:
    """Convert a block coordinate to its containing 16-block player section.

    Equivalent to ``block // 16`` for non-negative, ``block >> 4`` in general
    (arithmetic right-shift floors toward −∞, matching Java's ``>>``).
    """
    return block >> SECTION_SHIFT


def section_to_block_min(section: int) -> int:
    """Lowest block coordinate covered by this player section."""
    return section << SECTION_SHIFT


def section_to_block_max(section: int) -> int:
    """Highest block coordinate covered by this player section (inclusive)."""
    return (section << SECTION_SHIFT) + (BLOCKS_PER_SECTION - 1)


# ── Block ↔ WorldSection (level-aware) ────────────────────────────────────


def block_to_world_section(block: int, level: int) -> int:
    """Convert a block coordinate to its containing WorldSection at octree level L.

    Level 0 world sections are 32 blocks; each level doubles, so level L covers
    ``32 × 2^L`` blocks.

    Parameters
    ----------
    block : int
        Block coordinate (any axis).
    level : int
        Octree level (0–4).
    """
    return block >> (L0_SHIFT + level)


def world_section_to_block_min(ws: int, level: int) -> int:
    """Lowest block coordinate covered by this world section."""
    return ws << (L0_SHIFT + level)


def world_section_to_block_max(ws: int, level: int) -> int:
    """Highest block coordinate covered by this world section (inclusive)."""
    return ((ws + 1) << (L0_SHIFT + level)) - 1


def world_section_width(level: int) -> int:
    """Width in blocks of one world section at the given level."""
    return BLOCKS_PER_L0_WS << level  # 32 × 2^level


# ── PlayerSection ↔ WorldSection ──────────────────────────────────────────


def section_to_world_section(section: int, level: int) -> int:
    """Convert a player section (16-block) to the containing WorldSection at level L.

    Equivalent to ``block_to_world_section(section_to_block_min(section), level)``,
    but avoids the intermediate block step.  The shift is ``level + 1``
    because ``(block >> 4) >> (level + 1) = block >> (5 + level)``.
    """
    return section >> (level + 1)


def world_section_to_section_min(ws: int, level: int) -> int:
    """Lowest player-section coordinate covered by this WorldSection."""
    return ws << (level + 1)


def world_section_to_section_max(ws: int, level: int) -> int:
    """Highest player-section coordinate covered by this WorldSection (inclusive)."""
    return ((ws + 1) << (level + 1)) - 1


def sections_per_world_section(level: int) -> int:
    """Number of player-sections (16-block) per WorldSection axis at *level*."""
    return 1 << (level + 1)  # 2^(level+1)


# ── L0 WorldSection ↔ Voxy Section (16-block native sections) ─────────────


def l0_to_voxy_section_min(l0ws: int) -> int:
    """Lower of the two Voxy 16-block sections within an L0 WorldSection."""
    return l0ws << 1


def l0_to_voxy_section_max(l0ws: int) -> int:
    """Upper of the two Voxy 16-block sections within an L0 WorldSection."""
    return (l0ws << 1) + 1


def l0_to_voxy_section(l0ws: int, offset: int) -> int:
    """Voxy section for a sub-section within an L0 WorldSection.

    Parameters
    ----------
    l0ws : int
        L0 WorldSection coordinate.
    offset : int
        Sub-section offset (0 or 1).
    """
    return (l0ws << 1) + offset


# ── Voxy Section ↔ Block ──────────────────────────────────────────────────


def voxy_section_to_block_min(voxy_section: int) -> int:
    """Lowest block covered by this Voxy 16-block section."""
    return voxy_section << SECTION_SHIFT


def voxy_section_to_block_max(voxy_section: int) -> int:
    """Highest block covered by this Voxy 16-block section (inclusive)."""
    return (voxy_section << SECTION_SHIFT) + (BLOCKS_PER_SECTION - 1)


# ── Octree child expansion ────────────────────────────────────────────────


def child_x(parent_x: int, octant: int) -> int:
    """Child WorldSection X from parent X and octant (bit 0 = X)."""
    return (parent_x << 1) + (octant & 1)


def child_y(parent_y: int, octant: int) -> int:
    """Child WorldSection Y from parent Y and octant (bit 2 = Y)."""
    return (parent_y << 1) + ((octant >> 2) & 1)


def child_z(parent_z: int, octant: int) -> int:
    """Child WorldSection Z from parent Z and octant (bit 1 = Z)."""
    return (parent_z << 1) + ((octant >> 1) & 1)


def octant_index(lx: int, ly: int, lz: int) -> int:
    """Compute octant index (0–7) from local offsets within a parent cell.

    Parameters
    ----------
    lx, ly, lz : int
        Local offsets (0 or 1) within the parent.
    """
    return (lx & 1) | ((lz & 1) << 1) | ((ly & 1) << 2)


# ── Containment checks ────────────────────────────────────────────────────


def world_section_contains(ws: int, level: int, block: int) -> bool:
    """Does the world section at level L contain the given block coordinate?"""
    return world_section_to_block_min(ws, level) <= block <= world_section_to_block_max(ws, level)


def voxy_section_contains(voxy_section: int, block: int) -> bool:
    """Does the Voxy 16-block section contain the given block coordinate?"""
    return (
        voxy_section_to_block_min(voxy_section) <= block <= voxy_section_to_block_max(voxy_section)
    )


# ── Diagnostic / human-readable helpers ───────────────────────────────────


def describe(ws: int, level: int) -> str:
    """Describe a world section at level L in human-readable form.

    Example: ``"L4 ws=-1 blocks=[-512, -1] (512 wide)"``
    """
    return (
        f"L{level} ws={ws} "
        f"blocks=[{world_section_to_block_min(ws, level)}, "
        f"{world_section_to_block_max(ws, level)}] "
        f"({world_section_width(level)} wide)"
    )


def trace_block(block_x: int, block_y: int, block_z: int) -> str:
    """Trace a block position through the full L4 → L0 → Voxy hierarchy.

    Returns a multi-line string showing all intermediate coordinates.
    Useful for debugging coordinate mismatches.
    """
    lines = [
        f"Block ({block_x}, {block_y}, {block_z})",
        f"  Section: ({block_to_section(block_x)}, "
        f"{block_to_section(block_y)}, {block_to_section(block_z)})",
    ]
    for level in range(4, -1, -1):
        ws_x = block_to_world_section(block_x, level)
        ws_y = block_to_world_section(block_y, level)
        ws_z = block_to_world_section(block_z, level)
        lines.append(
            f"  L{level}: ({ws_x}, {ws_y}, {ws_z})  "
            f"X[{world_section_to_block_min(ws_x, level)}..{world_section_to_block_max(ws_x, level)}]  "
            f"Y[{world_section_to_block_min(ws_y, level)}..{world_section_to_block_max(ws_y, level)}]  "
            f"Z[{world_section_to_block_min(ws_z, level)}..{world_section_to_block_max(ws_z, level)}]"
        )
    l0_x = block_to_world_section(block_x, 0)
    l0_y = block_to_world_section(block_y, 0)
    l0_z = block_to_world_section(block_z, 0)
    lines.append(
        f"  Voxy: X{{{l0_to_voxy_section_min(l0_x)},{l0_to_voxy_section_max(l0_x)}}}  "
        f"Y{{{l0_to_voxy_section_min(l0_y)},{l0_to_voxy_section_max(l0_y)}}}  "
        f"Z{{{l0_to_voxy_section_min(l0_z)},{l0_to_voxy_section_max(l0_z)}}}"
    )
    return "\n".join(lines)
