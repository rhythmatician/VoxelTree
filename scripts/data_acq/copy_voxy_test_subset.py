"""Copy a small subset of extracted Voxy octree data into the test data directory.

This makes unit tests deterministic and independent of the full extracted dataset.

It picks one L4 parent section (0,0,0) and all available L3 children, plus any
L2 children needed to compute non_empty_children.
"""

from pathlib import Path

from VoxelTree.scripts.build_octree_pairs import child_coords_from_parent


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        fdst.write(fsrc.read())


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "data" / "voxy_octree"
    dst_root = repo_root / "tests" / "data" / "voxy_octree"

    parent_coords = (0, 0, 0)

    # L4 parent section
    l4_src = src_root / "level_4" / f"voxy_L4_x{parent_coords[0]}_y{parent_coords[1]}_z{parent_coords[2]}.npz"
    l4_dst = dst_root / "level_4" / l4_src.name
    _copy_file(l4_src, l4_dst)

    # L3 children
    for octant in range(8):
        l3_coords = child_coords_from_parent(*parent_coords, octant)
        l3_src = src_root / "level_3" / f"voxy_L3_x{l3_coords[0]}_y{l3_coords[1]}_z{l3_coords[2]}.npz"
        if not l3_src.exists():
            continue
        l3_dst = dst_root / "level_3" / l3_src.name
        _copy_file(l3_src, l3_dst)

        # Copy any L2 children referenced by this L3 (for non_empty_children validation)
        for l2_oct in range(8):
            l2_coords = child_coords_from_parent(*l3_coords, l2_oct)
            l2_src = src_root / "level_2" / f"voxy_L2_x{l2_coords[0]}_y{l2_coords[1]}_z{l2_coords[2]}.npz"
            if not l2_src.exists():
                continue
            l2_dst = dst_root / "level_2" / l2_src.name
            _copy_file(l2_src, l2_dst)

    print("Copied Voxy subset into", dst_root)


if __name__ == "__main__":
    main()
