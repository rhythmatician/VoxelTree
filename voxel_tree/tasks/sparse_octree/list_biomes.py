import os
import json
from collections import Counter
import sys

# Add VoxelTree/scripts to sys.path for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))
try:
    from biome_mapping import BIOME_ID_TO_NAME
except ImportError:
    print("Could not import BIOME_ID_TO_NAME from biome_mapping.py")
    sys.exit(1)

# Path to the sparse_octree_dumps directory
DUMPS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../../tools/fabric-server/runtime/sparse_octree_dumps"
    )  # Dumps dir location
)

biome_counter = Counter()

for fname in os.listdir(DUMPS_DIR):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(DUMPS_DIR, fname)
    try:
        with open(fpath, "r") as f:
            data = json.load(f)
            ids = data.get("biome_ids")
            if ids:
                biome_counter.update(ids)
    except Exception as e:
        print(f"Error reading {fname}: {e}")

print("Biome IDs found:")
for bid, count in sorted(biome_counter.items()):
    name = BIOME_ID_TO_NAME.get(bid, f"unknown({bid})")
    print(f"{bid:3}: {name:30}  count={count}")

print("\nTotal unique biomes:", len(biome_counter))
