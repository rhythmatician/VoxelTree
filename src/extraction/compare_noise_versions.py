# compare_noise_versions.py
# Compares each overworld.json file in data/noise_settings/*/ to the 1.21.5 reference

from pathlib import Path
import json
from deepdiff import DeepDiff

base_path = Path("data/noise_settings")
for filename in [
    "amplified.json",
    "end.json",
    "large_biomes.json",
    "overworld.json",
    "caves.json",
    "floating_islands.json",
    "nether.json",
]:
    reference_file = base_path / "1.21.5" / filename

    # Load reference JSON
    with open(reference_file, "r", encoding="utf-8") as f:
        reference = json.load(f)

    # Compare each other overworld.json file
    for version_dir in base_path.iterdir():
        if not version_dir.is_dir() or version_dir.name == "1.21.5":
            continue

        target_file = version_dir / filename
        if not target_file.exists():
            # print(f"⚠ No overworld.json in: {version_dir.name}")
            continue

        with open(target_file, "r", encoding="utf-8") as f:
            candidate = json.load(f)

        diff = DeepDiff(reference, candidate, ignore_order=True)

        if not diff:
            pass  # print("✅ No differences")
        else:
            print(f"\n===== {version_dir.name} =====")
            print("❌ Differences found:")
            print(diff.pretty())
