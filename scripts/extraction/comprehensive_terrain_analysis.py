#!/usr/bin/env python3
"""
Comprehensive Minecraft terrain block analysis.
Analyzes ALL blocks that can appear in vanilla world generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Set


def load_blocks() -> List[Dict]:
    """Load all Minecraft blocks from JSON."""
    blocks_file = Path("scripts/extraction/blocks/1.21.5.json")
    with open(blocks_file) as f:
        return json.load(f)


def analyze_terrain_blocks(blocks: List[Dict]) -> Dict[str, Set[str]]:
    """
    Comprehensive analysis of blocks that can appear in vanilla terrain.

    Returns dict of category -> set of block names that can generate.
    """
    categories: dict[str, set[str]] = {
        # Basic terrain
        "stone_variants": set(),
        "dirt_variants": set(),
        "sand_variants": set(),
        "gravel_variants": set(),
        # Ores (ALL of them)
        "ores": set(),
        # Wood (ALL variants)
        "logs_and_wood": set(),
        "planks": set(),
        "leaves": set(),
        "saplings": set(),
        # Water/Ice/Snow
        "water_ice_snow": set(),
        # Nether terrain
        "nether_terrain": set(),
        "nether_structures": set(),
        # End terrain
        "end_terrain": set(),
        # Ocean biomes
        "ocean_blocks": set(),
        "coral_and_sea_life": set(),
        # Cold biomes
        "ice_and_snow": set(),
        # Hot biomes (desert, mesa)
        "desert_blocks": set(),
        "mesa_blocks": set(),
        # Mushroom biome
        "mushroom_blocks": set(),
        # Underground
        "cave_blocks": set(),
        "underground_ores": set(),
        # Village generation
        "village_blocks": set(),
        "crop_blocks": set(),
        # Dungeon/structure generation
        "dungeon_blocks": set(),
        "temple_blocks": set(),
        "mansion_blocks": set(),
        "monument_blocks": set(),
        "fortress_blocks": set(),
        "city_blocks": set(),
        # Plants and flowers
        "flowers_and_plants": set(),
        "mushrooms": set(),
        # Redstone (can generate)
        "redstone_blocks": set(),
        # Misc terrain
        "misc_terrain": set(),
    }

    for block in blocks:
        name = block["name"]

        # Stone variants
        if any(
            word in name
            for word in [
                "stone",
                "granite",
                "diorite",
                "andesite",
                "deepslate",
                "tuff",
                "calcite",
                "cobblestone",
                "bedrock",
                "obsidian",
            ]
        ):
            categories["stone_variants"].add(name)

        # Dirt variants
        if any(
            word in name
            for word in ["dirt", "grass_block", "podzol", "mycelium", "coarse_dirt", "rooted_dirt"]
        ):
            categories["dirt_variants"].add(name)

        # Sand and gravel
        if "sand" in name and "sandstone" not in name:
            categories["sand_variants"].add(name)
        if "gravel" in name:
            categories["gravel_variants"].add(name)

        # ALL ores
        if "ore" in name:
            categories["ores"].add(name)
            categories["underground_ores"].add(name)

        # Wood variants (ALL 9+ wood types)
        wood_types = [
            "oak",
            "spruce",
            "birch",
            "jungle",
            "acacia",
            "dark_oak",
            "mangrove",
            "cherry",
            "bamboo",
            "warped",
            "crimson",
        ]
        for wood_type in wood_types:
            if wood_type in name:
                if "log" in name or "wood" in name:
                    categories["logs_and_wood"].add(name)
                elif "planks" in name:
                    categories["planks"].add(name)
                elif "leaves" in name:
                    categories["leaves"].add(name)
                elif "sapling" in name:
                    categories["saplings"].add(name)

        # Water, ice, snow
        if any(word in name for word in ["water", "ice", "snow", "powder_snow"]):
            if "water" in name:
                categories["water_ice_snow"].add(name)
            else:
                categories["ice_and_snow"].add(name)

        # Nether terrain and structures
        if any(
            word in name
            for word in [
                "nether",
                "soul",
                "warped",
                "crimson",
                "blackstone",
                "basalt",
                "magma",
                "glowstone",
                "quartz",
                "gold_block",
            ]
        ):
            if any(word in name for word in ["nether", "soul", "warped", "crimson"]):
                categories["nether_terrain"].add(name)
            if any(word in name for word in ["blackstone", "basalt", "quartz", "gold"]):
                categories["nether_structures"].add(name)

        # End terrain
        if any(word in name for word in ["end", "chorus", "purpur", "shulker"]):
            categories["end_terrain"].add(name)

        # Ocean blocks
        if any(
            word in name
            for word in ["prismarine", "sea", "kelp", "seagrass", "sponge", "turtle_egg"]
        ):
            categories["ocean_blocks"].add(name)

        # Coral and sea life
        if any(
            word in name
            for word in ["coral", "sea_pickle", "brain", "bubble", "fire", "horn", "tube"]
        ):
            categories["coral_and_sea_life"].add(name)

        # Desert blocks
        if any(word in name for word in ["sandstone", "cactus", "dead_bush"]):
            categories["desert_blocks"].add(name)

        # Mesa blocks (ALL terracotta)
        if any(word in name for word in ["terracotta", "glazed"]):
            categories["mesa_blocks"].add(name)

        # Mushroom biome
        if any(word in name for word in ["mushroom", "mycelium"]):
            categories["mushroom_blocks"].add(name)

        # Cave blocks
        if any(
            word in name
            for word in [
                "dripstone",
                "pointed_dripstone",
                "amethyst",
                "budding_amethyst",
                "moss",
                "azalea",
                "rooted_dirt",
                "hanging_roots",
            ]
        ):
            categories["cave_blocks"].add(name)

        # Village blocks (EXTENSIVE)
        if any(
            word in name
            for word in [
                "brick",
                "cobblestone",
                "planks",
                "log",
                "hay",
                "wool",
                "concrete",
                "glass",
                "iron_door",
                "iron_bars",
                "ladder",
                "torch",
                "lantern",
                "bell",
                "grindstone",
                "stonecutter",
                "smithing_table",
                "fletching_table",
                "cartography_table",
                "brewing_stand",
                "cauldron",
                "composter",
                "barrel",
            ]
        ):
            categories["village_blocks"].add(name)

        # Crop blocks
        if any(
            word in name
            for word in [
                "wheat",
                "carrots",
                "potatoes",
                "beetroots",
                "pumpkin",
                "melon",
                "sweet_berry",
                "glow_berries",
                "cocoa",
            ]
        ):
            categories["crop_blocks"].add(name)

        # Dungeon blocks
        if any(
            word in name
            for word in ["spawner", "chest", "mossy_cobblestone", "cracked_stone_bricks"]
        ):
            categories["dungeon_blocks"].add(name)

        # Temple blocks
        if any(
            word in name
            for word in [
                "sandstone",
                "chiseled_sandstone",
                "smooth_sandstone",
                "cut_sandstone",
                "jungle_log",
                "jungle_planks",
                "mossy_cobblestone",
                "dispenser",
                "tripwire_hook",
                "redstone_wire",
                "pressure_plate",
                "lever",
            ]
        ):
            categories["temple_blocks"].add(name)

        # Mansion blocks
        if any(
            word in name
            for word in ["dark_oak", "cobblestone", "wool", "carpet", "bookshelf", "loot_table"]
        ):
            categories["mansion_blocks"].add(name)

        # Monument blocks
        if any(word in name for word in ["prismarine", "sea_lantern", "sponge"]):
            categories["monument_blocks"].add(name)

        # Fortress blocks
        if any(
            word in name for word in ["nether_brick", "nether_wart", "blaze", "wither_skeleton"]
        ):
            categories["fortress_blocks"].add(name)

        # Ancient city blocks
        if any(word in name for word in ["deepslate", "sculk", "soul", "redstone_lamp", "candle"]):
            categories["city_blocks"].add(name)

        # Flowers and plants
        if any(
            word in name
            for word in [
                "flower",
                "tulip",
                "orchid",
                "allium",
                "houstonia",
                "poppy",
                "dandelion",
                "cornflower",
                "lily",
                "wither_rose",
                "sunflower",
                "lilac",
                "peony",
                "rose_bush",
                "grass",
                "fern",
                "vine",
                "sugar_cane",
            ]
        ):
            categories["flowers_and_plants"].add(name)

        # Mushrooms
        if any(word in name for word in ["mushroom", "fungus"]) and "mycelium" not in name:
            categories["mushrooms"].add(name)

        # Redstone (can naturally generate)
        if any(
            word in name
            for word in [
                "redstone_ore",
                "redstone_wire",
                "observer",
                "piston",
                "dispenser",
                "dropper",
                "hopper",
                "pressure_plate",
                "tripwire",
                "lever",
            ]
        ):
            categories["redstone_blocks"].add(name)

        # Misc terrain that doesn't fit elsewhere
        if any(
            word in name
            for word in ["clay", "bone_block", "honey_block", "honeycomb_block", "slime_block"]
        ):
            categories["misc_terrain"].add(name)

    return categories


def main():
    """Run comprehensive terrain analysis."""
    print("Loading Minecraft 1.21.5 blocks...")
    blocks = load_blocks()
    print(f"Total blocks in game: {len(blocks)}")

    print("\nAnalyzing terrain generation blocks...")
    categories = analyze_terrain_blocks(blocks)

    # Count unique blocks across all categories
    all_terrain_blocks = set()
    for category_blocks in categories.values():
        all_terrain_blocks.update(category_blocks)

    print("\n=== COMPREHENSIVE TERRAIN ANALYSIS ===")
    print(f"Total unique blocks that can generate in terrain: {len(all_terrain_blocks)}")
    print(f"Percentage of all blocks: {len(all_terrain_blocks)/len(blocks)*100:.1f}%")

    print("\n=== CATEGORY BREAKDOWN ===")
    total_by_category = 0
    for category, blocks_set in categories.items():
        count = len(blocks_set)
        total_by_category += count
        print(f"{category:25}: {count:3d} blocks")

    print(f"\nTotal blocks across categories: {total_by_category}")
    print("(Note: some blocks appear in multiple categories)")

    # Check if we exceed 1024 limit
    if len(all_terrain_blocks) > 1024:
        excess = len(all_terrain_blocks) - 1024
        print(f"\n*** CRITICAL: {excess} blocks OVER our 1024 limit! ***")
        print("We need to prioritize which blocks to include in training.")
    else:
        remaining = 1024 - len(all_terrain_blocks)
        print(f"\nGood news: {remaining} slots remaining in our 1024 vocabulary")

    # Show some examples from key categories
    print("\n=== EXAMPLES FROM KEY CATEGORIES ===")
    key_categories = [
        "ores",
        "mesa_blocks",
        "nether_terrain",
        "village_blocks",
        "coral_and_sea_life",
    ]
    for cat in key_categories:
        if cat in categories and categories[cat]:
            examples = sorted(list(categories[cat]))[:10]  # First 10
            print(f"{cat}: {examples[:5]}...")  # Show first 5

    return len(all_terrain_blocks)


if __name__ == "__main__":
    terrain_block_count = main()
