# extract_noise_settings.py
# Extracts all terrain noise_settings files from .jar files in .minecraft/versions/

import zipfile
from pathlib import Path
import shutil

# Input: official Minecraft version JARs
jar_dir = Path.home() / "AppData/Roaming/.minecraft/versions"
output_dir = Path("data/noise_settings")
output_dir.mkdir(parents=True, exist_ok=True)

# Prefix path to scan inside JARs
noise_dir_prefix = "data/minecraft/worldgen/noise_settings/"

for version_folder in jar_dir.iterdir():
    jar_file = version_folder / f"{version_folder.name}.jar"
    if not jar_file.exists():
        continue

    version = version_folder.name
    version_output_dir = output_dir / version
    version_output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(jar_file, "r") as jar:
        found_any = False
        for name in jar.namelist():
            if name.startswith(noise_dir_prefix) and name.endswith(".json"):
                found_any = True
                relative_name = name.split("noise_settings/")[-1]
                target_path = version_output_dir / relative_name
                with jar.open(name) as source, open(target_path, "wb") as dest:
                    shutil.copyfileobj(source, dest)
        if found_any:
            print(f"✔ Extracted: {version} -> {version_output_dir}/")
        else:
            print(f"⚠ No noise_settings folder in: {version}")
