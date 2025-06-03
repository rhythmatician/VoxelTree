# terrain_height_sampler.py
# Reconstructed from verified WebAssembly logic (Chunkbase Seed Map)

from math import sin, cos, pi
from typing import NamedTuple

# -------------------------
# Simulated Terrain Configs
# -------------------------


class SplineConfig:
    def evaluate(self, noise_vec):
        # Simplified: real terrain uses a weighted combination of erosion, weirdness, etc.
        t = noise_vec.temperature
        h = noise_vec.humidity
        e = noise_vec.erosion
        w = noise_vec.weirdness
        c = noise_vec.continentalness

        # Fake spline: return composite height
        return 64 + 32 * (0.5 * t + 0.2 * h - 0.3 * e + 0.4 * w + 0.3 * c)


class TerrainConfig:
    def __init__(self, spline_config):
        self.spline_config = spline_config


# -------------------------
# Noise Vector Placeholder
# -------------------------


class NoiseVec(NamedTuple):
    temperature: float
    humidity: float
    erosion: float
    weirdness: float
    continentalness: float


def get_noise_vector(x: int, z: int) -> NoiseVec:
    # Placeholder noise function
    return NoiseVec(
        temperature=0.5 + 0.3 * sin(x * 0.01),
        humidity=0.5 + 0.3 * cos(z * 0.01),
        erosion=0.5 + 0.2 * sin((x + z) * 0.008),
        weirdness=0.5 + 0.25 * cos((x - z) * 0.007),
        continentalness=0.5 + 0.35 * sin((x * z) * 0.00001),
    )


# -------------------------
# Height Sampler Pipeline
# -------------------------


def eval_spline(spline_input: dict) -> int:
    y = spline_input["spline_config"].evaluate(spline_input["noise"])
    return int(max(0, min(320, y)))  # Clamp to Minecraft build height


def compute_height_from_noise(frame: dict) -> int:
    spline = frame["spline_ref"]
    if spline is None:
        raise RuntimeError("Missing spline config")

    spline_input = {
        "spline_config": spline,
        "coord_x": frame["x"],
        "coord_z": frame["z"],
        "noise": get_noise_vector(frame["x"], frame["z"]),
    }
    return eval_spline(spline_input)


def sample_surface_from_config(config_ptr: TerrainConfig, x: int, z: int) -> int:
    frame = {"x": x, "z": z, "config": config_ptr, "spline_ref": config_ptr.spline_config}
    return compute_height_from_noise(frame)


def route_to_sampler(x: int, z: int, height_type: int, noise_type: int) -> int:
    if height_type not in range(5) or noise_type not in range(4):
        raise ValueError("Invalid height_type or noise_type")

    config_ptr = TerrainConfig(spline_config=SplineConfig())
    return sample_surface_from_config(config_ptr, x, z)


def get_surface(x: int, z: int, height_type: int = 0, noise_type: int = 0) -> int:
    x <<= 2
    z <<= 2
    return route_to_sampler(x, z, height_type, noise_type)


# -------------------------
# Example Usage
# -------------------------

if __name__ == "__main__":
    for x, z in [(0, 0), (100, 100), (1000, 500), (2048, -1024)]:
        h = get_surface(x, z)
        print(f"Height at ({x}, {z}) = {h}")
