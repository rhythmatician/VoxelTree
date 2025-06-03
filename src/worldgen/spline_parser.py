# spline_parser.py
# Parses Minecraft-style spline JSON configs into evaluable Python objects

import json
from pathlib import Path
from typing import Union

# ----------------------------
# Load JSON from 1.21.5 file
# ----------------------------


def load_noise_config(
    json_path: Union[str, Path] = "data/noise_settings/1.19.3+/overworld.json",
) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["noise"]["noise_router"]["final_density"]


# ----------------------------
# Spline Evaluator Classes
# ----------------------------


class Constant:
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, context: dict) -> float:
        return self.value


class Spline:
    def __init__(self, coordinate: str, points: list):
        self.coordinate = coordinate  # e.g. 'continentalness'
        self.points = points  # list of (location_expr, value_expr, optional_deriv)

    def evaluate(self, context: dict) -> float:
        x = context[self.coordinate]
        for i in range(len(self.points) - 1):
            (x0, y0, _), (x1, y1, _) = self.points[i], self.points[i + 1]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                return y0 * (1 - t) + y1 * t  # linear interp
        return self.points[-1][1]  # fallback to last value


# ----------------------------
# Recursive Parser
# ----------------------------


def parse_expr(expr: dict) -> Union[Spline, Constant]:
    t = expr["type"]

    if t == "minecraft:constant":
        return Constant(expr["value"])

    elif t == "minecraft:spline":
        coord = expr["coordinate"]
        points = []
        for point in expr["points"]:
            loc = point["location"]
            val = parse_expr(point["value"])
            der = point.get("derivative", None)
            points.append((loc, val, der))
        return Spline(coord, [(l, v.evaluate({coord: l}), d) for l, v, d in points])

    else:
        raise NotImplementedError(f"Unsupported type: {t}")


# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    expr = load_noise_config()
    spline = parse_expr(expr)
    print(
        spline.evaluate(
            {
                "temperature": 0.5,
                "humidity": 0.5,
                "continentalness": 0.3,
                "erosion": 0.4,
                "weirdness": 0.2,
                "depth": 0.0,
                "ridges": 0.0,
            }
        )
    )
