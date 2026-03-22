#!/usr/bin/env python3
"""
TerrainShaperSpline MLP Training

Trains a small 4->128->128->3 MLP to approximate Minecraft's nested cubic splines
that map (continents, erosion, ridges, weirdness) -> (offset, factor, jaggedness).

Correctly ports TerrainProvider.java + CubicSpline.java logic to Python:
  - Nested CubicSpline (values at control points can be other CubicSplines)
  - Minecraft's exact Hermite formula:
      lerp(t, y1, y2) + t*(1-t)*lerp(t, a, b)
      where a = d1*dx - (y2-y1),  b = -d2*dx + (y2-y1)
  - Linear extrapolation outside the known input range

Input layout: [continents, erosion, ridges, weirdness]  (all in roughly [-1, 1])
"""

import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json

# ============================================================================
# Minecraft CubicSpline — exact port of CubicSpline.java (Multipoint + Constant)
# ============================================================================


class _ConstantSpline:
    """CubicSpline.Constant — always returns the same float."""

    def __init__(self, value: float):
        self.value = float(value)

    def apply(self, c):  # c is a 4-float list/array
        return self.value

    def min_value(self):
        return self.value

    def max_value(self):
        return self.value


class _MultipointSpline:
    """CubicSpline.Multipoint — Hermite spline with possibly-spline values.

    Evaluation exactly matches CubicSpline.Multipoint.apply() from Minecraft:
        t = (input - x1) / (x2 - x1)
        a = d1*(x2-x1) - (y2-y1)
        b = -d2*(x2-x1) + (y2-y1)
        result = lerp(t, y1, y2) + t*(1-t)*lerp(t, a, b)
    """

    def __init__(self, coordinate_fn, locations, values, derivatives):
        # coordinate_fn : (4-float array) -> float
        # values        : list of _ConstantSpline | _MultipointSpline
        assert len(locations) == len(values) == len(derivatives)
        self.coordinate_fn = coordinate_fn
        self.locations = list(locations)
        self.values = list(values)
        self.derivatives = list(derivatives)

    @staticmethod
    def _linear_extend(t, locations, edge_value, derivatives, index):
        """Extrapolate linearly beyond the spline's boundary."""
        d = derivatives[index]
        if d == 0.0:
            return edge_value
        return edge_value + d * (t - locations[index])

    def apply(self, c):
        t = self.coordinate_fn(c)
        locs = self.locations
        n = len(locs)
        last = n - 1

        # Binary search for interval start
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) >> 1
            if t < locs[mid]:
                hi = mid
            else:
                lo = mid + 1
        start = lo - 1  # matches findIntervalStart in Java

        if start < 0:
            # Below first knot — linear extrapolation
            return self._linear_extend(t, locs, self.values[0].apply(c), self.derivatives, 0)
        if start >= last:
            # Above last knot — linear extrapolation
            return self._linear_extend(t, locs, self.values[last].apply(c), self.derivatives, last)

        x1 = locs[start]
        x2 = locs[start + 1]
        d1 = self.derivatives[start]
        d2 = self.derivatives[start + 1]
        y1 = self.values[start].apply(c)
        y2 = self.values[start + 1].apply(c)

        u = (t - x1) / (x2 - x1)
        dx = x2 - x1
        a = d1 * dx - (y2 - y1)
        b = -d2 * dx + (y2 - y1)
        # Minecraft's cubic Hermite formula (equivalent to standard, but this exact form)
        return (1.0 - u) * y1 + u * y2 + u * (1.0 - u) * ((1.0 - u) * a + u * b)


class SplineBuilder:
    """Port of CubicSpline.Builder — fluent API for building Minecraft-style splines."""

    def __init__(self, coordinate_fn, value_transformer=None):
        self.coordinate_fn = coordinate_fn
        # value_transformer: float -> float applied to FLOAT values only, not to sub-splines
        self.value_transformer = value_transformer or (lambda x: x)
        self._locations = []
        self._values = []  # list of _ConstantSpline | _MultipointSpline
        self._derivatives = []

    def add_point(self, location: float, value, derivative: float = 0.0):
        """Add a control point.  value may be a float or another spline object."""
        if self._locations and location <= self._locations[-1]:
            raise ValueError(
                f"Locations must be strictly ascending; got {location} after {self._locations[-1]}"
            )
        self._locations.append(float(location))
        if isinstance(value, (_ConstantSpline, _MultipointSpline)):
            self._values.append(value)
        else:
            self._values.append(_ConstantSpline(self.value_transformer(float(value))))
        self._derivatives.append(float(derivative))
        return self

    def build(self):
        if not self._locations:
            raise ValueError("No points added")
        if len(self._locations) == 1:
            return self._values[0]  # degenerate single-point spline
        return _MultipointSpline(
            self.coordinate_fn, self._locations, self._values, self._derivatives
        )


# ============================================================================
# Coordinate extractors (C = float[4], extractors pick one element)
# ============================================================================


def _continents(c):
    return c[0]


def _erosion(c):
    return c[1]


def _ridges(c):
    return c[2]


def _weirdness(c):
    return c[3]


def _no_transform(x):
    return x


def _amp_offset(x):
    return x if x < 0.0 else x * 2.0


def _amp_factor(x):
    return 1.25 - 6.25 / (x + 5.0)


def _amp_jaggedness(x):
    return x * 2.0


def _peaks_and_valleys(x):
    # NoiseRouterData.peaksAndValleys(float):
    # -(abs(abs(x) - 0.6666667) - 0.33333334) * 3.0f
    return -(abs(abs(x) - 0.6666667) - 0.33333334) * 3.0


# ============================================================================
# TerrainProvider — exact port of TerrainProvider.java
# ============================================================================


def _build_weirdness_jaggedness_spline(jaggedness_factor, jaggedness_transformer):
    max_neg = 0.63 * jaggedness_factor
    max_pos = 0.3 * jaggedness_factor
    return (
        SplineBuilder(_weirdness, jaggedness_transformer)
        .add_point(-0.01, max_neg)
        .add_point(0.01, max_pos)
        .build()
    )


def _build_ridge_jaggedness_spline(
    jaggedness_factor_at_peak, jaggedness_factor_at_high, jaggedness_transformer
):
    high_start = _peaks_and_valleys(0.4)
    high_end = _peaks_and_valleys(0.56666666)
    high_middle = (high_start + high_end) / 2.0

    b = SplineBuilder(_ridges, jaggedness_transformer)
    b.add_point(high_start, 0.0)

    if jaggedness_factor_at_high > 0.0:
        sub = _build_weirdness_jaggedness_spline(jaggedness_factor_at_high, jaggedness_transformer)
        b.add_point(high_middle, sub)
    else:
        b.add_point(high_middle, 0.0)

    if jaggedness_factor_at_peak > 0.0:
        sub = _build_weirdness_jaggedness_spline(jaggedness_factor_at_peak, jaggedness_transformer)
        b.add_point(1.0, sub)
    else:
        b.add_point(1.0, 0.0)

    return b.build()


def _build_erosion_jaggedness_spline(
    jf_peak_e0, jf_peak_e1, jf_high_e0, jf_high_e1, jaggedness_transformer
):
    ridge0 = _build_ridge_jaggedness_spline(jf_peak_e0, jf_high_e0, jaggedness_transformer)
    ridge1 = _build_ridge_jaggedness_spline(jf_peak_e1, jf_high_e1, jaggedness_transformer)
    return (
        SplineBuilder(_erosion, jaggedness_transformer)
        .add_point(-1.0, ridge0)
        .add_point(-0.78, ridge1)
        .add_point(-0.5775, ridge1)
        .add_point(-0.375, 0.0)
        .build()
    )


def overworld_jaggedness(amplified=False):
    """Returns a CubicSpline implementing TerrainProvider.overworldJaggedness()."""
    jt = _amp_jaggedness if amplified else _no_transform
    return (
        SplineBuilder(_continents, jt)
        .add_point(-0.11, 0.0)
        .add_point(0.03, _build_erosion_jaggedness_spline(1.0, 0.5, 0.0, 0.0, jt))
        .add_point(0.65, _build_erosion_jaggedness_spline(1.0, 1.0, 1.0, 0.0, jt))
        .build()
    )


def _get_erosion_factor(base_value, shattered, factor_transformer):
    base_spline = (
        SplineBuilder(_weirdness, factor_transformer)
        .add_point(-0.2, 6.3)
        .add_point(0.2, base_value)
        .build()
    )
    b = (
        SplineBuilder(_erosion, factor_transformer)
        .add_point(-0.6, base_spline)
        .add_point(
            -0.5,
            SplineBuilder(_weirdness, factor_transformer)
            .add_point(-0.05, 6.3)
            .add_point(0.05, 2.67)
            .build(),
        )
        .add_point(-0.35, base_spline)
        .add_point(-0.25, base_spline)
        .add_point(
            -0.1,
            SplineBuilder(_weirdness, factor_transformer)
            .add_point(-0.05, 2.67)
            .add_point(0.05, 6.3)
            .build(),
        )
        .add_point(0.03, base_spline)
    )
    if shattered:
        weirdness_shattered = (
            SplineBuilder(_weirdness, factor_transformer)
            .add_point(0.0, base_value)
            .add_point(0.1, 0.625)
            .build()
        )
        ridges_shattered = (
            SplineBuilder(_ridges, factor_transformer)
            .add_point(-0.9, base_value)
            .add_point(-0.69, weirdness_shattered)
            .build()
        )
        (
            b.add_point(0.35, base_value)
            .add_point(0.45, ridges_shattered)
            .add_point(0.55, ridges_shattered)
            .add_point(0.62, base_value)
        )
    else:
        extreme_hills = (
            SplineBuilder(_ridges, factor_transformer)
            .add_point(-0.7, base_spline)
            .add_point(-0.15, 1.37)
            .build()
        )
        peaks_only = (
            SplineBuilder(_ridges, factor_transformer)
            .add_point(0.45, base_spline)
            .add_point(0.7, 1.56)
            .build()
        )
        (
            b.add_point(0.05, peaks_only)
            .add_point(0.40, peaks_only)
            .add_point(0.45, extreme_hills)
            .add_point(0.55, extreme_hills)
            .add_point(0.58, base_value)
        )
    return b.build()


def overworld_factor(amplified=False):
    """Returns a CubicSpline implementing TerrainProvider.overworldFactor()."""
    ft = _amp_factor if amplified else _no_transform
    nt = _no_transform
    return (
        SplineBuilder(_continents, nt)
        .add_point(-0.19, 3.95)
        .add_point(-0.15, _get_erosion_factor(6.25, True, nt))
        .add_point(-0.1, _get_erosion_factor(5.47, True, ft))
        .add_point(0.03, _get_erosion_factor(5.08, True, ft))
        .add_point(0.06, _get_erosion_factor(4.69, False, ft))
        .build()
    )


def _mountain_continentalness(ridge, modulation, allow_rivers_below):
    ridge_offset = 1.17
    ridge_amplitude = 0.46082947
    ridge_slope = 1.0 - (1.0 - modulation) * 0.5
    ridge_intersect = 0.5 * (1.0 - modulation)
    adjusted = (ridge + ridge_offset) * ridge_amplitude
    cont = adjusted * ridge_slope - ridge_intersect
    if ridge < allow_rivers_below:
        return max(cont, -0.2222)
    return max(cont, 0.0)


def _mountain_ridge_zero_point(modulation):
    # calculateMountainRidgeZeroContinentalnessPoint
    ridge_amplitude = 0.46082947
    ridge_slope = 1.0 - (1.0 - modulation) * 0.5
    ridge_intersect = 0.5 * (1.0 - modulation)
    return ridge_intersect / (ridge_amplitude * ridge_slope) - 1.17


def _lerp(t, a, b):
    return a + t * (b - a)


def _build_mountain_ridge_spline(modulation, saddle, offset_transformer):
    """Port of buildMountainRidgeSplineWithPoints."""
    ALLOW_RIVERS = -0.7
    min_c = _mountain_continentalness(-1.0, modulation, ALLOW_RIVERS)
    max_c = _mountain_continentalness(1.0, modulation, ALLOW_RIVERS)
    zero_r = _mountain_ridge_zero_point(modulation)

    b = SplineBuilder(_ridges, offset_transformer)

    if -0.65 < zero_r < 1.0:
        after_river_c = _mountain_continentalness(-0.65, modulation, ALLOW_RIVERS)
        before_river_c = _mountain_continentalness(-0.75, modulation, ALLOW_RIVERS)
        d_min = (before_river_c - min_c) / (-0.75 - (-1.0))
        b.add_point(-1.0, min_c, d_min)
        b.add_point(-0.75, before_river_c)
        b.add_point(-0.65, after_river_c)
        zero_c = _mountain_continentalness(zero_r, modulation, ALLOW_RIVERS)
        d_max = (max_c - zero_c) / (1.0 - zero_r)
        b.add_point(zero_r - 0.01, zero_c)
        b.add_point(zero_r, zero_c, d_max)
        b.add_point(1.0, max_c, d_max)
    else:
        simple_d = (max_c - min_c) / 2.0  # over range [-1, 1]
        if saddle:
            b.add_point(-1.0, max(0.2, min_c))
            b.add_point(0.0, _lerp(0.5, min_c, max_c), simple_d)
        else:
            b.add_point(-1.0, min_c, simple_d)
        b.add_point(1.0, max_c, simple_d)

    return b.build()


def _ridge_spline(valley, low, mid, high, peaks, min_valley_steepness, offset_transformer):
    """Port of ridgeSpline()."""
    d1 = max(0.5 * (low - valley), min_valley_steepness)
    d2 = 5.0 * (mid - low)
    return (
        SplineBuilder(_ridges, offset_transformer)
        .add_point(-1.0, valley, d1)
        .add_point(-0.4, low, min(d1, d2))
        .add_point(0.0, mid, d2)
        .add_point(0.4, high, 2.0 * (high - mid))
        .add_point(1.0, peaks, 0.7 * (peaks - high))
        .build()
    )


def _build_erosion_offset_spline(
    low_valley,
    hill,
    tall_hill,
    mountain_factor,
    plain,
    swamp,
    include_extreme_hills,
    saddle,
    offset_transformer,
):
    """Port of buildErosionOffsetSpline()."""
    mf = mountain_factor
    very_low_mountains = _build_mountain_ridge_spline(
        _lerp(mf, 0.6, 1.5), saddle, offset_transformer
    )
    low_mountains = _build_mountain_ridge_spline(_lerp(mf, 0.6, 1.0), saddle, offset_transformer)
    mountains = _build_mountain_ridge_spline(mf, saddle, offset_transformer)

    wide_plateau = _ridge_spline(
        low_valley - 0.15,
        0.5 * mf,
        _lerp(0.5, 0.5, 0.5) * mf,
        0.5 * mf,
        0.6 * mf,
        0.5,
        offset_transformer,
    )
    narrow_plateau = _ridge_spline(
        low_valley, plain * mf, hill * mf, 0.5 * mf, 0.6 * mf, 0.5, offset_transformer
    )
    plains = _ridge_spline(low_valley, plain, plain, hill, tall_hill, 0.5, offset_transformer)
    plains_far_in = _ridge_spline(
        low_valley, plain, plain, hill, tall_hill, 0.5, offset_transformer
    )

    extreme_hills = (
        SplineBuilder(_ridges, offset_transformer)
        .add_point(-1.0, low_valley)
        .add_point(-0.4, plains)
        .add_point(0.0, tall_hill + 0.07)
        .build()
    )
    swamps = _ridge_spline(-0.02, swamp, swamp, hill, tall_hill, 0.0, offset_transformer)

    b = (
        SplineBuilder(_erosion, offset_transformer)
        .add_point(-0.85, very_low_mountains)
        .add_point(-0.70, low_mountains)
        .add_point(-0.40, mountains)
        .add_point(-0.35, wide_plateau)
        .add_point(-0.10, narrow_plateau)
        .add_point(0.20, plains)
    )

    if include_extreme_hills:
        (
            b.add_point(0.40, plains_far_in)
            .add_point(0.45, extreme_hills)
            .add_point(0.55, extreme_hills)
            .add_point(0.58, plains_far_in)
        )

    b.add_point(0.70, swamps)
    return b.build()


def overworld_offset(amplified=False):
    """Returns a CubicSpline implementing TerrainProvider.overworldOffset()."""
    ot = _amp_offset if amplified else _no_transform
    beach = _build_erosion_offset_spline(-0.15, 0.0, 0.0, 0.1, 0.0, -0.03, False, False, ot)
    low = _build_erosion_offset_spline(-0.10, 0.03, 0.1, 0.1, 0.01, -0.03, False, False, ot)
    mid = _build_erosion_offset_spline(-0.10, 0.03, 0.1, 0.7, 0.01, -0.03, True, True, ot)
    high = _build_erosion_offset_spline(-0.05, 0.03, 0.1, 1.0, 0.01, 0.01, True, True, ot)
    return (
        SplineBuilder(_continents, ot)
        .add_point(-1.10, 0.044)
        .add_point(-1.02, -0.2222)
        .add_point(-0.51, -0.2222)
        .add_point(-0.44, -0.12)
        .add_point(-0.18, -0.12)
        .add_point(-0.16, beach)
        .add_point(-0.15, beach)
        .add_point(-0.10, low)
        .add_point(0.25, mid)
        .add_point(1.00, high)
        .build()
    )


def evaluate_terrain_shaper(
    continents, erosion, ridges, weirdness, amplified=False, _cache=[None, None, None]
):
    """Evaluate all three terrain shaper outputs for a single 4-float input.

    Returns (offset, factor, jaggedness).

    Spline objects are cached on first call (building them is ~5ms, evaluating is O(1)).
    """
    if _cache[0] is None:
        _cache[0] = overworld_offset(amplified)
        _cache[1] = overworld_factor(amplified)
        _cache[2] = overworld_jaggedness(amplified)

    c = (continents, erosion, ridges, weirdness)
    return (_cache[0].apply(c), _cache[1].apply(c), _cache[2].apply(c))


# ============================================================================
# MLP Model  (wider than before — real function uses all 4 inputs)
# ============================================================================


class TerrainShaperMLP(nn.Module):
    """4-input, 3-output MLP for terrain shaper splines."""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Training
# ============================================================================


def generate_training_data(num_samples=500_000):
    """
    Generate training data using the correctly ported TerrainProvider splines.
    """
    print(f"Generating {num_samples:,} training samples using real TerrainProvider splines...")

    # Build splines once
    offset_spline = overworld_offset(amplified=False)
    factor_spline = overworld_factor(amplified=False)
    jaggedness_spline = overworld_jaggedness(amplified=False)

    rng = np.random.default_rng(42)
    inputs = rng.uniform(-1.0, 1.0, size=(num_samples, 4)).astype(np.float32)
    outputs = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,}/{num_samples:,}")
        c = inputs[i]  # [continents, erosion, ridges, weirdness]
        outputs[i, 0] = offset_spline.apply(c)
        outputs[i, 1] = factor_spline.apply(c)
        outputs[i, 2] = jaggedness_spline.apply(c)

    print("\nOutput ranges:")
    for j, name in enumerate(("offset", "factor", "jaggedness")):
        print(
            f"  {name}: [{outputs[:, j].min():.4f}, {outputs[:, j].max():.4f}]  "
            f"mean={outputs[:, j].mean():.4f}  std={outputs[:, j].std():.4f}"
        )

    return torch.from_numpy(inputs), torch.from_numpy(outputs)


def _weight_l2_norm(model: nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += p.pow(2).sum().item()
    return total**0.5


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[3] / "runs" / "terrain_shaper_checkpoint.pt"
)  # VoxelTree/runs/


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    epochs_without_improvement,
    loss_history,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "loss_history": loss_history,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return (
        ckpt["epoch"],
        ckpt["best_val_loss"],
        ckpt["epochs_without_improvement"],
        ckpt["loss_history"],
    )


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=5000,
    lr=1e-3,
    early_stopping_patience=300,
    target_loss=1e-6,
    weight_decay=1e-5,
    checkpoint_path=None,
    checkpoint_interval=50,
):
    """Train the MLP against the real TerrainProvider outputs."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-8
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    loss_history = []
    start_epoch = 0

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        start_epoch, best_val_loss, epochs_without_improvement, loss_history = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        start_epoch += 1  # resume from next epoch
        print(f"Resumed from checkpoint at epoch {start_epoch} (best val_loss={best_val_loss:.2e})")

    for epoch in range(start_epoch, epochs):
        # Training
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                val_loss += loss_fn(pred, batch_y).item()
        val_loss /= len(val_loader)

        weight_norm = _weight_l2_norm(model)
        loss_history.append((train_loss, val_loss))
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            pct_str = ""
            if len(loss_history) > 10:
                prev = loss_history[max(0, epoch - 10)][1]
                pct = (prev - val_loss) / prev * 100
                pct_str = f" | pct_improve={pct:.4f}%"
            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.5e}, val_loss={val_loss:.5e}  "
                f"w_norm={weight_norm:.3f}{pct_str}"
            )

        if val_loss < target_loss:
            print(f"\n[DONE] Target loss {target_loss:.1e} reached at epoch {epoch+1}!")
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    epochs_without_improvement,
                    loss_history,
                )
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    epochs_without_improvement,
                    loss_history,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best val_loss: {best_val_loss:.2e})")
                break

        if checkpoint_path is not None and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                epochs_without_improvement,
                loss_history,
            )

    return model


def export_onnx(model, output_path):
    """Export the trained model as ONNX."""
    dummy_input = torch.randn(1, 4, dtype=torch.float32)

    # Ensure human-readable prints don't fail on Windows consoles with limited encodings.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

    # Export should be done in eval mode to avoid training-time behavior (dropout, etc.).
    model.eval()

    def _try_export(obj):
        torch.onnx.export(
            obj,
            (dummy_input,),
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=False,
            export_params=True,
        )

    try:
        # Prefer exporting the raw nn.Module rather than tracing to a ScriptModule.
        _try_export(model)
        print(f"ONNX model exported to {output_path}")
        return
    except Exception as e1:
        # PyTorch 2.x may require converting ScriptModule to ExportedProgram
        if "Exporting a ScriptModule is not supported" in str(e1):
            try:
                from torch.export import TS2EPConverter  # type: ignore[attr-defined]

                ep = TS2EPConverter(model, (dummy_input,)).convert()
                _try_export(ep)
                print(f"ONNX model (via ExportedProgram) exported to {output_path}")
                return
            except Exception as e2:
                print(f"ONNX export via ExportedProgram also failed: {e2}")

        print(f"ONNX export failed: {e1}")

    checkpoint_path = output_path.replace(".onnx", ".pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "TerrainShaperMLP",
            "hidden_size": 128,
        },
        checkpoint_path,
    )
    print(f"PyTorch checkpoint saved to {checkpoint_path}")


class _Tee:
    def __init__(self, log_path: Path):
        self._stdout = sys.stdout
        self._log = log_path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = self

    def write(self, data):
        try:
            self._stdout.write(data)
        except UnicodeEncodeError:
            self._stdout.write(
                data.encode(self._stdout.encoding, errors="replace").decode(self._stdout.encoding)
            )
        self._log.write(data)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout
        self._log.close()


def main():
    log_dir = Path(__file__).resolve().parents[3] / "logs"  # VoxelTree/logs/
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_terrain_shaper_{timestamp}.log"
    tee = _Tee(log_path)
    print(f"Logging to: {log_path}\n")
    try:
        _main()
    finally:
        tee.close()


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and start over",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and export ONNX from the existing checkpoint",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to write the exported ONNX model (default is the standard output path)",
    )
    args, _ = parser.parse_known_args()

    checkpoint_path = CHECKPOINT_PATH
    if args.fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Deleted existing checkpoint, starting fresh.")

    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TerrainShaperMLP(hidden_size=128).to(device)

    # If the user only wants to export the ONNX model, load weights from checkpoint
    # and skip the expensive training procedure.
    if args.export_only:
        if not checkpoint_path.exists():
            raise SystemExit(
                f"No checkpoint found at {checkpoint_path}; cannot export without training."
            )
        print(f"Loading checkpoint from {checkpoint_path} for ONNX export...")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        output_dir = Path(__file__).parent / "model"
        output_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = Path(args.onnx_path) if args.onnx_path else output_dir / "terrain_shaper.onnx"
        model.cpu()
        export_onnx(model, str(onnx_path))
        return

    inputs, outputs = generate_training_data(num_samples=500_000)

    # Shuffle
    idx = torch.randperm(len(inputs))
    inputs, outputs = inputs[idx], outputs[idx]

    split = int(0.9 * len(inputs))
    train_inputs, val_inputs = inputs[:split], inputs[split:]
    train_outputs, val_outputs = outputs[:split], outputs[split:]

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_outputs), batch_size=4096, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(val_inputs, val_outputs), batch_size=4096, shuffle=False)

    train_loader = [(bx.to(device), by.to(device)) for bx, by in train_loader]
    val_loader = [(bx.to(device), by.to(device)) for bx, by in val_loader]

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training against REAL TerrainProvider splines (correct nested CubicSpline port)")
    print("Target val_loss < 1e-6\n")

    if checkpoint_path.exists():
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found, starting from epoch 1.")

    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=10_000,
        lr=1e-3,
        early_stopping_patience=300,
        target_loss=1e-6,
        weight_decay=1e-5,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=50,
    )

    output_dir = Path(__file__).parent / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.cpu()
    export_onnx(model, str(output_dir / "terrain_shaper.onnx"))

    metadata = {
        "model": "TerrainShaperMLP",
        "inputs": ["continents", "erosion", "ridges", "weirdness"],
        "outputs": ["offset", "factor", "jaggedness"],
        "input_ranges": {
            "continents": [-1.1, 1.0],
            "erosion": [-1.0, 1.0],
            "ridges": [-1.0, 1.0],
            "weirdness": [-1.0, 1.0],
        },
        "output_ranges": {"offset": [-0.5, 1.0], "factor": [0.625, 6.3], "jaggedness": [0.0, 2.0]},
        "hidden_size": 128,
        "layers": "4->128->128->64->3",
        "training_samples": 500_000,
        "notes": "Trained on correct TerrainProvider CubicSpline port (v2)",
    }
    metadata_path = output_dir / "terrain_shaper.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
