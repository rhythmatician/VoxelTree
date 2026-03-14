"""
Visualizes the progressive grokking network architecture:
  Stage 0 (green)  : 4 → 32 → 32 → [offset†, factor†, jag†]
  Stage 1 (purple) : [offset†, factor†, jag†] + 9 GLSL → 12 → 64 → 64 → [finalDensity†]
  Stage 2 (teal)   : [finalDensity†] + 7 GLSL → 8 → 128 → 128 → [block_logits K]

New inputs at each concat point are drawn COLINEAR (same x column) with the
checkpoint outputs they join, separated by a dashed divider.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── palette ────────────────────────────────────────────────────────────────────
S0_COLOR = "#3a8a3a"  # green  – Stage 0
S1_COLOR = "#6a3aaa"  # purple – Stage 1
S2_COLOR = "#1a7a7a"  # teal   – Stage 2
NEW_COLOR = "#b06000"  # amber  – injected inputs
CP_EDGE = "#ffd700"  # gold   – semantic checkpoint border
BG_COLOR = "#0e0e14"
WIRE_ALPHA = 0.18
WIRE_LW = 0.55
Y_SPAN = 9.0
NODE_R = 0.13

# ── layer spec ─────────────────────────────────────────────────────────────────
# Each entry: (true_size, display_count, node_color, column_label, is_checkpoint,
#              cp_node_labels,            # per-node text for checkpoint nodes (or None)
#              extra_labels, extra_color) # inline-concat nodes appended below (or None)
LAYERS = [
    # Stage 0
    (4, 4, S0_COLOR, "Inputs\n(C, E, RF, W)", False, ["C", "E", "RF", "W"], None, None),
    (32, 18, S0_COLOR, "Dense 4→32\nReLU", False, None, None, None),
    (32, 18, S0_COLOR, "Dense 32→32\nReLU", False, None, None, None),
    # Concat column: 3 checkpoint + 9 new = 12 total (Stage 0 → Stage 1 boundary)
    (
        3,
        3,
        S0_COLOR,
        "CONCAT\n(3 + 9 = 12)",
        True,
        ["offset†", "factor†", "jag†"],
        [
            "y",
            "surfDens",
            "slopedCheese",
            "entrances",
            "cheeseCaves",
            "spaghetti2D",
            "roughness",
            "noodleToggle",
            "noodleVal",
        ],
        NEW_COLOR,
    ),
    # Stage 1
    (64, 18, S1_COLOR, "Dense 12→64\nReLU", False, None, None, None),
    (64, 18, S1_COLOR, "Dense 64→64\nReLU", False, None, None, None),
    # Concat column: 1 checkpoint + 7 new = 8 total (Stage 1 → Stage 2 boundary)
    (
        1,
        1,
        S1_COLOR,
        "CONCAT\n(1 + 7 = 8)",
        True,
        ["finalDensity†"],
        ["temp", "veg", "continents", "erosion", "depth", "weirdness", "y"],
        NEW_COLOR,
    ),
    # Stage 2
    (128, 20, S2_COLOR, "Dense 8→128\nReLU", False, None, None, None),
    (128, 20, S2_COLOR, "Dense 128→128\nReLU", False, None, None, None),
    (20, 14, S2_COLOR, "block_logits\n(K classes)", False, None, None, None),
]

# x positions – tighten gaps slightly since concat columns are now self-contained
X_POS = [0, 2, 4, 6, 8.5, 10.5, 12.5, 15.0, 17.0, 19.0]

# ── helpers ────────────────────────────────────────────────────────────────────


def node_ys(n, y_span=Y_SPAN):
    if n == 1:
        return np.array([0.0])
    return np.linspace(-y_span / 2, y_span / 2, n)


def draw_plain_layer(ax, x, n_display, true_size, color, label, y_span=Y_SPAN):
    """Standard dense layer column. Returns ys array."""
    ys = node_ys(n_display, y_span)
    for y in ys:
        ax.add_patch(
            plt.Circle((x, y), NODE_R, color=color, ec=color, lw=0.8, zorder=4)
        )
    if true_size > n_display:
        ax.text(
            x,
            ys[-1] + 0.45,
            f"···\n({true_size})",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#aaaaaa",
            zorder=5,
        )
    ax.text(
        x,
        -y_span / 2 - 0.85,
        label,
        ha="center",
        va="top",
        fontsize=6.2,
        color="#dddddd",
        zorder=5,
        linespacing=1.4,
    )
    return ys


def draw_concat_column(
    ax, x, cp_labels, cp_color, extra_labels, extra_color, col_label, y_span=Y_SPAN
):
    """
    Draw a concat column: checkpoint nodes (top) + new-input nodes (bottom),
    all at the same x, colinear.  A dashed separator sits between the two groups.

    Returns (ys_cp, ys_all):
      ys_cp  – y-positions of the checkpoint nodes only  (wires FROM previous layer)
      ys_all – y-positions of ALL nodes                  (wires TO next layer)
    """
    n_cp = len(cp_labels)
    n_extra = len(extra_labels) if extra_labels else 0
    n_total = n_cp + n_extra

    # Distribute nodes evenly with a small gap between the two groups
    GAP = 0.5
    total_height = y_span
    cp_height = total_height * (n_cp / n_total) - GAP / 2
    extra_height = total_height * (n_extra / n_total) - GAP / 2

    top_center = y_span / 2 - cp_height / 2
    bottom_center = -y_span / 2 + extra_height / 2

    if n_cp == 1:
        ys_cp = np.array([top_center])
    else:
        ys_cp = np.linspace(
            top_center + cp_height / 2, top_center - cp_height / 2, n_cp
        )

    if n_extra == 0:
        ys_extra = np.array([])
    elif n_extra == 1:
        ys_extra = np.array([bottom_center])
    else:
        ys_extra = np.linspace(
            bottom_center + extra_height / 2, bottom_center - extra_height / 2, n_extra
        )

    ys_all = np.concatenate([ys_cp, ys_extra])

    # ── draw checkpoint nodes ──────────────────────────────────────────────
    for lbl, y in zip(cp_labels, ys_cp):
        ax.add_patch(
            plt.Circle((x, y), NODE_R, color=cp_color, ec=CP_EDGE, lw=2.2, zorder=4)
        )
        ax.text(
            x + NODE_R + 0.08,
            y,
            lbl,
            ha="left",
            va="center",
            fontsize=5.5,
            color=CP_EDGE,
            zorder=5,
        )

    # gold checkpoint box around cp nodes only
    cp_margin = 0.28
    box_y0 = ys_cp[-1] - cp_margin
    box_h = (ys_cp[0] - ys_cp[-1]) + 2 * cp_margin
    ax.add_patch(
        FancyBboxPatch(
            (x - cp_margin * 2, box_y0),
            cp_margin * 4 + 0.55,
            box_h,
            boxstyle="round,pad=0.05",
            linewidth=1.8,
            edgecolor=CP_EDGE,
            facecolor="none",
            zorder=3,
        )
    )
    ax.text(
        x,
        ys_cp[0] + cp_margin + 0.12,
        "SEMANTIC CHECKPOINT",
        ha="center",
        va="bottom",
        fontsize=4.5,
        color=CP_EDGE,
        zorder=5,
    )

    # ── dashed separator ──────────────────────────────────────────────────
    sep_y = (ys_cp[-1] + ys_extra[0]) / 2 if n_extra > 0 else ys_cp[-1] - 0.5
    ax.plot(
        [x - 0.55, x + 0.55],
        [sep_y, sep_y],
        color="#888888",
        lw=0.6,
        ls="--",
        zorder=3,
        alpha=0.6,
    )
    ax.text(
        x,
        sep_y,
        " CONCAT ",
        ha="center",
        va="center",
        fontsize=4.5,
        color="#aaaaaa",
        bbox=dict(facecolor=BG_COLOR, edgecolor="none", pad=1.5),
        zorder=4,
    )

    # ── draw extra (new-input) nodes ──────────────────────────────────────
    if extra_labels:
        for lbl, y in zip(extra_labels, ys_extra):
            ax.add_patch(
                plt.Circle(
                    (x, y), NODE_R, color=extra_color, ec="#ffaa44", lw=0.9, zorder=4
                )
            )
            ax.text(
                x + NODE_R + 0.08,
                y,
                lbl,
                ha="left",
                va="center",
                fontsize=5.5,
                color="#ffcc88",
                zorder=5,
            )

        # brace annotation for the amber group
        brace_x = x - 0.45
        ax.annotate(
            "",
            xy=(brace_x, ys_extra[-1]),
            xytext=(brace_x, ys_extra[0]),
            arrowprops=dict(
                arrowstyle="<->", color=NEW_COLOR, lw=0.8, mutation_scale=6
            ),
            zorder=3,
        )
        ax.text(
            brace_x - 0.12,
            (ys_extra[0] + ys_extra[-1]) / 2,
            f"+{n_extra}\nnew",
            ha="right",
            va="center",
            fontsize=5.0,
            color=NEW_COLOR,
            zorder=5,
            linespacing=1.3,
        )

    # ── column label below ────────────────────────────────────────────────
    ax.text(
        x,
        -y_span / 2 - 0.85,
        col_label,
        ha="center",
        va="top",
        fontsize=6.2,
        color="#dddddd",
        zorder=5,
        linespacing=1.4,
    )

    return ys_cp, ys_all


def draw_input_layer(ax, x, labels, color, col_label, y_span=Y_SPAN):
    """Input layer with per-node labels."""
    ys = node_ys(len(labels), y_span)
    for lbl, y in zip(labels, ys):
        ax.add_patch(
            plt.Circle((x, y), NODE_R, color=color, ec=color, lw=0.8, zorder=4)
        )
        ax.text(
            x - NODE_R - 0.08,
            y,
            lbl,
            ha="right",
            va="center",
            fontsize=6.0,
            color="#dddddd",
            zorder=5,
        )
    ax.text(
        x,
        -y_span / 2 - 0.85,
        col_label,
        ha="center",
        va="top",
        fontsize=6.2,
        color="#dddddd",
        zorder=5,
        linespacing=1.4,
    )
    return ys


def draw_wires(ax, x0, ys0, x1, ys1, color):
    rng = np.random.default_rng(42)
    pairs = [(a, b) for a in ys0 for b in ys1]
    if len(pairs) > 200:
        idx = rng.choice(len(pairs), 200, replace=False)
        pairs = [pairs[i] for i in idx]
    for a, b in pairs:
        ax.plot([x0, x1], [a, b], color=color, alpha=WIRE_ALPHA, lw=WIRE_LW, zorder=1)


# ── build figure ───────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(24, 11))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_aspect("equal")
ax.axis("off")

# ── stage background bands ─────────────────────────────────────────────────────
# Stage 0 spans columns 0-3 (inputs through first concat column)
# Stage 1 spans columns 4-6 (dense layers through second concat column)
# Stage 2 spans columns 7-9
stage_ranges = [
    (
        X_POS[0] - 0.6,
        X_POS[3] + 0.85,
        S0_COLOR,
        "Stage 0 — Terrain Shaper MLP\n(frozen after grokking)",
    ),
    (
        X_POS[4] - 0.6,
        X_POS[6] + 0.85,
        S1_COLOR,
        "Stage 1 — Cave + Density\n(frozen after grokking)",
    ),
    (
        X_POS[7] - 0.6,
        X_POS[9] + 0.6,
        S2_COLOR,
        "Stage 2 — Block Selection\n(fine-tuned end-to-end)",
    ),
]
for x0b, x1b, col, lbl in stage_ranges:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x0b, -Y_SPAN / 2 - 2.8),
            x1b - x0b,
            Y_SPAN + 3.6,
            boxstyle="round,pad=0.2",
            facecolor=col,
            alpha=0.08,
            edgecolor=col,
            lw=1.2,
            zorder=0,
        )
    )
    ax.text(
        (x0b + x1b) / 2,
        Y_SPAN / 2 + 1.1,
        lbl,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=col,
        alpha=0.9,
        zorder=2,
        linespacing=1.4,
    )

# ── draw columns and collect y-arrays ─────────────────────────────────────────
# col_ys[i] = the y-array to USE FOR WIRES OUTGOING from column i
# col_ys_in[i] = the y-array representing nodes RECEIVING wires (differs at concat cols)
col_ys = {}  # ys used as wire source / fan-out
col_ys_in = {}  # ys used as wire destination (= subset for plain layers)

for i, row in enumerate(LAYERS):
    true_sz, disp_sz, color, label, is_cp, cp_lbls, extra_lbls, extra_col = row
    x = X_POS[i]

    if is_cp:
        # Concat column
        ys_cp, ys_all = draw_concat_column(
            ax, x, cp_lbls, color, extra_lbls, extra_col, label
        )
        col_ys[i] = ys_all  # wires TO next layer use all nodes
        col_ys_in[i] = ys_cp  # wires FROM previous layer target only cp nodes
    elif cp_lbls is not None:
        # Input layer with per-node labels
        ys = draw_input_layer(ax, x, cp_lbls, color, label)
        col_ys[i] = ys
        col_ys_in[i] = ys
    else:
        # Plain dense layer
        ys = draw_plain_layer(ax, x, disp_sz, true_sz, color, label)
        col_ys[i] = ys
        col_ys_in[i] = ys

# ── draw wires ─────────────────────────────────────────────────────────────────
for i in range(len(LAYERS) - 1):
    c0 = LAYERS[i][2]
    c1 = LAYERS[i + 1][2]
    col = c0 if c0 == c1 else "#888888"
    src = col_ys[i]  # full fan-out from this column
    dst = col_ys_in[i + 1]  # only the receiving nodes of next column
    draw_wires(ax, X_POS[i], src, X_POS[i + 1], dst, col)

# ── legend ─────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=S0_COLOR, label="Stage 0 weights (frozen)"),
    mpatches.Patch(color=S1_COLOR, label="Stage 1 weights (frozen after S1)"),
    mpatches.Patch(color=S2_COLOR, label="Stage 2 weights"),
    mpatches.Patch(color=NEW_COLOR, label="New GLSL inputs injected at concat"),
    mpatches.Patch(
        facecolor="none",
        edgecolor=CP_EDGE,
        lw=2,
        label="Semantic checkpoint (interpretable)",
    ),
]
ax.legend(
    handles=legend_handles,
    loc="lower right",
    fontsize=7,
    framealpha=0.25,
    labelcolor="white",
    facecolor="#222",
    edgecolor="#555",
)

# ── param count annotations ────────────────────────────────────────────────────
for xi, txt, col in [
    (X_POS[3], "1,315 params\n~5 KB", S0_COLOR),
    (X_POS[6], "+5,057 params\n+20 KB", S1_COLOR),
    (X_POS[9], "+18,836 params\n+74 KB", S2_COLOR),
]:
    ax.text(
        xi,
        -Y_SPAN / 2 - 2.5,
        txt,
        ha="center",
        va="top",
        fontsize=6,
        color=col,
        alpha=0.8,
    )

ax.text(
    (X_POS[0] + X_POS[9]) / 2,
    -Y_SPAN / 2 - 3.2,
    "Full network: ~25,000 params  (~100 KB)",
    ha="center",
    va="top",
    fontsize=8,
    color="#cccccc",
    fontweight="bold",
)

ax.set_xlim(-1.5, X_POS[-1] + 1.8)
ax.set_ylim(-Y_SPAN / 2 - 4.0, Y_SPAN / 2 + 2.5)

plt.tight_layout(pad=0.3)
out = "architecture.png"
plt.savefig(out, dpi=180, facecolor=BG_COLOR)
print(f"Saved → {out}")
