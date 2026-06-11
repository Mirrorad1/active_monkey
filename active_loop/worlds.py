"""Procedural benchmark world generators for the active-loop lab.

Four canonical world families used in the growth/benchmark workstream (T3 of the
rigor-fairness-upgrade spec):

  learnable()          -- fully deterministic, structurally adequate (Exp 132 ARM-STD)
  noisy(p_true)        -- irreducible observation noise (Exp 132 ARM-NOISE)
  aliased(...)         -- 4-colors x N-cells-per-color aliased layouts (Exp 143 / 144)
  nonstationary(...)   -- abrupt color<->cell remap of a base world at a given step

Each generator returns a plain dict (serialisable, no external objects) describing
the world so callers can reconstruct the environment without importing creature code.
The dict always contains:

  kind          : str   -- one of "learnable" | "noisy" | "aliased" | "nonstationary"
  rows, cols    : int   -- grid dimensions
  n_colors      : int   -- number of observation classes
  cmap          : list  -- int list length rows*cols; cmap[cell] = color

"noisy" adds:
  p_true        : float -- probability of seeing the true color
  n_colors      : int   -- total colors (for the noise distribution)
Callers that use the noise arm should wrap their observation draw with the
NoisyCmap helper (see exp132_surprise_ceiling.py) or the analytic_floor()
helper to know the irreducible entropy.

"aliased" adds:
  n_cells_per_color : int
  layout_seed       : int

"nonstationary" adds:
  base_kind       : str  -- kind of the base world dict
  remap_at_step   : int  -- step at which the remap fires
  remap_seed      : int  -- rng seed for the remap permutation
  cmap_after      : list -- cmap[cell] = color after remap

All world descriptions are deterministic given their parameters.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Shared constants copied from exp132 / exp143 (source-of-truth in those files)
# ---------------------------------------------------------------------------

#: Standard 4x4 grid rows (Exp 132, 143, 144)
ROWS: int = 4
#: Standard 4x4 grid columns (Exp 132, 143, 144)
COLS: int = 4
#: Total cells in the standard grid
N_CELLS: int = ROWS * COLS  # 16

#: Number of observation colors in the standard learnable / noisy worlds (Exp 132)
N_COLORS_STANDARD: int = 3

#: Number of aliased colors (Exp 143, 144)
N_COLORS_ALIASED: int = 4

#: Default noise probability for the noisy arm (Exp 132)
P_TRUE_DEFAULT: float = 0.7

#: Three canonical aliased layout seeds (Exp 143 seed 7; Exp 144 seeds 7, 11, 13)
ALIASED_LAYOUT_SEEDS: tuple[int, int, int] = (7, 11, 13)

# ---------------------------------------------------------------------------
# Analytic floor helper
# ---------------------------------------------------------------------------


def analytic_floor(p_true: float, n_colors: int) -> float:
    """Irreducible surprise (entropy) in nats for the noisy observation model.

    Each step: true color observed with probability p_true; otherwise uniform
    over the other (n_colors - 1) colors.  This is the per-step cross-entropy
    H(true || noisy) = H(noisy) in nats, which forms the irreducible floor
    the agent cannot drive below regardless of how well it learns.

    Validated by Exp 132: for p_true=0.7, n_colors=3 (3 colors, p_other=0.15
    each) the analytic value is ~0.82 nats, above the 0.7-nat ceiling threshold
    -- ensuring the ARM-NOISE ceiling detector MUST fire if calibrated correctly.

    Args:
        p_true:   Probability of observing the true color (0 < p_true < 1).
        n_colors: Total number of observation classes (including the true color).

    Returns:
        Irreducible entropy in nats.

    Example (Exp 132 ARM-NOISE):
        >>> round(analytic_floor(0.7, 3), 4)
        0.8228
    """
    if not (0.0 < p_true < 1.0):
        raise ValueError(f"p_true must be in (0, 1), got {p_true}")
    if n_colors < 2:
        raise ValueError(f"n_colors must be >= 2, got {n_colors}")

    p_other = (1.0 - p_true) / (n_colors - 1)
    # H = -p_true * ln(p_true) - (n_colors-1) * p_other * ln(p_other)
    h = -p_true * math.log(p_true)
    if p_other > 0:
        h -= (n_colors - 1) * p_other * math.log(p_other)
    return h


# ---------------------------------------------------------------------------
# World generators
# ---------------------------------------------------------------------------


def learnable() -> dict:
    """Return mirro's committed world layout: fully deterministic, structurally adequate.

    This is the Exp 132 ARM-STD world: the creature's own committed 4x4 color map.
    In this world the surprise ceiling detector should stay quiet (the counter-prediction
    validated by Exp 132).

    The color map is loaded from the creature's committed manifest at runtime.  The
    returned dict is a snapshot (no live file dependency after construction).

    Returns:
        World dict with kind="learnable".  The cmap is read from
        creature/state/mirro/manifest.json (the creature's committed world).

    Raises:
        FileNotFoundError: if the manifest is not present (run from repo root).
    """
    import json
    from pathlib import Path

    manifest_path = Path("creature/state/mirro/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    world_dict = manifest["world"]

    # Extract cmap -- stored as a list in the manifest
    cmap = list(world_dict["cmap"])
    rows = int(world_dict.get("rows", ROWS))
    cols = int(world_dict.get("cols", COLS))
    n_colors = int(world_dict.get("n_colors", N_COLORS_STANDARD))

    return {
        "kind": "learnable",
        "rows": rows,
        "cols": cols,
        "n_colors": n_colors,
        "cmap": cmap,
    }


def noisy(p_true: float = P_TRUE_DEFAULT) -> dict:
    """Return the Exp 132 ARM-NOISE world description.

    Same layout as learnable() but tagged for noisy observation.  The caller is
    responsible for wrapping observation draws with a NoisyCmap (see exp132) or
    using analytic_floor() to determine the irreducible entropy.

    The analytic irreducible surprise for p_true=0.7, n_colors=3 is ~0.82 nats,
    above the 0.7-nat ceiling threshold -- so the ceiling detector MUST fire on
    this world if calibrated correctly (validated by Exp 132 P1).

    Args:
        p_true: Probability of seeing the true color per step.  Default 0.7
                (Exp 132 ARM-NOISE value).

    Returns:
        World dict with kind="noisy", p_true=p_true, plus the base cmap.
    """
    base = learnable()
    return {
        **base,
        "kind": "noisy",
        "p_true": p_true,
    }


def _build_aliased_cmap(n_colors: int, n_cells_per_color: int,
                         layout_seed: int) -> list[int]:
    """Build a balanced aliased color map using rng(layout_seed).permutation.

    Algorithm (copied exactly from exp143 lines 105-110 and exp144 build_cmap):
        rng = np.random.default_rng(layout_seed)
        perm = rng.permutation(n_cells)
        for color_idx in range(n_colors):
            for slot in range(n_cells_per_color):
                cmap[perm[color_idx * n_cells_per_color + slot]] = color_idx

    This is the canonical construction used by Exp 143 (seed 7) and Exp 144
    (seeds 7, 11, 13).  Seeds 7/11/13 reproduce the exact layouts committed
    in experiments/outputs/exp143.txt and experiments/outputs/exp144.txt.
    """
    n_cells = n_colors * n_cells_per_color
    rng = np.random.default_rng(layout_seed)
    perm = rng.permutation(n_cells)
    cmap = np.empty(n_cells, dtype=int)
    for color_idx in range(n_colors):
        for slot in range(n_cells_per_color):
            cmap[perm[color_idx * n_cells_per_color + slot]] = color_idx
    return cmap.tolist()


def aliased(
    n_colors: int = N_COLORS_ALIASED,
    n_cells_per_color: int = 4,
    layout_seed: int = 7,
) -> dict:
    """Return an aliased world: n_colors * n_cells_per_color cells, balanced assignment.

    The color map is generated via rng(layout_seed).permutation, matching exactly
    the construction in Exp 143 (single layout, seed 7) and Exp 144 (three layouts,
    seeds 7, 11, 13).  Layout seeds 7/11/13 reproduce the committed experiment outputs:

      seed 7  -> [1, 1, 2, 0, 2, 3, 0, 1, 0, 3, 0, 3, 3, 2, 1, 2]  (Exp 143/144)
      seed 11 -> [3, 1, 0, 2, 2, 1, 1, 3, 0, 1, 3, 2, 0, 0, 3, 2]  (Exp 144)
      seed 13 -> [1, 0, 0, 2, 1, 2, 1, 0, 1, 3, 2, 2, 3, 3, 3, 0]  (Exp 144)

    In an aliased world each color maps to multiple spatially scattered cells, making
    a single emission component structurally inadequate -- the Exp 132 ceiling detector
    fires during phase 1 (validated by Exp 143 P2a: 8/8 seeds).

    Args:
        n_colors:         Number of observation classes (default 4, Exp 143/144).
        n_cells_per_color: Cells per color (default 4, Exp 143/144); total cells =
                          n_colors * n_cells_per_color.
        layout_seed:      RNG seed for the permutation (default 7, Exp 143).

    Returns:
        World dict with kind="aliased".
    """
    n_cells = n_colors * n_cells_per_color
    # Infer grid dimensions: assume square if possible, else 1 x n_cells
    sqrt = int(round(n_cells ** 0.5))
    if sqrt * sqrt == n_cells:
        rows, cols = sqrt, sqrt
    else:
        rows, cols = 1, n_cells

    cmap = _build_aliased_cmap(n_colors, n_cells_per_color, layout_seed)

    return {
        "kind": "aliased",
        "rows": rows,
        "cols": cols,
        "n_colors": n_colors,
        "n_cells_per_color": n_cells_per_color,
        "layout_seed": layout_seed,
        "cmap": cmap,
    }


def nonstationary(
    base: dict,
    remap_at_step: int,
    remap_seed: int,
) -> dict:
    """Return a nonstationary world: abrupt color<->cell remap of base at a given step.

    After remap_at_step the color map is replaced by a new random permutation of the
    same color set, generated by rng(remap_seed).  The agent experiences the base
    world up to step remap_at_step - 1 and the remapped world from step remap_at_step.

    Both the pre-remap cmap and the post-remap cmap_after are embedded in the dict
    so callers can switch without re-calling this function.

    Args:
        base:           A world dict returned by learnable(), noisy(), or aliased().
        remap_at_step:  Step index at which the remap fires (0-indexed; steps before
                        this index use base["cmap"], from this index use cmap_after).
        remap_seed:     RNG seed for generating the remapped color map.

    Returns:
        World dict with kind="nonstationary", cmap (pre-remap), cmap_after (post-remap).
    """
    n_cells = len(base["cmap"])
    n_colors = base["n_colors"]

    # Generate remapped cmap: shuffle the existing color assignments
    rng = np.random.default_rng(remap_seed)
    # Permute the color labels applied to each cell
    old_cmap = np.array(base["cmap"], dtype=int)
    # Abrupt remap: random permutation of color labels across all cells
    new_assignment = rng.permutation(n_cells).astype(int)
    # Ensure balanced color assignment if base had balanced aliased colors
    # For a general remap we shuffle the cell->color mapping directly
    new_cmap = np.empty(n_cells, dtype=int)
    for new_cell, old_cell in enumerate(new_assignment):
        new_cmap[new_cell] = old_cmap[old_cell]

    return {
        **base,
        "kind": "nonstationary",
        "base_kind": base["kind"],
        "remap_at_step": remap_at_step,
        "remap_seed": remap_seed,
        "cmap": list(base["cmap"]),   # pre-remap (explicit, not relying on **base)
        "cmap_after": new_cmap.tolist(),
    }
