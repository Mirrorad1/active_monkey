"""ecology/continuous_world.py — ContinuousWorld for continuous-locomotion (Exp 238).

Implements the same call-site seams used by engine/creature (resource_at, consume,
step_regen, neighbors) with a bounded continuous arena and a fixed deterministic
sum-of-Gaussian-bumps resource density field rho(x,y).

ANTI-CHEAT: intake is the line integral of the PROVIDED rho field along the swept
segment. The heritable trait (locomotor_speed) keys ONLY the swept distance
d = locomotor_speed * dt. rho never receives any f(trait) term.

Geometry: bounded arena [0, ARENA_W] x [0, ARENA_H] (default 12x12 to mirror the
12x12 discrete grid scale). Positions are continuous 2-tuples (float, float).

Resource density:
  BUMP field (default): fixed sum-of-Gaussians bumps, zero rng, fully deterministic.
  FLAT field: uniform rho = FLAT_RHO, no bumps.
  NEUTRAL field: different bump arrangement (same number/sigma, different centers).

The resource density is stored as a mutable 2D grid of cells of width CELL_SIZE for
efficient consume() — each cell holds a depletable resource proportional to rho. The
line-integral intake is computed from the DENSITY FIELD rho (not from cell values),
while consume() depletes the underlying cell grid along the swept segment (K samples).

Regrowth: each step each sub-cell regenerates at regen_rate (capped at capacity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


# Arena size (mirrors 12x12 discrete grid scale).
ARENA_W: float = 12.0
ARENA_H: float = 12.0

# Sub-cell grid for consume() resource depletion.
# We use a 24x24 sub-grid (0.5 unit cells) for fine-grained depletion.
_GRID_CELLS: int = 24  # cells per axis
_CELL_SIZE: float = ARENA_W / _GRID_CELLS  # 0.5 units

# Line-integral sub-samples per sweep segment (K fixed for determinism).
_K_SAMPLES: int = 16

# Gaussian bump parameters (BUMP field).
_BUMP_SIGMA: float = 1.5
_BUMP_AMPLITUDE: float = 1.0  # peak density

# Bump centers for BUMP field.
_BUMP_CENTERS_BUMP: tuple[tuple[float, float], ...] = (
    (3.0,  3.0),
    (3.0,  9.0),
    (9.0,  3.0),
    (9.0,  9.0),
    (6.0,  6.0),
)

# Bump centers for NEUTRAL-LAYOUT field (same count/sigma, different arrangement).
_BUMP_CENTERS_NEUTRAL: tuple[tuple[float, float], ...] = (
    (2.0,  5.0),
    (5.0,  2.0),
    (10.0, 7.0),
    (7.0, 10.0),
    (4.0,  8.0),
)

# Flat field density.
_FLAT_RHO: float = 0.5


def _rho_bump(x: float, y: float,
              centers: tuple[tuple[float, float], ...],
              sigma: float,
              amplitude: float) -> float:
    """Sum-of-Gaussians density at (x, y). Pure math, no rng."""
    total = 0.0
    s2 = (sigma * sigma)
    for (cx, cy) in centers:
        dx = x - cx
        dy = y - cy
        total = total + (amplitude * math.exp(-(((dx * dx) + (dy * dy)) / (2.0 * s2))))
    return total


def _rho_flat(x: float, y: float) -> float:  # noqa: ARG001
    return _FLAT_RHO


@dataclass
class ContinuousWorld:
    """Continuous-space world implementing the same seams as GridWorld.

    resource_at(pos2)   -> float  density at continuous position (reads density FIELD)
    consume(pos2, d, locomotor_speed, dt) -> float  line-integral intake + deplete
    step_regen()        -> None   regenerate sub-cell grid
    neighbors(pos2)     -> ignored (movement is continuous; heading computed by creature)

    pos2: tuple[float, float] — (x, y) continuous position in [0, ARENA_W] x [0, ARENA_H].

    The sub-cell grid holds depletable resource in each cell; its state is updated by
    consume().  The density field rho is separate and does NOT decay (it is the structural
    landscape); the sub-cell grid is the AVAILABLE resource (can be depleted and regenerates).

    Exp 240: logistic_regen — OFF by default (byte-identical to Exp 238-239).
    When True, each sub-cell regens at:
        regen_amount = regen_rate * resource_cell * (1 - resource_cell / capacity)
    instead of the flat regen_rate. This logistic formula is maximised at 50% resource
    (regen_rate * capacity / 4) and approaches 0 at both 0 (depleted) and capacity (full).
    The result: heavily-depleted cells recover SLOWLY, so a large population that depletes
    the bump cells cannot keep harvesting at the same rate — a genuine negative feedback that
    produces a stable carrying capacity instead of a commons-tragedy runaway.
    OFF path (logistic_regen=False) is byte-identical to Exp 238-239 (the block never runs).
    """
    layout: Literal["bump", "flat", "neutral"] = "bump"
    regen_rate: float = 0.05
    capacity: float = 2.0
    # Exp 240: logistic regeneration gate — OFF (False) by default, byte-identical to Exp 238-239.
    logistic_regen: bool = False

    # Sub-cell resource grid: shape (_GRID_CELLS, _GRID_CELLS), each cell in [0, capacity].
    # Initialised to capacity (full) at construction.
    _resource: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._resource:
            # Initialize full sub-cell grid.
            self._resource = [
                [self.capacity] * _GRID_CELLS
                for _ in range(_GRID_CELLS)
            ]

    # ------------------------------------------------------------------
    # Density field
    # ------------------------------------------------------------------

    def rho(self, x: float, y: float) -> float:
        """Resource density at continuous position (x, y). Pure; does not deplete."""
        if self.layout == "flat":
            return _FLAT_RHO
        elif self.layout == "neutral":
            return _rho_bump(x, y, _BUMP_CENTERS_NEUTRAL, _BUMP_SIGMA, _BUMP_AMPLITUDE)
        else:
            return _rho_bump(x, y, _BUMP_CENTERS_BUMP, _BUMP_SIGMA, _BUMP_AMPLITUDE)

    # ------------------------------------------------------------------
    # Sub-cell indexing
    # ------------------------------------------------------------------

    def _cell_idx(self, x: float, y: float) -> tuple[int, int]:
        """Return sub-cell (row, col) for continuous position (x, y)."""
        ci = int(x / _CELL_SIZE)
        ri = int(y / _CELL_SIZE)
        # Clamp to grid bounds.
        ci = max(0, min(_GRID_CELLS - 1, ci))
        ri = max(0, min(_GRID_CELLS - 1, ri))
        return ri, ci

    # ------------------------------------------------------------------
    # Resource seams
    # ------------------------------------------------------------------

    def resource_at(self, pos2: tuple[float, float]) -> float:
        """Return sub-cell resource at pos2. Used for depletion-check."""
        ri, ci = self._cell_idx(pos2[0], pos2[1])
        return float(self._resource[ri][ci])

    def line_integral_intake(
        self,
        x0: float, y0: float,
        x1: float, y1: float,
        energy_deficit: float,
    ) -> float:
        """Compute the line integral of rho along the segment (x0,y0)→(x1,y1).

        Uses K fixed equal sub-samples (fixed parenthesization for determinism).
        Returns the intake amount, capped by energy_deficit and available resource.
        Does NOT mutate state; call consume_segment() afterward.

        ANTI-CHEAT: intake is integral of the PROVIDED rho field.
        The caller controls the segment length (via locomotor_speed*dt).
        """
        k = _K_SAMPLES
        # Segment length d = sqrt((x1-x0)^2 + (y1-y0)^2).
        ddx = (x1 - x0)
        ddy = (y1 - y0)
        d = math.sqrt((ddx * ddx) + (ddy * ddy))
        if d < 1e-12:
            return 0.0
        # Trapezoidal-rule line integral: sum rho at k equally-spaced points, weight by d/k.
        total = 0.0
        for i in range(k):
            t = i / (k - 1) if k > 1 else 0.0
            xi = x0 + t * ddx
            yi = y0 + t * ddy
            # Clamp to arena.
            xi = max(0.0, min(ARENA_W, xi))
            yi = max(0.0, min(ARENA_H, yi))
            total = total + self.rho(xi, yi)
        # Weight by d/k (trapezoid step size).
        raw_intake = (total * d) / k
        # Cap by energy deficit.
        return min(raw_intake, energy_deficit)

    def consume_segment(
        self,
        x0: float, y0: float,
        x1: float, y1: float,
        amount: float,
    ) -> None:
        """Deplete resource along the swept segment proportional to each sub-cell's density.

        Distributes `amount` across the K sample points' sub-cells, weighted by local rho.
        ANTI-CHEAT: depletion is of the PROVIDED sub-cell grid, not any f(trait) term.
        """
        k = _K_SAMPLES
        ddx = (x1 - x0)
        ddy = (y1 - y0)
        # Collect sub-cell rho weights for each sample.
        sample_rho: list[float] = []
        sample_ri: list[int] = []
        sample_ci: list[int] = []
        for i in range(k):
            t = i / (k - 1) if k > 1 else 0.0
            xi = x0 + t * ddx
            yi = y0 + t * ddy
            xi = max(0.0, min(ARENA_W, xi))
            yi = max(0.0, min(ARENA_H, yi))
            ri, ci = self._cell_idx(xi, yi)
            sample_rho.append(self.rho(xi, yi))
            sample_ri.append(ri)
            sample_ci.append(ci)
        total_rho = sum(sample_rho)
        if total_rho < 1e-12 or amount < 1e-12:
            return
        # Deplete each sub-cell proportional to its density contribution.
        for i in range(k):
            share = amount * (sample_rho[i] / total_rho)
            ri = sample_ri[i]
            ci = sample_ci[i]
            avail = float(self._resource[ri][ci])
            depleted = min(share, avail)
            self._resource[ri][ci] = float(self._resource[ri][ci]) - depleted

    def consume(
        self,
        x0: float, y0: float,
        x1: float, y1: float,
        energy_deficit: float,
    ) -> float:
        """Full eat step: compute line-integral intake and deplete.

        Returns actually consumed energy. ANTI-CHEAT: see line_integral_intake.
        """
        intake = self.line_integral_intake(x0, y0, x1, y1, energy_deficit)
        if intake > 1e-12:
            self.consume_segment(x0, y0, x1, y1, intake)
        return intake

    def step_regen(self) -> None:
        """Regenerate each sub-cell by regen_rate, capped at capacity.

        Exp 240: when logistic_regen=True, uses logistic formula instead of flat rate:
            regen_amount = regen_rate * resource_cell * (1 - resource_cell / capacity)
        This is the classic renewable-commons regulation: depleted cells recover slowly
        (near-zero resource → near-zero regen), full cells don't overshoot (1 - 1.0 = 0),
        and maximum regen occurs at 50% fullness.  The OFF path (logistic_regen=False) is
        BYTE-IDENTICAL to Exp 238-239 — the logistic branch is never entered when False.
        """
        cap = self.capacity
        rr = self.regen_rate
        if self.logistic_regen:
            # Exp 240: logistic regen — gated ON branch only (OFF is byte-identical).
            # regen_amount = regen_rate * v * (1 - v / cap), capped at cap.
            # This creates genuine negative feedback: a depleted cell (v ≈ 0) regens near 0,
            # so a large population that strips the field cannot harvest at the same rate.
            inv_cap = 1.0 / cap
            for ri in range(_GRID_CELLS):
                row = self._resource[ri]
                for ci in range(_GRID_CELLS):
                    v = float(row[ci])
                    delta = rr * v * (1.0 - v * inv_cap)
                    v = v + delta
                    if v > cap:
                        v = cap
                    elif v < 0.0:
                        v = 0.0
                    row[ci] = v
        else:
            # Original flat-rate regen (byte-identical to Exp 238-239 when logistic_regen=False).
            for ri in range(_GRID_CELLS):
                row = self._resource[ri]
                for ci in range(_GRID_CELLS):
                    v = float(row[ci]) + rr
                    if v > cap:
                        v = cap
                    row[ci] = v

    # ------------------------------------------------------------------
    # Heading helper (food-gradient sensing)
    # ------------------------------------------------------------------

    def best_heading(self, x: float, y: float, n_probes: int = 8) -> tuple[float, float]:
        """Return the unit heading (dx, dy) toward the richest nearby field.

        Scans `n_probes` equally-spaced directions at radius ARENA_W/4, picks the
        direction with highest rho.  Pure arithmetic, no rng.
        ANTI-CHEAT: heading is toward the PROVIDED field; locomotor_speed keys only d.
        """
        radius = ARENA_W / 4.0
        best_rho = float("-inf")
        best_dx = 1.0
        best_dy = 0.0
        for k in range(n_probes):
            angle = (2.0 * math.pi * k) / n_probes
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            # Clamp to arena.
            px = max(0.0, min(ARENA_W, px))
            py = max(0.0, min(ARENA_H, py))
            r = self.rho(px, py)
            if r > best_rho:
                best_rho = r
                best_dx = math.cos(angle)
                best_dy = math.sin(angle)
        # Normalise.
        mag = math.sqrt((best_dx * best_dx) + (best_dy * best_dy))
        if mag < 1e-12:
            return (1.0, 0.0)
        return (best_dx / mag, best_dy / mag)

    @classmethod
    def from_config(
        cls,
        layout: Literal["bump", "flat", "neutral"] = "bump",
        regen_rate: float = 0.05,
        capacity: float = 2.0,
        logistic_regen: bool = False,
    ) -> "ContinuousWorld":
        """Build a ContinuousWorld from config parameters.  No rng used.

        logistic_regen=False (default): byte-identical to Exp 238-239.
        logistic_regen=True (Exp 240): logistic resource renewal for stable carrying capacity.
        """
        return cls(layout=layout, regen_rate=regen_rate, capacity=capacity,
                   logistic_regen=logistic_regen)
