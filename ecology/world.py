"""ecology/world.py — GridWorld with regenerating resources and local sensing.

Resource regeneration rule (documented):
  Each step, every cell (whether depleted or not) gains `regen_rate` units,
  capped at `capacity`.  If `regen_rate == 0`, cells that were already at
  exactly 0.0 receive a minimum floor bump of `_FLOOR_REGEN = 0.05` so that
  permanent dead-cell lock-in is impossible.  Formally:
    1. resource[i] += regen_rate  (all cells)
    2. resource[i] = min(capacity, resource[i])
    3. if regen_rate == 0 and resource[i] == 0:
         resource[i] = min(_FLOOR_REGEN, capacity)

Grid indexing: cell = row * cols + col.  Positions are flat integers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_FLOOR_REGEN: float = 0.05  # minimum regrowth on depleted cells


@dataclass
class GridWorld:
    rows: int
    cols: int
    resource: np.ndarray          # shape (rows*cols,), dtype float64, >= 0
    capacity: float
    regen_rate: float

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------
    def _rc(self, pos: int) -> tuple[int, int]:
        return divmod(pos, self.cols)

    def _pos(self, r: int, c: int) -> int:
        return r * self.cols + c

    def size(self) -> int:
        return self.rows * self.cols

    # ------------------------------------------------------------------
    # Resource interface
    # ------------------------------------------------------------------
    def resource_at(self, pos: int) -> float:
        return float(self.resource[pos])

    def consume(self, pos: int, amount: float) -> float:
        """Consume up to `amount` from cell `pos`.  Returns actually consumed."""
        available = float(self.resource[pos])
        consumed = min(amount, available)
        self.resource[pos] -= consumed
        return consumed

    def step_regen(self) -> None:
        """Regenerate resources by one step.

        All cells gain regen_rate, capped at capacity.  The _FLOOR_REGEN bump
        applies ONLY when regen_rate == 0 and the cell is still exactly 0 after
        adding regen_rate (preventing permanent dead-cell lock-in).
        See module docstring for the formal rule.
        """
        depleted = self.resource == 0.0
        self.resource += self.regen_rate
        self.resource = np.clip(self.resource, 0.0, self.capacity)
        # Floor bump: only for cells that were 0 AND are still 0 (regen_rate == 0 case)
        still_zero = depleted & (self.resource == 0.0)
        if np.any(still_zero):
            self.resource[still_zero] = min(_FLOOR_REGEN, self.capacity)

    def local_reading(
        self, pos: int, sensor_precision: float, rng: np.random.Generator
    ) -> float:
        """Return a (possibly noisy) resource reading at pos.

        With probability `sensor_precision`, returns the true value.
        Otherwise returns a uniform sample from [0, capacity].
        Deterministic given rng.
        """
        if rng.random() < sensor_precision:
            return self.resource_at(pos)
        else:
            return float(rng.uniform(0.0, self.capacity))

    def neighbors(self, pos: int) -> list[int]:
        """Von-Neumann neighbors (up/down/left/right), wall-clamped, no wrap.
        Returned in a fixed deterministic order: up, down, left, right
        (only included when within bounds).
        """
        r, c = self._rc(pos)
        result: list[int] = []
        if r > 0:
            result.append(self._pos(r - 1, c))
        if r < self.rows - 1:
            result.append(self._pos(r + 1, c))
        if c > 0:
            result.append(self._pos(r, c - 1))
        if c < self.cols - 1:
            result.append(self._pos(r, c + 1))
        return result

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        rows: int,
        cols: int,
        capacity: float,
        regen_rate: float,
        initial_resource: float,
        rng: np.random.Generator,
    ) -> "GridWorld":
        """Build initial resource field.

        initial_resource is the fraction of capacity each cell starts at
        (0.0 = empty, 1.0 = full).  A tiny uniform perturbation is added
        to break symmetry while remaining deterministic with rng.
        """
        base = capacity * max(0.0, min(1.0, initial_resource))
        perturb = rng.uniform(-0.05 * capacity, 0.05 * capacity, size=rows * cols)
        resource = np.clip(base + perturb, 0.0, capacity)
        return cls(
            rows=rows,
            cols=cols,
            resource=resource,
            capacity=capacity,
            regen_rate=regen_rate,
        )
