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
    # Exp 197: optional temperature field (None when enable_temperature=False)
    temperature: "np.ndarray | None" = None   # shape (rows*cols,) float64, [0,1]
    # Thermal policy parameters (used by creature.py; stored here to avoid cfg threading)
    temperature_comfort: float = 0.5
    # current_comfort: dynamic comfort center (updated by Ecology.step() each tick).
    # Initialised to temperature_comfort; when comfort_amplitude==0 it never changes.
    # creature.py and temperature_stress() read this field rather than the static cfg value.
    current_comfort: float = 0.5
    thermosense_noise_base: float = 0.5
    thermal_avoidance_weight: float = 1.0
    thermosense_active_threshold: float = 0.05

    # Exp 200: foraging-sense — food concentrated in a drifting thermal band.
    # All new fields have defaults that preserve previous behaviour exactly.
    #
    # forage_mode: when True, the policy thermal branch steers TOWARD food (the
    #   food-optimal temperature) rather than AWAY from thermal stress.
    # current_food_optimal: drifting food-optimal temperature; updated by
    #   Ecology.step() when enable_food_coupling is True. Initialised to
    #   food_optimal_base (set via from_config).
    # enable_food_coupling: gates the in-band regen concentration in step_regen().
    #   When False, step_regen() is EXACTLY the old code (byte-identical).
    # food_band_width / food_concentration: parameters for the regen boost.
    forage_mode: bool = False
    current_food_optimal: float = 0.5
    enable_food_coupling: bool = False
    food_band_width: float = 0.15
    food_concentration: float = 1.0

    # Exp 201: band-staleness foraging — the food-optimal temperature DRIFTS and is
    # no longer handed to the policy for free (exp200 read current_food_optimal
    # directly).  Each creature must privately ESTIMATE the drifting center via an
    # EMA tracker whose responsiveness (alpha) and reading-noise are keyed to
    # thermosense_intensity.  All defaults are no-ops (byte-identical to exp194-200).
    #
    # enable_band_staleness: gates the new forage sub-branch in creature.py.
    # band_responsiveness: scales the tracker EMA alpha = clamp(intensity*resp, 0, 1).
    # food_optimal_base: the STATIC drift center; the tracker is lazily initialised
    #   here (a neutral start, NOT the moving current_food_optimal) so a crude tracker
    #   does not get a free accurate starting point.
    enable_band_staleness: bool = False
    band_responsiveness: float = 1.0
    food_optimal_base: float = 0.5

    # Exp 204: residue / false-positive discrimination — the field where eaten food
    # leaves a misleading trace.  None (default) ⇒ the residue mechanic is OFF and the
    # engine's eat step is byte-identical to exp194-203.  When enable_residue is True the
    # engine allocates a zeros(rows*cols) array here, accumulates residue on eating, and
    # decays it each step.  A pure state array (NOT in events_hash); the discrimination
    # logic + params live in the engine (cfg) so the world only holds the trace field.
    residue: "np.ndarray | None" = None

    # Exp 206: rotating-class niche fields — all None/0 (default) ⇒ the niche mechanic is OFF
    # and behaviour is byte-identical to exp194-205.  Allocated by the engine ONLY when
    # enable_niche=True.  class_phase = per-cell hidden phase (pure arithmetic, NO rng);
    # class_signal = frac(class_phase + omega(t)), the ROTATING per-cell class signal recomputed
    # each step; class_occ_prev/cur = per-class occupancy counts (frozen-prev read by the
    # crowding discount + routing).  enable_niche/niche_* mirror cfg so creature.py can read them
    # without threading cfg.  niche_read_perm = the BARCODE_SHUFFLED placebo permutation (None
    # normally) decorrelating the routing read-class from the true crowding class.  None of these
    # enter events_hash.
    class_phase: "np.ndarray | None" = None
    class_signal: "np.ndarray | None" = None
    class_occ_prev: "np.ndarray | None" = None
    class_occ_cur: "np.ndarray | None" = None
    enable_niche: bool = False
    niche_classes: int = 2
    niche_confusion: float = 0.6
    niche_weight: float = 4.0
    niche_read_perm: "np.ndarray | None" = None

    # PERF (not part of state/equality): lazily-built static neighbor table (see neighbors()).
    _neighbor_table: "list[list[int]] | None" = field(default=None, init=False, repr=False, compare=False)

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

        Exp 200 food-coupling extension: when enable_food_coupling is True AND a
        temperature field is present, cells within food_band_width of
        current_food_optimal get regen_rate * food_concentration; out-of-band
        cells get regen_rate * out_factor, where out_factor is chosen so total
        regen is approximately conserved.  Deterministic (no rng).
        When enable_food_coupling is False, this method is byte-identical to before.
        """
        # Exp 200 food-coupling: concentrated regen — only when enabled AND temperature
        # field is present (temperature carries the spatial gradient for the band).
        if self.enable_food_coupling and self.temperature is not None:
            n_cells = self.rows * self.cols
            in_band = np.abs(self.temperature - self.current_food_optimal) <= self.food_band_width
            n_in = int(np.sum(in_band))
            n_out = n_cells - n_in

            # Choose out_factor so total regen ≈ n_cells * regen_rate (conserved).
            # Total regen with boost = n_in * regen_rate * food_concentration
            #                        + n_out * regen_rate * out_factor
            # Setting equal to n_cells * regen_rate and solving for out_factor:
            #   out_factor = (n_cells - n_in * food_concentration) / max(1, n_out)
            # Clamped to >= 0 so we never give negative regen.
            if n_out > 0:
                out_factor = max(0.0, (n_cells - n_in * self.food_concentration) / n_out)
            else:
                out_factor = 0.0

            depleted = self.resource == 0.0
            # Apply per-cell regen in-place using vectorised arithmetic.
            regen_amounts = np.where(in_band, self.regen_rate * self.food_concentration,
                                     self.regen_rate * out_factor)
            self.resource += regen_amounts
            self.resource = np.clip(self.resource, 0.0, self.capacity)
            # Floor bump for cells that were depleted AND are still zero.
            still_zero = depleted & (self.resource == 0.0)
            if np.any(still_zero):
                self.resource[still_zero] = min(_FLOOR_REGEN, self.capacity)
            return

        # --- Original path (byte-identical when enable_food_coupling is False) ---
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

        PERF: the grid is static, so the neighbor table is identical every step. It is
        built ONCE (lazily) and returned by lookup — byte-identical values/order to the
        old per-call computation, but eliminates ~1.8M redundant recomputations per run
        (it was ~10% of run time at large populations). Callers read the list read-only
        (verified: choose_action/reproduction only iterate / max / min over it), so the
        cached list is returned directly (no copy).
        """
        cache = self._neighbor_table
        if cache is None:
            cache = self._neighbor_table = self._build_neighbor_table()
        return cache[pos]

    def _build_neighbor_table(self) -> list[list[int]]:
        """Precompute the von-Neumann neighbor list for every cell, in up/down/left/right
        order — identical to the old neighbors() output for each pos."""
        table: list[list[int]] = []
        for pos in range(self.rows * self.cols):
            r, c = self._rc(pos)
            nb: list[int] = []
            if r > 0:
                nb.append(self._pos(r - 1, c))
            if r < self.rows - 1:
                nb.append(self._pos(r + 1, c))
            if c > 0:
                nb.append(self._pos(r, c - 1))
            if c < self.cols - 1:
                nb.append(self._pos(r, c + 1))
            table.append(nb)
        return table

    # ------------------------------------------------------------------
    # Exp 197: temperature stress query
    # ------------------------------------------------------------------
    def temperature_stress(
        self, pos: int, tolerance: float, stress_scale: float
    ) -> float:
        """Return the thermal stress energy drain at pos.

        If the temperature field is absent, returns 0.0 (OFF path — no cost).
        Uses self.current_comfort as the comfort center (updated each tick by
        Ecology.step(); equals temperature_comfort when comfort_amplitude==0).
        Stress = stress_scale * max(0, |temperature[pos] - current_comfort| - tolerance).
        A creature within its tolerance band of the comfort zone incurs 0 stress.
        """
        if self.temperature is None:
            return 0.0
        return stress_scale * max(
            0.0, abs(float(self.temperature[pos]) - self.current_comfort) - tolerance
        )

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
        *,
        enable_temperature: bool = False,
        temperature_comfort: float = 0.5,
        thermosense_noise_base: float = 0.5,
        thermal_avoidance_weight: float = 1.0,
        thermosense_active_threshold: float = 0.05,
        # Exp 200: foraging-sense parameters — all default to OFF (no-op).
        forage_mode: bool = False,
        food_optimal_base: float = 0.5,
        enable_food_coupling: bool = False,
        food_band_width: float = 0.15,
        food_concentration: float = 1.0,
        # Exp 201: band-staleness parameters — all default to OFF (no-op).
        enable_band_staleness: bool = False,
        band_responsiveness: float = 1.0,
    ) -> "GridWorld":
        """Build initial resource field and optional temperature gradient.

        initial_resource is the fraction of capacity each cell starts at
        (0.0 = empty, 1.0 = full).  A tiny uniform perturbation is added
        to break symmetry while remaining deterministic with rng.

        If enable_temperature is True, a STATIC deterministic left-to-right
        gradient is built: temperature[r*cols + c] = c / (cols - 1).
        The gradient requires no rng draws, so the resource-perturbation rng
        stream is unchanged relative to the no-temperature case (regression safe).

        Exp 200 foraging parameters are stored in the world so step_regen() and
        creature.py can read them without threading cfg through every call.
        All defaults are no-ops — byte-identical to Exp 194–199 when not set.
        """
        base = capacity * max(0.0, min(1.0, initial_resource))
        perturb = rng.uniform(-0.05 * capacity, 0.05 * capacity, size=rows * cols)
        resource = np.clip(base + perturb, 0.0, capacity)

        temperature: np.ndarray | None = None
        if enable_temperature:
            # Static left-to-right gradient: 0.0 at left column, 1.0 at right.
            # Pure arithmetic, no rng — resource rng stream unchanged.
            temperature = np.array(
                [c / max(cols - 1, 1) for r in range(rows) for c in range(cols)],
                dtype=np.float64,
            )

        return cls(
            rows=rows,
            cols=cols,
            resource=resource,
            capacity=capacity,
            regen_rate=regen_rate,
            temperature=temperature,
            temperature_comfort=temperature_comfort,
            current_comfort=temperature_comfort,  # initialised to static value; updated per-tick
            thermosense_noise_base=thermosense_noise_base,
            thermal_avoidance_weight=thermal_avoidance_weight,
            thermosense_active_threshold=thermosense_active_threshold,
            # Exp 200 foraging fields — set from config; defaults are no-ops.
            forage_mode=forage_mode,
            current_food_optimal=food_optimal_base,  # initialised to static value; updated per-tick
            enable_food_coupling=enable_food_coupling,
            food_band_width=food_band_width,
            food_concentration=food_concentration,
            # Exp 201 band-staleness fields — defaults are no-ops.
            enable_band_staleness=enable_band_staleness,
            band_responsiveness=band_responsiveness,
            food_optimal_base=food_optimal_base,  # stored for neutral tracker init
        )
