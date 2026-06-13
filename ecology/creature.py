"""ecology/creature.py — Phenotype, Policy protocol, HomeostaticPolicy, Creature.

Policy pluggability seam:
  The Policy protocol (choose_action signature) is the intended interface for
  future navigation strategies, including a pymdp value-iteration navigator
  that would implement the same `choose_action(creature, world, rng) -> int`
  contract without any changes to the engine.

HomeostaticPolicy heuristic:
  The creature maintains a learned resource map m[cell] initialized to an
  optimistic prior.  Each observation updates m via EMA with learning_rate.
  On each step:
    - If the current cell's resource is below a depletion threshold:
        with prob exploration_bias: EXPLORE (move to least-recently-visited
        neighbor, tie-broken by lowest cell index; rng used for the coin).
        else: EXPLOIT (move to the neighbor with highest expected m[cell]).
    - Else (resource plentiful here): stay or exploit based on exploration_bias.
  This minimises expected energy-deficit: the creature moves toward cells where
  it expects high resource (exploitation) and occasionally explores to update
  beliefs about unknown cells (free-energy-reducing curiosity).  The tie-break
  on lowest cell index is fully deterministic (no set/dict ordering).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

import numpy as np

from ecology.genotype import Genotype, thermosense_active

if TYPE_CHECKING:
    from ecology.world import GridWorld


_DEPLETION_THRESHOLD: float = 0.5   # resource below this triggers possible move
_OPTIMISTIC_PRIOR: float = 1.0      # initial belief about unseen cells' resource


# ---------------------------------------------------------------------------
# Phenotype
# ---------------------------------------------------------------------------
@dataclass
class Phenotype:
    """Runtime developmental state of a creature."""
    energy: float
    age: int
    pos: int
    stress: float = 0.0
    damage: float = 0.0
    offspring_count: int = 0
    steps_moved: int = 0
    steps_explored: int = 0
    resource_eaten: float = 0.0
    alive: bool = True
    cause_of_death: str | None = None  # free string; "starvation" used here;
                                        # Exp 195 may add "temperature", "damage", etc.
    birth_t: int = 0                    # Exp 197: timestep the creature was created;
                                        # founders keep the default 0 (t=0 at __init__);
                                        # NOT emitted in any event dict / events_hash.


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------
class Policy(Protocol):
    """Navigation policy interface.

    A future pymdp value-iteration navigator implements this same protocol:
      def choose_action(self, creature: Creature, world: GridWorld,
                        rng: np.random.Generator) -> int

    Returns the target cell (a valid flat position).  May return current pos
    for "stay".  Must be deterministic given rng state.
    """

    def choose_action(
        self,
        creature: "Creature",
        world: "GridWorld",
        rng: np.random.Generator,
    ) -> int: ...


# ---------------------------------------------------------------------------
# HomeostaticPolicy
# ---------------------------------------------------------------------------
class HomeostaticPolicy:
    """Simple heuristic policy maintaining a learned resource map.

    Internal state:
      m[cell]  — EMA-updated expected resource value per cell.
      visit_t[cell] — timestep of last visit (or -1 if never).

    See module docstring for the full choice logic.
    """

    def __init__(self, n_cells: int, learning_rate: float, exploration_bias: float) -> None:
        self.n_cells = n_cells
        self.learning_rate = learning_rate
        self.exploration_bias = exploration_bias
        # Optimistic prior: assume all cells have some resource
        self.m: np.ndarray = np.full(n_cells, _OPTIMISTIC_PRIOR, dtype=np.float64)
        self.visit_t: np.ndarray = np.full(n_cells, -1, dtype=np.int64)

    def update_belief(self, pos: int, observed: float, t: int) -> None:
        """Update learned map at pos with EMA; record visit time."""
        self.m[pos] += self.learning_rate * (observed - self.m[pos])
        self.visit_t[pos] = t

    def choose_action(
        self,
        creature: "Creature",
        world: "GridWorld",
        rng: np.random.Generator,
    ) -> int:
        pos = creature.phenotype.pos
        neighbors = world.neighbors(pos)
        if not neighbors:
            return pos  # trapped (shouldn't happen on 12x12 but handle it)

        # Exp 197: thermal-aware branch — ONLY when temperature field is present AND
        # the creature has an active thermosense organ.  When use_thermo is False,
        # the exact existing logic below runs with IDENTICAL rng draws (regression guard).
        use_thermo = (
            world.temperature is not None
            and thermosense_active(creature.genotype, world.thermosense_active_threshold)
        )

        if not use_thermo:
            # ----------------------------------------------------------------
            # EXACT existing logic — verbatim, unchanged — rng stream identical
            # ----------------------------------------------------------------
            current_resource = world.resource_at(pos)

            # Decide explore vs exploit
            if current_resource < _DEPLETION_THRESHOLD:
                # Resource depleted here — consider moving
                if rng.random() < self.exploration_bias:
                    # EXPLORE: go to least-recently-visited neighbor
                    # Sort by visit_t ascending (oldest / never visited first),
                    # tie-break by lowest cell index (deterministic)
                    target = min(neighbors, key=lambda c: (self.visit_t[c], c))
                    creature.phenotype.steps_explored += 1
                    return target
                else:
                    # EXPLOIT: go to neighbor with highest expected resource
                    # tie-break by lowest cell index
                    target = max(neighbors, key=lambda c: (self.m[c], -c))
                    # Negate c so larger index loses ties: max by (m, -c) means
                    # higher m wins; equal m -> lower c wins (since -c is larger)
                    return target
            else:
                # Resource plentiful here; occasionally explore, else stay
                if rng.random() < self.exploration_bias * 0.3:
                    target = min(neighbors, key=lambda c: (self.visit_t[c], c))
                    creature.phenotype.steps_explored += 1
                    return target
                return pos  # stay and eat

        elif world.forage_mode:
            # ----------------------------------------------------------------
            # Exp 200: FORAGE mode — steer TOWARD food-optimal temperature.
            # Runs ONLY when thermosense is active AND world.forage_mode is True.
            # Avoidance mode is completely bypassed; rng draws happen here only
            # in this branch so regression conditions are unaffected.
            # ----------------------------------------------------------------
            food_opt = world.current_food_optimal
            intensity = creature.genotype.thermosense_intensity
            forage_weight = world.thermal_avoidance_weight

            # Noise decreases with higher intensity (better organ = better signal).
            noise_sd = max(0.0, world.thermosense_noise_base * (1.0 - intensity))

            # Noisy temperature reading at each neighbor; steer toward food_opt.
            # Score = learned resource estimate - forage_weight * |noisy_temp - food_opt|
            # Higher score = more food expected AND closer to the food band.
            noisy_temp: dict[int, float] = {}
            for n in neighbors:
                noisy_temp[n] = float(world.temperature[n]) + rng.normal(0.0, noise_sd)

            current_resource = world.resource_at(pos)
            if current_resource < _DEPLETION_THRESHOLD:
                # Resource depleted here — move to best neighbor for foraging.
                target = max(
                    neighbors,
                    key=lambda n: (self.m[n] - forage_weight * abs(noisy_temp[n] - food_opt), -n),
                )
                return target
            else:
                # Resource available here; check if a neighbor is significantly better
                # (close to food band AND has expected resource).
                best_neighbor = max(
                    neighbors,
                    key=lambda n: (self.m[n] - forage_weight * abs(noisy_temp[n] - food_opt), -n),
                )
                stay_score = self.m[pos] - forage_weight * abs(
                    float(world.temperature[pos]) - food_opt
                )
                move_score = (
                    self.m[best_neighbor]
                    - forage_weight * abs(noisy_temp[best_neighbor] - food_opt)
                )
                if move_score > stay_score:
                    return best_neighbor
                return pos

        else:
            # ----------------------------------------------------------------
            # Thermal-aware branch (Exp 197 avoid-mode) — runs ONLY when
            # temperature is on AND creature has active thermosense AND
            # forage_mode is False.
            # This branch may consume rng differently; that is intentional and
            # safe because it never executes in control/regression conditions.
            # ----------------------------------------------------------------
            comfort = world.current_comfort
            tolerance = creature.genotype.temperature_tolerance
            intensity = creature.genotype.thermosense_intensity
            avoidance = world.thermal_avoidance_weight

            # Noise decreases with higher intensity (better organ = better signal).
            noise_sd = max(0.0, world.thermosense_noise_base * (1.0 - intensity))

            # Estimate thermal stress at each neighbor via noisy temperature reading.
            stress_est: dict[int, float] = {}
            for n in neighbors:
                noisy = float(world.temperature[n]) + rng.normal(0.0, noise_sd)
                stress_est[n] = max(0.0, abs(noisy - comfort) - tolerance)

            # Estimate stress at current cell (no rng draw — use true value for stability).
            current_temp_stress = max(
                0.0, abs(float(world.temperature[pos]) - comfort) - tolerance
            )

            # If the current cell is highly stressed (above tolerance), prefer moving
            # to the lowest-stress neighbor regardless of resource.
            if current_temp_stress > 0.0:
                target = min(neighbors, key=lambda n: (stress_est[n], n))
                return target

            # Otherwise: balance resource and thermal stress (exploit with stress penalty).
            current_resource = world.resource_at(pos)
            if current_resource < _DEPLETION_THRESHOLD:
                # Resource depleted — move to best neighbor (high resource, low stress).
                target = max(
                    neighbors,
                    key=lambda n: (self.m[n] - avoidance * stress_est[n], -n),
                )
                return target
            else:
                # Resource plentiful here; stay unless thermally stressed (handled above).
                return pos


# ---------------------------------------------------------------------------
# Creature
# ---------------------------------------------------------------------------
class Creature:
    """A single creature with genotype, phenotype, and homeostatic policy.

    Attributes:
      creature_id   — unique integer id (ascending from 0)
      parent_id     — id of parent or None for founders
      generation    — 0 for founders, parent.generation+1 for offspring
      lineage_root  — id of the founding ancestor in this lineage
      genotype      — frozen inherited config
      phenotype     — mutable runtime state
      policy        — navigation policy (HomeostaticPolicy or any Policy)
      _t            — internal timestep counter for visit tracking
    """

    def __init__(
        self,
        creature_id: int,
        parent_id: int | None,
        generation: int,
        lineage_root: int,
        genotype: Genotype,
        phenotype: Phenotype,
        n_cells: int,
    ) -> None:
        self.creature_id = creature_id
        self.parent_id = parent_id
        self.generation = generation
        self.lineage_root = lineage_root
        self.genotype = genotype
        self.phenotype = phenotype
        self._t: int = 0
        self.policy: HomeostaticPolicy = HomeostaticPolicy(
            n_cells=n_cells,
            learning_rate=genotype.learning_rate,
            exploration_bias=genotype.exploration_bias,
        )

    # ------------------------------------------------------------------
    # Sense + act
    # ------------------------------------------------------------------
    def sense(self, world: "GridWorld", rng: np.random.Generator) -> float:
        """Get a (possibly noisy) resource reading at current cell; update belief."""
        pos = self.phenotype.pos
        reading = world.local_reading(pos, self.genotype.sensor_precision, rng)
        self.policy.update_belief(pos, reading, self._t)
        return reading

    def act(self, world: "GridWorld", rng: np.random.Generator) -> int:
        """Choose and execute a move.  Returns new position.
        Movement cost is ONLY paid if the cell changed (caller does energy deduction).
        """
        target = self.policy.choose_action(self, world, rng)
        old_pos = self.phenotype.pos
        self.phenotype.pos = target
        if target != old_pos:
            self.phenotype.steps_moved += 1
        self._t += 1
        return target

    def is_alive(self) -> bool:
        return self.phenotype.alive

    def genotype_dict(self) -> dict:
        """Return a plain dict of genotype fields for logging."""
        from dataclasses import asdict
        return asdict(self.genotype)
