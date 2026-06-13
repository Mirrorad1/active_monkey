"""ecology/genotype.py — Genotype dataclass, mutation, validation, clamping.

All trait values are deterministically derived from seeds; no wall-clock or
unordered-set iteration affects the genotype stream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Trait bounds (inclusive [lo, hi])
# ---------------------------------------------------------------------------
TRAIT_BOUNDS: dict[str, tuple[float, float]] = {
    "movement_cost":                    (0.01,  2.0),
    "baseline_metabolic_cost":          (0.01,  2.0),
    "energy_capacity":                  (5.0,  50.0),
    "reproduction_energy_threshold":    (2.0,  50.0),   # also <= energy_capacity
    "reproduction_energy_transfer_fraction": (0.05, 0.60),
    "reproduction_cost_fraction":       (0.02,  0.30),
    "maturity_age":                     (1,    30),      # int
    "aging_cost":                       (0.0,   0.5),
    "exploration_bias":                 (0.0,   1.0),
    "learning_rate":                    (0.01,  1.0),
    "memory_length":                    (1,    20),      # int
    "temperature_tolerance":            (0.0,   1.0),    # UNUSED (reserved Exp195)
    "sensor_precision":                 (0.5,   1.0),
}

INT_TRAITS: frozenset[str] = frozenset({"maturity_age", "memory_length"})


def clamp_traits(d: dict[str, Any]) -> dict[str, Any]:
    """Clamp each trait to its bounds; round INT_TRAITS; enforce threshold≤capacity."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        lo, hi = TRAIT_BOUNDS[k]
        v = float(v)
        v = max(lo, min(hi, v))
        if k in INT_TRAITS:
            v = int(round(v))
            v = max(int(lo), min(int(hi), v))
        out[k] = v
    # Enforce threshold ≤ capacity
    out["reproduction_energy_threshold"] = min(
        out["reproduction_energy_threshold"],
        out["energy_capacity"],
    )
    return out


@dataclass(frozen=True)
class Genotype:
    """Inherited configuration for a creature.  All fields within TRAIT_BOUNDS.

    temperature_tolerance is carried but UNUSED in Exp 194; it is reserved for
    Exp 195 to add environment temperature stress.
    """
    movement_cost: float
    baseline_metabolic_cost: float
    energy_capacity: float
    reproduction_energy_threshold: float          # invariant: <= energy_capacity
    reproduction_energy_transfer_fraction: float  # fraction of current energy transferred
    reproduction_cost_fraction: float             # parent overhead fraction
    maturity_age: int
    aging_cost: float
    exploration_bias: float
    learning_rate: float
    memory_length: int
    temperature_tolerance: float                  # UNUSED (reserved)
    sensor_precision: float


def is_valid(g: Genotype) -> bool:
    """Return True iff every trait is within TRAIT_BOUNDS and threshold ≤ capacity."""
    d = asdict(g)
    for k, v in d.items():
        lo, hi = TRAIT_BOUNDS[k]
        if not (lo <= v <= hi):
            return False
    if g.reproduction_energy_threshold > g.energy_capacity:
        return False
    return True


def mutate(g: Genotype, rng: np.random.Generator, rate: float) -> Genotype:
    """Return a new Genotype with each trait independently perturbed by
    N(0, rate*(hi-lo)) and clamped into valid range.  Deterministic given rng.
    Result always satisfies is_valid().
    """
    d = asdict(g)
    new_d: dict[str, Any] = {}
    for k, v in d.items():
        lo, hi = TRAIT_BOUNDS[k]
        sigma = rate * (hi - lo)
        perturbed = v + rng.normal(0.0, sigma)
        new_d[k] = perturbed
    clamped = clamp_traits(new_d)
    result = Genotype(**clamped)
    assert is_valid(result), f"mutate produced invalid genotype: {result}"
    return result


def founder() -> Genotype:
    """The base ancestor used for all Exp 194 scenarios.

    Values chosen after tuning trials (documented in ecology/scenarios.py) to
    produce meaningful ecology dynamics — reproduction is energetically costly
    (high threshold + high transfer + high overhead) so the population does not
    trivially explode, while metabolic costs ensure mortality pressure.

    Reported values (all within TRAIT_BOUNDS):
      movement_cost=0.3, baseline_metabolic_cost=0.5, energy_capacity=20.0,
      reproduction_energy_threshold=17.0 (85% of capacity),
      reproduction_energy_transfer_fraction=0.45 (transfers 45% of current energy),
      reproduction_cost_fraction=0.15 (parent overhead),
      maturity_age=5, aging_cost=0.02,
      exploration_bias=0.4, learning_rate=0.3, memory_length=5,
      temperature_tolerance=0.5 (unused, reserved Exp195), sensor_precision=0.85.
    """
    d = {
        "movement_cost": 0.3,
        "baseline_metabolic_cost": 0.5,
        "energy_capacity": 20.0,
        "reproduction_energy_threshold": 17.0,  # requires 85% energy to reproduce
        "reproduction_energy_transfer_fraction": 0.45,  # large transfer to child
        "reproduction_cost_fraction": 0.15,   # significant parent overhead
        "maturity_age": 5,
        "aging_cost": 0.02,
        "exploration_bias": 0.4,
        "learning_rate": 0.3,
        "memory_length": 5,
        "temperature_tolerance": 0.5,
        "sensor_precision": 0.85,
    }
    clamped = clamp_traits(d)
    g = Genotype(**clamped)
    assert is_valid(g), f"founder() produced invalid genotype: {g}"
    return g
