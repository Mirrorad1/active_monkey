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
    # Exp 197: thermosense organ traits — defaults produce zero-organ founders.
    "thermosense_intensity":            (0.0,   1.0),    # 0 = organ absent
    "thermosense_inefficiency":         (0.2,   1.0),    # upkeep multiplier; 0.2 = floor
    # Exp hidden-state-mode: memory horizon for noisy cue integration.
    "memory_horizon":                   (0,    12),      # int; 0 = no memory
    # Phase 3 rung-1b: CONTINUOUS belief persistence (EMA weight) — the local (small-ε)
    # analog of memory_horizon, so a genuinely small heritable step can be tested.
    "belief_persistence":               (0.0,  0.95),    # 0 = react to current cue only
    # Phase 4: active sensing — per-step probability of paying to draw extra cues.
    "information_sampling_rate":        (0.0,  1.0),
    # Exp 235: locomotion / terrain climbing ability — LAST, WITH DEFAULT (regression-safe).
    "climb_ability":                    (0.0,  1.0),
    # Exp 238: continuous locomotion speed — LAST, WITH DEFAULT (regression-safe).
    "locomotor_speed":                  (0.25, 4.0),
}

INT_TRAITS: frozenset[str] = frozenset({"maturity_age", "memory_length", "memory_horizon"})

# Thermosense trait names — used to gate rng draws in mutate() (regression guard).
THERMOSENSE_TRAITS: frozenset[str] = frozenset({"thermosense_intensity", "thermosense_inefficiency"})

# Memory trait names — used to gate rng draws in mutate() (regression guard).
MEMORY_TRAITS: frozenset[str] = frozenset({"memory_horizon", "belief_persistence"})

# Active-sensing trait names — used to gate rng draws in mutate() (regression guard).
ACTIVE_SENSING_TRAITS: frozenset[str] = frozenset({"information_sampling_rate"})

# Exp 235: locomotion / terrain climbing trait — used to gate rng draws in mutate() (regression guard).
LOCOMOTION_TRAITS: frozenset[str] = frozenset({"climb_ability"})

# Exp 238: continuous locomotion speed trait — skip rng draw when OFF (regression guard).
LOCOMOTION_CONTINUOUS_TRAITS: frozenset[str] = frozenset({"locomotor_speed"})


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

    thermosense_intensity / thermosense_inefficiency (Exp 197): the evolvable
    thermosense organ.  Defaults to 0.0 / 1.0 so founders have no active organ
    (intensity=0 ⇒ inactive, upkeep=0).  Both fields are LAST with defaults so
    that existing Genotype(...) construction without them still works.
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
    # Exp 197 thermosense organ — LAST, WITH DEFAULTS (regression-safe)
    thermosense_intensity: float = 0.0            # 0.0 = organ absent (inactive)
    thermosense_inefficiency: float = 1.0         # upkeep multiplier; evolved down ⇒ cheaper
    # Exp hidden-state-mode: cue integration window — LAST, WITH DEFAULT (regression-safe)
    memory_horizon: int = 0                        # 0 = no cue buffer; int in [0, 12]
    belief_persistence: float = 0.0                # continuous EMA persistence; 0 = none
    information_sampling_rate: float = 0.0         # Phase 4: per-step probe probability (active sensing); 0 = never probe
    # Exp 235: locomotion / terrain climbing ability — LAST, WITH DEFAULT (regression-safe).
    # 0.05 = low but non-zero (founders can cross rim with very low probability).
    climb_ability: float = 0.05
    # Exp 238: continuous locomotion speed — LAST, WITH DEFAULT (regression-safe).
    # Default 1.0 = midpoint of [0.25, 4.0]; founders start at a neutral speed.
    locomotor_speed: float = 1.0


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


def mutate(
    g: Genotype,
    rng: np.random.Generator,
    rate: float,
    mutate_thermosense: bool = False,
    freeze_learning_rate: bool = False,
    freeze_thermosense: bool = False,
    mutate_memory: bool = False,
    mutate_active_sensing: bool = False,
    mutate_locomotion: bool = False,
    mutate_continuous_locomotion: bool = False,
) -> Genotype:
    """Return a new Genotype with each trait independently perturbed by
    N(0, rate*(hi-lo)) and clamped into valid range.  Deterministic given rng.
    Result always satisfies is_valid().

    REGRESSION GUARD: when mutate_thermosense=False (the default), thermosense
    traits are copied unchanged WITHOUT any rng draw, so the rng stream for all
    base traits is byte-identical to the pre-Exp-197 behaviour.  When
    mutate_thermosense=True, draws are made AFTER all base-trait draws (in field
    order) — the base-trait stream is unaffected.

    Exp 201 confound-killer: freeze_learning_rate=True PINS learning_rate to the
    parent value so the learned resource map cannot be sharpened as a non-thermal
    substitute for the costed thermosense organ.  The rng draw for learning_rate
    STILL happens (the perturbation is computed then DISCARDED), so the rng stream
    for every trait after learning_rate is identical to an ordinary mutation —
    only the learning_rate RESULT is pinned.  Default False ⇒ byte-identical to
    Exp 194-200.

    Exp 203 selection-gradient audit: freeze_thermosense=True PINS the thermosense
    organ traits (intensity + inefficiency) to the PARENT value so a clamped sensor
    value BREEDS TRUE across generations while upkeep is STILL CHARGED (cost on).
    Like freeze_learning_rate it keeps the rng draw and DISCARDS the perturbation,
    so the stream for every later trait is identical to an ordinary thermosense
    mutation — only the result is pinned.  Only meaningful when mutate_thermosense
    is True (otherwise the traits already skip the draw); default False ⇒
    byte-identical to Exp 194-202.

    Hidden-state-mode REGRESSION GUARD: when mutate_memory=False (the default),
    memory_horizon is copied unchanged WITHOUT any rng draw, so the rng stream for
    all base + thermosense traits is byte-identical to the pre-hidden-mode behaviour.
    When mutate_memory=True, the draw is made AFTER all other traits (memory_horizon
    is LAST in field order) — the upstream stream is unaffected.  Default False ⇒
    byte-identical to Exp 194-206.

    Phase 4 REGRESSION GUARD: when mutate_active_sensing=False (the default),
    information_sampling_rate is copied unchanged WITHOUT any rng draw, so the rng
    stream for all base + thermosense + memory traits is byte-identical to the
    pre-Phase-4 behaviour.  information_sampling_rate is the LAST field in field
    order, so skipping its draw leaves the rng stream for all upstream traits
    byte-identical.  Default False ⇒ byte-identical to Exp 194-209.

    Exp 235 REGRESSION GUARD: when mutate_locomotion=False (the default),
    climb_ability is copied unchanged WITHOUT any rng draw, so the rng stream for
    all base + thermosense + memory + active-sensing traits is byte-identical to the
    pre-Exp-235 behaviour.  climb_ability is the LAST field in field order, so
    skipping its draw leaves the rng stream for all upstream traits byte-identical.
    Default False ⇒ byte-identical to Exp 194-213.

    Exp 238 REGRESSION GUARD: when mutate_continuous_locomotion=False (the default),
    locomotor_speed is copied unchanged WITHOUT any rng draw, so the rng stream for
    all prior traits is byte-identical to the pre-Exp-238 behaviour.
    locomotor_speed is LAST in field order.  Default False ⇒ byte-identical to Exp 194-237.
    """
    d = asdict(g)
    new_d: dict[str, Any] = {}
    for k, v in d.items():
        # Thermosense traits: skip rng draw when mutation is disabled — this is
        # THE regression guard.  No rng.normal call is made, so the stream for
        # all base traits before these fields is untouched.
        if k in THERMOSENSE_TRAITS and not mutate_thermosense:
            new_d[k] = v
            continue
        # Memory traits: skip rng draw when mutation is disabled — mirrors the
        # thermosense skip guard above.  memory_horizon is LAST in field order
        # so no upstream trait's draw is affected when the skip fires.
        if k in MEMORY_TRAITS and not mutate_memory:
            new_d[k] = v
            continue
        # Active-sensing traits: skip rng draw when mutation is disabled — mirrors
        # the memory skip guard above.  information_sampling_rate is the LAST field
        # in field order so no upstream trait's draw is affected when the skip fires.
        if k in ACTIVE_SENSING_TRAITS and not mutate_active_sensing:
            new_d[k] = v
            continue
        # Locomotion traits: skip rng draw when mutation is disabled — mirrors the
        # active-sensing skip guard above.  climb_ability is the LAST field in field
        # order so no upstream trait's draw is affected when the skip fires.
        if k in LOCOMOTION_TRAITS and not mutate_locomotion:
            new_d[k] = v
            continue
        # Continuous locomotion traits: skip rng draw when mutation is disabled — mirrors the
        # locomotion skip guard above.  locomotor_speed is the LAST field in field order
        # so no upstream trait's draw is affected when the skip fires.
        if k in LOCOMOTION_CONTINUOUS_TRAITS and not mutate_continuous_locomotion:
            new_d[k] = v
            continue
        lo, hi = TRAIT_BOUNDS[k]
        sigma = rate * (hi - lo)
        perturbed = v + rng.normal(0.0, sigma)
        # Exp 201: pin learning_rate to the parent value but KEEP the rng draw
        # above so the downstream stream is unchanged (gated; default no-op).
        if k == "learning_rate" and freeze_learning_rate:
            new_d[k] = v
            continue
        # Exp 203: pin the thermosense organ traits to the parent value but KEEP
        # the rng draw above so the downstream stream is unchanged (gated; default
        # no-op).  This makes a clamped sensor value breed true with cost ON.
        if k in THERMOSENSE_TRAITS and freeze_thermosense:
            new_d[k] = v
            continue
        new_d[k] = perturbed
    clamped = clamp_traits(new_d)
    result = Genotype(**clamped)
    assert is_valid(result), f"mutate produced invalid genotype: {result}"
    return result


def complexity(g: Genotype) -> float:
    """Return the normalised complexity blend [0, 1] for a genotype.

    This is the SINGLE canonical definition shared by both the reproduction-overhead
    path and the senescence-degradation path.  Changing the blend here propagates
    automatically to both — the two paths cannot diverge.

    Complexity = mean of three normalised traits:
      - energy_capacity  in [5, 50]  -> [0, 1]
      - sensor_precision in [0.5, 1] -> [0, 1]
      - memory_length    in [1, 20]  -> [0, 1]
    """
    norm_cap    = (g.energy_capacity    - 5.0) / 45.0
    norm_sensor = (g.sensor_precision   - 0.5) / 0.5
    norm_mem    = (g.memory_length      - 1.0) / 19.0
    return (norm_cap + norm_sensor + norm_mem) / 3.0


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
      thermosense_intensity=0.0, thermosense_inefficiency=1.0 (no organ; must emerge
      by mutation under enable_thermosense treatment).
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
        # Exp 197: no thermosense organ at founding; must emerge by mutation.
        "thermosense_intensity": 0.0,
        "thermosense_inefficiency": 1.0,
        # Exp hidden-state-mode: no cue memory at founding; must be set explicitly.
        "memory_horizon": 0,
        "belief_persistence": 0.0,
        # Phase 4: no active sensing at founding; must emerge by mutation under enable_active_sensing.
        "information_sampling_rate": 0.0,
        # Exp 235: low but non-zero climbing ability at founding; must evolve under enable_terrain.
        "climb_ability": 0.05,
        # Exp 238: neutral starting speed; must evolve under enable_continuous_locomotion.
        "locomotor_speed": 1.0,
    }
    clamped = clamp_traits(d)
    g = Genotype(**clamped)
    assert is_valid(g), f"founder() produced invalid genotype: {g}"
    return g


# ---------------------------------------------------------------------------
# Exp 197: thermosense organ helpers
# ---------------------------------------------------------------------------

def thermosense_active(g: Genotype, threshold: float) -> bool:
    """Return True if the thermosense organ is expressed (intensity above threshold)."""
    return g.thermosense_intensity > threshold


def thermosense_upkeep(g: Genotype, floor: float, threshold: float) -> float:
    """Return the energy upkeep cost of the expressed thermosense organ.

    Upkeep = floor + inefficiency_multiplier * intensity.
    When inactive (intensity <= threshold), returns 0.0.
    The floor > 0 and inefficiency >= 0.2 guarantee the organ is never free.
    """
    if not thermosense_active(g, threshold):
        return 0.0
    return floor + g.thermosense_inefficiency * g.thermosense_intensity


def expressed_complexity(g: Genotype, threshold: float) -> float:
    """Return expressed phenotypic complexity: base body (1 unit) + active thermosense.

    This is a READOUT of active machinery; it does NOT feed into the existing
    complexity() blend (which governs reproduction overhead and senescence).
    """
    return 1.0 + (g.thermosense_intensity if thermosense_active(g, threshold) else 0.0)
