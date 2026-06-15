"""
ecology.evolvability.trait_axis — generic adapter for one heritable trait.

Provides TraitAxis, a frozen dataclass for reading/writing/clamping a single
evolvable trait in a Genotype.  JSON-serializable; no engine imports.
"""
from __future__ import annotations

import dataclasses as D
from typing import Optional

from ecology.genotype import Genotype, TRAIT_BOUNDS, INT_TRAITS


@D.dataclass(frozen=True)
class TraitAxis:
    """Adapter for a single heritable trait.

    All methods are pure (no side effects, no engine imports).
    """
    name: str
    resident_value: float
    mutant_value: float
    low_value: Optional[float] = None
    high_value: Optional[float] = None
    cost_enabled: bool = True
    h_trait: str = "thermosense_intensity"
    inefficiency_trait: Optional[str] = "thermosense_inefficiency"
    inefficiency_value: float = 0.20
    freeze_flag: Optional[str] = "freeze_thermosense"
    enable_flag: str = "enable_thermosense"
    active_threshold: float = 0.05
    cost_floor: float = 0.0
    cost_inefficiency: float = 0.20
    backend: str = "thermosense"
    # EcologyConfig overrides that FULLY disconnect the trait from outcomes — every
    # channel the trait feeds into (cost AND every steering/percept term) turned off,
    # so changing the trait value cannot affect the run. The null-guard battery uses
    # this to assert byte-identical events across trait values (the anti-cheat test).
    # Declaring this dict is the key step when ADDING A NEW TRAIT AXIS: enumerate the
    # config fields that, when set, make the trait causally inert. Empty ⇒ the guard
    # falls back to {enable_flag: False}, which only removes the COST channel and is
    # usually insufficient (the percept/steering channels still leak the trait).
    disconnect_overrides: dict = D.field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, g: Genotype) -> float:
        """Return the trait value from a Genotype."""
        return getattr(g, self.h_trait)

    def clamp(self, v: float) -> float:
        """Clamp v into TRAIT_BOUNDS[h_trait]; round if it is an int trait."""
        lo, hi = TRAIT_BOUNDS[self.h_trait]
        v = float(v)
        v = max(lo, min(hi, v))
        if self.h_trait in INT_TRAITS:
            v = float(int(round(v)))
            v = max(float(int(lo)), min(float(int(hi)), v))
        return v

    def set(self, g: Genotype, v: float) -> Genotype:
        """Return a new Genotype with h_trait set to clamp(v) and inefficiency updated."""
        kw: dict = {self.h_trait: self.clamp(v)}
        if self.inefficiency_trait is not None:
            kw[self.inefficiency_trait] = self.inefficiency_value
        return D.replace(g, **kw)

    def clamp_founder(
        self,
        base: Genotype,
        v: float,
        inefficiency: Optional[float] = None,
    ) -> Genotype:
        """Like set but uses an explicit inefficiency (default self.inefficiency_value)."""
        ineff = self.inefficiency_value if inefficiency is None else inefficiency
        kw: dict = {self.h_trait: self.clamp(v)}
        if self.inefficiency_trait is not None:
            kw[self.inefficiency_trait] = ineff
        return D.replace(base, **kw)

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------

    def cost(self, h: float) -> float:
        """Upkeep cost for trait value h.  0 at or below active_threshold; monotone above."""
        if h <= self.active_threshold:
            return 0.0
        return self.cost_floor + self.cost_inefficiency * h

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain JSON-serializable dict."""
        return D.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TraitAxis":
        """Construct from a dict; unknown keys are silently ignored."""
        field_names = {f.name for f in D.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in field_names}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Built-in axes
# ---------------------------------------------------------------------------

THERMOSENSE_AXIS = TraitAxis(
    name="thermosense",
    resident_value=0.10,
    mutant_value=0.15,
    low_value=0.0,
    high_value=0.60,
    # Every channel the thermosense organ feeds into, turned off (verified by probe:
    # only with ALL of these set are runs byte-identical across thermosense_intensity).
    disconnect_overrides={
        "enable_thermosense": False,       # no upkeep cost
        "enable_food_coupling": False,     # no forage target
        "thermosense_forage_mode": False,  # no forage steering
        "thermal_avoidance_weight": 0.0,   # no avoidance steering
        "enable_temperature": False,       # no temperature field to sense/steer on
    },
)

# Phase 3: the first NON-thermosense trait. Uses backend="memory" => the binding gate
# (local_pairwise_gradient) runs through the GENERIC common garden (Phase 2.5 / PR #49),
# NOT sense_axis. memory cost lives in the engine (memory_cost_slope), not an inefficiency
# trait, so inefficiency_trait=None. No engine freeze hook for memory_horizon => freeze_flag
# None (the gate freezes via mutation_rate=0). disconnect = enable_hidden_mode off (full).
MEMORY_AXIS = TraitAxis(
    name="memory_horizon",
    resident_value=1,
    mutant_value=2,
    low_value=0,
    high_value=8,
    cost_enabled=True,
    h_trait="memory_horizon",
    inefficiency_trait=None,
    inefficiency_value=0.0,
    freeze_flag=None,
    enable_flag="enable_hidden_mode",
    active_threshold=0,
    cost_floor=0.0,
    cost_inefficiency=0.0,
    backend="memory",
    disconnect_overrides={"enable_hidden_mode": False},
)

# Phase 3 rung-1b: the CONTINUOUS analog of memory_horizon — belief_persistence (EMA weight).
# resident 0.5 -> mutant 0.55 is a genuinely SMALL eps step (vs the integer's 100% 1->2 jump),
# so this is the faithful LOCAL-gradient test. Same backend/freeze/disconnect as MEMORY_AXIS.
BELIEF_PERSISTENCE_AXIS = TraitAxis(
    name="belief_persistence",
    resident_value=0.50,
    mutant_value=0.55,
    low_value=0.0,
    high_value=0.95,
    cost_enabled=True,
    h_trait="belief_persistence",
    inefficiency_trait=None,
    inefficiency_value=0.0,
    freeze_flag=None,
    enable_flag="enable_hidden_mode",
    active_threshold=0.0,
    cost_floor=0.0,
    cost_inefficiency=0.0,
    backend="memory",
    disconnect_overrides={"enable_hidden_mode": False},
)

# Phase 4 rung-3: information_sampling_rate (active sensing probe probability).
# backend="active_sensing" => generic gate path (not thermosense).
# freeze via mutation_rate=0 (no engine freeze hook) => freeze_flag=None.
# disconnect = enable_active_sensing off (full disconnect: no probe draws, no probe cost).
ACTIVE_SENSING_AXIS = TraitAxis(
    name="information_sampling_rate",
    resident_value=0.0,
    mutant_value=0.10,
    low_value=0.0,
    high_value=1.0,
    cost_enabled=True,
    h_trait="information_sampling_rate",
    inefficiency_trait=None,
    inefficiency_value=0.0,
    freeze_flag=None,                       # freeze via mutation_rate=0 (no engine freeze hook)
    enable_flag="enable_active_sensing",
    active_threshold=0.0,
    cost_floor=0.0,
    cost_inefficiency=0.0,
    backend="active_sensing",               # non-thermosense ⇒ generic gate path
    disconnect_overrides={"enable_active_sensing": False},
)

BUILTIN_AXES: dict[str, TraitAxis] = {
    "thermosense": THERMOSENSE_AXIS,
    "memory_horizon": MEMORY_AXIS,
    "belief_persistence": BELIEF_PERSISTENCE_AXIS,
    "information_sampling_rate": ACTIVE_SENSING_AXIS,
}


def make_axis(spec: "str | dict") -> TraitAxis:
    """Resolve a TraitAxis from a string key or a dict spec.

    str  -> look up BUILTIN_AXES (raises ValueError if not found)
    dict -> TraitAxis.from_dict(spec)
    """
    if isinstance(spec, str):
        if spec not in BUILTIN_AXES:
            raise ValueError(
                f"Unknown built-in axis {spec!r}. "
                f"Available: {sorted(BUILTIN_AXES)}"
            )
        return BUILTIN_AXES[spec]
    return TraitAxis.from_dict(spec)
