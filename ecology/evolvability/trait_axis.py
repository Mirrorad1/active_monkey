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
)

BUILTIN_AXES: dict[str, TraitAxis] = {
    "thermosense": THERMOSENSE_AXIS,
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
