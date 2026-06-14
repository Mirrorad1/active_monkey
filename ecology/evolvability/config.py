"""
ecology.evolvability.config — PreflightConfig dataclass + loaders.

Supports JSON natively; YAML only if PyYAML is installed (import is deferred
to the from_yaml() function so the module loads cleanly without pyyaml).
"""
from __future__ import annotations

import dataclasses as D
import hashlib
import json
import pathlib
from typing import Optional

from .trait_axis import TraitAxis, make_axis


# ---------------------------------------------------------------------------
# ControllerAxis
# ---------------------------------------------------------------------------

@D.dataclass(frozen=True)
class ControllerAxis:
    """Describes the ecological controller variable used as the experimental knob."""
    name: str
    low_value: float
    high_value: float
    config_field: str  # EcologyConfig field used as the controller proxy

    def to_dict(self) -> dict:
        return D.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ControllerAxis":
        field_names = {f.name for f in D.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in field_names}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# PreflightConfig
# ---------------------------------------------------------------------------

@D.dataclass(frozen=True)
class PreflightConfig:
    """Full specification for one Evolvability Preflight run."""
    slug: str
    description: str = ""
    base_scenario: str = "balanced"
    base_overrides: dict = D.field(default_factory=dict)
    founder_overrides: dict = D.field(default_factory=dict)  # Genotype trait overrides for the base founder
    trait: TraitAxis = D.field(default_factory=lambda: make_axis("thermosense"))
    controller: Optional[ControllerAxis] = None
    seeds: tuple = (0, 1, 2)
    horizon: int = 1500
    settle_steps: int = 0
    measurement_window: tuple = (100, 700)
    output_dir: str = "results/preflight"
    replicates: int = 1
    gates: tuple = (
        "gifted_benefit",
        "monomorphic_sweep",
        "local_pairwise_gradient",
        "invasion_from_rarity",
        "null_guards",
    )
    gate_params: dict = D.field(default_factory=dict)
    monomorphic_grid: tuple = (0.0, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60)
    cost_values: tuple = (0.10, 0.20, 0.40)
    win_threshold: Optional[int] = None
    lose_threshold: Optional[int] = None
    min_valid_seeds: int = 3
    min_population: int = 10
    deterministic: bool = True
    null_toggles: dict = D.field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain JSON-serializable dict."""
        d: dict = {}
        for f in D.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, TraitAxis):
                d[f.name] = v.to_dict()
            elif isinstance(v, ControllerAxis):
                d[f.name] = v.to_dict()
            elif isinstance(v, tuple):
                d[f.name] = list(v)
            else:
                d[f.name] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PreflightConfig":
        """Build a PreflightConfig from a dict; unknown keys are ignored."""
        field_names = {f.name for f in D.fields(cls)}
        kw: dict = {}

        for f in D.fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]

            if f.name == "trait":
                if isinstance(v, str):
                    kw["trait"] = make_axis(v)
                elif isinstance(v, dict):
                    kw["trait"] = TraitAxis.from_dict(v)
                else:
                    kw["trait"] = v

            elif f.name == "controller":
                if v is None:
                    kw["controller"] = None
                elif isinstance(v, dict):
                    kw["controller"] = ControllerAxis.from_dict(v)
                else:
                    kw["controller"] = v

            elif f.name in ("seeds", "measurement_window", "gates",
                            "monomorphic_grid", "cost_values"):
                kw[f.name] = tuple(v) if isinstance(v, list) else v

            elif f.name in ("base_overrides", "gate_params", "null_toggles", "founder_overrides"):
                kw[f.name] = dict(v) if v is not None else {}

            elif f.name in field_names:
                kw[f.name] = v

        return cls(**kw)

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    def config_hash(self) -> str:
        """SHA-256 hex digest of the canonical JSON representation."""
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------

    def effective_thresholds(self) -> tuple:
        """(win_threshold, lose_threshold) — derived from seeds length if not set."""
        if self.win_threshold is not None and self.lose_threshold is not None:
            return (self.win_threshold, self.lose_threshold)
        from .metrics import default_thresholds
        return default_thresholds(len(self.seeds))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def from_yaml(path) -> PreflightConfig:
    """Load a PreflightConfig from a YAML file.

    Requires PyYAML.  Raises RuntimeError if it is not installed.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "PyYAML not installed; use a .json config or `uv pip install pyyaml`"
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return PreflightConfig.from_dict(data)


def from_json(path) -> PreflightConfig:
    """Load a PreflightConfig from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return PreflightConfig.from_dict(data)


def load_config(path) -> PreflightConfig:
    """Dispatch to from_yaml or from_json based on file suffix."""
    p = pathlib.Path(path)
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return from_yaml(p)
    elif suffix == ".json":
        return from_json(p)
    else:
        raise ValueError(
            f"Unsupported config file extension {suffix!r}. "
            "Use .json, .yaml, or .yml."
        )
