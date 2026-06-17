"""Schema definitions for the coalescence artifact layer."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from active_loop.state import (
    SchemaMismatch,
    canonical_json,
    sha256_bytes,
    _check_schema as _state_check_schema,
)
from active_loop.artifacts import hash_file, repo_commit  # noqa: F401 (re-exported)

SCHEMA_VERSION = "0.1.0"

CONFIDENCE_LEVELS = ("high", "medium", "low", "unknown")

BACKFILL_LEVEL_NAMES = (
    "index_only",
    "summary_bundle",
    "metrics_bundle",
    "repro_bundle",
    "trajectory_bundle",
    "checkpoint_bundle",
    "mechanism_bundle",
)

MECHANISM_STATUS = ("validated", "falsified", "constrained", "speculative", "scaffold")

MECHANISM_TYPES = (
    "functional-valence-learning",
    "hidden-state-belief",
    "costed-sensing",
    "uncertainty-gated-probing",
    "costed-signaling",
    "selection-stabilized-trait",
    "transfer-invariant-abstraction",
    "identity-self-modeling",
)


def backfill_level_name(i: int) -> str:
    return BACKFILL_LEVEL_NAMES[i]


def backfill_level_index(name: str) -> int:
    return BACKFILL_LEVEL_NAMES.index(name)


# ── Serialization helpers ────────────────────────────────────────────────────

def to_canonical_json(obj: Any) -> str:
    return canonical_json(obj)


def write_json(obj_dict: dict, path: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj_dict) + "\n", encoding="utf-8")


def read_json(path: Any) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(obj_dict: dict, path: Any) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.dump(obj_dict, sort_keys=True, default_flow_style=False),
                encoding="utf-8",
            )
            return
        except ImportError:
            pass
    write_json(obj_dict, path)


def load(path: Any) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        except ImportError:
            return read_json(path)
    return read_json(path)


def check_schema_version(stored: str, allow_mismatch: bool = False) -> None:
    _state_check_schema(stored, allow_mismatch)


# ── Dataclass base helpers ───────────────────────────────────────────────────

def _from_dict_kwargs(cls, d: dict) -> dict:
    """Return only the kwargs that match known fields, using defaults for missing optional ones."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in field_names}


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentBundle:
    """A distilled record of one experiment: question, result, verdict, and refs."""

    ARTIFACT_TYPE: str = dataclasses.field(default="experiment_bundle", init=False, repr=False)

    # REQUIRED
    experiment_id: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    direction: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    question: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    hypothesis: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    status: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    verdict: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    repo_commit: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    created_at: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    confidence: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    backfill_level: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    # OPTIONAL
    source_files: list = field(default_factory=list)
    raw_data_refs: list = field(default_factory=list)
    metrics_refs: list = field(default_factory=list)
    scorer_refs: list = field(default_factory=list)
    state_refs: list = field(default_factory=list)
    mechanism_refs: list = field(default_factory=list)
    geometry_refs: list = field(default_factory=list)
    caveats: list = field(default_factory=list)
    reproduction_command: Optional[str] = field(default=None)
    schema_version: str = field(default=SCHEMA_VERSION)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["artifact_type"] = self.ARTIFACT_TYPE
        d.pop("ARTIFACT_TYPE", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, allow_schema_mismatch: bool = False) -> "ExperimentBundle":
        check_schema_version(d.get("schema_version", SCHEMA_VERSION), allow_schema_mismatch)
        kw = _from_dict_kwargs(cls, d)
        kw.pop("ARTIFACT_TYPE", None)
        return cls(**kw)


@dataclass
class ExperimentSpec:
    """Formal spec of variables, dynamics, and stop/pass/fail conditions for an experiment."""

    ARTIFACT_TYPE: str = dataclasses.field(default="experiment_spec", init=False, repr=False)

    # REQUIRED
    experiment_id: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    # OPTIONAL
    variables: Any = field(default=None)
    hidden_state: Any = field(default=None)
    observations: Any = field(default=None)
    actions: Any = field(default=None)
    reward_or_fitness: Any = field(default=None)
    costs: Any = field(default=None)
    update_rules: Any = field(default=None)
    selection_rules: Any = field(default=None)
    environment_dynamics: Any = field(default=None)
    agent_dynamics: Any = field(default=None)
    stop_condition: Any = field(default=None)
    pass_condition: Any = field(default=None)
    fail_condition: Any = field(default=None)
    schema_version: str = field(default=SCHEMA_VERSION)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["artifact_type"] = self.ARTIFACT_TYPE
        d.pop("ARTIFACT_TYPE", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, allow_schema_mismatch: bool = False) -> "ExperimentSpec":
        check_schema_version(d.get("schema_version", SCHEMA_VERSION), allow_schema_mismatch)
        kw = _from_dict_kwargs(cls, d)
        kw.pop("ARTIFACT_TYPE", None)
        return cls(**kw)


@dataclass
class TrajectoryRow:
    """One row in a trajectory log.

    Parquet and CSV are acceptable alternates for bulk storage, but JSONL is the
    canonical interchange format for this layer.  Old experiments that lack raw
    trajectories MUST NOT be forced into this format — leave raw_data_refs empty.
    """

    ARTIFACT_TYPE: str = dataclasses.field(default="trajectory_row", init=False, repr=False)

    # REQUIRED
    experiment_id: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    seed: int = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    t: int = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    # OPTIONAL
    direction: Optional[str] = field(default=None)
    episode: Optional[int] = field(default=None)
    agent_id: Optional[str] = field(default=None)
    environment_id: Optional[str] = field(default=None)
    hidden_state: Any = field(default=None)
    observation: Any = field(default=None)
    belief_state: Any = field(default=None)
    action: Any = field(default=None)
    message: Any = field(default=None)
    reward_or_valence: Optional[float] = field(default=None)
    fitness: Optional[float] = field(default=None)
    next_observation: Any = field(default=None)
    next_hidden_state: Any = field(default=None)
    metadata: dict = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["artifact_type"] = self.ARTIFACT_TYPE
        d.pop("ARTIFACT_TYPE", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, allow_schema_mismatch: bool = False) -> "TrajectoryRow":
        check_schema_version(d.get("schema_version", SCHEMA_VERSION), allow_schema_mismatch)
        kw = _from_dict_kwargs(cls, d)
        kw.pop("ARTIFACT_TYPE", None)
        return cls(**kw)


_TRAJECTORY_REQUIRED = ("experiment_id", "seed", "t")


def write_trajectory_jsonl(rows: Iterable, path: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, TrajectoryRow):
                fh.write(canonical_json(row.to_dict()) + "\n")
            else:
                fh.write(canonical_json(row) + "\n")


def read_trajectory_jsonl(path: Any) -> list:
    path = Path(path)
    result = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


def validate_trajectory_row(d: dict) -> list:
    """Return list of missing required keys; empty list means valid."""
    return [k for k in _TRAJECTORY_REQUIRED if k not in d]


@dataclass
class ScorerCard:
    """Provenance record for a frozen scorer file."""

    ARTIFACT_TYPE: str = dataclasses.field(default="scorer_card", init=False, repr=False)

    # REQUIRED
    scorer_id: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    scorer_version: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    file_path: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    sha256: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    # OPTIONAL
    metrics: list = field(default_factory=list)
    required_controls: list = field(default_factory=list)
    pass_conditions: list = field(default_factory=list)
    fail_conditions: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    schema_version: str = field(default=SCHEMA_VERSION)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["artifact_type"] = self.ARTIFACT_TYPE
        d.pop("ARTIFACT_TYPE", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, allow_schema_mismatch: bool = False) -> "ScorerCard":
        check_schema_version(d.get("schema_version", SCHEMA_VERSION), allow_schema_mismatch)
        kw = _from_dict_kwargs(cls, d)
        kw.pop("ARTIFACT_TYPE", None)
        return cls(**kw)

    @classmethod
    def from_file(
        cls,
        file_path: Any,
        scorer_id: str,
        scorer_version: str,
        repo: Any = ".",
        **kw: Any,
    ) -> "ScorerCard":
        fp = Path(repo) / file_path
        h = hash_file(fp)
        return cls(
            scorer_id=scorer_id,
            scorer_version=scorer_version,
            file_path=str(file_path),
            sha256=h,
            **kw,
        )


@dataclass
class MechanismCard:
    """A reusable mechanistic claim distilled from one or more experiments."""

    ARTIFACT_TYPE: str = dataclasses.field(default="mechanism_card", init=False, repr=False)

    # REQUIRED
    mechanism_id: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    mechanism_type: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    claim: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    status: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    source_experiments: list = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    # OPTIONAL
    works_when: list = field(default_factory=list)
    fails_when: list = field(default_factory=list)
    required_conditions: list = field(default_factory=list)
    reusable_interface: Any = field(default=None)
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    state_requirements: list = field(default_factory=list)
    costs: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    falsifiers: list = field(default_factory=list)
    known_confounds: list = field(default_factory=list)
    next_compositions: list = field(default_factory=list)
    schema_version: str = field(default=SCHEMA_VERSION)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["artifact_type"] = self.ARTIFACT_TYPE
        d.pop("ARTIFACT_TYPE", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, allow_schema_mismatch: bool = False) -> "MechanismCard":
        check_schema_version(d.get("schema_version", SCHEMA_VERSION), allow_schema_mismatch)
        kw = _from_dict_kwargs(cls, d)
        kw.pop("ARTIFACT_TYPE", None)
        return cls(**kw)


@dataclass
class GeometryMap:
    """A map of parameter regions where a mechanism holds or fails."""

    ARTIFACT_TYPE: str = dataclasses.field(default="geometry_map", init=False, repr=False)

    # REQUIRED
    geometry_id: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    source_experiments: list = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    # OPTIONAL
    mechanism_id: Optional[str] = field(default=None)
    swept_parameters: Any = field(default=None)
    fixed_parameters: dict = field(default_factory=dict)
    metrics: list = field(default_factory=list)
    outcome_regions: list = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    negative_regions: list = field(default_factory=list)
    positive_regions: list = field(default_factory=list)
    confounds_excluded: list = field(default_factory=list)
    next_experiment_suggestions: list = field(default_factory=list)
    schema_version: str = field(default=SCHEMA_VERSION)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["artifact_type"] = self.ARTIFACT_TYPE
        d.pop("ARTIFACT_TYPE", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, allow_schema_mismatch: bool = False) -> "GeometryMap":
        check_schema_version(d.get("schema_version", SCHEMA_VERSION), allow_schema_mismatch)
        kw = _from_dict_kwargs(cls, d)
        kw.pop("ARTIFACT_TYPE", None)
        return cls(**kw)


@dataclass
class AdapterCard:
    """Describes how to compose two mechanisms."""

    ARTIFACT_TYPE: str = dataclasses.field(default="adapter_card", init=False, repr=False)

    # REQUIRED
    adapter_id: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    from_mechanism: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    to_mechanism: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    # OPTIONAL
    input_contract: Any = field(default=None)
    output_contract: Any = field(default=None)
    required_state: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)
    tests: list = field(default_factory=list)
    schema_version: str = field(default=SCHEMA_VERSION)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["artifact_type"] = self.ARTIFACT_TYPE
        d.pop("ARTIFACT_TYPE", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, allow_schema_mismatch: bool = False) -> "AdapterCard":
        check_schema_version(d.get("schema_version", SCHEMA_VERSION), allow_schema_mismatch)
        kw = _from_dict_kwargs(cls, d)
        kw.pop("ARTIFACT_TYPE", None)
        return cls(**kw)


@dataclass
class BoundaryNote:
    """Documents where a mechanism fails and what that implies."""

    ARTIFACT_TYPE: str = dataclasses.field(default="boundary_note", init=False, repr=False)

    # REQUIRED
    boundary_id: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    source_experiments: list = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    failed_mechanism: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    observed_failure: str = field(default=dataclasses.MISSING)  # type: ignore[assignment]
    # OPTIONAL
    tested_conditions: list = field(default_factory=list)
    excluded_confounds: list = field(default_factory=list)
    implication: Optional[str] = field(default=None)
    next_safe_region_to_test: Optional[str] = field(default=None)
    schema_version: str = field(default=SCHEMA_VERSION)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["artifact_type"] = self.ARTIFACT_TYPE
        d.pop("ARTIFACT_TYPE", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, allow_schema_mismatch: bool = False) -> "BoundaryNote":
        check_schema_version(d.get("schema_version", SCHEMA_VERSION), allow_schema_mismatch)
        kw = _from_dict_kwargs(cls, d)
        kw.pop("ARTIFACT_TYPE", None)
        return cls(**kw)


ARTIFACT_REGISTRY: dict = {
    "experiment_bundle": ExperimentBundle,
    "experiment_spec": ExperimentSpec,
    "trajectory_row": TrajectoryRow,
    "scorer_card": ScorerCard,
    "mechanism_card": MechanismCard,
    "geometry_map": GeometryMap,
    "adapter_card": AdapterCard,
    "boundary_note": BoundaryNote,
}
