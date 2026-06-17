"""Substrate-independent AgentState / checkpoint abstraction (Artifact Infrastructure M9).

This module is the portable spine for *copyable agent artifacts*.  It is deliberately
free of any agent-runtime import (no ``pymdp``, no JAX) so that saving, loading,
hashing, and inspecting an artifact works on a fresh clone even when the agent's
inference engine is not installed.  Numeric payloads use **safetensors** (never
pickle); non-tensor metadata is canonical (sorted-key) JSON.

What "AgentState" represents (substrate-independent):
  - architecture_id / agent_class  — what the thing is
  - tensors                        — parameter tensors / probability tables / learned
                                     count tensors (Dirichlet pseudo-counts), keyed by name
  - belief_state                   — current posterior over hidden state, if present
  - history_hashes                 — hashes of observation/action history (not raw logs)
  - rng_state                      — seedable RNG state (list[int], deterministic)
  - provenance                     — repo commit, source experiment ids, created_at
  - scorer_compat                  — which frozen scorer this state is meant to be judged by
  - schema_version                 — for graceful migration / mismatch refusal
  - metadata                       — arbitrary JSON-serializable extras

Honest framing: "weights" here means probability tables / Dirichlet counts /
generative-model tensors, NOT neural-network weights.  "belief" means a posterior
distribution over a hidden state, NOT subjective belief.  Nothing here claims
sentience, feeling, or understanding.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# safetensors is an explicit dependency (see pyproject.toml).  Public artifacts must
# never use pickle; numeric payloads go through safetensors.numpy.
from safetensors.numpy import save_file as _st_save_file
from safetensors.numpy import load_file as _st_load_file

# Bump only with a documented migration.  load() refuses a mismatched MAJOR.MINOR
# unless explicitly allowed (graceful failure on schema mismatch).
SCHEMA_VERSION = "0.1.0"


# ── Canonical JSON helpers ───────────────────────────────────────────────────

def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, ensure_ascii.

    Two equal Python structures always serialize to byte-identical strings, which is
    what makes content hashing stable across processes and machines.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_array(arr: np.ndarray) -> str:
    """Stable content hash of a single array (dtype + shape + C-contiguous bytes)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


class SchemaMismatch(ValueError):
    """Raised when an artifact's schema_version is incompatible and not explicitly allowed."""


# ── Provenance / compatibility records ───────────────────────────────────────

@dataclass
class AgentProvenance:
    """Where this state came from (best-effort; 'unknown' is an honest default)."""

    repo_commit: str = "unknown"
    source_experiments: list[int] = field(default_factory=list)
    created_at: str = ""  # ISO-8601 UTC; filled by now() if empty
    source_repo: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "repo_commit": self.repo_commit,
            "source_experiments": list(self.source_experiments),
            "created_at": self.created_at,
            "source_repo": self.source_repo,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentProvenance":
        return cls(
            repo_commit=d.get("repo_commit", "unknown"),
            source_experiments=list(d.get("source_experiments", [])),
            created_at=d.get("created_at", ""),
            source_repo=d.get("source_repo", ""),
            notes=d.get("notes", ""),
        )

    @staticmethod
    def now() -> str:
        return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ScorerCompatibility:
    """The frozen scorer this state is meant to be evaluated by, pinned by hash."""

    scorer_path: str
    scorer_hash: str
    scorer_version: str = "affect-score-1e"
    metric_name: str = "mean_last_third_pos"

    def to_dict(self) -> dict:
        return {
            "scorer_path": self.scorer_path,
            "scorer_hash": self.scorer_hash,
            "scorer_version": self.scorer_version,
            "metric_name": self.metric_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScorerCompatibility":
        return cls(
            scorer_path=d["scorer_path"],
            scorer_hash=d["scorer_hash"],
            scorer_version=d.get("scorer_version", "affect-score-1e"),
            metric_name=d.get("metric_name", "mean_last_third_pos"),
        )


# ── The core state object ────────────────────────────────────────────────────

@dataclass
class AgentState:
    """A portable snapshot of one agent's numeric content + provenance.

    tensors: all numeric arrays (parameter tables, Dirichlet counts) keyed by name.
    belief_state: current posterior arrays, if present (kept separate from parameters).
    history_hashes: e.g. {"observations": sha, "actions": sha} — NOT raw logs.
    rng_state: a deterministic list[int] (e.g. a JAX PRNGKey serialized), or None.
    """

    architecture_id: str
    agent_class: str
    tensors: dict[str, np.ndarray] = field(default_factory=dict)
    belief_state: dict[str, np.ndarray] | None = None
    history_hashes: dict[str, str] = field(default_factory=dict)
    rng_state: list[int] | None = None
    provenance: AgentProvenance = field(default_factory=AgentProvenance)
    scorer_compat: ScorerCompatibility | None = None
    schema_version: str = SCHEMA_VERSION
    metadata: dict = field(default_factory=dict)

    # ── derived views ────────────────────────────────────────────────────────

    def _all_arrays(self) -> dict[str, np.ndarray]:
        """Flat name->array map for serialization; belief arrays get a 'belief::' prefix."""
        out: dict[str, np.ndarray] = {k: np.ascontiguousarray(v) for k, v in self.tensors.items()}
        if self.belief_state:
            for k, v in self.belief_state.items():
                out[f"belief::{k}"] = np.ascontiguousarray(v)
        return out

    def metadata_view(self, include_volatile: bool = True) -> dict:
        """The non-tensor metadata as a plain dict.

        include_volatile=False drops created_at / repo_commit so that the *scientific
        content* of two independent exports hashes identically (stable content hash).
        """
        prov = self.provenance.to_dict()
        if not include_volatile:
            prov = {k: v for k, v in prov.items() if k not in ("created_at", "repo_commit")}
        return {
            "architecture_id": self.architecture_id,
            "agent_class": self.agent_class,
            "schema_version": self.schema_version,
            "tensor_keys": sorted(self.tensors.keys()),
            "belief_keys": sorted(self.belief_state.keys()) if self.belief_state else [],
            "history_hashes": dict(self.history_hashes),
            "rng_state": list(self.rng_state) if self.rng_state is not None else None,
            "provenance": prov,
            "scorer_compat": self.scorer_compat.to_dict() if self.scorer_compat else None,
            "metadata": self.metadata,
        }

    def content_hash(self) -> str:
        """Stable sha256 over scientific content (metadata sans volatile fields + tensors).

        Excludes created_at / repo_commit so identical agents exported at different
        times hash identically.  Tensor bytes are included in sorted-key order.
        """
        h = hashlib.sha256()
        h.update(canonical_json(self.metadata_view(include_volatile=False)).encode())
        for k in sorted(self._all_arrays().keys()):
            h.update(k.encode())
            h.update(hash_array(self._all_arrays()[k]).encode())
        return h.hexdigest()

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str | Path, stem: str = "state") -> dict[str, Path]:
        """Write <stem>.safetensors (numeric) + <stem>.config.json (metadata).

        Returns the paths written.  The safetensors header also embeds schema_version
        and content_hash so the tensor file is self-describing.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        st_path = directory / f"{stem}.safetensors"
        cfg_path = directory / f"{stem}.config.json"

        arrays = self._all_arrays()
        if not arrays:
            # safetensors refuses an empty tensor dict; store a zero-byte sentinel.
            arrays = {"__empty__": np.zeros((0,), dtype=np.float32)}
        st_meta = {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash(),
            "agent_class": self.agent_class,
        }
        _st_save_file(arrays, str(st_path), metadata={k: str(v) for k, v in st_meta.items()})

        cfg = self.metadata_view(include_volatile=True)
        cfg["content_hash"] = self.content_hash()
        cfg_path.write_text(canonical_json(cfg) + "\n")
        return {"tensors": st_path, "config": cfg_path}

    @classmethod
    def load(
        cls,
        directory: str | Path,
        stem: str = "state",
        allow_schema_mismatch: bool = False,
    ) -> "AgentState":
        """Load an AgentState written by save(); refuse incompatible schema.

        Raises FileNotFoundError if the config/tensor files are missing, and
        SchemaMismatch if the stored MAJOR.MINOR differs from this module's
        SCHEMA_VERSION unless allow_schema_mismatch=True.
        """
        directory = Path(directory)
        cfg_path = directory / f"{stem}.config.json"
        st_path = directory / f"{stem}.safetensors"
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing config: {cfg_path}")
        if not st_path.exists():
            raise FileNotFoundError(f"missing tensors: {st_path}")

        cfg = json.loads(cfg_path.read_text())
        stored_schema = cfg.get("schema_version", "unknown")
        _check_schema(stored_schema, allow_schema_mismatch)

        arrays = _st_load_file(str(st_path))
        arrays.pop("__empty__", None)
        tensors: dict[str, np.ndarray] = {}
        belief: dict[str, np.ndarray] = {}
        for k, v in arrays.items():
            if k.startswith("belief::"):
                belief[k[len("belief::"):]] = v
            else:
                tensors[k] = v

        prov = AgentProvenance.from_dict(cfg.get("provenance", {}))
        sc = cfg.get("scorer_compat")
        return cls(
            architecture_id=cfg["architecture_id"],
            agent_class=cfg["agent_class"],
            tensors=tensors,
            belief_state=belief or None,
            history_hashes=dict(cfg.get("history_hashes", {})),
            rng_state=cfg.get("rng_state"),
            provenance=prov,
            scorer_compat=ScorerCompatibility.from_dict(sc) if sc else None,
            schema_version=stored_schema,
            metadata=cfg.get("metadata", {}),
        )


# ── Checkpoint (a named AgentState + its on-disk location) ───────────────────

@dataclass
class AgentCheckpoint:
    """A named AgentState — e.g. 'init' or 'learned_example' — with its files."""

    name: str
    state: AgentState
    tensors_path: Path | None = None
    config_path: Path | None = None

    def save(self, directory: str | Path) -> "AgentCheckpoint":
        paths = self.state.save(directory, stem=self.name)
        self.tensors_path = paths["tensors"]
        self.config_path = paths["config"]
        return self

    @classmethod
    def load(cls, directory: str | Path, name: str, allow_schema_mismatch: bool = False) -> "AgentCheckpoint":
        state = AgentState.load(directory, stem=name, allow_schema_mismatch=allow_schema_mismatch)
        directory = Path(directory)
        return cls(
            name=name,
            state=state,
            tensors_path=directory / f"{name}.safetensors",
            config_path=directory / f"{name}.config.json",
        )

    def content_hash(self) -> str:
        return self.state.content_hash()


# ── Manifest ─────────────────────────────────────────────────────────────────

# Minimum required keys for a well-formed artifact manifest.
REQUIRED_MANIFEST_KEYS = (
    "artifact_id",
    "schema_version",
    "agent_class",
    "frozen_scorer",
    "scorer_hash",
    "init_checkpoint_hash",
)


@dataclass
class ArtifactManifest:
    """The artifact-level manifest.json (validated on load)."""

    data: dict

    def validate(self) -> None:
        missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in self.data]
        if missing:
            raise ValueError(f"manifest missing required keys: {missing}")
        _check_schema(self.data.get("schema_version", "unknown"), allow_mismatch=False)

    def to_json(self) -> str:
        return canonical_json(self.data)

    @classmethod
    def from_path(cls, path: str | Path, allow_schema_mismatch: bool = False) -> "ArtifactManifest":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"missing manifest: {path}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"corrupt manifest JSON at {path}: {e}") from e
        m = cls(data)
        # validate required keys; schema check honors the allow flag
        missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in data]
        if missing:
            raise ValueError(f"manifest missing required keys: {missing}")
        _check_schema(data.get("schema_version", "unknown"), allow_schema_mismatch)
        return m


# ── schema gate ──────────────────────────────────────────────────────────────

def _check_schema(stored: str, allow_mismatch: bool) -> None:
    """Refuse a MAJOR.MINOR mismatch unless explicitly allowed."""
    if allow_mismatch:
        return
    if stored == SCHEMA_VERSION:
        return
    cur = SCHEMA_VERSION.split(".")
    got = str(stored).split(".")
    if len(got) >= 2 and got[:2] == cur[:2]:
        return  # same MAJOR.MINOR, patch differs -> compatible
    raise SchemaMismatch(
        f"artifact schema {stored!r} is incompatible with loader {SCHEMA_VERSION!r}; "
        f"pass allow_schema_mismatch=True to override (at your own risk)."
    )
