"""Validation layer for coalescence artifacts."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from active_loop.state import SchemaMismatch, sha256_bytes, ArtifactManifest, REQUIRED_MANIFEST_KEYS
from active_loop.artifacts import hash_file
from active_loop.coalescence import schema as _schema
from active_loop.coalescence.schema import (
    ARTIFACT_REGISTRY,
    SCHEMA_VERSION,
    check_schema_version,
    load,
    read_json,
)

# Re-export for convenience
__all__ = [
    "MissingFieldError",
    "required_fields",
    "validate_dict",
    "validate_artifact_file",
    "validate_bundle",
    "validate_all",
    "sha256_bytes",
    "hash_file",
]


class MissingFieldError(ValueError):
    def __init__(self, missing: list, artifact_type: str = ""):
        self.missing = missing
        msg = f"artifact {artifact_type!r} missing required fields: {missing}"
        super().__init__(msg)


def required_fields(artifact_type: str) -> tuple:
    cls = ARTIFACT_REGISTRY.get(artifact_type)
    if cls is None:
        raise ValueError(f"unknown artifact_type {artifact_type!r}")
    result = []
    for f in dataclasses.fields(cls):
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            # schema_version has a default string; ARTIFACT_TYPE is init=False
            if f.init:
                result.append(f.name)
    return tuple(result)


def validate_dict(d: dict, artifact_type: str = None) -> dict:
    if artifact_type is None:
        artifact_type = d.get("artifact_type")
    if not artifact_type:
        raise ValueError("artifact_type missing from dict and not provided")
    if artifact_type not in ARTIFACT_REGISTRY:
        raise ValueError(f"unknown artifact_type {artifact_type!r}")
    check_schema_version(d.get("schema_version", SCHEMA_VERSION))
    req = required_fields(artifact_type)
    missing = [k for k in req if k not in d]
    if missing:
        raise MissingFieldError(missing, artifact_type)
    return d


def validate_artifact_file(path: Any, allow_schema_mismatch: bool = False) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"artifact file not found: {path}")
    d = load(path)
    return validate_dict(d)


def validate_bundle(bundle_dir: Any, allow_schema_mismatch: bool = False) -> dict:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {bundle_dir}")
    try:
        manifest = read_json(manifest_path)
    except Exception as e:
        raise ValueError(f"corrupt manifest.json at {manifest_path}: {e}") from e
    validate_dict(manifest, artifact_type="experiment_bundle")

    ref_lists = ["source_files", "raw_data_refs", "metrics_refs", "scorer_refs", "state_refs"]
    checked = 0
    missing_refs = []
    repo_root = Path(".")
    for key in ref_lists:
        for ref in manifest.get(key, []):
            checked += 1
            rel_bundle = bundle_dir / ref
            rel_repo = repo_root / ref
            if not rel_bundle.exists() and not rel_repo.exists():
                missing_refs.append(ref)
    if missing_refs:
        raise FileNotFoundError(
            f"bundle {bundle_dir} claims refs that do not exist: {missing_refs}"
        )

    return {"ok": True, "artifact_type": "experiment_bundle", "checked_refs": checked}


def validate_all(root: str = ".") -> dict:
    root_path = Path(root)
    passed = []
    failed = []
    skipped = []
    search_dirs = [
        root_path / "experiment_bundles",
        root_path / "mechanisms",
        root_path / "geometry_maps",
        root_path / "boundary_notes",
        root_path / "artifacts",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix not in (".yaml", ".yml", ".json"):
                continue
            try:
                raw = load(p)
            except Exception:
                skipped.append(str(p))
                continue
            if not isinstance(raw, dict):
                # e.g. a plain-list scorer_refs.json
                skipped.append(str(p))
                continue
            artifact_type = raw.get("artifact_type")
            if artifact_type in ARTIFACT_REGISTRY:
                # Coalescence artifact — validate fully
                try:
                    validate_dict(raw, artifact_type=artifact_type)
                    passed.append(str(p))
                except Exception as e:
                    failed.append({"path": str(p), "error": str(e)})
            elif p.name == "manifest.json" and all(k in raw for k in REQUIRED_MANIFEST_KEYS):
                # Checkpoint artifact manifest — delegate to ArtifactManifest
                try:
                    ArtifactManifest.from_path(p)
                    passed.append(str(p))
                except Exception as e:
                    failed.append({"path": str(p), "error": str(e)})
            else:
                # Plain data / config / sidecar — not an artifact, skip silently
                skipped.append(str(p))
    return {"passed": passed, "failed": failed, "skipped": skipped}


# make Any importable without re-importing typing in callers
from typing import Any  # noqa: E402
