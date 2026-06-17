"""Tests for active_loop.coalescence.export.

All tests use real repo data (run from repo root).  No network; deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from active_loop.coalescence.export import export_bundle
from active_loop.coalescence.validate import validate_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_export_exp222_checkpoint(tmp_path):
    """exp222 at checkpoint_bundle: manifest written, bundle validates, state_refs ok."""
    bundle_dir = tmp_path / "exp222_bundle"
    manifest = export_bundle("exp222", "checkpoint_bundle", bundle_dir)

    assert (bundle_dir / "manifest.json").exists(), "manifest.json not written"

    result = validate_bundle(bundle_dir)
    assert result["ok"] is True, f"validate_bundle failed: {result}"

    # state_refs must reference the affect-dyad artifact
    state_refs = manifest.get("state_refs", [])
    assert any("affect-dyad" in ref for ref in state_refs), (
        f"state_refs does not reference affect-dyad artifact: {state_refs}"
    )


def test_export_refuses_overclaim(tmp_path):
    """exp210 has no raw trajectories; exporting at trajectory_bundle must raise ValueError."""
    bundle_dir = tmp_path / "exp210_bundle"
    with pytest.raises(ValueError, match="cannot export exp210"):
        export_bundle("exp210", "trajectory_bundle", bundle_dir)


def test_export_metrics_bundle_no_raw_claim(tmp_path):
    """exp217 at metrics_bundle must have empty raw_data_refs (no trajectories)."""
    bundle_dir = tmp_path / "exp217_bundle"
    manifest = export_bundle("exp217", "metrics_bundle", bundle_dir)

    assert manifest["raw_data_refs"] == [], (
        f"raw_data_refs should be empty at metrics_bundle: {manifest['raw_data_refs']}"
    )
    result = validate_bundle(bundle_dir)
    assert result["ok"] is True


def test_exported_bundle_validates(tmp_path):
    """exp199 at trajectory_bundle: validate_bundle passes; raw_data_refs non-empty and all exist."""
    bundle_dir = tmp_path / "exp199_bundle"
    manifest = export_bundle("exp199", "trajectory_bundle", bundle_dir)

    result = validate_bundle(bundle_dir)
    assert result["ok"] is True, f"validate_bundle failed: {result}"

    raw_refs = manifest.get("raw_data_refs", [])
    assert raw_refs, "raw_data_refs must be non-empty for exp199 at trajectory_bundle"

    repo_root = Path(".")
    for ref in raw_refs:
        assert (repo_root / ref).exists(), f"raw_data_ref does not exist: {ref}"


def test_refs_are_original_paths(tmp_path):
    """source_files in the exported manifest point under 'experiments/', not into the bundle."""
    bundle_dir = tmp_path / "exp222_src_test"
    manifest = export_bundle("exp222", "checkpoint_bundle", bundle_dir)

    source_files = manifest.get("source_files", [])
    assert source_files, "exp222 must have at least one source_file"

    for ref in source_files:
        assert ref.startswith("experiments/"), (
            f"source_file {ref!r} does not start with 'experiments/' — "
            "raw data must be referenced in place, not copied into the bundle"
        )
        # Must NOT be inside the bundle dir
        assert not ref.startswith(str(bundle_dir)), (
            f"source_file {ref!r} appears to be inside the bundle dir"
        )
