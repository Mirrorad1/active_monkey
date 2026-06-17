"""Tests for the hardened validate_all dispatch logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from active_loop.coalescence.schema import (
    SCHEMA_VERSION,
    MechanismCard,
    dump,
)
from active_loop.coalescence.validate import validate_all, validate_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mechanism_card_dict() -> dict:
    return MechanismCard(
        mechanism_id="m-test-01",
        mechanism_type="functional-valence-learning",
        claim="agent learns functional valence",
        status="validated",
        source_experiments=["exp001"],
    ).to_dict()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_validate_all_passes_existing_checkpoint_artifact():
    """The checkpoint artifact manifest.json must appear in 'passed', NOT 'failed'.
    Its config.json and eval_results/*.json must appear in 'skipped'."""
    result = validate_all(".")
    passed = result["passed"]
    failed = result["failed"]
    skipped = result["skipped"]

    manifest_path = "artifacts/active-monkey-affect-dyad-v0/manifest.json"
    # Find by suffix since paths may be relative to cwd or absolute
    passed_suffixes = [p.replace("\\", "/") for p in passed]
    failed_paths = [f["path"].replace("\\", "/") for f in failed]
    skipped_paths = [s.replace("\\", "/") for s in skipped]

    assert any(manifest_path in p for p in passed_suffixes), (
        f"manifest.json not in passed; passed={passed_suffixes[:5]}, failed={failed_paths[:5]}"
    )
    assert not any(manifest_path in p for p in failed_paths), (
        f"manifest.json unexpectedly in failed: {failed_paths}"
    )

    # config.json and eval_results files should be skipped, not failed
    config_path = "artifacts/active-monkey-affect-dyad-v0/config.json"
    assert any(config_path in s for s in skipped_paths), (
        f"config.json not in skipped; skipped={skipped_paths[:10]}"
    )
    # At least one eval_results file should be skipped
    assert any("eval_results" in s for s in skipped_paths), (
        f"eval_results files not in skipped; skipped={skipped_paths[:10]}"
    )


def test_validate_all_no_failures_on_clean_repo():
    """The current repo state must produce zero failures."""
    result = validate_all(".")
    assert result["failed"] == [], (
        f"Unexpected failures: {result['failed']}"
    )


def test_bundle_missing_ref_rejected(tmp_path):
    """A bundle manifest that references a nonexistent raw_data_ref raises FileNotFoundError."""
    bundle_dir = tmp_path / "experiment_bundles" / "exp_test"
    bundle_dir.mkdir(parents=True)
    manifest = {
        "artifact_type": "experiment_bundle",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "exp_test",
        "direction": "valence-learning",
        "question": "q",
        "hypothesis": "h",
        "status": "complete",
        "verdict": "CONFIRMED",
        "repo_commit": "abc",
        "created_at": "2026-01-01T00:00:00Z",
        "confidence": "high",
        "backfill_level": "metrics_bundle",
        "raw_data_refs": ["nonexistent_file.json"],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        validate_bundle(bundle_dir)


def test_unknown_artifact_type_skipped_not_failed(tmp_path):
    """A stray data.json with no 'artifact_type' is skipped, not failed."""
    artifacts_dir = tmp_path / "artifacts" / "some-stray"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "data.json").write_text(json.dumps({"foo": 1}), encoding="utf-8")
    result = validate_all(str(tmp_path))
    assert result["failed"] == []
    skipped = [s.replace("\\", "/") for s in result["skipped"]]
    assert any("data.json" in s for s in skipped), (
        f"stray data.json not in skipped; skipped={skipped}"
    )


def test_coalescence_card_validates(tmp_path):
    """A valid MechanismCard written to mechanisms/ under a temp root appears in 'passed'."""
    mechanisms_dir = tmp_path / "mechanisms"
    mechanisms_dir.mkdir()
    card_path = mechanisms_dir / "m_test.yaml"
    dump(_mechanism_card_dict(), card_path)
    result = validate_all(str(tmp_path))
    passed = [p.replace("\\", "/") for p in result["passed"]]
    assert any("m_test.yaml" in p for p in passed), (
        f"MechanismCard not in passed; passed={passed}, failed={result['failed']}"
    )
    assert result["failed"] == []
