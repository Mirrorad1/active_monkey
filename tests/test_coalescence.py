"""Tests for the coalescence artifact layer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import active_loop.coalescence as c
from active_loop.coalescence.schema import (
    ARTIFACT_REGISTRY,
    BACKFILL_LEVEL_NAMES,
    SCHEMA_VERSION,
    AdapterCard,
    BoundaryNote,
    ExperimentBundle,
    ExperimentSpec,
    GeometryMap,
    MechanismCard,
    ScorerCard,
    TrajectoryRow,
    backfill_level_index,
    backfill_level_name,
    dump,
    load,
    validate_trajectory_row,
    write_trajectory_jsonl,
    read_trajectory_jsonl,
)
from active_loop.coalescence.validate import (
    MissingFieldError,
    sha256_bytes,
    validate_bundle,
    validate_dict,
)


def _min_bundle_dict(**overrides) -> dict:
    d = {
        "artifact_type": "experiment_bundle",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "exp001",
        "direction": "valence-learning",
        "question": "Does the agent learn?",
        "hypothesis": "Yes",
        "status": "complete",
        "verdict": "CONFIRMED",
        "repo_commit": "abc123",
        "created_at": "2026-01-01T00:00:00Z",
        "confidence": "high",
        "backfill_level": "metrics_bundle",
    }
    d.update(overrides)
    return d


def test_experiment_bundle_validates_good_example():
    d = _min_bundle_dict()
    result = validate_dict(d)
    assert result["experiment_id"] == "exp001"


def test_schema_rejects_missing_required_fields():
    d = _min_bundle_dict()
    del d["verdict"]
    with pytest.raises(MissingFieldError) as exc_info:
        validate_dict(d)
    assert "verdict" in exc_info.value.missing


def test_hash_helper_deterministic():
    h1 = sha256_bytes(b"x")
    h2 = sha256_bytes(b"x")
    assert h1 == h2
    assert h1 == hashlib.sha256(b"x").hexdigest()


def test_scorer_card_records_hash(tmp_path):
    scorer_file = tmp_path / "scorer.py"
    scorer_file.write_text("# frozen scorer\n", encoding="utf-8")
    card = ScorerCard.from_file(
        file_path="scorer.py",
        scorer_id="sc-001",
        scorer_version="1.0",
        repo=str(tmp_path),
    )
    assert len(card.sha256) == 64
    assert all(c in "0123456789abcdef" for c in card.sha256)
    d = card.to_dict()
    card2 = ScorerCard.from_dict(d)
    assert card2.to_dict() == d


def test_metrics_bundle_does_not_claim_raw_trajectories():
    bundle = ExperimentBundle(
        experiment_id="exp002",
        direction="d",
        question="q",
        hypothesis="h",
        status="complete",
        verdict="CONFIRMED",
        repo_commit="abc",
        created_at="2026-01-01T00:00:00Z",
        confidence="high",
        backfill_level="metrics_bundle",
        raw_data_refs=[],
    )
    assert bundle.to_dict()["raw_data_refs"] == []


def test_corrupt_or_missing_manifest_rejected(tmp_path):
    # No manifest.json at all
    with pytest.raises(FileNotFoundError):
        validate_bundle(tmp_path)

    # manifest.json present but missing required fields
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"artifact_type": "experiment_bundle", "schema_version": SCHEMA_VERSION}),
        encoding="utf-8",
    )
    with pytest.raises(MissingFieldError):
        validate_bundle(tmp_path)


def test_roundtrip_all_card_types():
    instances = [
        ExperimentBundle(
            experiment_id="e1", direction="d", question="q", hypothesis="h",
            status="complete", verdict="CONFIRMED", repo_commit="abc",
            created_at="2026-01-01T00:00:00Z", confidence="high",
            backfill_level="metrics_bundle",
        ),
        ExperimentSpec(experiment_id="e2"),
        TrajectoryRow(experiment_id="e3", seed=0, t=1),
        ScorerCard(scorer_id="s1", scorer_version="1.0", file_path="x.py", sha256="a" * 64),
        MechanismCard(
            mechanism_id="m1", mechanism_type="functional-valence-learning",
            claim="agent learns valence", status="validated",
            source_experiments=[1, 2],
        ),
        GeometryMap(geometry_id="g1", source_experiments=[3]),
        AdapterCard(adapter_id="a1", from_mechanism="m1", to_mechanism="m2"),
        BoundaryNote(
            boundary_id="b1", source_experiments=[4],
            failed_mechanism="m1", observed_failure="no learning at k=1",
        ),
    ]
    for inst in instances:
        d = inst.to_dict()
        cls = ARTIFACT_REGISTRY[d["artifact_type"]]
        d2 = cls.from_dict(d).to_dict()
        assert d == d2, f"roundtrip failed for {cls.__name__}"


def test_trajectory_jsonl_roundtrip(tmp_path):
    rows = [
        TrajectoryRow(experiment_id="e1", seed=0, t=i, action=i * 2)
        for i in range(3)
    ]
    path = tmp_path / "traj.jsonl"
    write_trajectory_jsonl(rows, path)
    loaded = read_trajectory_jsonl(path)
    assert len(loaded) == 3
    for i, row in enumerate(loaded):
        assert row["t"] == i
        assert row["action"] == i * 2

    # validate_trajectory_row flags missing 't'
    bad = {"experiment_id": "e1", "seed": 0}
    missing = validate_trajectory_row(bad)
    assert "t" in missing
    # good row has no missing keys
    good = {"experiment_id": "e1", "seed": 0, "t": 0}
    assert validate_trajectory_row(good) == []


def test_backfill_level_helpers():
    for i, name in enumerate(BACKFILL_LEVEL_NAMES):
        assert backfill_level_name(i) == name
        assert backfill_level_index(name) == i


def test_dump_load_yaml_or_json_fallback(tmp_path):
    d = _min_bundle_dict()

    # JSON path
    json_path = tmp_path / "card.json"
    dump(d, json_path)
    loaded = load(json_path)
    assert loaded == d

    # YAML path (if available)
    try:
        import yaml  # noqa: F401
        yaml_path = tmp_path / "card.yaml"
        dump(d, yaml_path)
        loaded_yaml = load(yaml_path)
        assert loaded_yaml == d
    except ImportError:
        pass  # yaml not available; JSON-only fallback already tested
