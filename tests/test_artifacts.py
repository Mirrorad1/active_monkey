"""Tests for the AgentState / artifact infrastructure (Artifact Infrastructure M9).

Fast tests (default): pure-numpy serialization, hashing, manifest validation, CLI
export/inspect of the init tensors, and the no-network guard — none require pymdp/JAX.
Slow tests (`-m slow`): real DirectHeadAgent export/score (JAX compile is ~minutes).
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from active_loop.state import (
    SCHEMA_VERSION,
    AgentCheckpoint,
    AgentProvenance,
    AgentState,
    ArtifactManifest,
    ScorerCompatibility,
    SchemaMismatch,
    canonical_json,
)
from active_loop import artifacts
from active_loop.cli.main import main as cli_main


# ── fixtures ─────────────────────────────────────────────────────────────────

def _toy_state(seed: int = 0) -> AgentState:
    rng = np.random.default_rng(seed)
    return AgentState(
        architecture_id="toy-arch",
        agent_class="ToyAgent",
        tensors={
            "a0": rng.random((3, 4)).astype(np.float32),
            "pa0": rng.random((3, 4)).astype(np.float32),
        },
        belief_state={"q0": np.array([0.25, 0.25, 0.5], dtype=np.float32)},
        history_hashes={"obs": "deadbeef"},
        rng_state=[0, seed],
        provenance=AgentProvenance(
            repo_commit="abc123", source_experiments=[222, 225],
            created_at="2026-01-01T00:00:00Z",
        ),
        scorer_compat=ScorerCompatibility(scorer_path="eval/affect_score.py", scorer_hash="ff" * 32),
        metadata={"checkpoint": "toy"},
    )


# ── 1. AgentState serialization roundtrip ────────────────────────────────────

def test_agent_state_roundtrip(tmp_path: Path):
    st = _toy_state()
    st.save(tmp_path, stem="state")
    loaded = AgentState.load(tmp_path, stem="state")

    assert loaded.architecture_id == st.architecture_id
    assert loaded.agent_class == st.agent_class
    assert set(loaded.tensors) == set(st.tensors)
    for k in st.tensors:
        np.testing.assert_array_equal(loaded.tensors[k], st.tensors[k])
    assert loaded.belief_state is not None
    np.testing.assert_array_equal(loaded.belief_state["q0"], st.belief_state["q0"])
    assert loaded.history_hashes == st.history_hashes
    assert loaded.rng_state == st.rng_state
    assert loaded.provenance.source_experiments == [222, 225]
    assert loaded.scorer_compat.scorer_path == "eval/affect_score.py"
    # content hash is preserved across a save/load roundtrip
    assert loaded.content_hash() == st.content_hash()


# ── 2. Checkpoint tensor save/load roundtrip ─────────────────────────────────

def test_checkpoint_roundtrip(tmp_path: Path):
    ckpt = AgentCheckpoint(name="init", state=_toy_state())
    ckpt.save(tmp_path)
    assert (tmp_path / "init.safetensors").exists()
    assert (tmp_path / "init.config.json").exists()

    loaded = AgentCheckpoint.load(tmp_path, "init")
    assert loaded.content_hash() == ckpt.content_hash()
    np.testing.assert_array_equal(loaded.state.tensors["a0"], ckpt.state.tensors["a0"])


def test_empty_tensors_roundtrip(tmp_path: Path):
    """A state with no tensors still saves/loads (safetensors sentinel handling)."""
    st = AgentState(architecture_id="x", agent_class="Y", tensors={})
    st.save(tmp_path, stem="empty")
    loaded = AgentState.load(tmp_path, stem="empty")
    assert loaded.tensors == {}


# ── 3. Manifest validation ───────────────────────────────────────────────────

def test_manifest_validate_ok():
    m = ArtifactManifest({
        "artifact_id": "x", "schema_version": SCHEMA_VERSION, "agent_class": "DirectHeadAgent",
        "frozen_scorer": "eval/affect_score.py", "scorer_hash": "ab" * 32,
        "init_checkpoint_hash": "cd" * 32,
    })
    m.validate()  # no raise


def test_manifest_validate_missing_key():
    m = ArtifactManifest({"artifact_id": "x", "schema_version": SCHEMA_VERSION})
    with pytest.raises(ValueError):
        m.validate()


# ── 4. File/hash stability ───────────────────────────────────────────────────

def test_content_hash_stable_across_instances():
    """Two states with identical scientific content hash identically even with different
    created_at / repo_commit (volatile fields are excluded from the content hash)."""
    a = _toy_state()
    b = _toy_state()
    b.provenance.created_at = "2099-12-31T23:59:59Z"
    b.provenance.repo_commit = "different"
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_with_tensor():
    a = _toy_state()
    b = _toy_state()
    b.tensors["a0"] = b.tensors["a0"] + 1.0
    assert a.content_hash() != b.content_hash()


def test_hash_file_and_directory_stable(tmp_path: Path):
    (tmp_path / "x.txt").write_text("hello")
    (tmp_path / "y.txt").write_text("world")
    h1 = artifacts.hash_directory(tmp_path)
    h2 = artifacts.hash_directory(tmp_path)
    assert h1 == h2
    assert set(h1["files"]) == {"x.txt", "y.txt"}


def test_canonical_json_deterministic():
    a = {"b": 1, "a": [3, 2, 1], "c": {"z": 1, "y": 2}}
    assert canonical_json(a) == canonical_json(dict(reversed(list(a.items()))))


# ── 5. Frozen scorer hash recorded in manifest ───────────────────────────────

def test_export_records_scorer_hash(tmp_path: Path):
    out = tmp_path / "art"
    manifest = artifacts.export_affect_dyad_artifact(out, run_learned=False, run_eval=False)
    expected = artifacts.hash_file("eval/affect_score.py")
    assert manifest["scorer_hash"] == expected
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["scorer_hash"] == expected
    assert on_disk["init_checkpoint_hash"]


# ── 6 & 7. CLI export + inspect (fast, no agent) ─────────────────────────────

def test_cli_export_creates_files(tmp_path: Path):
    out = tmp_path / "art"
    rc = cli_main(["artifact", "export", "--preset", "affect-dyad-v0", "--out", str(out),
                   "--no-learned", "--no-eval"])
    assert rc == 0
    for f in ("manifest.json", "config.json", "model_card.yaml", "README.md",
              "init.safetensors", "init.config.json"):
        assert (out / f).exists(), f"missing {f}"
    assert (out / "examples" / "converse_demo.py").exists()
    assert (out / "examples" / "score_model.py").exists()


def test_cli_inspect_succeeds(tmp_path: Path, capsys):
    out = tmp_path / "art"
    cli_main(["artifact", "export", "--preset", "affect-dyad-v0", "--out", str(out),
              "--no-learned", "--no-eval"])
    capsys.readouterr()  # drain the export output so only the inspect JSON remains
    rc = cli_main(["artifact", "inspect", str(out), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_id"] == artifacts.DEFAULT_ARTIFACT_ID
    assert payload["checks"]["init_hash_ok"] is True


def test_cli_unknown_preset_nonzero(tmp_path: Path):
    rc = cli_main(["artifact", "export", "--preset", "nope", "--out", str(tmp_path / "x")])
    assert rc == 2


# ── 8. score: structured failure on scorer-hash drift (fast, no agent) ───────

def test_score_reports_hash_drift(tmp_path: Path):
    out = tmp_path / "art"
    artifacts.export_affect_dyad_artifact(out, run_learned=False, run_eval=False)
    # tamper the manifest's pinned scorer hash -> score must refuse, not silently rescore
    m = json.loads((out / "manifest.json").read_text())
    m["scorer_hash"] = "00" * 32
    (out / "manifest.json").write_text(canonical_json(m) + "\n")
    result = artifacts.score_artifact(out)
    assert result["status"] == "error"
    assert result["scorable"] is False
    assert "drift" in result["reason"]


# ── 9. load rejects missing / corrupt manifest ───────────────────────────────

def test_load_missing_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        artifacts.load_manifest(tmp_path)


def test_load_corrupt_manifest(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{not valid json")
    with pytest.raises(ValueError):
        artifacts.load_manifest(tmp_path)


def test_cli_inspect_bad_artifact_nonzero(tmp_path: Path):
    rc = cli_main(["artifact", "inspect", str(tmp_path)])
    assert rc == 1


# ── 10. schema mismatch refusal unless allowed ───────────────────────────────

def test_schema_mismatch_refused(tmp_path: Path):
    st = _toy_state()
    st.save(tmp_path, stem="state")
    cfg = json.loads((tmp_path / "state.config.json").read_text())
    cfg["schema_version"] = "9.9.9"
    (tmp_path / "state.config.json").write_text(canonical_json(cfg) + "\n")
    with pytest.raises(SchemaMismatch):
        AgentState.load(tmp_path, stem="state")
    # explicit override loads
    loaded = AgentState.load(tmp_path, stem="state", allow_schema_mismatch=True)
    assert loaded.schema_version == "9.9.9"


def test_manifest_schema_mismatch_refused(tmp_path: Path):
    artifacts.export_affect_dyad_artifact(tmp_path, run_learned=False, run_eval=False)
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["schema_version"] = "9.9.9"
    (tmp_path / "manifest.json").write_text(canonical_json(m) + "\n")
    with pytest.raises(SchemaMismatch):
        artifacts.load_manifest(tmp_path)
    assert artifacts.load_manifest(tmp_path, allow_schema_mismatch=True)["schema_version"] == "9.9.9"


# ── 11. no hidden network calls ──────────────────────────────────────────────

def test_no_network_on_export_inspect(tmp_path: Path, monkeypatch):
    import socket

    def _blocked(*a, **k):
        raise AssertionError("network access attempted during local artifact op")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    out = tmp_path / "art"
    artifacts.export_affect_dyad_artifact(out, run_learned=False, run_eval=False)
    artifacts.inspect_artifact(out)
    # state save/load too
    _toy_state().save(tmp_path / "s", stem="state")
    AgentState.load(tmp_path / "s", stem="state")


# ── slow: real agent export + score ──────────────────────────────────────────

@pytest.mark.slow
def test_full_export_and_score(tmp_path: Path):
    out = tmp_path / "art"
    manifest = artifacts.export_affect_dyad_artifact(
        out, learn_turns=30, run_learned=True, run_eval=True,
        eval_seeds=(20,), eval_turns=30,
    )
    assert manifest["learned_checkpoint_hash"] is not None
    assert (out / "learned_example.safetensors").exists()
    summary = artifacts.inspect_artifact(out)
    assert summary["checks"]["init_hash_ok"] is True
    assert summary["checks"].get("learned_hash_ok") is True
    # quick real score returns a structured ok report
    result = artifacts.score_artifact(out, seeds=(20,), turns=30)
    assert result["status"] == "ok"
    assert "metric" in result


@pytest.mark.slow
def test_load_agent_and_converse(tmp_path: Path):
    out = tmp_path / "art"
    artifacts.export_affect_dyad_artifact(out, run_learned=False, run_eval=False)
    agent = artifacts.load_agent_from_artifact(out, which="init", seed=0)
    agent.perceive(0)
    r = agent.act()
    assert isinstance(r, int)
