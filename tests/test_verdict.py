"""Tests for active_loop.verdict — round-trip, schema fields, validation errors."""
from __future__ import annotations

import json
import pytest

from active_loop.verdict import write_verdict, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_arms():
    return {
        "arm_a": {"pass": True, "reason": "criterion met"},
        "arm_b": {"pass": False, "reason": "below threshold"},
    }


# ---------------------------------------------------------------------------
# Round-trip: write then read back
# ---------------------------------------------------------------------------

def test_round_trip_writes_valid_json(tmp_path):
    out = tmp_path / "exp999_verdict.json"
    write_verdict(
        path=out,
        experiment="exp999",
        arms=_minimal_arms(),
        verdict="POSITIVE",
        halted=False,
        notes="integration smoke test",
    )
    data = json.loads(out.read_text())
    assert data["experiment"] == "exp999"
    assert data["verdict"] == "POSITIVE"
    assert data["halted"] is False
    assert data["notes"] == "integration smoke test"


def test_round_trip_arms_preserved(tmp_path):
    out = tmp_path / "exp999_verdict.json"
    arms = _minimal_arms()
    write_verdict(out, "exp999", arms, "MIXED", halted=True)
    data = json.loads(out.read_text())
    assert data["arms"]["arm_a"]["pass"] is True
    assert data["arms"]["arm_b"]["pass"] is False
    assert data["arms"]["arm_b"]["reason"] == "below threshold"


def test_round_trip_returns_absolute_path(tmp_path):
    out = tmp_path / "exp999_verdict.json"
    result = write_verdict(out, "exp999", _minimal_arms(), "NEGATIVE", halted=False)
    assert result.is_absolute()
    assert result == out.resolve()


# ---------------------------------------------------------------------------
# Schema fields all present
# ---------------------------------------------------------------------------

def test_schema_version_field_present(tmp_path):
    out = tmp_path / "v.json"
    write_verdict(out, "exp1", _minimal_arms(), "POSITIVE", halted=False)
    data = json.loads(out.read_text())
    assert "schema_version" in data
    assert data["schema_version"] == SCHEMA_VERSION


def test_all_required_fields_present(tmp_path):
    out = tmp_path / "v.json"
    write_verdict(out, "exp1", _minimal_arms(), "NEGATIVE", halted=True, notes="ok")
    data = json.loads(out.read_text())
    for field in ("schema_version", "experiment", "verdict", "halted", "arms", "notes"):
        assert field in data, f"missing field: {field}"


def test_notes_defaults_to_empty_string(tmp_path):
    out = tmp_path / "v.json"
    write_verdict(out, "exp1", _minimal_arms(), "NEGATIVE", halted=False)
    data = json.loads(out.read_text())
    assert data["notes"] == ""


# ---------------------------------------------------------------------------
# Verdict validation
# ---------------------------------------------------------------------------

def test_invalid_verdict_raises_value_error(tmp_path):
    out = tmp_path / "v.json"
    with pytest.raises(ValueError, match="verdict must be one of"):
        write_verdict(out, "exp1", _minimal_arms(), "INCONCLUSIVE", halted=False)


def test_empty_verdict_raises_value_error(tmp_path):
    out = tmp_path / "v.json"
    with pytest.raises(ValueError):
        write_verdict(out, "exp1", _minimal_arms(), "", halted=False)


# ---------------------------------------------------------------------------
# Arms validation
# ---------------------------------------------------------------------------

def test_arms_not_dict_raises_value_error(tmp_path):
    out = tmp_path / "v.json"
    with pytest.raises(ValueError, match="arms must be a dict"):
        write_verdict(out, "exp1", [{"pass": True, "reason": "x"}], "POSITIVE", halted=False)  # type: ignore[arg-type]


def test_arm_entry_not_dict_raises_value_error(tmp_path):
    out = tmp_path / "v.json"
    with pytest.raises(ValueError):
        write_verdict(out, "exp1", {"arm_a": "bad"}, "POSITIVE", halted=False)  # type: ignore[arg-type]


def test_arm_missing_pass_key_raises_value_error(tmp_path):
    out = tmp_path / "v.json"
    with pytest.raises(ValueError, match="missing required keys"):
        write_verdict(out, "exp1", {"arm_a": {"reason": "x"}}, "POSITIVE", halted=False)  # type: ignore[arg-type]


def test_arm_missing_reason_key_raises_value_error(tmp_path):
    out = tmp_path / "v.json"
    with pytest.raises(ValueError, match="missing required keys"):
        write_verdict(out, "exp1", {"arm_a": {"pass": True}}, "POSITIVE", halted=False)  # type: ignore[arg-type]


def test_arm_pass_wrong_type_raises_value_error(tmp_path):
    out = tmp_path / "v.json"
    with pytest.raises(ValueError, match="must be bool"):
        write_verdict(
            out, "exp1",
            {"arm_a": {"pass": 1, "reason": "x"}},  # int not bool
            "POSITIVE", halted=False,
        )


def test_arm_reason_wrong_type_raises_value_error(tmp_path):
    out = tmp_path / "v.json"
    with pytest.raises(ValueError, match="must be str"):
        write_verdict(
            out, "exp1",
            {"arm_a": {"pass": True, "reason": 42}},  # int not str
            "POSITIVE", halted=False,
        )


def test_empty_arms_raises_value_error(tmp_path):
    out = tmp_path / "v.json"
    with pytest.raises(ValueError, match="at least one entry"):
        write_verdict(out, "exp1", {}, "POSITIVE", halted=False)


# ---------------------------------------------------------------------------
# All three valid verdict strings accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["POSITIVE", "NEGATIVE", "MIXED"])
def test_all_valid_verdicts_accepted(tmp_path, verdict):
    out = tmp_path / f"v_{verdict}.json"
    write_verdict(out, "exp1", _minimal_arms(), verdict, halted=False)
    data = json.loads(out.read_text())
    assert data["verdict"] == verdict
