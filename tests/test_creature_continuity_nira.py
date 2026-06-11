"""Continuity guard for nira — the persistent continuous-substrate creature.

Mirrors the invariant logic of test_creature_continuity.py, pointed at
creature/state/nira.

SKIP cleanly if the nira directory does not yet exist — so the suite stays
green before first commit of state.
"""
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent
SPINE = ROOT / "creature" / "state" / "nira"

RESET_EVENTS = {"rebirth", "reset", "restart"}


def _spine_present() -> bool:
    return (SPINE / "manifest.json").exists()


def _require_spine() -> None:
    if not _spine_present():
        pytest.skip("nira state directory does not exist yet — skipping continuity guard")


def _manifest():
    return json.loads((SPINE / "manifest.json").read_text(encoding="utf-8"))


def _biography():
    path = SPINE / "BIOGRAPHY.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _ages():
    return [e["age_steps"] for e in _biography() if "age_steps" in e]


def test_nira_snapshot_present():
    """The nira spine has a committed snapshot (manifest + biography + arrays)."""
    _require_spine()
    assert (SPINE / "manifest.json").exists(), "nira manifest missing"
    assert (SPINE / "BIOGRAPHY.jsonl").exists(), "nira biography missing"
    assert (SPINE / "arrays.npz").exists(), "nira arrays missing"


def test_nira_age_never_goes_backwards():
    """nira's age never resets SILENTLY — only via a logged restart event."""
    _require_spine()
    violations = []
    last_age = None
    reset_pending = False
    for e in _biography():
        if e.get("event") in RESET_EVENTS:
            reset_pending = True
            continue
        if "age_steps" not in e:
            continue
        a = e["age_steps"]
        if last_age is not None and a < last_age:
            if not reset_pending:
                violations.append((last_age, a))
        last_age = a
        reset_pending = False

    assert not violations, (
        "nira age went backwards SILENTLY (a reset/rewind without a logged "
        f"{sorted(RESET_EVENTS)} event): " + repr(violations)
    )


def test_nira_manifest_matches_biography_head():
    """The committed manifest age == the LATEST age in the biography."""
    _require_spine()
    manifest_age = _manifest()["age_steps"]
    ages = _ages()
    assert ages, "nira biography has no age-bearing events"
    assert manifest_age == ages[-1], (
        f"manifest age_steps={manifest_age} but biography head={ages[-1]} — "
        f"the committed snapshot is not the latest point of nira's life"
    )
