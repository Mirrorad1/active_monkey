"""Tests for active_loop.coalescence.inventory.

Uses real repo data — run from repo root.  No network, deterministic.
"""
from __future__ import annotations

import pytest

from active_loop.coalescence.inventory import build_inventory, inventory_json
from active_loop.coalescence.schema import CONFIDENCE_LEVELS, BACKFILL_LEVEL_NAMES


# Required keys every inventory experiment record must have.
_REQUIRED_KEYS = {
    "experiment_id",
    "n",
    "direction",
    "chapter",
    "status",
    "verdict",
    "available_sources",
    "has_script",
    "has_output",
    "has_metrics",
    "has_raw_trajectories",
    "has_scorer",
    "has_checkpoint",
    "has_repro_command",
    "confidence",
    "backfill_level_possible",
    "notes",
}


def test_inventory_runs_and_covers_all():
    """build_inventory() returns >= 225 experiments; all required keys present; n unique."""
    inv = build_inventory()
    experiments = inv["experiments"]

    assert inv["count"] >= 225, f"Expected >= 225, got {inv['count']}"
    assert len(experiments) >= 225

    ns = [e["n"] for e in experiments]
    assert len(ns) == len(set(ns)), "Duplicate n values found"

    for rec in experiments:
        missing = _REQUIRED_KEYS - rec.keys()
        assert not missing, f"exp{rec.get('n')} missing keys: {missing}"


def test_no_hallucinated_trajectories():
    """exp210 (summary/script only) must NOT report trajectories; exp199 MUST."""
    inv = build_inventory()
    exp_map = {e["n"]: e for e in inv["experiments"]}

    # exp210: active-sensing preflight — has script + txt, no raw traj subdir
    e210 = exp_map[210]
    assert e210["has_raw_trajectories"] is False, (
        f"exp210 falsely reports trajectories; notes={e210['notes']}"
    )
    assert e210["backfill_level_possible"] != "trajectory_bundle", (
        f"exp210 backfill_level_possible should not be trajectory_bundle, "
        f"got {e210['backfill_level_possible']}"
    )

    # exp199: has experiments/outputs/exp199_n5_valley_sweep/ with traj_*.json
    e199 = exp_map[199]
    assert e199["has_raw_trajectories"] is True, (
        f"exp199 should report trajectories; notes={e199['notes']}"
    )


def test_checkpoint_detection():
    """exp222 and exp225 must have has_checkpoint True; exp201 must not."""
    inv = build_inventory()
    exp_map = {e["n"]: e for e in inv["experiments"]}

    for n in (222, 225):
        rec = exp_map[n]
        assert rec["has_checkpoint"] is True, (
            f"exp{n} should have checkpoint; available_sources={rec['available_sources']}"
        )

    e201 = exp_map[201]
    assert e201["has_checkpoint"] is False, (
        f"exp201 should NOT have checkpoint; got artifact sources={e201['available_sources']}"
    )


def test_confidence_in_enum():
    """Every confidence and backfill_level_possible must be valid enum values."""
    inv = build_inventory()
    for rec in inv["experiments"]:
        assert rec["confidence"] in CONFIDENCE_LEVELS, (
            f"exp{rec['n']} confidence {rec['confidence']!r} not in {CONFIDENCE_LEVELS}"
        )
        assert rec["backfill_level_possible"] in BACKFILL_LEVEL_NAMES, (
            f"exp{rec['n']} backfill_level_possible "
            f"{rec['backfill_level_possible']!r} not in {BACKFILL_LEVEL_NAMES}"
        )


def test_inventory_json_deterministic():
    """inventory_json() must return the same string on two consecutive calls."""
    j1 = inventory_json()
    j2 = inventory_json()
    assert j1 == j2, "inventory_json() is not deterministic"
