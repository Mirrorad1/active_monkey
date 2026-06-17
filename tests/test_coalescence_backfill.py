"""Tests for active_loop.coalescence.backfill.

Uses real repo data — run from repo root.  No network, deterministic.
"""
from __future__ import annotations

import pytest

from active_loop.coalescence.backfill import backfill_plan, backfill_plan_json
from active_loop.coalescence.schema import BACKFILL_LEVEL_NAMES, backfill_level_index

_PLAN_REQUIRED_KEYS = {
    "experiment_id",
    "n",
    "direction",
    "verdict",
    "current_level",
    "target_level",
    "actions",
    "rerun_commands",
    "blocked_on",
}

_TOP_LEVEL_REQUIRED_KEYS = {"schema_version", "repo_commit", "summary", "plans"}
_SUMMARY_REQUIRED_KEYS = {
    "can_backfill_immediately",
    "need_rerun_for_trajectories",
    "need_checkpoint_export",
    "counts_by_current_level",
}


def test_plan_runs():
    """backfill_plan() returns plans for all 225+; every plan has the required keys."""
    result = backfill_plan()

    # Top-level structure
    missing_top = _TOP_LEVEL_REQUIRED_KEYS - result.keys()
    assert not missing_top, f"Top-level keys missing: {missing_top}"

    plans = result["plans"]
    assert len(plans) >= 225, f"Expected >= 225 plans, got {len(plans)}"

    ns = [p["n"] for p in plans]
    assert len(ns) == len(set(ns)), "Duplicate n values in plans"

    for plan in plans:
        missing = _PLAN_REQUIRED_KEYS - plan.keys()
        assert not missing, f"plan for exp{plan.get('n')} missing keys: {missing}"
        # current_level and target_level must be valid enum values
        assert plan["current_level"] in BACKFILL_LEVEL_NAMES, (
            f"exp{plan['n']} current_level {plan['current_level']!r} not in BACKFILL_LEVEL_NAMES"
        )
        assert plan["target_level"] in BACKFILL_LEVEL_NAMES, (
            f"exp{plan['n']} target_level {plan['target_level']!r} not in BACKFILL_LEVEL_NAMES"
        )
        # target_level must never be lower than current_level (no demotion)
        assert backfill_level_index(plan["target_level"]) >= backfill_level_index(plan["current_level"]), (
            f"exp{plan['n']} target_level {plan['target_level']!r} < current_level {plan['current_level']!r}"
        )
        # actions and blocked_on must be lists
        assert isinstance(plan["actions"], list), f"exp{plan['n']} actions not a list"
        assert isinstance(plan["blocked_on"], list), f"exp{plan['n']} blocked_on not a list"
        assert isinstance(plan["rerun_commands"], list), f"exp{plan['n']} rerun_commands not a list"


def test_marks_missing_trajectories_honestly():
    """exp210 (no raw trajectories) must not claim trajectory data.

    - current_level must equal 'repro_bundle' (has script but no traj subdir).
    - If target_level >= trajectory_bundle, 'raw trajectories absent' must appear
      in blocked_on, OR rerun_commands must be non-empty.
    - The plan must NOT assert has_raw_trajectories anywhere (not our field, but
      we verify target_level is honest by checking it doesn't assert trajectory
      completion without evidence).
    """
    result = backfill_plan()
    plan_map = {p["n"]: p for p in result["plans"]}

    e210 = plan_map[210]
    assert e210["current_level"] == "repro_bundle", (
        f"exp210 current_level should be 'repro_bundle', got {e210['current_level']!r}"
    )

    traj_idx = backfill_level_index("trajectory_bundle")
    target_idx = backfill_level_index(e210["target_level"])

    if target_idx >= traj_idx:
        # If targeting trajectory or above, must surface the gap honestly
        has_blocked = "raw trajectories absent" in e210["blocked_on"]
        has_rerun = len(e210["rerun_commands"]) > 0
        assert has_blocked or has_rerun, (
            f"exp210 targets {e210['target_level']!r} but neither blocks on missing trajectories "
            f"nor provides a rerun command: blocked_on={e210['blocked_on']}, "
            f"rerun_commands={e210['rerun_commands']}"
        )


def test_metrics_only_not_overclaimed():
    """An experiment with current_level 'metrics_bundle' in a non-mechanism direction
    must NOT be auto-promoted to trajectory_bundle or checkpoint_bundle.

    Find any exp with metrics_bundle that is NOT in a mechanism direction and
    confirm target_level == current_level.
    """
    from active_loop.coalescence.backfill import _MECHANISM_DIRECTIONS

    result = backfill_plan()
    found_candidate = False
    for plan in result["plans"]:
        if (
            plan["current_level"] == "metrics_bundle"
            and plan["direction"] not in _MECHANISM_DIRECTIONS
        ):
            found_candidate = True
            assert plan["target_level"] == "metrics_bundle", (
                f"exp{plan['n']} (direction={plan['direction']!r}) with metrics_bundle "
                f"incorrectly promoted to {plan['target_level']!r}"
            )
    # If no candidate was found that is acceptable too (structure guard still runs in test_plan_runs)
    # but we soft-warn if no candidate found (the test should find at least one)
    # Directions like 'persistent-creature', 'language', etc. would qualify
    if not found_candidate:
        pytest.skip("No metrics_bundle experiment outside mechanism directions found — inventory may have changed")


def test_summary_lists_immediate_vs_rerun():
    """summary.can_backfill_immediately must be non-empty.

    The two lists need not be disjoint (an exp can need a rerun to go further
    but already qualifies for metrics-level export), but can_backfill_immediately
    must have entries since many experiments have at least metrics_bundle.
    """
    result = backfill_plan()
    summary = result["summary"]

    missing_summary = _SUMMARY_REQUIRED_KEYS - summary.keys()
    assert not missing_summary, f"Summary keys missing: {missing_summary}"

    assert isinstance(summary["can_backfill_immediately"], list)
    assert isinstance(summary["need_rerun_for_trajectories"], list)
    assert isinstance(summary["need_checkpoint_export"], list)
    assert isinstance(summary["counts_by_current_level"], dict)

    assert len(summary["can_backfill_immediately"]) > 0, (
        "can_backfill_immediately is empty — expected many experiments at metrics_bundle or above"
    )

    # counts_by_current_level must cover all BACKFILL_LEVEL_NAMES
    for lvl in BACKFILL_LEVEL_NAMES:
        assert lvl in summary["counts_by_current_level"], (
            f"counts_by_current_level missing level {lvl!r}"
        )

    # Total count from summary must match total plans
    total_from_summary = sum(summary["counts_by_current_level"].values())
    assert total_from_summary == len(result["plans"]), (
        f"counts_by_current_level total {total_from_summary} != plans count {len(result['plans'])}"
    )


def test_plan_json_deterministic():
    """backfill_plan_json() must return the same string on two consecutive calls."""
    j1 = backfill_plan_json()
    j2 = backfill_plan_json()
    assert j1 == j2, "backfill_plan_json() is not deterministic"
    # Also ensure it is valid JSON
    import json
    parsed = json.loads(j1)
    assert "plans" in parsed
    assert "summary" in parsed
