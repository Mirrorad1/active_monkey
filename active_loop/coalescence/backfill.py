"""Backfill planning for the active_loop coalescence layer.

Inspects the inventory and produces a concrete, honest action plan for each
experiment: what level it is currently at, what level is honest to target,
what actions are needed, and what blocks promotion.

HONESTY contract (binding):
- Never claim a backfill level the evidence does not support.
- Never invent rerun commands for non-existent scripts.
- Never promote an experiment beyond what its evidence justifies.
- target_level == current_level unless explicit mechanism/boundary evidence
  exists for that direction and the promotion is justified.

Public API
----------
backfill_plan(repo_root='.', inventory=None) -> dict
backfill_plan_json(repo_root='.') -> str   (deterministic canonical JSON)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from active_loop.coalescence.inventory import build_inventory
from active_loop.coalescence.schema import (
    BACKFILL_LEVEL_NAMES,
    SCHEMA_VERSION,
    backfill_level_index,
    to_canonical_json,
)
from active_loop.artifacts import repo_commit


# ---------------------------------------------------------------------------
# Directions that have distilled mechanism/boundary evidence anchoring a card.
# Only these are eligible for target_level promotion to mechanism_bundle.
# ---------------------------------------------------------------------------

_MECHANISM_DIRECTIONS: frozenset[str] = frozenset(
    {
        "affective-dyad",
        "costed-sensing",
        "active-sensing",
        "uncertainty-gated-active-sensing",
        "hidden-state-memory",
    }
)

# Experiment ranges per direction (for the specific promoted directions only).
# These define which exp numbers belong to the mechanism-eligible directions.
_MECHANISM_DIRECTION_RANGES: list[tuple[int, int, str]] = [
    (214, 226, "affective-dyad"),
    (125, 132, "affective-dyad"),
    (199, 207, "costed-sensing"),
    (210, 210, "active-sensing"),
    (211, 211, "uncertainty-gated-active-sensing"),
    (208, 209, "hidden-state-memory"),
]


def _is_mechanism_eligible(n: int, direction: str) -> bool:
    """Return True if experiment n/direction is eligible for mechanism_bundle promotion."""
    if direction not in _MECHANISM_DIRECTIONS:
        return False
    for lo, hi, d in _MECHANISM_DIRECTION_RANGES:
        if lo <= n <= hi and d == direction:
            return True
    return False


# ---------------------------------------------------------------------------
# Levels that qualify as "promotable" for mechanism_bundle targeting.
# The experiment must already be at one of these levels AND in a mechanism
# direction to be targeted at mechanism_bundle.
# ---------------------------------------------------------------------------

_PROMOTABLE_LEVELS: frozenset[str] = frozenset(
    {
        "metrics_bundle",
        "repro_bundle",
        "trajectory_bundle",
        "checkpoint_bundle",
    }
)


# ---------------------------------------------------------------------------
# Plan builder for a single experiment
# ---------------------------------------------------------------------------

def _build_plan(exp: dict) -> dict:
    """Build the backfill plan record for one experiment from its inventory record."""
    exp_id = exp["experiment_id"]
    n = exp["n"]
    direction = exp["direction"]
    verdict = exp["verdict"]
    current_level_name = exp["backfill_level_possible"]
    current_idx = backfill_level_index(current_level_name)

    has_script = exp["has_script"]
    has_raw_trajectories = exp["has_raw_trajectories"]
    has_checkpoint = exp["has_checkpoint"]
    has_repro_command = exp["has_repro_command"]

    # --- Target level (honest: only promote when distilled evidence justifies it) ---
    if (
        _is_mechanism_eligible(n, direction)
        and current_level_name in _PROMOTABLE_LEVELS
    ):
        target_level_name = "mechanism_bundle"
    else:
        target_level_name = current_level_name

    target_idx = backfill_level_index(target_level_name)

    # --- Actions, rerun_commands, blocked_on ---
    actions: list[str] = []
    rerun_commands: list[str] = []
    blocked_on: list[str] = []

    # Find the primary script path from available_sources
    script_path: str | None = None
    for src in exp.get("available_sources", []):
        if src.startswith("experiments/") and src.endswith(".py"):
            script_path = src
            break

    # Base export action at current level
    if current_level_name != "index_only":
        actions.append(f"export at {current_level_name}")

    # Handle script-based promotion
    if has_script and script_path:
        if not has_raw_trajectories and target_idx >= backfill_level_index("trajectory_bundle"):
            actions.append("rerun to regenerate raw trajectories (new reproduction run, does not mutate historical data)")
            rerun_commands.append(
                f"PYTHONPATH=. uv run --python .venv python {script_path}"
            )
            blocked_on.append("raw trajectories absent")
    elif not has_script:
        blocked_on.append("no committed script")
        if current_level_name == "index_only":
            actions.append("locate or reconstruct script before any backfill")

    # Checkpoint note for affective-dyad without a checkpoint
    if direction == "affective-dyad" and not has_checkpoint:
        actions.append(
            "checkpoint exportable via 'active-monkey artifact export' once a checkpoint is pinned"
        )

    # If targeting mechanism_bundle but at a lower level with no traj
    if target_level_name == "mechanism_bundle":
        if not has_raw_trajectories and not has_checkpoint:
            if has_script:
                if "raw trajectories absent" not in blocked_on:
                    blocked_on.append("raw trajectories absent")
            # mechanism card distillation is a manual authoring step
            actions.append(
                "author mechanism card distilling claim, boundary, and reusable interface from existing evidence"
            )

    # No-op case: already at index_only and not promotable
    if current_level_name == "index_only" and target_level_name == "index_only":
        if not blocked_on:
            blocked_on.append("no committed script")
        if not actions:
            actions.append("locate or reconstruct script before any backfill")

    return {
        "experiment_id": exp_id,
        "n": n,
        "direction": direction,
        "verdict": verdict,
        "current_level": current_level_name,
        "target_level": target_level_name,
        "actions": actions,
        "rerun_commands": rerun_commands,
        "blocked_on": blocked_on,
    }


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(plans: list[dict]) -> dict:
    """Build the top-level summary dict from all plans."""
    can_backfill_immediately: list[str] = []
    need_rerun_for_trajectories: list[str] = []
    need_checkpoint_export: list[str] = []
    counts_by_current_level: dict[str, int] = {lvl: 0 for lvl in BACKFILL_LEVEL_NAMES}

    traj_idx = backfill_level_index("trajectory_bundle")
    metrics_idx = backfill_level_index("metrics_bundle")

    for plan in plans:
        current = plan["current_level"]
        counts_by_current_level[current] = counts_by_current_level.get(current, 0) + 1

        curr_idx = backfill_level_index(current)

        if curr_idx >= metrics_idx:
            can_backfill_immediately.append(plan["experiment_id"])

        if any("rerun to regenerate" in a for a in plan.get("actions", [])):
            need_rerun_for_trajectories.append(plan["experiment_id"])

        if any("active-monkey artifact export" in a for a in plan.get("actions", [])):
            need_checkpoint_export.append(plan["experiment_id"])

    return {
        "can_backfill_immediately": sorted(can_backfill_immediately),
        "need_rerun_for_trajectories": sorted(need_rerun_for_trajectories),
        "need_checkpoint_export": sorted(need_checkpoint_export),
        "counts_by_current_level": counts_by_current_level,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def backfill_plan(repo_root: str | Path = ".", inventory: dict | None = None) -> dict:
    """Build the full backfill plan.

    Parameters
    ----------
    repo_root : str or Path
        Root of the repository (used only if *inventory* is None).
    inventory : dict or None
        Pre-built inventory dict (from build_inventory).  If None, calls
        build_inventory(repo_root) automatically.

    Returns
    -------
    dict with keys:
        schema_version, repo_commit, summary, plans
    """
    repo_root = Path(repo_root).resolve()

    if inventory is None:
        inventory = build_inventory(repo_root)

    plans: list[dict] = [
        _build_plan(exp) for exp in inventory.get("experiments", [])
    ]

    summary = _build_summary(plans)

    return {
        "schema_version": SCHEMA_VERSION,
        "repo_commit": inventory.get("repo_commit", repo_commit(repo_root)),
        "summary": summary,
        "plans": plans,
    }


def backfill_plan_json(repo_root: str | Path = ".") -> str:
    """Return deterministic canonical JSON of the backfill plan."""
    return to_canonical_json(backfill_plan(repo_root))
