"""Fast tests for structure.py Phase 2-3 scaffolds.

All tests are self-contained and fast (no I/O, no large worlds).
"""
from __future__ import annotations

import importlib
import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest

from active_loop.structure import (
    STRUCTURE_LEARNING_ENABLED,
    _GATE_MSG,
    add_state,
    bmr_delta_f,
    candidate_score,
    load_replay,
    merge_states,
    prune_pass,
    select_variant,
    spawn_rule_check,
    spawn_state,
    split_state,
)
from active_loop.creature import Creature, World


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_world() -> World:
    """3x3 world with a simple 3-color pattern."""
    cmap = [0, 1, 2, 0, 1, 2, 0, 1, 2]  # 9 cells, 3 colors
    return World(rows=3, cols=3, cmap=cmap, n_colors=3)


# ---------------------------------------------------------------------------
# test_flag_gate
# ---------------------------------------------------------------------------

def test_flag_gate():
    """Every public function must raise RuntimeError without enabled=True."""
    pA_dummy = np.ones((3, 4)) * 0.5
    B_dummy = np.zeros((4, 4, 4))
    for s in range(4):
        for a in range(4):
            B_dummy[s, s, a] = 1.0  # identity transition

    qs_dummy = np.ones(4) / 4
    obs_dummy = np.zeros(5, dtype=np.uint8)
    acts_dummy = np.zeros(5, dtype=np.uint8)

    # load_replay: needs a real path; create a minimal temp file
    with tempfile.TemporaryDirectory() as tmp:
        rp = Path(tmp) / "replay.bin"
        rp.write_bytes(bytes([0, 1, 2, 3]))  # two valid pairs
        with pytest.raises(RuntimeError, match="flag-gated"):
            load_replay(rp)

    with pytest.raises(RuntimeError, match="flag-gated"):
        candidate_score(pA_dummy, B_dummy, qs_dummy, obs_dummy, acts_dummy)

    with pytest.raises(RuntimeError, match="flag-gated"):
        bmr_delta_f(pA_dummy, pA_dummy, pA_dummy)

    with pytest.raises(RuntimeError, match="flag-gated"):
        prune_pass(pA_dummy, 0.5)

    with pytest.raises(RuntimeError, match="flag-gated"):
        spawn_state(pA_dummy, np.ones(3) / 3)

    with pytest.raises(RuntimeError, match="flag-gated"):
        add_state(pA_dummy)

    with pytest.raises(RuntimeError, match="flag-gated"):
        split_state(pA_dummy, 0)

    with pytest.raises(RuntimeError, match="flag-gated"):
        merge_states(pA_dummy, 0, 1)

    with pytest.raises(RuntimeError, match="flag-gated"):
        select_variant(1.0, 0.5)

    with pytest.raises(RuntimeError, match="flag-gated"):
        spawn_rule_check(0.8, 0.7, 5, 3)


# ---------------------------------------------------------------------------
# test_bmr_prune_unused_state_favored
# ---------------------------------------------------------------------------

def test_bmr_prune_unused_state_favored():
    """BMR should favor pruning a state that was never visited.

    Setup:
      - 3 observation colors, 4 hidden states.
      - Prior: uniform 0.5 on all cells.
      - Posterior: states 0-2 receive +50 peaked counts; state 3 untouched
        (posterior == prior for that column).
    Expectation (pinned sign convention):
      - bmr_delta_f for reducing state 3 must be POSITIVE (reduction favored).
    """
    n_obs = 3
    n_states = 4
    a0_prior = np.ones((n_obs, n_states)) * 0.5

    # Posterior: states 0-2 heavily visited; state 3 untouched
    a_post = a0_prior.copy()
    # Peaked: each of states 0-2 accumulates +50 on ONE observation row
    a_post[0, 0] += 50.0  # state 0 predicts color 0
    a_post[1, 1] += 50.0  # state 1 predicts color 1
    a_post[2, 2] += 50.0  # state 2 predicts color 2
    # state 3: a_post[:, 3] == a0_prior[:, 3] (untouched)

    # Reduce state 3: set its prior column to epsilon
    a0_reduced = a0_prior.copy()
    a0_reduced[:, 3] = 1e-10

    delta_f = bmr_delta_f(a_post, a0_prior, a0_reduced, enabled=True)

    # POSITIVE delta_f means: ln p(data|reduced) > ln p(data|full) => pruning favored
    assert delta_f > 0, (
        f"Expected positive delta_f (pruning unused state 3 favored), got {delta_f:.6f}"
    )


# ---------------------------------------------------------------------------
# test_bmr_prune_used_state_rejected
# ---------------------------------------------------------------------------

def test_bmr_prune_used_state_rejected():
    """BMR should REJECT pruning a heavily-used state.

    Same setup as test_bmr_prune_unused_state_favored, but we attempt to prune
    state 0 (which received +50 peaked counts).  Expectation: delta_f is NEGATIVE
    (full model is better).
    """
    n_obs = 3
    n_states = 4
    a0_prior = np.ones((n_obs, n_states)) * 0.5

    a_post = a0_prior.copy()
    a_post[0, 0] += 50.0
    a_post[1, 1] += 50.0
    a_post[2, 2] += 50.0

    # Reduce state 0 (heavily used)
    a0_reduced = a0_prior.copy()
    a0_reduced[:, 0] = 1e-10

    delta_f = bmr_delta_f(a_post, a0_prior, a0_reduced, enabled=True)

    # NEGATIVE delta_f means: ln p(data|full) > ln p(data|reduced) => keep state
    assert delta_f < 0, (
        f"Expected negative delta_f (pruning used state 0 rejected), got {delta_f:.6f}"
    )


# ---------------------------------------------------------------------------
# test_candidate_score_runs
# ---------------------------------------------------------------------------

def test_candidate_score_runs():
    """Candidate score runs on creature-generated history; own model beats shuffled."""
    world = _tiny_world()
    creature = Creature.birth("test_cand", world, seed=42)
    creature.live(50)

    obs = np.array([o for o, a in creature._replay], dtype=np.uint8)
    acts = np.array([a for o, a in creature._replay], dtype=np.uint8)

    n_states = world.n_cells
    pA_own = creature.pA.copy()
    B_own = world.transition_matrix()
    qs0_own = np.ones(n_states) / n_states

    F_own = candidate_score(pA_own, B_own, qs0_own, obs, acts, enabled=True)
    assert np.isfinite(F_own), f"F_own not finite: {F_own}"
    assert F_own > 0, f"Expected F_own > 0, got {F_own}"

    # Wrong candidate: shuffle columns of pA
    rng = np.random.default_rng(7)
    perm = rng.permutation(n_states)
    pA_wrong = pA_own[:, perm]

    F_wrong = candidate_score(pA_wrong, B_own, qs0_own, obs, acts, enabled=True)
    assert np.isfinite(F_wrong), f"F_wrong not finite: {F_wrong}"

    # Own model should score lower F (better fit) on its own history
    assert F_own < F_wrong, (
        f"Expected own model (F={F_own:.4f}) < wrong model (F={F_wrong:.4f})"
    )


# ---------------------------------------------------------------------------
# test_spawn_and_merge_shapes
# ---------------------------------------------------------------------------

def test_spawn_and_merge_shapes():
    """spawn_state adds a column; merge_states removes one; split conserves sum."""
    n_obs, n_states = 4, 5
    pA = np.abs(np.random.default_rng(0).standard_normal((n_obs, n_states))) + 0.1

    # spawn adds a column
    obs_dist = np.ones(n_obs) / n_obs
    pA_spawned = spawn_state(pA, obs_dist, enabled=True)
    assert pA_spawned.shape == (n_obs, n_states + 1), (
        f"spawn: expected ({n_obs}, {n_states+1}), got {pA_spawned.shape}"
    )
    assert pA.shape == (n_obs, n_states), "spawn must not mutate input"

    # merge removes a column
    pA_merged = merge_states(pA, 0, 1, enabled=True)
    assert pA_merged.shape == (n_obs, n_states - 1), (
        f"merge: expected ({n_obs}, {n_states-1}), got {pA_merged.shape}"
    )
    assert pA.shape == (n_obs, n_states), "merge must not mutate input"

    # split: total count sum is preserved within 1e-9
    pA_split = split_state(pA, 2, jitter=0.0, seed=0, enabled=True)
    assert pA_split.shape == (n_obs, n_states + 1), (
        f"split: expected ({n_obs}, {n_states+1}), got {pA_split.shape}"
    )
    # With jitter=0, original col 2 counts should be conserved across the two halves
    original_sum_col2 = pA[:, 2].sum()
    # col 2 in split result + the new appended col
    split_sum = pA_split[:, 2].sum() + pA_split[:, n_states].sum()
    assert abs(split_sum - original_sum_col2) < 1e-9, (
        f"split count conservation failed: orig={original_sum_col2}, split={split_sum}"
    )
    assert pA.shape == (n_obs, n_states), "split must not mutate input"
