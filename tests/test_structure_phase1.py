"""Phase 1 structure-learning instrumentation tests.

Fast, numpy-only.  No pymdp dependency.

Covers:
1. test_behavior_invariance   — determinism contract + instrumentation exists
2. test_surprise_drops        — learning reduces surprise in a static world
3. test_replay_roundtrip      — save/load replay.bin round-trip correctness
4. test_ceiling_quiet_on_static — no false ceiling alarm after 1500 steps
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from active_loop.creature import Creature, World


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_world_3x3() -> World:
    """3x3 world, n_colors=3, cmap cycles through 0,1,2."""
    return World(rows=3, cols=3, cmap=[0, 1, 2, 0, 1, 2, 0, 1, 2], n_colors=3)


# ---------------------------------------------------------------------------
# Test 1 — determinism contract + instrumentation presence
# ---------------------------------------------------------------------------

def test_behavior_invariance():
    """Determinism: birth(seed=7) → live(300) is bit-identical across two runs.

    The true before/after-git diff is validated externally; here we verify
    the determinism contract holds AND that the new instrumentation fields exist
    and are populated correctly.
    """
    world = make_world_3x3()

    # --- Run A ---
    c_a = Creature.birth("det-a", world, seed=7)
    c_a.live(300, seed=7)
    hash_a = c_a._state_hash()

    # --- Run B (same seed) ---
    c_b = Creature.birth("det-b", world, seed=7)
    c_b.live(300, seed=7)
    hash_b = c_b._state_hash()

    assert hash_a == hash_b, (
        f"state_hash must be identical for two seeded runs: {hash_a} vs {hash_b}"
    )

    # Instrumentation fields exist and are populated
    assert hasattr(c_a, "_surprise_window"), "_surprise_window must exist"
    assert hasattr(c_a, "_replay"), "_replay must exist"
    assert hasattr(c_a, "_ceiling_events"), "_ceiling_events must exist"

    # Window is populated (300 steps > SURPRISE_WINDOW=200 so window is full)
    assert len(c_a._surprise_window) == c_a._surprise_window.maxlen, (
        f"Window should be full after 300 steps; got {len(c_a._surprise_window)}"
    )

    # Replay accumulates (obs, action) per step; _replay holds post-live buffer
    # After 300 steps the replay buffer should have exactly 300 entries
    assert len(c_a._replay) == 300, (
        f"_replay should have 300 entries after live(300); got {len(c_a._replay)}"
    )

    # All obs values are valid color indices
    obs_vals = [t[0] for t in c_a._replay]
    assert all(0 <= o < world.n_colors for o in obs_vals), (
        f"All obs in replay must be in [0, n_colors): got {set(obs_vals)}"
    )

    # All action values are in [0, 4)
    act_vals = [t[1] for t in c_a._replay]
    assert all(0 <= a < 4 for a in act_vals), (
        f"All actions in replay must be in [0, 4): got {set(act_vals)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — learning reduces surprise in a static world
# ---------------------------------------------------------------------------

def test_surprise_drops():
    """After 600 steps, mean surprise over the last 100 < mean over the first 100."""
    world = make_world_3x3()
    c = Creature.birth("learner", world, seed=42)

    # Collect per-step surprise by living in small batches and reading the window
    # Use a small surprise_window so we can sample early and late cleanly
    c2 = Creature.birth("learner2", world, seed=42, surprise_window=600)
    c2.live(600, seed=42)

    win_arr = np.array(c2._surprise_window)
    assert len(win_arr) == 600, f"Expected 600 surprise values, got {len(win_arr)}"

    early_mean = float(win_arr[:100].mean())
    late_mean  = float(win_arr[-100:].mean())

    assert late_mean < early_mean, (
        f"Surprise should drop as learning proceeds: "
        f"early_mean={early_mean:.4f} late_mean={late_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — replay round-trip via save()
# ---------------------------------------------------------------------------

def test_replay_roundtrip(tmp_path):
    """save() flushes replay.bin; parsed pairs match expected format and length."""
    world = make_world_3x3()
    STEPS = 250

    c = Creature.birth("replay-test", world, seed=5)
    c.live(STEPS, seed=5)

    save_dir = tmp_path / "replay-test"
    c.save(str(save_dir))  # triggers save_replay() automatically

    replay_path = save_dir / "replay.bin"
    assert replay_path.exists(), "replay.bin must be created by save()"

    raw = replay_path.read_bytes()
    assert len(raw) == STEPS * 2, (
        f"replay.bin should be {STEPS*2} bytes (uint8 pairs); got {len(raw)}"
    )

    pairs = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2)
    assert pairs.shape == (STEPS, 2), (
        f"Expected ({STEPS}, 2) array of pairs; got {pairs.shape}"
    )

    obs_vals = pairs[:, 0]
    act_vals = pairs[:, 1]

    assert all(0 <= o < world.n_colors for o in obs_vals), (
        f"All obs in replay.bin must be in [0, n_colors); got unique={set(obs_vals.tolist())}"
    )
    assert all(0 <= a < 4 for a in act_vals), (
        f"All actions in replay.bin must be in [0, 4); got unique={set(act_vals.tolist())}"
    )

    # After save, the in-memory buffer is cleared
    assert c._replay == [], "_replay must be cleared after save_replay()"


# ---------------------------------------------------------------------------
# Test 4 — no false ceiling alarm after learning converges
# ---------------------------------------------------------------------------

def test_ceiling_quiet_on_static():
    """After 1500 steps in a static world, surprise is low so ceiling_flag is False.

    The creature learns the colormap well, surprise falls below CEILING_MEAN_THRESH
    (0.7 nats), so no false alarm is raised.
    """
    world = make_world_3x3()
    c = Creature.birth("no-ceiling", world, seed=0)
    c.live(1500, seed=0)

    m = c.surprise_metrics()

    assert m["window_len"] == c._surprise_window.maxlen, (
        f"Window should be full after 1500 steps; got {m['window_len']}"
    )
    assert m["mean"] is not None, "mean must be computed when window is full"
    assert m["ceiling_flag"] is False, (
        f"After 1500 steps of learning, ceiling_flag must be False "
        f"(surprise fell below threshold); mean={m['mean']:.4f}"
    )
