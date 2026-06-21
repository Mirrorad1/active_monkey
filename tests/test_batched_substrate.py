"""Tests for embodied.batched_world — buffer ops (Task 1) + vmap advance (Task 2)."""
import numpy as np
import pytest


def _world(max_pop=4):
    from embodied.foodfield import FoodField, FoodFieldConfig
    from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
    from embodied.batched_world import BatchedEmbodiedWorld
    return BatchedEmbodiedWorld(
        FoodField(FoodFieldConfig(), 0),
        PolicyRunner(DEFAULT_CKPT),
        max_pop,
        bout_steps=4,
    )


def test_init_buffer_and_spawn():
    w = _world(4)
    state, alive = w.init_buffer(n_founders=2, seed=0, founder_xys=[(1.0, 0.0), (-1.0, 2.0)])

    # Buffer has max_pop slots; only first n_founders are alive
    assert state.q.shape[0] == 4
    assert alive.tolist() == [True, True, False, False]

    # Founders placed at requested xy
    assert np.allclose(np.asarray(state.q[0, 0:2]), [1.0, 0.0])
    assert np.allclose(np.asarray(state.q[1, 0:2]), [-1.0, 2.0])

    # spawn_into_slot sets target slot, leaves others unchanged
    s2 = w.spawn_into_slot(state, 3, (4.0, 4.0), seed=99)
    assert np.allclose(np.asarray(s2.q[3, 0:2]), [4.0, 4.0])          # slot 3 set
    assert np.allclose(np.asarray(s2.q[0]), np.asarray(state.q[0]))   # slot 0 q unchanged
    assert np.allclose(np.asarray(s2.qd[0]), np.asarray(state.qd[0]))  # slot 0 qd unchanged (spawn writes qd too)


def test_advance_batch_equals_sequential():
    """Task 2: vmap advance must agree with the sequential reference up to float noise."""
    w = _world(4)
    state, _ = w.init_buffer(2, 0, [(1.0, 0.0), (-1.0, 2.0)])
    targets = np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    sb, pb = w.advance_batch(state, targets)
    ss, ps = w.advance_sequential(state, targets)

    # Shape: [max_pop, bout_steps, 2]
    assert pb.shape == (4, w.bout_steps, 2), f"expected (4, {w.bout_steps}, 2), got {pb.shape}"
    assert ps.shape == (4, w.bout_steps, 2), f"sequential shape {ps.shape}"

    # vmap vs loop: equal up to float noise (NOT exact equality)
    max_path_diff = float(np.max(np.abs(np.asarray(pb) - np.asarray(ps))))
    max_q_diff = float(np.max(np.abs(np.asarray(sb.q) - np.asarray(ss.q))))
    print(f"max path diff batched vs sequential: {max_path_diff:.2e}")
    print(f"max q diff batched vs sequential:    {max_q_diff:.2e}")

    assert np.allclose(np.asarray(pb), np.asarray(ps), atol=1e-4), (
        f"paths disagree: max abs diff = {max_path_diff:.2e}"
    )
    assert np.allclose(np.asarray(sb.q), np.asarray(ss.q), atol=1e-4), (
        f"final q disagrees: max abs diff = {max_q_diff:.2e}"
    )
