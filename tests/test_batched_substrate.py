"""Tests for embodied.batched_world — buffer ops (Task 1: init_buffer + spawn_into_slot)."""
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
