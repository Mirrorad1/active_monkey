import jax.numpy as jnp
import numpy as np

from active_loop.model_spec import build_controller_arrays, DIMS
from active_loop.signals import NUM_OBS


def test_dims():
    assert DIMS.num_states == [3]
    assert DIMS.num_controls == [3]
    assert DIMS.num_obs == NUM_OBS
    assert DIMS.batch_size == 1


def test_shapes_have_leading_batch_axis():
    A, B, C, D, pA, pB = build_controller_arrays()
    assert len(A) == 4 and len(B) == 1 and len(C) == 4 and len(D) == 1
    for m, n_o in enumerate(NUM_OBS):
        assert A[m].shape == (1, n_o, 3), f"A[{m}] shape {A[m].shape}"
        assert C[m].shape == (1, n_o)
    assert B[0].shape == (1, 3, 3, 3)
    assert D[0].shape == (1, 3)
    assert pA is not None and len(pA) == 4
    assert pA[0].shape == A[0].shape


def test_A_and_B_are_normalized_over_outcome_axis():
    A, B, C, D, pA, pB = build_controller_arrays()
    for a in A:
        col_sums = jnp.sum(a, axis=1)
        assert jnp.allclose(col_sums, 1.0, atol=1e-5)
    col_sums = jnp.sum(B[0], axis=1)
    assert jnp.allclose(col_sums, 1.0, atol=1e-5)


def test_success_modality_likelihood_is_informative():
    A, *_ = build_controller_arrays()
    p_success_on_track = float(A[0][0, 1, 0])
    p_success_wrong = float(A[0][0, 1, 1])
    assert p_success_on_track > p_success_wrong
