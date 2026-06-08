import jax.numpy as jnp
from active_loop.lang_model_spec import build_lang_arrays, LANG_DIMS
from active_loop.alphabet import V


def test_dims():
    assert LANG_DIMS.num_obs == [V]
    assert LANG_DIMS.num_controls == [1]
    assert LANG_DIMS.batch_size == 1
    assert LANG_DIMS.K >= 2


def test_shapes():
    K = LANG_DIMS.K
    A, B, D, pA, pB = build_lang_arrays()
    assert A[0].shape == (1, V, K)
    assert B[0].shape == (1, K, K, 1)
    assert D[0].shape == (1, K)
    assert pA[0].shape == (1, V, K) and pB[0].shape == (1, K, K, 1)


def test_columns_normalized():
    A, B, D, pA, pB = build_lang_arrays()
    assert jnp.allclose(jnp.sum(A[0], axis=1), 1.0, atol=1e-5)
    assert jnp.allclose(jnp.sum(B[0], axis=1), 1.0, atol=1e-5)
    assert jnp.allclose(jnp.sum(D[0], axis=1), 1.0, atol=1e-5)


def test_seed_is_deterministic():
    a1, *_ = build_lang_arrays(seed=7)
    a2, *_ = build_lang_arrays(seed=7)
    assert jnp.allclose(a1[0], a2[0])
