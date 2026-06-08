"""MUTABLE surface (M3): the character HMM generative model (A/B/D priors, K states).

The M3b autopilot loop may edit this file (e.g. K, Dirichlet concentration, init
randomness) to lower held-out free energy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from active_loop.alphabet import V

K = 14          # number of latent "meaning" states
A_CONC = 0.1    # Dirichlet prior concentration for A
B_CONC = 0.1    # Dirichlet prior concentration for B
INIT_SEED = 0


@dataclass(frozen=True)
class LangDims:
    num_obs: list[int]
    num_states: list[int]
    num_controls: list[int]
    batch_size: int
    K: int


LANG_DIMS = LangDims(num_obs=[V], num_states=[K], num_controls=[1], batch_size=1, K=K)


def _norm(x: np.ndarray, axis: int) -> np.ndarray:
    return x / x.sum(axis=axis, keepdims=True)


def _batch(x: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(x[None, ...])


def build_lang_arrays(seed: int = INIT_SEED):
    """Return (A, B, D, pA, pB) as lists of batched JAX arrays for the char HMM.

    Each list element carries the leading batch axis: A[0].shape == (1, V, K), etc.
    """
    rng = np.random.default_rng(seed)
    A_b = _batch(_norm(rng.random((V, K)) + 0.1, axis=0))     # (1, V, K)
    B_b = _batch(_norm(rng.random((K, K, 1)) + 0.1, axis=0))  # (1, K, K, 1)
    D_b = _batch(np.ones((K,)) / K)                           # (1, K)
    pA = [_batch(np.ones((V, K)) * A_CONC)]
    pB = [_batch(np.ones((K, K, 1)) * B_CONC)]
    return [A_b], [B_b], [D_b], pA, pB
