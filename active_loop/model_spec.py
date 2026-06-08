"""MUTABLE surface: the controller's generative model (A/B/C/D priors).

This is the only file the M2 outer loop is allowed to let an LLM edit.
Shapes carry a leading batch axis (batch_size=1) as required by pymdp's Agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from active_loop.signals import NUM_OBS

NUM_STATES = [3]
NUM_CONTROLS = [3]  # ACT, ASK, SWITCH
BATCH = 1


@dataclass(frozen=True)
class Dims:
    num_obs: list[int] = field(default_factory=lambda: list(NUM_OBS))
    num_states: list[int] = field(default_factory=lambda: list(NUM_STATES))
    num_controls: list[int] = field(default_factory=lambda: list(NUM_CONTROLS))
    batch_size: int = BATCH


DIMS = Dims()


def _batch(x: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(x[None, ...])


def _normalize(x: np.ndarray, axis: int) -> np.ndarray:
    return x / x.sum(axis=axis, keepdims=True)


def build_controller_arrays():
    """Return (A, B, C, D, pA, pB) as lists of batched JAX arrays."""
    ns = NUM_STATES[0]

    A_np = []
    a0 = np.array([
        [0.2, 0.8, 0.5],
        [0.8, 0.2, 0.5],
    ])
    A_np.append(a0)
    a1 = np.array([
        [0.1, 0.1, 0.6],
        [0.3, 0.2, 0.3],
        [0.6, 0.7, 0.1],
    ])
    A_np.append(a1)
    a2 = np.array([
        [0.7, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.1, 0.5, 0.3],
    ])
    A_np.append(a2)
    a3 = np.array([
        [0.8, 0.8, 0.8],
        [0.05, 0.15, 0.1],
        [0.15, 0.05, 0.1],
    ])
    A_np.append(a3)
    A = [_batch(_normalize(a, axis=0)) for a in A_np]

    B_np = np.zeros((ns, ns, NUM_CONTROLS[0]))
    B_np[:, :, 0] = np.array([
        [0.7, 0.2, 0.4],
        [0.2, 0.7, 0.3],
        [0.1, 0.1, 0.3],
    ])
    B_np[:, :, 1] = np.array([
        [0.8, 0.3, 0.5],
        [0.15, 0.6, 0.4],
        [0.05, 0.1, 0.1],
    ])
    B_np[:, :, 2] = np.array([
        [0.3, 0.3, 0.3],
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
    ])
    B = [_batch(_normalize(B_np, axis=0))]

    C_np = [
        np.array([0.0, 2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, -0.5]),
        np.array([0.0, -1.0, 1.0]),
    ]
    C = [_batch(c) for c in C_np]

    D = [_batch(np.array([0.25, 0.15, 0.6]))]

    pA = [_batch(_normalize(a, axis=0) * 2.0 + 0.1) for a in A_np]
    pB = [_batch(_normalize(B_np, axis=0) * 2.0 + 0.1)]

    return A, B, C, D, pA, pB
