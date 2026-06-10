"""M4a affect model spec (MUTABLE): the dyad generative model per
docs/specs/m4-affective-dyad.md section 2, with the discovered substrate
requirements designed in: Dirichlet count decay (the window, LV=0.999, Exp
85-91) applied in the agent's learn step; the exploration reflex via EFE's
epistemic term + the ASK response.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

# ── Dimensions ──────────────────────────────────────────────────────────────
K = 4   # latent intent states
U = 6   # utterance-code outcomes
R = 5   # responses: GREET=0, MIRROR=1, SOOTHE=2, PLAY=3, ASK=4
V = 3   # valence outcomes: NEG=0, NEU=1, POS=2
LV = 0.999  # Dirichlet window decay (Exp 85-91 requirement)

NEG, NEU, POS = 0, 1, 2
GREET, MIRROR, SOOTHE, PLAY, ASK = 0, 1, 2, 3, 4


def build_dyad_model(seed: int = 0) -> dict:
    """Build the M4a dyadic generative model arrays in pymdp (JAX) conventions.

    Returns a dict with keys: A, pA, B, pB, C, D — all in batch-first JAX
    arrays with batch_size=1, matching the conventions in controller.py.

    A[0]: (1, U, K) utterance emission — column-stochastic, uniform + jitter.
    A[1]: (1, V, K) valence emission — column-stochastic, uniform + jitter.
    pA:   scaled Dirichlet concentrations A*1.0 + 0.1 (weak, for fast learning).
    B[0]: (1, K, K, R) intent transition — sticky (0.8) + response mixing.
    pB:   matching concentrations B*1.0 + 0.1.
    C[0]: (1, U) zeros — no utterance preference.
    C[1]: (1, V) [-2.0, 0.0, 3.0] — strong positive preference.
    D[0]: (1, K) uniform over intent states.
    """
    rng = np.random.default_rng(seed)

    # ── A matrices ──────────────────────────────────────────────────────────
    # Uniform column-stochastic + weak seeded jitter (the agent learns the real one)
    A0_raw = np.ones((U, K)) / U + rng.uniform(0.0, 0.05, (U, K))
    A0 = A0_raw / A0_raw.sum(axis=0, keepdims=True)  # column-normalize → (U, K)

    A1_raw = np.ones((V, K)) / V + rng.uniform(0.0, 0.05, (V, K))
    A1 = A1_raw / A1_raw.sum(axis=0, keepdims=True)  # (V, K)

    # ── B matrix ─────────────────────────────────────────────────────────────
    # Sticky identity (0.8) + response-dependent seeded mixing, column-stochastic
    # per action slice.  Shape: (K, K, R) → pymdp batched: (1, K, K, R)
    B0 = np.zeros((K, K, R))
    for r in range(R):
        mix = rng.uniform(0.0, 0.1, (K, K))  # small off-diagonal mass
        for k in range(K):
            col = mix[:, k].copy()
            col[k] += 0.8  # sticky diagonal
            col = col / col.sum()
            B0[:, k, r] = col

    # ── C (preferences) ──────────────────────────────────────────────────────
    C0 = np.zeros(U)
    C1 = np.array([-2.0, 0.0, 3.0])  # NEG, NEU, POS

    # ── D (prior over intent) ─────────────────────────────────────────────────
    D0 = np.ones(K) / K

    # ── Dirichlet concentrations ──────────────────────────────────────────────
    pA0 = A0 * 1.0 + 0.1
    pA1 = A1 * 1.0 + 0.1
    pB0 = B0 * 1.0 + 0.1

    # ── Batch-first JAX arrays (batch_size=1) ─────────────────────────────────
    def _b(x: np.ndarray) -> jnp.ndarray:
        return jnp.array(x[None])  # prepend batch dim

    return dict(
        A  = [_b(A0),  _b(A1)],
        pA = [_b(pA0), _b(pA1)],
        B  = [_b(B0)],
        pB = [_b(pB0)],
        C  = [_b(C0),  _b(C1)],
        D  = [_b(D0)],
    )
