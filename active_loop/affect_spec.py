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


def build_direct_head_model(seed: int = 0, k: int = K) -> dict:
    """M4a increment 1d: the DIRECT response->valence head (Exp 214 redesign).

    Exp 214 showed the response->valence credit path was too INDIRECT (response moved
    intent via B, intent emitted valence via A1; the within-turn B signal was ~0).  This
    model makes valence depend DIRECTLY on (intent, response) by adding a second hidden
    factor `last_response` (R states) that the action DETERMINISTICALLY sets, and routing
    the valence emission through it: A1 = P(valence | intent, last_response).  All the
    learning lives in A (learn_A=True, learn_B=False): A1 is a direct Dirichlet count table
    P(valence | intent, response), so a non-vanishing gradient runs straight from action to
    feedback.  B0 (intent) = identity per response (intent is observation-driven, unmoved by
    the response); B1 (last_response) = deterministic action-set.

    Hidden factors: [intent (K), last_response (R)].  Observations: [utterance (U), valence (V)].
    Control: [response (R)].  Shapes are batch-first JAX (batch=1), per controller.py.
    """
    rng = np.random.default_rng(seed)

    # A[0] = P(utterance | intent, last_response): depends on intent ONLY (broadcast over R).
    A0_2d = np.ones((U, k)) / U + rng.uniform(0.0, 0.05, (U, k))
    A0_2d = A0_2d / A0_2d.sum(axis=0, keepdims=True)
    A0 = np.repeat(A0_2d[:, :, None], R, axis=2)              # (U, k, R)

    # A[1] = P(valence | intent, last_response): THE DIRECT HEAD — learnable, ~uniform+jitter.
    A1 = np.ones((V, k, R)) / V + rng.uniform(0.0, 0.05, (V, k, R))
    A1 = A1 / A1.sum(axis=0, keepdims=True)                   # (V, k, R) column-stochastic over V

    # B[0] = P(intent' | intent): identity, UNCONTROLLED (num_controls=1) — the response does
    # not move intent (intent is observation-driven).  Shape (k, k, 1) so pymdp treats factor 0
    # as a single no-op control; the ONLY controllable factor is last_response (below), so the
    # policy space is exactly the R responses (not R x R).
    B0 = np.eye(k)[:, :, None]                               # (k, k, 1)

    # B[1] = P(last_response' | last_response, response): deterministic — next = the action taken.
    # This is the ONLY controlled factor (R actions = the response repertoire).
    B1 = np.zeros((R, R, R))
    for a in range(R):
        for rp in range(R):
            B1[a, rp, a] = 1.0                               # B1[r'=a, r_prev, action=a] = 1

    C0 = np.zeros(U); C1 = np.array([-2.0, 0.0, 3.0])         # strong POS preference
    D0 = np.ones(k) / k; D1 = np.ones(R) / R                 # uniform priors over both factors

    pA0 = A0 * 1.0 + 0.1
    pA1 = A1 * 1.0 + 0.1                                      # weak Dirichlet on the head -> fast learning
    pB0 = B0 * 50.0 + 0.1                                     # strong concentration: B stays structural
    pB1 = B1 * 50.0 + 0.1                                     # (learn_B=False, but keep them sharp)

    def _b(x: np.ndarray) -> jnp.ndarray:
        return jnp.array(x[None])

    return dict(
        A  = [_b(A0),  _b(A1)],
        pA = [_b(pA0), _b(pA1)],
        B  = [_b(B0),  _b(B1)],
        pB = [_b(pB0), _b(pB1)],
        C  = [_b(C0),  _b(C1)],
        D  = [_b(D0),  _b(D1)],
    )
