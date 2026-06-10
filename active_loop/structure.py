"""Structure learning, Phases 2-3 (SCAFFOLD, flag-gated): replay scoring, Bayesian model
reduction, and expansion operators for the creature's discrete generative model.

NOT wired into any default run — every entry point requires
``STRUCTURE_LEARNING_ENABLED=True`` passed explicitly.

Math: docs/specs/structure-learning.md.

Phase 1 (the surprise-ceiling detector + replay buffer) lives in creature.py.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Module-level flag default (never flip to True here; callers must be explicit)
# ---------------------------------------------------------------------------

STRUCTURE_LEARNING_ENABLED: bool = False

_GATE_MSG = "structure learning is flag-gated; pass enabled=True explicitly"


def _gate(enabled: bool) -> None:
    """Raise RuntimeError unless caller explicitly passes enabled=True."""
    if not enabled:
        raise RuntimeError(_GATE_MSG)


# ---------------------------------------------------------------------------
# Phase 2: replay I/O
# ---------------------------------------------------------------------------

def load_replay(path, *, enabled: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Parse a replay.bin file into parallel obs and action arrays.

    Format: flat uint8 pairs ``[obs, action] * N``.  Each step occupies exactly
    2 bytes; ``obs < n_colors`` (< 256) and ``action < 4``.

    Parameters
    ----------
    path : path-like
        Path to the ``replay.bin`` file written by ``Creature.save_replay()``.
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    obs_array : np.ndarray, dtype=uint8, shape (N,)
    act_array : np.ndarray, dtype=uint8, shape (N,)
    """
    _gate(enabled)
    data = Path(path).read_bytes()
    if len(data) % 2 != 0:
        raise ValueError(
            f"replay.bin has odd byte count ({len(data)}); expected flat uint8 pairs"
        )
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 2)
    return arr[:, 0].copy(), arr[:, 1].copy()


# ---------------------------------------------------------------------------
# Phase 2: free-energy scoring under a candidate model
# ---------------------------------------------------------------------------

def candidate_score(
    pA: np.ndarray,
    world_B: np.ndarray,
    qs0: np.ndarray,
    obs: np.ndarray,
    acts: np.ndarray,
    *,
    enabled: bool = False,
) -> float:
    """Score an observation+action history under a candidate generative model.

    Re-runs the creature's exact inference chain over the logged history under
    the CANDIDATE model parameters ``(pA, world_B)`` and returns the accumulated
    negative log-evidence F.

    Algorithm (per step t):
      1. ``A_hat = pA / pA.sum(axis=0, keepdims=True)``  (column-normalised).
      2. ``p_o = A_hat[obs_t] @ qs``
      3. ``F += -ln(p_o + 1e-300)``  (accumulate neg-log-evidence)
      4. ``qs ∝ A_hat[obs_t] * qs``  (Bayesian update; uniform fallback if denom=0)
      5. ``qs = world_B[:, :, act_t] @ qs``  (advance through transition model)

    Notes
    -----
    (a) F here is the accumulated negative log-evidence of the observation stream
        under exact filtering — the variational bound is tight for this model
        class (discrete HMM with exact Bayesian filtering).
    (b) Candidates are comparable ONLY on the SAME history; F values across
        different histories are not commensurable.
    (c) The history is biased toward the data distribution of the OLD policy
        (active-data bias): a candidate that would change behavior is scored on
        data it did not generate.  See docs/specs/structure-learning.md §6.

    Parameters
    ----------
    pA : np.ndarray, shape (n_obs, n_states)
        Candidate Dirichlet counts for any n_states (need not equal creature's
        current state count).
    world_B : np.ndarray, shape (n_states, n_states, 4)
        Candidate transition tensor.
    qs0 : np.ndarray, shape (n_states,)
        Initial belief vector (will be normalised internally).
    obs : np.ndarray, dtype=uint8, shape (N,)
        Observation sequence (indices into axis 0 of pA).
    acts : np.ndarray, dtype=uint8, shape (N,)
        Action sequence.
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    float
        Accumulated negative log-evidence (lower = better fit).
    """
    _gate(enabled)

    pA = np.asarray(pA, dtype=float)
    world_B = np.asarray(world_B, dtype=float)

    # Column-normalise pA once
    col_sums = pA.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    A_hat = pA / col_sums

    n_states = pA.shape[1]
    qs = np.asarray(qs0, dtype=float).copy()
    denom_qs = qs.sum()
    if denom_qs > 0:
        qs = qs / denom_qs
    else:
        qs = np.ones(n_states) / n_states

    eps = 1e-300
    F = 0.0

    for obs_t, act_t in zip(obs, acts):
        p_o = float(A_hat[int(obs_t)] @ qs)
        F += -np.log(p_o + eps)

        # Belief update: qs ∝ A_hat[obs_t] * qs
        qs_new = A_hat[int(obs_t)] * qs
        denom = qs_new.sum()
        if denom > 0:
            qs_new = qs_new / denom
        else:
            qs_new = np.ones(n_states) / n_states

        # Advance through transition
        qs = world_B[:, :, int(act_t)] @ qs_new

    return float(F)


# ---------------------------------------------------------------------------
# Phase 2: Bayesian Model Reduction
# ---------------------------------------------------------------------------

def _log_beta(x: np.ndarray) -> float:
    """Multivariate log-Beta: sum(gammaln(x)) - gammaln(sum(x))."""
    return float(np.sum(gammaln(x)) - gammaln(np.sum(x)))


def bmr_delta_f(
    a_post: np.ndarray,
    a0_prior: np.ndarray,
    a0_reduced: np.ndarray,
    *,
    enabled: bool = False,
) -> float:
    r"""Closed-form Dirichlet BMR delta-F (log Bayes factor), summed over columns.

    Sign convention (PINNED)
    ------------------------
    ΔF = F_reduced − F_full in the log-evidence-change sense, computed as the
    log Bayes factor:

        ΔF = ln p(data | reduced) − ln p(data | full)

    POSITIVE ΔF favors the reduced model (reduction is beneficial).
    NEGATIVE ΔF favors the full model (reduction throws away useful structure).

    Formula (per column j)
    ----------------------
    Let ``a_tilde_j = a_post_j + a0_reduced_j − a0_prior_j`` (clipped at 1e-10).

        ΔF_j = ln B(a_post_j) + ln B(a0_reduced_j)
               − ln B(a0_prior_j) − ln B(a_tilde_j)

    where ``ln B(x) = Σ gammaln(x_i) − gammaln(Σ x_i)``.

    If any entry of ``a_tilde_j`` would be ≤ 0 before clipping, the reduction is
    technically incompatible at that column (the reduced prior has placed zero mass
    where the posterior has weight); the clip to 1e-10 approximates the limit but
    will produce a strongly negative ΔF for that column, effectively penalising the
    incompatible reduction.

    Parameters
    ----------
    a_post : np.ndarray, shape (n_obs, n_states)
        Posterior Dirichlet counts (after observing data).
    a0_prior : np.ndarray, same shape
        Original prior Dirichlet counts before observing data.
    a0_reduced : np.ndarray, same shape
        Reduced prior Dirichlet counts (e.g., column zeroed to epsilon for
        state pruning).
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    float
        ΔF (log Bayes factor); positive = reduction favored.
    """
    _gate(enabled)

    a_post = np.asarray(a_post, dtype=float)
    a0_prior = np.asarray(a0_prior, dtype=float)
    a0_reduced = np.asarray(a0_reduced, dtype=float)

    if a_post.shape != a0_prior.shape or a_post.shape != a0_reduced.shape:
        raise ValueError("a_post, a0_prior, a0_reduced must all have the same shape")

    n_states = a_post.shape[1]
    delta_f = 0.0

    for j in range(n_states):
        a_tilde_j = a_post[:, j] + a0_reduced[:, j] - a0_prior[:, j]
        a_tilde_j = np.clip(a_tilde_j, 1e-10, None)

        delta_f_j = (
            _log_beta(a_post[:, j])
            + _log_beta(a0_reduced[:, j])
            - _log_beta(a0_prior[:, j])
            - _log_beta(a_tilde_j)
        )
        delta_f += delta_f_j

    return float(delta_f)


# ---------------------------------------------------------------------------
# Phase 2: pruning pass
# ---------------------------------------------------------------------------

def prune_pass(
    pA: np.ndarray,
    a0_prior_scale: float,
    *,
    enabled: bool = False,
) -> list[dict]:
    """Score each hidden state as a candidate for removal; return ranked report.

    For each hidden state ``s`` (column index), computes the BMR ΔF for a
    reduction that effectively removes state ``s`` by setting its prior column to
    a near-zero (epsilon = 1e-10) concentration — i.e., flattening-to-epsilon
    ("removal-by-flattening").  This makes the reduced prior's column ``s``
    functionally uniform at near-zero mass, concentrating all evidence
    weight on the remaining states.

    Candidates are ranked by ΔF descending (most favorable to prune first, i.e.
    most-positive ΔF at the top).

    This function NEVER mutates its inputs and NEVER applies any reduction
    automatically — it is a pure report.  Callers decide whether to act.

    Parameters
    ----------
    pA : np.ndarray, shape (n_obs, n_states)
        Current Dirichlet counts (posterior).
    a0_prior_scale : float
        Uniform prior concentration scale.  The original prior is constructed
        as ``np.full_like(pA, a0_prior_scale)``.
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    list of dict
        Each entry: ``dict(state=s, delta_f=score)``, sorted descending by
        ``delta_f``.  Positive score = evidence for pruning; negative = keep.
    """
    _gate(enabled)

    pA = np.asarray(pA, dtype=float)
    n_obs, n_states = pA.shape
    a0_prior = np.full_like(pA, float(a0_prior_scale))

    results = []
    for s in range(n_states):
        a0_reduced = a0_prior.copy()
        a0_reduced[:, s] = 1e-10  # removal-by-flattening

        score = bmr_delta_f(pA, a0_prior, a0_reduced, enabled=True)
        results.append({"state": s, "delta_f": score})

    results.sort(key=lambda d: d["delta_f"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Phase 3: expansion operators (scaffold only)
# ---------------------------------------------------------------------------

def spawn_state(
    pA: np.ndarray,
    obs_dist: np.ndarray,
    weak: float = 0.1,
    *,
    enabled: bool = False,
) -> np.ndarray:
    """Append a new hidden state column seeded at obs_dist + weak uniform counts.

    The new column is initialised as::

        new_col = obs_dist * 1.0 + weak * uniform

    where ``uniform = np.ones(n_obs) / n_obs`` (so counts sum to 1 + weak).

    PROVISIONAL: the new state must survive the next ``prune_pass`` or be deleted
    by the caller.  B-tensor adjustments for changed state counts are the CALLER's
    responsibility in Phase 3 proper (near-uniform init for new rows/cols); this
    operator handles pA only.

    Parameters
    ----------
    pA : np.ndarray, shape (n_obs, n_states)
    obs_dist : np.ndarray, shape (n_obs,)
        Seed distribution for the new state (e.g., the offending observation
        distribution at the surprise-ceiling step).
    weak : float
        Weak uniform regularisation added to the seeded column (default 0.1).
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    np.ndarray, shape (n_obs, n_states + 1)
        New pA with one appended column; input is NOT mutated.
    """
    _gate(enabled)

    pA = np.asarray(pA, dtype=float)
    obs_dist = np.asarray(obs_dist, dtype=float)
    n_obs = pA.shape[0]
    if obs_dist.shape != (n_obs,):
        raise ValueError(
            f"obs_dist shape {obs_dist.shape} must match pA n_obs={n_obs}"
        )

    uniform = np.ones(n_obs) / n_obs
    new_col = obs_dist.copy() + weak * uniform
    return np.column_stack([pA, new_col])


def add_state(pA: np.ndarray, *, enabled: bool = False) -> np.ndarray:
    """Append a new hidden state with uniform observation distribution.

    Convenience wrapper around ``spawn_state`` with ``obs_dist = uniform``.

    B-tensor adjustments for changed state counts are the CALLER's
    responsibility in Phase 3 proper (near-uniform init for new rows/cols);
    this operator handles pA only.

    Parameters
    ----------
    pA : np.ndarray, shape (n_obs, n_states)
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    np.ndarray, shape (n_obs, n_states + 1)
    """
    _gate(enabled)

    pA = np.asarray(pA, dtype=float)
    n_obs = pA.shape[0]
    obs_dist = np.ones(n_obs) / n_obs
    return spawn_state(pA, obs_dist, enabled=True)


def split_state(
    pA: np.ndarray,
    s: int,
    jitter: float = 0.05,
    seed: int = 0,
    *,
    enabled: bool = False,
) -> np.ndarray:
    """Duplicate column ``s`` with seeded multiplicative jitter, halving original counts.

    The original column ``s`` has its counts halved.  A new column is appended,
    initialised as the halved column with multiplicative jitter applied::

        half = pA[:, s] / 2.0
        rng = np.random.default_rng(seed)
        noise = 1.0 + jitter * (2 * rng.random(n_obs) - 1)   # uniform in [1-j, 1+j]
        new_col = half * noise

    The total counts over {s, new} ≈ original column s (exactly when jitter=0).

    B-tensor adjustments for changed state counts are the CALLER's
    responsibility in Phase 3 proper (near-uniform init for new rows/cols);
    this operator handles pA only.

    Parameters
    ----------
    pA : np.ndarray, shape (n_obs, n_states)
    s : int
        Index of column to split.
    jitter : float
        Magnitude of multiplicative noise (default 0.05).
    seed : int
        Seed for jitter RNG (default 0; deterministic).
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    np.ndarray, shape (n_obs, n_states + 1)
        Input is NOT mutated.  Original column ``s`` has half counts; appended
        column has the jittered half.  Sum of both columns == sum of original
        column when jitter=0 (within floating-point precision).
    """
    _gate(enabled)

    pA = np.asarray(pA, dtype=float).copy()
    n_obs, n_states = pA.shape
    if not (0 <= s < n_states):
        raise IndexError(f"state index s={s} out of range for n_states={n_states}")

    half = pA[:, s] / 2.0
    pA[:, s] = half

    rng = np.random.default_rng(seed)
    noise = 1.0 + jitter * (2.0 * rng.random(n_obs) - 1.0)
    new_col = half * noise

    return np.column_stack([pA, new_col])


def merge_states(
    pA: np.ndarray,
    s1: int,
    s2: int,
    *,
    enabled: bool = False,
) -> np.ndarray:
    """Sum columns ``s1`` and ``s2`` into ``s1``, drop column ``s2``.

    B-tensor adjustments for changed state counts are the CALLER's
    responsibility in Phase 3 proper (near-uniform init for new rows/cols);
    this operator handles pA only.

    Parameters
    ----------
    pA : np.ndarray, shape (n_obs, n_states)
    s1 : int
        Index of the column to merge INTO.
    s2 : int
        Index of the column to be absorbed and dropped.
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    np.ndarray, shape (n_obs, n_states - 1)
        Input is NOT mutated.  Column ``s1`` contains the sum; column ``s2``
        is removed.
    """
    _gate(enabled)

    pA = np.asarray(pA, dtype=float).copy()
    n_obs, n_states = pA.shape
    for idx, name in [(s1, "s1"), (s2, "s2")]:
        if not (0 <= idx < n_states):
            raise IndexError(f"{name}={idx} out of range for n_states={n_states}")
    if s1 == s2:
        raise ValueError("s1 and s2 must be distinct states")

    pA[:, s1] = pA[:, s1] + pA[:, s2]
    return np.delete(pA, s2, axis=1)


# ---------------------------------------------------------------------------
# Phase 3: variant selection
# ---------------------------------------------------------------------------

def select_variant(
    F_current: float,
    F_candidate: float,
    *,
    enabled: bool = False,
) -> bool:
    """Accept candidate model iff its free energy is strictly lower than current.

    Decision rule: accept iff ``F_candidate < F_current`` strictly.

    Every proposal and decision (including dead ends where candidate is
    rejected) should be logged by the caller.  This function is a pure
    predicate — it does not log or mutate state.

    Parameters
    ----------
    F_current : float
        Accumulated negative log-evidence for the current model on the history.
    F_candidate : float
        Accumulated negative log-evidence for the candidate model on the same
        history.
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    bool
        True if candidate should replace current (strict improvement).
    """
    _gate(enabled)
    return bool(F_candidate < F_current)


# ---------------------------------------------------------------------------
# Phase 3: spawn trigger predicate
# ---------------------------------------------------------------------------

def spawn_rule_check(
    per_state_surprise_min: float,
    threshold: float,
    consecutive_flagged: int,
    K: int,
    *,
    enabled: bool = False,
) -> bool:
    """Phase-3 trigger predicate: should a new state be spawned?

    Fires when the minimum per-state predictive surprise exceeds ``threshold``
    for at least ``K`` consecutive ceiling-flagged steps::

        min_s -ln p(o_t | s) > threshold   AND   consecutive_flagged >= K

    This is a pure predicate — no side effects, no mutation.

    The criterion operationalises "irreducible surprise" in a factor/modality:
    even the best-fitting existing state cannot explain the observation stream,
    suggesting the model space itself is too small.

    Parameters
    ----------
    per_state_surprise_min : float
        ``min_s -ln p(o_t | s)`` — minimum surprise across all states at the
        current step (lower = at least one state explains the observation well).
    threshold : float
        Surprise threshold (nats) above which a step is deemed un-explainable.
        Corresponds to ``CEILING_MEAN_THRESH`` from creature.py (0.7 nats).
    consecutive_flagged : int
        Number of consecutive ceiling-flagged steps so far.
    K : int
        Minimum run length required to trigger spawn.
    enabled : bool
        Must be ``True``; raises ``RuntimeError`` otherwise.

    Returns
    -------
    bool
        True iff spawn should be triggered.
    """
    _gate(enabled)
    return bool(per_state_surprise_min > threshold and consecutive_flagged >= K)
