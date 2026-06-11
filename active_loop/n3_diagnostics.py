"""active_loop/n3_diagnostics.py — the N3 failure-mode diagnostic (shadow layer).

First-cut, **transparent rule-based** classifier that reads a creature's own
learning signals (NOT the world label) and classifies *why* its prediction error
is high.  This is the N2-order diagnostic half of the N3 bounded-map/open-world
direction (`loop/directions/n3-bounded-map-open-world.md`); the repair-controller
half (N3b) consumes its output.

Honesty (binding, VALIDATION.md "provided vs self-formed"):
  - The classifier is **provided** (designer-written rules + thresholds). It is a
    baseline-grade readout, not a learned/self-formed model. The empirical question
    N3a tests is NOT whether these rules are clever — it is whether the creature's
    own signals *carry separable regime information at all*. A learned diagnostic is
    later work; this establishes (or refutes) separability cheaply.
  - All thresholds are tied to **already-validated** growth constants
    (``ALARM_THRESH`` = 0.7, ``K_PENALTY`` = 0.05 from ``active_loop.growth``,
    confirmed Exp 145/154), fixed by design BEFORE any fresh-seed run — not tuned
    on results. The only new constant is the temporal-jump margin (declared below).

The load-bearing discriminator (structural vs. noise) is the **shadow analog of the
validated live-probation accept test** (Exp 145/154). A naive offline replay-NLL
split test is INSUFFICIENT and was shown to fail in Exp 155's smoke run: noisy
observations also have multi-cluster spatial structure (a true block plus scattered
noise), so an offline split "improves" replay NLL even though it cannot reduce real
surprise. The faithful test instead fits the split on a TRAIN subset and measures
**held-out predictive surprise** against the full competing mixture, demanding that
the split (a) drives held-out surprise BELOW the irreducible floor threshold
(structure is *reducible*; noise is not) AND (b) beats the single-component fit by
the validated keep margin. The classifier never sees the world kind.

Design doc: docs/specs/n3-open-world.md (§2A, §9 first minimal experiment).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from active_loop.continuous import NIW
from active_loop.growth import (
    ALARM_THRESH,
    KEEP_MARGIN,
    MIN_REPLAY_PAIRS,
    batch_jump_em,
    em_result_to_components,
    mixture_predictive_logprobs,
    select_best_k,
)

# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------

#: The four regimes the four already-existing worlds instantiate (N3a-min scope).
#: The full 10-way set (docs/specs/n3-open-world.md §2A) is reached once the
#: expanding / spreading / horizon worlds are built (N3c).
REGIMES_4: list[str] = [
    "stable_known",
    "structural_inadequacy",
    "irreducible_noise",
    "nonstationarity",
]

# ---------------------------------------------------------------------------
# New constant (declared; the only threshold not inherited from growth.py)
# ---------------------------------------------------------------------------

#: A nonstationary world shows a LOW early-phase mean that JUMPS high after the
#: remap. The jump must clear this margin (in nats) on top of crossing the alarm
#: threshold, so ordinary learning jitter near 0.7 is not read as a world change.
#: Set by design (a "clear step"), not tuned on run data.
JUMP_MARGIN: float = 0.3


# ---------------------------------------------------------------------------
# Run signals — the per-run feature bundle the classifier reads
# ---------------------------------------------------------------------------


@dataclass
class RunSignals:
    """Scalar/boolean features extracted from one shadow run (no world label).

    Attributes:
        loud_mean:     final-window mean surprise of the loudest color (nats).
        ceiling_fired: did the global surprise-ceiling detector fire in the run's
                       second half?
        early_mean:    global mean surprise over an early post-burn-in window.
        late_mean:     global mean surprise over the final window.
        structural:    did the loud color's replay pass the penalized-split test
                       (a >1-component fit beats the single-component fit on
                       held-out replay NLL)? See ``structural_gain``.
    """

    loud_mean: float
    ceiling_fired: bool
    early_mean: float
    late_mean: float
    structural: bool


# ---------------------------------------------------------------------------
# The structural-vs-noise discriminator (penalized held-out replay split test)
# ---------------------------------------------------------------------------


def _held_out_color_surprise(
    test_pairs: list[tuple[np.ndarray, np.ndarray]],
    components: list[list[tuple[float, NIW]]],
    color: int,
) -> float:
    """Mean −log p(color | place) over held-out positions, under the full mixture.

    ``test_pairs`` are ``(mu_place, Sigma_place_diag)`` at which ``color`` was the
    observed emission. Using the FULL competing mixture (all colors) is essential:
    structure is predictable because at a structured cell no other color competes,
    whereas under noise other colors overlap everywhere so the color stays
    unpredictable no matter how its own components are fit.
    """
    if not test_pairs:
        return float("inf")
    total = 0.0
    for mu, sig in test_pairs:
        log_probs = mixture_predictive_logprobs(mu, sig, components, convention="normalized")
        total += -float(log_probs[color])
    return total / len(test_pairs)


def structural_gain(
    components: list[list[tuple[float, NIW]]],
    loud_color: int,
    loud_pairs: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    *,
    floor_thresh: float = ALARM_THRESH,
    margin: float = KEEP_MARGIN,
) -> tuple[bool, dict]:
    """Shadow analog of live probation: would splitting the loud color reduce surprise?

    Fits the loud color's positions on a TRAIN subset (K=1 and the best of K in
    {2,3,4}), then evaluates **held-out predictive surprise** for that color on a
    disjoint TEST subset, against the full competing mixture (other colors kept as
    learned). Returns ``(structural, scores)`` where ``structural`` requires BOTH:

      (a) ``surprise_bestK < floor_thresh`` — the split makes the color *predictable*
          (structure is reducible; irreducible noise stays at its floor, which sits
          above ``floor_thresh`` = the validated 0.7 alarm threshold), AND
      (b) ``surprise_bestK <= surprise_K1 - margin`` — the *split* is what did it
          (the validated 0.1-nat keep margin from live probation), not the
          single-component fit already predicting well.

    Returns ``(False, {...})`` when there is insufficient evidence (fewer than
    ``MIN_REPLAY_PAIRS`` pairs, or a degenerate train/test split).
    """
    n = len(loud_pairs)
    if n < MIN_REPLAY_PAIRS:
        return False, {"reason": "insufficient_pairs", "n_pairs": n}

    n_train = int(round(0.7 * n))
    train = loud_pairs[:n_train]
    test = loud_pairs[n_train:]
    if len(train) < MIN_REPLAY_PAIRS or len(test) < 5:
        return False, {"reason": "degenerate_split", "n_train": len(train), "n_test": len(test)}

    # K=1 candidate: replace ONLY the loud color with a single-Gaussian train fit.
    w1, m1, c1 = batch_jump_em(train, 1, rng)
    comps_k1 = em_result_to_components(w1, m1, c1, np.array([float(len(train))]))
    cand_k1 = [list(cc) for cc in components]
    cand_k1[loud_color] = comps_k1

    # best-K candidate: replace the loud color with the penalized-best split.
    best_k, comps_best, n_eff, _ = select_best_k(train, rng)
    cand_best = [list(cc) for cc in components]
    cand_best[loud_color] = comps_best

    s_k1 = _held_out_color_surprise(test, cand_k1, loud_color)
    s_best = _held_out_color_surprise(test, cand_best, loud_color)

    structural = (s_best < floor_thresh) and (s_best <= s_k1 - margin)
    return structural, {
        "n_pairs": n,
        "n_train": len(train),
        "n_test": len(test),
        "best_k": best_k,
        "heldout_surprise_k1": round(s_k1, 4),
        "heldout_surprise_bestK": round(s_best, 4),
        "structural": structural,
    }


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


def classify_regime(
    sig: RunSignals,
    *,
    alarm_thresh: float = ALARM_THRESH,
    jump_margin: float = JUMP_MARGIN,
) -> str:
    """Classify the cause of surprise from one run's signals.

    Decision order (each branch is the cheapest sufficient discriminator):

      1. **stable_known** — surprise never got high: the loud color's final mean is
         below ``alarm_thresh`` AND the global ceiling never fired. The map is
         adequate; do nothing.
      2. **nonstationarity** — surprise was LOW early then JUMPED: ``early_mean`` is
         below ``alarm_thresh`` while ``late_mean`` is above it by at least
         ``jump_margin``. The model was adequate, then the world changed → forget,
         not grow. Checked before the structural test because a post-change color
         can also look splittable.
      3. **structural_inadequacy** — high and persistent, and the penalized split
         test fired (``sig.structural``): one component is hiding multiple causes →
         grow / split.
      4. **irreducible_noise** — high and persistent, but splitting does not help →
         quarantine, do not keep growing.

    Returns one of ``REGIMES_4``.
    """
    high = (sig.loud_mean >= alarm_thresh) or sig.ceiling_fired
    if not high:
        return "stable_known"
    if sig.early_mean < alarm_thresh and (sig.late_mean - sig.early_mean) >= jump_margin:
        return "nonstationarity"
    if sig.structural:
        return "structural_inadequacy"
    return "irreducible_noise"
