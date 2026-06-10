"""
Exp 133 — continuous-substrate rung 1: convergence to a point (continuous vs tabular twin).

Hypothesis: closed-form precision accumulation on a continuous latent s in R^2
localizes the posterior at a true concept point from one concept's word stream,
no slower (in held-out predictive NLL) than the tabular twin on the identical stream.

Declared modeling choices (per IDEAS.md build constraints, fixed for the ladder):
- The agent's emission likelihood is the UNNORMALIZED Gaussian footprint
  exp(-0.5 (s-mu_k)^T Sigma_k^-1 (s-mu_k)) — the conjugacy-buying approximation.
  The data GENERATOR uses the normalized mixture p_k propto N(s*; mu_k, Sigma_k).
  The mismatch (posterior fixed point = precision-weighted mean of observed word
  centers, analytically biased toward the centroid by neighbor-word leakage) is
  part of what's measured; predicted analytic bias at this geometry ~ 0.017,
  far below the P1 threshold.
- Anchor GIVEN to both agents (rung 1 does not learn emissions): the continuous
  agent gets the 6 (mu_k, Sigma_k); the tabular twin gets the true emission
  table A[k,c] = normalized mixture evaluated at mu_c. NIW learning enters at
  rung 3.

Geometry (fixed): 6 word-Gaussians at hexagon vertices, radius 1, Sigma_k =
0.35^2 I; concept = word-0 center s* = (1, 0); prior N(0, 4I); T = 200 training
observations, 200 held-out words from the same generator; seeds 0..7.

Predictions (hypothesis TRUE iff all hold):
- P1 localization: final ||mu_post - s*|| < 0.1 in >= 7/8 seeds.
- P2 contraction: tr(Sigma_post) non-increasing at every step in 8/8 seeds AND
  final tr(Sigma_post) < 1% of prior trace (instrument sanity).
- P3 twin race: mean held-out predictive NLL over the first 50 training steps —
  continuous <= tabular + 0.10 nats in >= 6/8 seeds.

Falsifier (any triggers NEGATIVE): >= 2/8 seeds end with error >= 0.1 (no
localization), OR P2 contraction violated in any seed, OR continuous worse than
tabular by > 0.10 nats early in >= 6/8 seeds (tabular decisively faster — log
the regime per the direction card's FAIL clause). Outcomes between P3-pass and
the P3 falsifier are MIXED, judged per the raw rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.linalg import inv

from active_loop.continuous import GaussianBelief, predictive_word_logprobs

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

N_WORDS = 6
SIGMA_K_SCALE = 0.35
RADIUS = 1.0
D = 2
T_TRAIN = 200
T_HOLDOUT = 200
N_SEEDS = 8
PRIOR_MU = np.zeros(D)
PRIOR_SIGMA = 4.0 * np.eye(D)

# Hexagon word centres
_angles = np.array([k * np.pi / 3.0 for k in range(N_WORDS)])
WORD_MUS = [np.array([np.cos(a), np.sin(a)]) for a in _angles]
WORD_SIGMAS = [SIGMA_K_SCALE**2 * np.eye(D) for _ in range(N_WORDS)]
WORD_LAMBDAS = [inv(S) for S in WORD_SIGMAS]

# True concept centre: word-0 (1, 0)
S_STAR = WORD_MUS[0].copy()

# Steps to log: dense 1..50, then every 10th up to 200
_LOG_STEPS = set(range(1, 51)) | set(range(60, T_TRAIN + 1, 10))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _emission_probs(s: np.ndarray) -> np.ndarray:
    """Normalized emission probabilities p_k = N(s; mu_k, Sigma_k) / Z."""
    log_p = np.array([
        -0.5 * float((s - mu_k) @ WORD_LAMBDAS[k] @ (s - mu_k))
        - 0.5 * D * np.log(2.0 * np.pi)
        - 0.5 * np.log(np.linalg.det(WORD_SIGMAS[k]))
        for k, mu_k in enumerate(WORD_MUS)
    ])
    log_p -= float(np.logaddexp.reduce(log_p))
    return np.exp(log_p)


def _generate_stream(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_words, holdout_words) for this seed."""
    rng = np.random.default_rng(seed)
    p = _emission_probs(S_STAR)
    p = p / p.sum()  # ensure exact normalisation
    train = rng.choice(N_WORDS, size=T_TRAIN, p=p)
    holdout = rng.choice(N_WORDS, size=T_HOLDOUT, p=p)
    return train, holdout


# ---------------------------------------------------------------------------
# Tabular twin
# ---------------------------------------------------------------------------

def _build_tabular_A() -> np.ndarray:
    """A[k, c] = p(word=k | concept=c) = N(mu_c; mu_k, Sigma_k), column-normalised.

    Columns index concepts (c=0..5 at hexagon vertices); rows index words (k=0..5).
    """
    A = np.zeros((N_WORDS, N_WORDS))
    for c in range(N_WORDS):
        mu_c = WORD_MUS[c]
        for k in range(N_WORDS):
            mu_k = WORD_MUS[k]
            diff = mu_c - mu_k
            log_density = (
                -0.5 * float(diff @ WORD_LAMBDAS[k] @ diff)
                - 0.5 * D * np.log(2.0 * np.pi)
                - 0.5 * np.log(np.linalg.det(WORD_SIGMAS[k]))
            )
            A[k, c] = np.exp(log_density)
    # Column-normalise
    col_sums = A.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    A = A / col_sums
    return A


TABULAR_A = _build_tabular_A()

# Tabular concept locations for localization diagnostic
CONCEPT_MUS = np.array(WORD_MUS)  # (6, 2)


def _tabular_holdout_nll(q: np.ndarray, holdout: np.ndarray) -> float:
    """Mean -log p(word | belief q) over holdout words."""
    eps = 1e-300
    total = 0.0
    for word in holdout:
        p_word = float(TABULAR_A[word, :] @ q)
        total += -np.log(p_word + eps)
    return total / len(holdout)


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------

def run_seed(seed: int) -> tuple[list[dict], dict]:
    """Run both agents for one seed; return (rows, summary)."""
    train, holdout = _generate_stream(seed)
    rows = []

    params = {
        "d": D,
        "sigma_k": SIGMA_K_SCALE,
        "radius": float(RADIUS),
        "T": T_TRAIN,
    }

    # ------------------------------------------------------------------
    # Continuous agent
    # ------------------------------------------------------------------
    belief = GaussianBelief(PRIOR_MU, PRIOR_SIGMA)
    prior_trace = belief.trace_sigma

    cont_early_nlls = []  # steps 1..50
    prev_trace_sigma = belief.trace_sigma
    p2_violated = False

    for t, word_k in enumerate(train, start=1):
        belief.observe(WORD_MUS[word_k], WORD_LAMBDAS[word_k])
        cur_trace_sigma = belief.trace_sigma
        if cur_trace_sigma > prev_trace_sigma + 1e-12:
            p2_violated = True
        prev_trace_sigma = cur_trace_sigma

        if t in _LOG_STEPS:
            mu_post = belief.mu
            Sigma_post = belief.Sigma
            loc_err = float(np.linalg.norm(mu_post - S_STAR))
            tr_sigma = belief.trace_sigma
            prec_tr = belief.precision_trace
            ent = belief.entropy

            # Held-out NLL
            hnll = -float(np.mean([
                predictive_word_logprobs(mu_post, Sigma_post, WORD_MUS, WORD_SIGMAS)[w]
                for w in holdout
            ]))

            if t <= 50:
                cont_early_nlls.append(hnll)

            for metric, val in [
                ("loc_err", loc_err),
                ("trace_sigma", tr_sigma),
                ("precision_trace", prec_tr),
                ("entropy", ent),
                ("holdout_nll", hnll),
            ]:
                rows.append({
                    "exp": 133, "rung": 1, "agent": "continuous",
                    "seed": seed, "step": t,
                    "metric": metric, "value": val,
                    "params": params,
                })

    final_mu = belief.mu
    final_loc_err = float(np.linalg.norm(final_mu - S_STAR))
    final_tr_sigma = belief.trace_sigma

    # ------------------------------------------------------------------
    # Tabular twin
    # ------------------------------------------------------------------
    q = np.ones(N_WORDS) / N_WORDS  # uniform prior
    tab_early_nlls = []

    for t, word_k in enumerate(train, start=1):
        q = q * TABULAR_A[word_k, :]
        denom = q.sum()
        if denom < 1e-300:
            q = np.ones(N_WORDS) / N_WORDS
        else:
            q = q / denom

        if t in _LOG_STEPS:
            tab_hnll = _tabular_holdout_nll(q, holdout)
            tab_ent = float(-np.sum(q * np.log(q + 1e-300)))
            tab_loc = float(np.linalg.norm(CONCEPT_MUS.T @ q - S_STAR))

            if t <= 50:
                tab_early_nlls.append(tab_hnll)

            for metric, val in [
                ("holdout_nll", tab_hnll),
                ("entropy", tab_ent),
                ("loc_diag", tab_loc),
            ]:
                rows.append({
                    "exp": 133, "rung": 1, "agent": "tabular",
                    "seed": seed, "step": t,
                    "metric": metric, "value": val,
                    "params": params,
                })

    # Summary
    cont_early_mean = float(np.mean(cont_early_nlls)) if cont_early_nlls else float("nan")
    tab_early_mean = float(np.mean(tab_early_nlls)) if tab_early_nlls else float("nan")
    summary = {
        "seed": seed,
        "final_loc_err": final_loc_err,
        "final_tr_sigma": final_tr_sigma,
        "prior_trace": float(prior_trace),
        "p2_violated": p2_violated,
        "cont_early_nll": cont_early_mean,
        "tab_early_nll": tab_early_mean,
        "gap_cont_minus_tab": cont_early_mean - tab_early_mean,
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: list[dict] = []
    summaries: list[dict] = []

    for seed in range(N_SEEDS):
        rows, summary = run_seed(seed)
        all_rows.extend(rows)
        summaries.append(summary)

    # ------------------------------------------------------------------
    # Write JSON output
    # ------------------------------------------------------------------
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp133_rows.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=None)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Exp 133 — continuous-substrate rung 1: convergence to a point")
    print("=" * 72)
    print()
    print(
        f"{'Seed':>4}  {'FinalLocErr':>11}  {'FinalTrSigma':>12}  "
        f"{'ContEarlyNLL':>12}  {'TabEarlyNLL':>11}  {'Gap(C-T)':>9}  "
        f"{'P2Viol':>6}"
    )
    print("-" * 72)
    for s in summaries:
        print(
            f"{s['seed']:>4}  {s['final_loc_err']:>11.6f}  "
            f"{s['final_tr_sigma']:>12.6f}  "
            f"{s['cont_early_nll']:>12.6f}  "
            f"{s['tab_early_nll']:>11.6f}  "
            f"{s['gap_cont_minus_tab']:>+9.6f}  "
            f"{'YES' if s['p2_violated'] else 'no':>6}"
        )
    print()

    # ------------------------------------------------------------------
    # Predicate tallies
    # ------------------------------------------------------------------
    p1_threshold = 0.1
    p1_pass = sum(1 for s in summaries if s["final_loc_err"] < p1_threshold)
    p2_ok = sum(1 for s in summaries if not s["p2_violated"])
    prior_trace = summaries[0]["prior_trace"]
    p2_sanity = sum(
        1 for s in summaries if s["final_tr_sigma"] < 0.01 * prior_trace
    )
    p3_threshold = 0.10
    p3_pass = sum(
        1 for s in summaries
        if s["gap_cont_minus_tab"] <= p3_threshold
    )

    print(f"Prior tr(Sigma) = {prior_trace:.4f}")
    print()
    print(f"P1 localization  (final err < {p1_threshold}): {p1_pass}/8 seeds  "
          f"(need >= 7 for PASS)")
    print(f"P2 contraction   (no violation in all steps): {p2_ok}/8 seeds  "
          f"(need 8/8 for PASS)")
    print(f"P2 sanity        (final tr < 1% prior tr):    {p2_sanity}/8 seeds  "
          f"(need 8/8 for PASS — second P2 conjunct)")
    print(f"P3 twin race     (gap <= +{p3_threshold} nats early): {p3_pass}/8 seeds  "
          f"(need >= 6 for PASS)")
    print()

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    # Falsifier conditions
    p1_localization_fail = (N_SEEDS - p1_pass) >= 2  # >=2 seeds end error >= 0.1
    p2_contraction_fail = p2_ok < N_SEEDS             # any seed violates
    p3_falsifier = (N_SEEDS - p3_pass) >= 6           # continuous worse in >=6/8
    p3_pass_verdict = p3_pass >= 6

    if p1_localization_fail or p2_contraction_fail or p3_falsifier:
        verdict = "NEGATIVE"
        reasons = []
        if p1_localization_fail:
            reasons.append(
                f"P1 fail: {N_SEEDS - p1_pass}/8 seeds have final error >= {p1_threshold}"
            )
        if p2_contraction_fail:
            reasons.append(
                f"P2 fail: contraction violated in {N_SEEDS - p2_ok}/8 seeds"
            )
        if p3_falsifier:
            reasons.append(
                f"P3 falsifier: continuous worse than tabular by >{p3_threshold} nats "
                f"in {N_SEEDS - p3_pass}/8 seeds"
            )
        print(f"VERDICT: {verdict}")
        for r in reasons:
            print(f"  - {r}")
    elif p1_pass >= 7 and p2_ok == 8 and p2_sanity == 8 and p3_pass_verdict:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict} — all P1/P2/P3 conditions satisfied "
              f"(both P2 conjuncts)")
    else:
        verdict = "MIXED"
        notes = []
        if p1_pass < 7:
            notes.append(f"P1 borderline: {p1_pass}/8 (need 7)")
        if p2_sanity < 8:
            notes.append(f"P2 sanity conjunct: {p2_sanity}/8 (need 8)")
        if not p3_pass_verdict:
            notes.append(f"P3 borderline: {p3_pass}/8 (need 6)")
        print(f"VERDICT: {verdict}")
        for n in notes:
            print(f"  - {n}")

    print()
    print(f"Rows written to: {out_path}")


if __name__ == "__main__":
    main()
