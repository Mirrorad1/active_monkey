# Structure learning: surprise ceilings, BMR, and endogenous model expansion

## 1. Motivation

The RECIPE's parameter-learning loop (`creature.py`) accumulates Dirichlet counts
`pA[obs, cell] += qs_cell` at every step.  Over many steps on a fixed-size grid,
the sensory map converges: `map_accuracy` approaches 1.0 and `surprise_mean` falls.

But this is parameter learning within a fixed state space.  If the world is more
complex than the creature's model — more hidden states than grid cells, aliased
observations that a single-level HMM cannot resolve, or latent structure (context,
objects, other agents) — the surprise floor will remain elevated even after the
parameters have converged.  The creature is stuck at an **irreducible surprise
ceiling** that parameter tuning alone cannot lower.

The human directive of 2026-06-10 introduced Phase 2-3 to address this:

- **Phase 2 (Bayesian Model Reduction)**: score the existing model structure,
  identify unused states, and produce a ranked pruning report.
- **Phase 3 (Expansion)**: when the surprise ceiling detector fires and persists,
  provision new hidden states seeded at the offending observation distribution.

Both phases are **flag-gated scaffolds** — not wired into any default run.
The flag gate ensures behavior-invariance of the production creature until the
full Phase 2-3 loop has been validated on held-out experiments.

---

## 2. F as a bound on log evidence

For the creature's exact-filtering discrete HMM, the accumulated negative log
predictive probability *is* the negative log-evidence (no bound slack):

```
F = Σ_t  -ln p(o_t | o_{<t}, a_{<t})
```

Each term is the negative log of the one-step predictive distribution under
exact Bayesian filtering.  For a discrete HMM with exact forward recursion,
the variational bound is tight — there is no free-energy gap.

The general complexity–accuracy decomposition of variational free energy is:

```
F = KL[q(s) || p(s)] - E_q[ln p(o | s)]
  = complexity - accuracy
```

where `complexity` is the KL divergence from prior to posterior (the cost of
updating beliefs) and `accuracy` is the expected log-likelihood under the
posterior.  In the exact-filtering case, the two terms collapse into the
accumulated negative log-predictive above.

**Key implication**: lower F on the same history = better model.  Candidates are
comparable only on the same observation-action history.

---

## 3. The surprise ceiling

Phase 1 (`creature.py`) instruments a rolling window of per-step surprise:

```
surprise_t = -ln p(o_t | o_{<t}, a_{<t})
```

A **surprise ceiling** is declared when the window (length 200) satisfies all
three conditions simultaneously:

| Condition | Threshold | Meaning |
|-----------|-----------|---------|
| `mean_surprise > 0.7` nats | `CEILING_MEAN_THRESH = 0.7` | Surprise is elevated (> half of ln(3) ≈ 1.099, the uniform-3-color baseline) |
| `\|slope\| < 5e-4` nats/step | `CEILING_SLOPE_THRESH = 5e-4` | Surprise is not declining |
| `learning_active = True` | always True in current variant | Learning is running but not helping |

Operationally: a surprise ceiling means **the model's predictive ability has
plateaued at a level that parameter updates cannot lower**.  This is a structural
signal — the model space is too small.

---

## 4. BMR: closed-form Dirichlet formula

Given:
- `a_post[:, j]` — posterior Dirichlet counts for column (hidden state) `j`
- `a0_prior[:, j]` — original prior counts
- `a0_reduced[:, j]` — reduced prior counts (e.g., column zeroed to epsilon)

Define the multivariate log-Beta:

```
ln B(x) = Σ_i gammaln(x_i) - gammaln(Σ_i x_i)
```

The combined posterior under the reduced prior:

```
a_tilde_j = a_post_j + a0_reduced_j - a0_prior_j   (clipped at 1e-10 component-wise)
```

The per-column BMR delta (log Bayes factor contribution):

```
ΔF_j = ln B(a_post_j) + ln B(a0_reduced_j) - ln B(a0_prior_j) - ln B(a_tilde_j)
```

Summed over all columns:

```
ΔF = Σ_j ΔF_j
```

### Sign convention (PINNED)

```
ΔF = ln p(data | reduced) - ln p(data | full)
```

- **POSITIVE ΔF** → reduction is favored (the reduced model explains the data at
  least as well as the full model).
- **NEGATIVE ΔF** → full model is better (reduction discards useful structure).

### Two canonical test cases

1. **Unused state (state 3 in a 4-state model, no data allocated to it)**:
   `a_post[:, 3] == a0_prior[:, 3]`.  Reducing state 3 yields **positive ΔF**
   (the prior was the posterior; the reduced model is not penalized).

2. **Heavily-used state (state 0, +50 peaked counts)**:
   `a_post[0, 0] = a0_prior[0, 0] + 50`.  Reducing state 0 yields **negative ΔF**
   (the posterior diverged strongly from the prior; the reduction discards evidence).

Both cases are pinned in `tests/test_structure_bmr.py`.

---

## 5. The spawn rule

Phase-3 trigger predicate (`spawn_rule_check` in `structure.py`):

```
min_s [ -ln p(o_t | s) ] > threshold
AND
consecutive_flagged >= K
```

Where:
- `threshold` = the surprise ceiling threshold (0.7 nats, matching `CEILING_MEAN_THRESH`).
- `K` = minimum run of consecutive ceiling-flagged steps required before spawning.
- `min_s` = the minimum surprise across all existing hidden states at step `t`.

The minimum-over-states criterion ensures the trigger fires only when **no existing
state** can explain the observation — not just the current MAP state.

When the predicate fires, a provisional state is appended via `spawn_state`:

```
new_col = obs_dist * 1.0 + weak * uniform
```

where `obs_dist` is the offending observation distribution (the one-hot or
distribution at the ceiling step) and `weak = 0.1` is a uniform regulariser.

**Survival rule**: the new state must survive the next `prune_pass` (obtain a
positive ΔF when tested as a *keep* rather than a *prune* candidate).  States
that do not earn their place are deleted by the caller.

---

## 6. The active-data bias caveat

Replay is generated under the **old policy** (random walk in the current
creature).  Candidate scoring via `candidate_score` computes F on this fixed
history.

This creates an **active-data bias**: a candidate model that would change
behavior (e.g., an expanded model that would explore different cells) is scored
on data it did not generate.  The comparison is valid only when:

1. Both models are evaluated on **the same history**.
2. The behavioral consequences of the structural change are **not** being
   evaluated (only the explanatory quality of past data is assessed).

**Named limitation**: candidates that would substantially change policy cannot be
fairly evaluated by replay scoring alone.  The improvement may be underestimated.

**Future mitigation**: on-policy evaluation epochs — short live() runs under the
candidate model, comparing the resulting F on fresh data.  Not implemented in
Phases 2-3 scaffolds.

---

## 7. Phase status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Surprise ceiling detector + replay buffer instrumentation | **Live** (instrumentation-only; behavior-invariance verified bit-exactly in `test_structure_phase1.py`) |
| 2 | Replay scoring (`candidate_score`), BMR delta-F (`bmr_delta_f`), pruning report (`prune_pass`) | **Scaffolded** — flag-gated, not wired into default runs |
| 3 | State expansion operators (`spawn_state`, `add_state`, `split_state`, `merge_states`), variant selection (`select_variant`), spawn trigger (`spawn_rule_check`) | **Scaffolded** — flag-gated, not wired into default runs |

No default run calls any function in `structure.py`.  Every entry point requires
`enabled=True` passed explicitly.  The FROZEN manifest paths are untouched.
