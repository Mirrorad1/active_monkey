# Problem 2 — the phase picture (synthesis of Exp 133–139)

> Synthesis document for the closed `continuous-substrate` direction (card:
> `loop/directions/continuous-substrate.md`, supplied program:
> `docs/research/problem2-continuous-substrate.md`). Every claim cites its
> experiment; every experiment's script + raw output is committed. This document
> adds NO new claims — it arranges the logged ones. Written at direction closure
> (Exp 138 stop condition; Exp 139 was the post-closure consolidation audit).

## The question

Does replacing the enumerated state `s ∈ {1..N}` with a continuous latent
`s ∈ ℝᵈ` — closed-form conjugate inference, online, gradient-free — remove the
tabular ceiling, and where exactly does it break?

## The answer, in one paragraph

The tabular substrate was **not load-bearing for the repo's collapse finding**
(noise erosion is conjugate arithmetic, identical in both substrates — Exp 135),
but it **is a genuine ceiling under out-of-model input** (blends are unboundedly
catastrophic for tables, bounded for continua — Exp 134). In prediction, the
closed-form continuous agent ties both its tabular twin (in-model: Exp 133),
and a trained amortized encoder and the exact posterior (Exp 138); the fork's
real currencies at toy scale are the training bill, the conjugacy tax, and two
design laws about what may be forgotten and what may be decayed.

## The six rungs

| Rung | Exp | Verdict | The datum |
|---|---|---|---|
| 1 Convergence | 133 | POSITIVE 8/8 | Precision accumulation localizes; well-specified tabular twin's early edge ≤ 0.03 nats, characterized |
| 2 Interpolation | 134 | NEGATIVE (sign-reversed) → NEW INSIGHT | Interpolation everywhere (0/96 snaps); Σ never widens; **blends break tables, not continua** (exact-Bayes majority-corner collapse 46/46, cost ~ (L²/σ²), unbounded; continuum floor ln 4) |
| 3 Collapse rematch | 135 | POSITIVE (the card's "deep negative" in lawful form) | Erosion law δ(n) = n/(κ_eff+n) within 0.015; half-dose **linear in banked mass**; substrate-independent (twin ratio 1.52); ν₀ null for means |
| 4 Dimensionality | 136 | MIXED | Quality d-invariant to d=32 under pinned Mahalanobis separation; cost alive (8.3 ms at d=32); **exponents unmeasurable at toy scale** |
| 5 Tracking | 137 | NEGATIVE → NEW INSIGHT | Static-prior forgetting loses to plain EMA (up to 6×); **decay counts, not location** → exact EMA tie + cube-root window law ([80,20,10] vs [62.6,24.8,9.9]) |
| 6 Amortized control | 138 | MIXED | Three-way predictive tie (closed −exact 37 mnat; encoder−closed −11 mnat); conjugacy taxes location ~4%, not prediction; **12.8M training samples vs zero** |

Post-closure audit: Exp 139 (fresh seeds) demoted Exp 135's post-hoc ν₀-NLL
observation to a baseline-midpoint metric confound; the ν₀-null law replicated
out-of-sample.

## The laws (with their binding ranges)

1. **Mass-linear erosion (Exp 135).** Under an anchored state, noise-phase drift
   of learned emission means is δ(n) = n/(κ₀ + n_struct + n); collapse
   resistance is bankable, linear in accumulated evidence, dialed by κ alone.
   Binds: anchored learning, equal known footprints, d=2. Substrate-independent
   (Dirichlet twin obeys the same law). This extends the social chapter's
   dose-vs-accumulated-mass law across substrates.
2. **The forgetting-window law, continuous form (Exp 137 + Exp 88 cross-link).**
   Tracking error vs forgetting window is U-shaped; the optimum follows the EMA
   ramp law N* = (σ²/2v²)^{1/3}. Holds ONLY under count-decay
   (`NIW.decay(keep_mean=True)`): the decayed quantity must be evidence mass,
   never location — the static-prior form adds a position-dependent bias that
   grows with distance travelled. At v=0 the unforgetting conjugate annealer
   beats every fixed rate.
3. **Out-of-model asymmetry (Exp 134).** A static-state categorical posterior's
   log-odds between candidate atoms scale as (count imbalance) × (separation²/σ²):
   one surplus word collapses exact Bayes to a corner with unbounded held-out
   cost. A unimodal continuous posterior interpolates instead and pays at most
   ~ln(M). The price of the continuum: Σ does not widen on blends — confidence
   is misplaced even when the mean is reasonable (Gap 1 of the program,
   confirmed exactly).

## Licensed claims and their bounds

- LICENSED: "closed-form ≈ amortized ≈ exact in prediction at toy scale; the
  differences are the training bill (12.8M samples vs 0) and a small conjugacy
  tax (37 mnat predictive, ~4% localization)" — bounded by: minimal encoder, one
  geometry, d=2, T=50 (Exp 138).
- LICENSED: "the collapse clause of the RECIPE is substrate-independent" (Exp 135).
- NOT LICENSED: any closed-form-vs-amortized claim **at scale** — rung 4 showed
  the cost exponents are invisible below d≈32 in numpy (overhead-dominated), so
  the crossover question is provably out of toy reach (Exp 136).
- NOT LICENSED: anything about unanchored symmetry breaking — every rung had the
  anchor PROVIDED; tabula-rasa remains the documented ceiling
  (`open_problem.html`).

## Open edges (named, not opened)

- **Problem 2b**: continuous observations (breaks conjugacy unless linear; the
  program's named next increment).
- Larger-d exponents (d ≥ 128) for the cost-crossover question.
- Regime-switch (non-constant-velocity) tracking, where conjugate adaptivity
  might genuinely beat fixed rates rather than tie.
- The κ₀=1 residual in Exp 139 (a possibly-real small effect beyond the metric
  artifact; parked).
- Substrate migration: porting the creature's RECIPE chain onto
  `active_loop/continuous.py` — reserved for an explicit human call
  (closure CONSULT in `loop/IDEAS.md`).

## Artifact index

- Core math: `active_loop/continuous.py` (GaussianBelief, gaussian_product, NIW
  with update/update_batch/update_moments/decay(keep_mean),
  log_categorical_posterior, predictive_word_logprobs) — 10 durable guards in
  `tests/test_continuous.py`.
- Experiments: `experiments/exp133–139_*.py` + `experiments/outputs/exp13*.txt`
  + JSON rows; mechanism checks: `exp134_exact_bayes_check.py`,
  `exp137_mechanism_check.py` (both bit-identity-verified against their
  originals).
- Process guards added during the ladder: log-space filtering rule (card),
  decay-form rule (card), three-way verdict rule (PROTOCOL step 3).
