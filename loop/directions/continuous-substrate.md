# direction: continuous-substrate

**Question.** Does replacing the enumerated state `s ∈ {1..N}` with a continuous latent
`s ∈ ℝᵈ` — closed-form conjugate inference (precision accumulation), online, gradient-free —
remove the tabular ceiling, and where exactly does it break?

**Why it matters.** This is Problem 2 of the frontier map: the field's live fork between
closed-form Bayes (AXIOM/VBGS) and amortized neural inference (deep-AIF). The repo's collapse
finding (Exp 31) and the M4a credit-assignment wall (Exp 125–128) both live on the tabular
substrate; this direction tests whether that substrate was load-bearing. Either verdict
sharpens the PREMISE — the hand-placed word-Gaussians ARE the one innate anchor, in
continuous form. The full supplied program (math, metrics, honest gaps) is
`docs/research/problem2-continuous-substrate.md`: read it before rung 1. Observations stay
DISCRETE and the action model stays FIXED until rungs 1–3 have verdicts.

**Experiment ladder.**
1. Convergence to a point: 6 hand-placed word-Gaussians, d=2, broad prior; one concept's
   words; measure ‖μ_post − μ_true‖ and tr(Σ_post) vs the tabular twin on identical streams.
   FAIL = posterior doesn't localize, or tabular is faster with no characterizable regime.
2. Interpolation to unseen blends: 4 concepts at unit-square corners; blended stream;
   posterior should land between, with widened Σ. FAIL = snaps to a corner across the whole
   (separation, Σ) sweep — no interpolation regime (map the unimodal-approximation boundary).
3. The Exp 31 rematch: NIW-learned emission means; structured phase then noise phase;
   collapse index tr(Σ_between)/tr(Σ_within); critical noise level vs (ν₀, κ₀). FAIL =
   continuous collapses like tabular (a deep negative: collapse would be a property of
   online Bayes, not of tables — log it as such).
4. Dimensionality: d = 2, 4, 8, 16, 32; convergence quality + wall clock (O(d²) update,
   O(d³) solve). FAIL = quality or cost is dead by d≈8.
5. Non-stationary tracking: drifting μ_true(t); with/without precision forgetting; tracking
   error vs drift velocity (NIW-as-learning-rate). FAIL = no hyperparameter setting beats a
   fixed-learning-rate baseline.
6. The amortized control: a minimal ELBO-trained encoder on the SAME generative model —
   without it, no closed-form-vs-amortized claim is licensed. FAIL = the comparison can't be
   made apples-to-apples at toy scale (log why; that bounds every claim from rungs 1–5).

**Stop condition.** The phase picture exists (collapse boundary, interpolation boundary,
d-scaling curve) AND the amortized comparison is logged (either verdict) — or two iterations
stuck on numerics (covariance conditioning, not the question) → log the wall and consult.
This direction touches neither the halted M4a thread (increment 1c awaits its own human
word) nor mirro's/vela's spines.

**Build rules (loop-added, META guards).**
- [Exp 134] Tabular twins MUST filter in log space via
  `active_loop.continuous.log_categorical_posterior` — multiply-then-renormalize
  filters underflow-ratchet at large separation (entries hit exact float 0 and never
  recover, making the filter order-dependent; 7/46 argmax anomalies, all artifacts).
  Guard test: tests/test_continuous.py::test_log_categorical_posterior_order_independent_no_ratchet.
- [Exp 137] Non-stationary tracking / forgetting MUST decay evidence counts, not
  location: `NIW.decay(..., keep_mean=True)`. The default static-prior re-anchor form
  loses to a plain EMA by up to 6x under sustained drift and distorts the forgetting-
  window optimum (mechanism pinned, bit-identical replication). Decay counts, not
  location. Guard test: tests/test_continuous.py::test_niw_decay_keep_mean.

**STATUS: CLOSED 2026-06-10 — stop condition met at Exp 138** (ladder Exp 133–138; the
phase picture exists and the amortized comparison is logged). Closure CONSULT with the
six-rung verdict and next-step options posted in loop/IDEAS.md; this card re-opens only
on an explicit human choice there.
