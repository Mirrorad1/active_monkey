# Exp282 verification record

Scope: deterministic checks of the finite-flux conformal-entropy formulas.
Numerical agreement is not global optimality, a novel theorem, a constructed
nonsaturating quantum state, or a publication decision.

## Parent source inspection

The parent inspected the runner before the full run. The ODE integrator carries
its numerical `(w,w',integral((w'/w)^2))` state across all four half-pulse
segments without resetting it to the analytic curve. DOP853 uses relative
tolerance `2e-12`, absolute tolerance `2e-13`, and bounded maximum steps.
The duration quadrature uses quadratic endpoint substitutions, not a second
evaluation of the inverse-trigonometric formula. Sinusoidal caps are analytic;
polynomial caps use all real derivative roots and endpoints in float64, with
that numerical limitation stated in the raw output.

The bounded smoke run took `0.005736625 s` internally and projected
`0.136476271 s` of serial ODE work for 30 cases. The complete run used four
threads for independent cases, took `0.254082708 s` internally, and retained
every case. This small timing underestimate changes no numerical criterion.
There was one source-matched parent smoke and one parent full execution with
saved outputs. The worker also ran an unsaved smoke during implementation.
Neither source nor thresholds changed after the saved smoke/full execution.

The executed runner, focused tests, and pre-run plan match every hash in
`exp282.json`'s `source_sha256` field. The full-run hashes are:

- JSON: `e5fb279df45cf0da3ab4b948d81583f31429ca8fdd1340a14fed897e36deb6b4`
- Text: `db1eaed622c814e724b8ed7c582b68d68ab6595983aafbe329dd6d3cb021a6d2`

The rational-algebra backend is Python `fractions`, not SymPy. Seven exact
residuals vanish. Maximum normalized residuals over all 30 cases:

| Quantity | Maximum | Declared bound |
| --- | ---: | ---: |
| Final stationary endpoint | 1.767950080122218e-13 | 2e-7 |
| Apex state | 7.116287590804858e-13 | 2e-7 |
| Switch matching | 1.181889124056471e-14 | 2e-7 |
| Sampled ODE profile | 1.994609617923251e-12 | 2e-7 |
| Integrated energy identity | 2.804877411176352e-11 | 2e-7 |
| Envelope duration quadrature | 1.8778264473673817e-15 | 2e-8 |

All 24 nonoptimal profiles have `T/Tmin >= 1.0774331745485477`, above the
declared `1-2e-7` threshold. Their constant continuations are `C1` and
`W^{2,infinity}`, not globally smooth. Six equal-cap inversions and sinusoidal
comparisons pass. The QNEC inverse example has root residual
`4.440892098500626e-16`, `bcrit=2.203056359208288`, and
`D/K=3.8534573218480777`. Its derivative error is
`2.9762858844151197e-11`; the large-b asymptotic sequence is diagnostic only,
with no acceptance threshold invented after the run.

## Independent mathematical review

The `hep_flux_verifier` worker received the assumptions and candidate proof,
then reviewed the complete manuscript. This review was independent but not
blinded. It checked the original Bianchi--Smerlak normalization and the
Stefanatos correspondence against primary sources. It found no substantive
mathematical or physical blocker. Specifically it checked:

- The connected-component speed envelope and Lipschitz proof, including
  arbitrarily many extra turns.
- Exact peak restoration after mollification and time dilation; both original
  caps remain satisfied, and duration tends to the weak-class optimum.
- The distinction between an attained piecewise-smooth optimum and a smooth
  infimum; no energy delta pulses at the finite-cap switches.
- QNEC must use the same null entropy/deformation, with the finite-interval
  vacuum-subtraction limitation explicit.
- The inverse-gap condition, smooth strictness, positive-cap floor and its
  divergence coefficient, and the absence of a realizability claim.
- Equal-cap asymptotics and sinusoidal peak-flux comparison.

The reviewer caught an exponent typo `bcrit^{,2}`, now corrected. We also
clarified which one-sided floor was meant, used the arbitrary `w` in the
uniqueness proof, sharpened the small-height remainder, and narrowed priority
wording for the elementary unequal-cap extension. The final PDF contains all
these corrections. No experiment source or output needed modification.

## Independent literature review

The `hep_flux_prior_art` worker searched primary literature independently.
Its decisive result was the exact equal-cap minimum-time theorem in
Stefanatos, arXiv:1301.5137, Theorem 4. The correspondence is
`x1=w`, `x2=w'`, `x2'=-u x1`, with bounded `u`; its one-way `(1,0)->(R,0)`
solution is the rising half of this entropy pulse. The no-extra-turn result
is also already within that theorem's scope. It cannot be sold as a new
optimal-control theorem.

Karmoudi--Daoud, arXiv:2610.03576v2, contains the integrated-energy optima
that defeated the initial research idea. Teo--Wong, gr-qc/0206066, contains
related hyperbolic/trigonometric square-pulse matching. The bounded search
found no exact published CFT application or inverse peak-QNEC-gap diagnostic,
but reported their significance as modest and priority as unproved. The
manuscript and README preserve these qualifications. There is no positive
publication verdict.

## Software and document checks

All 16 focused tests passed in both the worker run and the parent's fresh run.
The final source compiled successfully with the desktop LaTeX compiler and
Tectonic 0.17.0. `pdfinfo` reports four pages including references, letter paper,
no JavaScript, and no encryption. The parent inspected rendered images of all
four final pages: no clipping, overlapping equations, or broken references.
The TeX log contains no overfull/underfull-box warnings or undefined citations.
Final PDF SHA256:
`7201759056364696d8670920340aa2cb81830a933f7c2f1d9e3bc46282e0de7f`.

The manuscript has no invented author/affiliation. It is a research draft;
no arXiv submission, endorsement check, or editorial acceptance is claimed.

## Detached raw-data recomputation

The separate `hep_flux_raw_verifier` received only the runner's opening
predeclaration and raw JSON/text, not the parent interpretation or manuscript.
It disclosed that aggregate flags were visible and ignored them: this is
detached recomputation, **not strict blinding**. It did not inspect the runner
implementation or rerun the ODE integrator, and edited no repository files.

Verdict: **POSITIVE within the inspected numerical scope**. It reconstructed
all seven symbolic identities and all 30 endpoint, apex, switch, energy, and
archived-quadrature residuals from the supplied definitions and primitive
boundary values. Together with 24 baseline inequalities, all 181 requested
core conjuncts pass. It confirmed complete six-height/five-ratio coverage and
agreement of JSON/text on their 18 shared top-level fields.

Fresh endpoint-regularized 128-node Gauss--Legendre quadrature independently
agreed in every case (maximum normalized error `7.014239233985472e-15`).
Rederiving the baseline caps, with factored polynomial derivatives and refined
interior roots, gave a maximum normalized cap difference
`2.5535189461757313e-14` and duration-ratio difference
`4.321486253417685e-15`. All six ancillary equal-cap checks also passed.
A fresh inverse solve gave `bcrit=2.203056359208285` and
`D/K=3.8534573218480643`, agreeing at floating-point precision.

Raw-data limit: the archive retains five boundary states per ODE case and
120 segment-profile residual summaries, not all 7,800 sampled states or the
ODE right-hand side. The sample-summary maximum is below tolerance but cannot
be reconstructed from those boundary states alone. Parent implementation
inspection and focused tests are separate evidence, not a cure for that raw
reconstruction limit. No independently reproduced ODE-trajectory claim is made.
Floating-point extrema are not certified interval calculations. This verdict
does not establish global optimality, smooth sharpness, state realizability,
priority, publication readiness, or original-source fidelity.

## Repository checks

The targeted experiment, site-data, experiment-parser, and iteration-checker
test suite passed. The iteration rubric returned PASS with four reviewed
warnings: `0.005736625` and `0.136476271` are rounded values from the saved
smoke rather than the full-run text; `0.254082708` rounds the full-run runtime;
`2610.03576` is a cited arXiv identifier, not a numerical observation. No stale
scientific value was substituted. All 1,236,114 bytes of the pre-existing
EXPERIMENTS.md are preserved exactly as a prefix; only Exp282 was appended.
The staged whitespace check reports one extra blank line at the focused test
file's EOF. It is retained to preserve the test bytes already pinned in both
raw runs; it has no executable or scientific effect. No other whitespace
diagnostics were reported. The static-site source/deploy consistency check
also passed.
