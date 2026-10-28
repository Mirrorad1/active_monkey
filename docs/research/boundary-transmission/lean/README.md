# Lean proof supplement

This project formalizes the mathematical core of *Trait Growth against the
Selection Differential*. It uses Lean **4.19.0** and mathlib **v4.19.0**, pinned to
commit `c44e0c8ee63ca166450922a373c7409c5d26b00b` in `lake-manifest.json`.

## Check the proofs

With [Lean/elan installed](https://leanprover-community.github.io/install/project.html),
run from this directory:

```sh
lake update
lake exe cache get
lake build
lake env lean BoundaryTransmission/Audit.lean
```

The `lean-toolchain` file selects the compiler version. `lake build` checks all
modules and the axiom audit. The final command reruns the audit even when the build
is cached. No Python simulation or network access is needed after dependencies and
their compiled cache are installed. Dependency checkouts/build products live in
the ignored `.lake/`; proof sources and the dependency manifest are committed.

On this machine the native mathlib cache executable failed with the macOS loader
error `__DATA_CONST segment missing SG_READ_ONLY flag`. Running the same cache
program through Lean's interpreter successfully fetched the cache:

```sh
lake env lean --run .lake/packages/mathlib/Cache/Main.lean get
```

This is a setup workaround, not a change to Lean's proof checker. Here the compiler
was unpacked from the official `leanprover/lean4` release into a temporary directory;
the project itself contains no machine-specific compiler path.

For the compiler already downloaded in this session, the checked local command was:

```sh
PATH=/private/tmp/active-loop-lean-4.19/bin:$PATH lake build
```

## What is proved

All names below are in namespace `BoundaryTransmission`.

| Paper claim | Lean theorem | Explicit scope |
|---|---|---|
| Exact clipped-Gaussian displacement, Eq. (2) | `clippedGaussian_drift` in `Gaussian.lean` | Actual expectation under `gaussianReal 0 1`; `L ≤ U`, `σ > 0`; arbitrary parent value `h` |
| Interior Gaussian first moment | `standardGaussian_firstMoment` | Derived by differentiating mathlib's Gaussian PDF and applying the fundamental theorem of calculus |
| Clipped expectation exists | `clippedDisplacement_gaussianIntegrable` | Integrability proved from bounded clipping, independently of the formula |
| Positive innovation at zero implies positive expectation | `boundary_innovation_positive_expectation` in `Boundary.lean` | A probability measure, measurable `X`, a.e. `0 ≤ X ≤ U`, and positive probability of `X > 0` |
| Zero expectation iff zero a.s. | `expectation_eq_zero_iff_ae_zero` | Boundedness supplies integrability; no default-zero nonintegrable-integral loophole |
| No mean-preserving positive innovation at zero | `no_mean_unbiased_boundary_innovation` | Same bounded probability-space assumptions |
| One-step Price identity, Eq. (5) | `price_identity` in `Price.lean` | Arbitrary real traits and total ancestry map between nonempty finite censuses |
| Covariance form | `selection_eq_covariance` | Descendant weights are proved to be actual fiber cardinalities; covariance has population denominator `N` |
| Only newborns contribute to transmission | `transmission_only_newborns` | Assumes survivor traits equal their ancestors' values; retains the complete after-census denominator |
| Cumulative identity | `cumulative_price` | Any finite sequence satisfying the proved one-step identity; population sizes may change |
| Newborn endpoint decomposition | `newborn_endpoint_identity` | Newborns are a separate nonempty census mapped to their parents |
| Growth against nonpositive selection needs positive transmission | `positive_growth_requires_transmission` | Direct consequence of the exact census identity |
| Treatment contrast includes transmission contrast | `contrast_identity` | Algebraic accounting; no causal interpretation |
| Concrete possibility of positive growth against cumulative selection | `growth_against_selection_example` in `Examples.lean` | An abstract two-step bounded ledger: zero founders, final growth `1/10`, total selection `-1/5`, total transmission `3/10` |

`Examples.lean` is a mathematical witness and check on the definitions, not a new
simulation or evidence about a particular mutation sample. The Gaussian CDF is
defined as the integral of mathlib's standard Gaussian density on `(-∞,x]`.
The proof derives the interior integral and both tails; it does not assume the
closed-form expectation as a premise.

## Trust and coverage

There are no admitted proofs or new axioms in the proof modules. `Audit.lean`
inspects kernel-checked theorem dependencies and rejects anything outside the
standard `propext`, `Classical.choice`, and `Quot.sound` allowlist. It also rejects
local axioms and an empty audit. Generated helper theorem declarations are counted
alongside the named theorems, so the audit count is not a count of scientific claims.

The audit was also tested against two temporary negative controls: an extra local
axiom and an admitted theorem. Both failed with the expected diagnostic; the real
package passed. See `verification.txt` for the executed check summary.

These results concern exact real-number mathematics. They do **not** prove:

- Python engine, observer, ancestry extraction, or floating-point correctness;
- the recorded simulation outcomes or the printed decimal approximations;
- a causal effect of the focal trait, fitness benefit, or publication novelty.

The experiment's negative verdict remains unchanged. Extinction is deliberately
outside the census-mean theorem: both population types require `Nonempty`. The
Gaussian formula assumes positive mutation scale; the zero-scale special case is
not part of that theorem. General boundary innovation does not require Gaussian
mutation at all.

Independent source review found no actionable mismatch between these statements
and the paper. That review did not independently run Lean; the recorded build and
axiom checks are the implementing agent's executed verification.

Claudeception retrospective: the macOS native-cache failure and interpreter
workaround were non-obvious and reusable. Their exact symptom, verified command,
and limits are preserved in the setup section above. A separate skill would
duplicate that guide; no new global skill or machine configuration was installed.
