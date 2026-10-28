# Finite-flux Page-curve note: analysis and verification plan

User request (2026-10-27): independently choose a hep-th question, perform the
analysis, and prepare an article of at most five pages with a potentially
publishable result. The user explicitly delegates the choice. This is a new
theoretical-physics direction, not a continuation or repair of sleep C1.

## Question and claim to check

For a two-dimensional conformal vacuum with Bianchi--Smerlak flux
`F = K w''/w`, `K=c/(12 pi)`, `w=exp(6 S/c)`, what is the shortest complete
entropy excursion with height `Smax`, stationary endpoints, and pointwise caps
`-Fminus <= F <= Fplus`?

Let `R=exp(6 Smax/c)>1`, `a=sqrt(Fplus/K)`, `b=sqrt(Fminus/K)`, and
`y=sqrt((b*b*R*R+a*a)/(a*a+b*b))`. The candidate sharp bound is

`T >= 2 [acosh(y)/a + acos(y/R)/b]`.

Candidate optimizer: on the rising half, `w=cosh(a u)` followed by
`w=R cos(b(tau-u))`; reflect about the maximum. Its flux is positive,
negative across the maximum, and positive, at the specified caps. The proposed
proof uses a speed envelope, including for nonmonotone excursions. Smooth
sharpness requires mollification, restoration of the exact height, and time
dilation to preserve both caps.

## Prior-art boundary

The initially considered integrated-negative-energy question is already solved
by Karmoudi and Daoud, arXiv:2610.03576, including an integrated-energy Pareto
frontier. Those results are prior work, not our contribution. The proposed new
contribution is the exact optimization with *pointwise* flux caps, its unequal-cap
formula, smooth realizability, and the equal-cap peak-flux specialization.
Novelty remains provisional until a primary-source search and independent
literature check are complete; a search without a hit cannot prove priority.

## Verification predeclaration (before numerical execution)

The analysis above precedes computation. Numerical checks are verification of
an analytic candidate, not a blind discovery experiment or proof of novelty.

1. Verify algebraic matching and the Schwarzian normalization symbolically.
2. Integrate the switched ODE independently, using `a=1`,
   `log(R) in {0.01,0.1,0.5,1,2,5}` and `b/a in {0.1,0.3,1,3,10}`.
   Check endpoint, apex, and matching residuals below `2e-7` after normalization
   by `max(1,R,a*R,b*R)`; report every case. Check the integrated-flux identity
   against quadrature of `(w'/w)^2` to relative/absolute `2e-7`.
3. Independently integrate the speed-envelope quadrature and compare the closed
   form within `2e-8` relative/absolute. Use endpoint-removing substitutions.
4. Test regular nonoptimal entropy profiles, including multiple turnovers.
   For every tested profile derive its *actual* positive and negative caps and
   check `T/Tmin >= 1-2e-7`. This checks examples, not global optimality.
5. Verify exact equal-cap peak-flux formula and the sinusoidal comparison;
   tabulate dimensionless heights and limiting expressions. Numerical equality
   with a formula alone is not an independent proof.
6. Obtain independent mathematical review, literature review, and a restricted
   raw-output check. Disclose the scope of each review.

Falsifier: any valid analytic counterexample, matching failure, normalization
error, or numerical violation beyond these tolerances defeats the corresponding
claim. Retain failed checks and disclose any repair. No inference about general
QNEC states, gravitational island entropies, or four-dimensional backreaction is
licensed by conformal-vacuum calculations.

## Deliverables

- A standalone LaTeX article and a rendered PDF, at most five pages including
  references, with an explicit theorem and proof, construction, limits, and
  physical scope.
- Reproducible verification script and raw outputs under `experiments/`, with
  an append-only Exp 282 entry and source-linked site card.
- A concise literature/verification record distinguishing established results,
  the candidate contribution, and unresolved publication judgments.

Runtime preflight: 30 two-dimensional ODE checks plus one-dimensional quadratures
and a small deterministic profile set; bounded work, expected seconds rather
than minutes. First time a one-case smoke check; use parallel independent checks
if the full calculation is appreciable. No simulation of a quantum field or
black-hole backreaction is claimed.

## Disclosed refinements before the full numerical run

- The prior-art search found the exact equal-cap one-way control theorem in
  Stefanatos, arXiv:1301.5137, Theorem 4. The full equal-cap pulse follows by
  reflection. The contribution must be described as a CFT application with an
  elementary unequal-cap extension, not a newly discovered control theorem.
- The analytic follow-up is a peak-QNEC-gap diagnostic: if `0<=Delta_Q<=D`,
  substitute `b^2=(Fminus+D)/K`. Invert the strictly decreasing duration bound
  to obtain a necessary peak-gap threshold. The `R=e, a=b0=1, T=3.5` example
  and near-floor asymptotics were derived before their runner checks, and an
  independent reviewer supplied numbers. This is disclosed verification, not
  a blind empirical prediction. No nonsaturating-state realizability is claimed.
- Sinusoidal and polynomial examples with constant continuation are `C1` and
  `W^{2,infinity}`, with finite acceleration steps at the boundary. They are not
  globally `C-infinity`; smooth realizability is established by the separate
  mollification/height-restoration/dilation proof. No thresholds changed.
