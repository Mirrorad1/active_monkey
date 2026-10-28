# A finite-flux speed limit for Page-like entropy excursions

Research draft, 27 October 2026. [Four-page PDF](finite-flux-page-curves.pdf)
and [standalone LaTeX source](finite-flux-page-curves.tex), including references.
No author or affiliation has been invented. No submission has been made.

## Question and result

How quickly can nonnegative radiation entropy rise to a prescribed height and
return to its initial value if the positive and negative null-energy fluxes
are separately bounded? In the two-dimensional conformal-vacuum sector,
the Bianchi--Smerlak relation converts this into a bounded-acceleration problem
for `w=exp(6S/c)`. The note gives its exact duration formula, an elementary
global proof, the optimizer, and smooth approximants preserving both caps and
the exact height. It then gives a necessary peak-QNEC-gap threshold for an
excursion faster than the conformal-vacuum limit.

This is a prescribed-boundary/CFT calculation, not a backreacting evaporation
model. The entropy must be the one in the stated null-deformation inequality;
generic finite-interval vacuum subtraction is insufficient. No quantum state
realizing the nonsaturating diagnostic is constructed.

## What is established, and what might be new

The equal-cap minimum-time control theorem is **known**. Stefanatos's
quantum-piston theorem becomes the rising half of the entropy pulse under a
change of variables. Extra turnovers are already covered by that control
problem; their exclusion is not an additional new theorem. The unequal-cap
formula is an elementary extension and carries no established priority claim.

The candidate contribution is the CFT application together with the inverse
peak-QNEC-gap diagnostic and its divergence near the positive-cap duration
floor. A bounded primary-source search did not identify this exact application
or diagnostic. That does not establish priority or sufficient significance for
publication. Mathematical verification is positive; publication-level novelty
and significance remain provisional. This is a modest research note for expert
evaluation, not a certified original theorem or a promised arXiv acceptance.

The initial integrated-negative-energy idea was abandoned after an exact
overlap was found in Karmoudi--Daoud. The analysis-plan records that pivot and
the later discovery of the quantum-piston antecedent. Neither result is
presented as ours.

## Literature check

Primary sources inspected by the parent and independent literature reviewer:

| Source | Relationship to this note |
| --- | --- |
| [Bianchi--Smerlak, 1404.0602](https://arxiv.org/abs/1404.0602) | Exact flux--entropy relation and normalization. |
| [Abdolrahimi--Page, 1506.01018](https://arxiv.org/abs/1506.01018) | Energy--entropy relation in evaporation; integrated costs and impulsive negative energy. |
| [Karmoudi--Daoud, 2610.03576v2](https://arxiv.org/abs/2610.03576v2) | Integrated-energy optima and tradeoffs; defeats novelty of the initial question. |
| [Stefanatos, 1301.5137, Theorem 4](https://arxiv.org/abs/1301.5137) | Exact equal-cap one-way minimum-time problem; the central mathematical antecedent. |
| [Teo--Wong, gr-qc/0206066](https://arxiv.org/abs/gr-qc/0206066) | Hyperbolic/trigonometric matching for square energy pulses in quantum interest. |
| [Ecker et al., 1901.04499](https://arxiv.org/abs/1901.04499) | Two-dimensional QNEC saturation and nonsaturation. |
| [Fewster--Hollands, math-ph/0412028](https://arxiv.org/abs/math-ph/0412028) | Smeared quantum energy inequalities; distinct from imposed pointwise caps. |

Searches included Page-curve energy costs, bounded moving-mirror radiation,
minimum-time bounded controls of `w''=q w`, quantum-piston shortcuts, quantum
interest, and QNEC entropy-growth bounds. Adjacent entropy-growth papers
[1909.00919](https://arxiv.org/abs/1909.00919) and
[2109.09914](https://arxiv.org/abs/2109.09914) were screened; no comprehensive
citation-graph or unpublished-work audit is claimed. Search date: 27 October 2026.

## Reproduction and verification

The pre-run [analysis plan](analysis-plan.md) is pinned by the numerical
output's source hashes. Source and results:

- `experiments/exp282_finite_flux_page.py`
- `experiments/outputs/exp282.json` and `exp282.txt`
- `experiments/outputs/exp282_smoke.json` and `exp282_smoke.txt`
- `tests/test_exp282_finite_flux.py`
- `experiments/outputs/exp282_verification.md`

From the repository root:

```sh
uv run --no-sync --python .venv python experiments/exp282_finite_flux_page.py
uv run --no-sync --python .venv python -m pytest -q tests/test_exp282_finite_flux.py
```

The runner prints every case by default. To save a rerun, pass a **new**
`--output-prefix experiments/outputs/exp282_rerun_<label>`; existing outputs
are never overwritten. No random seeds, training data, creature state, hosted
models, or GPU are involved. The parent inspected the implementation, including
continuous propagation across all switching intervals and independent
endpoint-regularized quadrature.

All 30 ODE cases, 18 sinusoidal profiles with one, two, or three peaks, six
asymmetric polynomial profiles, seven rational-algebra identities, and six
equal-cap inversions passed their stated checks. Maximum normalized ODE
endpoint error was `1.768e-13`, apex error `7.117e-13`, energy-identity error
`2.805e-11`; maximum independent duration-quadrature error was `1.878e-15`.
The lowest tested baseline duration/minimum-duration ratio was `1.077433`.
The nonoptimal profiles have `C1` constant continuation and bounded flux
steps; they are not globally smooth. Global optimality and smooth sharpness
come from the proof, not from a finite grid. The full run took `0.254083 s`
internally on this machine; timing is not a scientific result.

At `R=e`, `a=b0=1`, and `T=3.5`, the gap calculation gives
`bcrit=2.203056359208288`, hence `D/K >= 3.8534573218480777`.
This example was chosen after analytic derivation, with reviewer-supplied
numeric values available before execution; it is not a blind prediction.

The manuscript received independent mathematical review and an independent
prior-art review. A separate restricted raw-data recomputation is documented
with its exposure limits; none of these reviews certifies publication novelty.
The raw reviewer confirmed all 181 requested core conjuncts and separately
recomputed duration quadrature and baseline caps. The full ODE sample states
are not archived, so its check of sampled profile errors is limited to stored
summaries; the integrator was not independently rerun.
The 16 focused software tests passed. The final PDF compiled with Tectonic
0.17.0 using standard TeX packages; `pdfinfo` reports four letter-size pages.
All four pages were rendered and visually inspected without clipping or
overlap. The built-in Codex LaTeX compiler also compiled the final source.

Full-run JSON SHA256:
`e5fb279df45cf0da3ab4b948d81583f31429ca8fdd1340a14fed897e36deb6b4`.
Full-run text SHA256:
`db1eaed622c814e724b8ed7c582b68d68ab6595983aafbe329dd6d3cb021a6d2`.
Final PDF SHA256:
`7201759056364696d8670920340aa2cb81830a933f7c2f1d9e3bc46282e0de7f`.

## Process lesson retained

Claudeception evaluation: the reusable finding is a project research-process
rule, not a new software workaround requiring a global skill. An existing
unrelated research skill mentions a novelty ledger but does not cover this
failure mode. `loop/PROTOCOL.md` and lesson L54 now require equation-level,
cross-field antecedent searches and separate judgments of correctness,
theorem novelty, application novelty, and significance. Numerical PASS cannot
promote any of the latter judgments.
