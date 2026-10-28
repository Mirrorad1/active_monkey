# Trait Growth against the Selection Differential

Five-page research draft, including references. Source: `trait-growth-selection.tex`.
PDF: `output/pdf/boundary-transmission/trait-growth-selection.pdf` from the repository root.
The author field is intentionally empty; supply the human authorship and a public
repository/release URL before submitting. No submission has been made.

The [Lean supplement](lean/README.md) now provides checked proofs of the exact
Gaussian formula, boundary innovation, and Price accounting, with pinned
dependencies and an axiom audit. The open `.tex` references it and makes the
nonempty-census and positive-scale assumptions explicit. The earlier standalone
PDF has not been regenerated; the document editor previews the current source.

## Question and result

Can a bounded heritable capability grow from zero while its realized selection
differential is negative? Experiment 283 instruments the existing ecology without
changing its mutation operator or engine. It combines a clipped-Gaussian expectation
with a standard overlapping-population Price decomposition.

The prospectively specified substantial-growth prediction **fails**: only 1/16
cost-only runs jointly complete, reach a living mean of .02, and have cumulative
selection at most -.02; 12/16 were required. The neutral conjunct passes 16/16.
All 48 runs finish. Cost-only mean living trait is .015910; recent newborn mean is
.029583. Replacing the registered endpoint with newborn mean would change the size
count from 1/16 to 16/16, but that descriptive sensitivity cannot rescue the test.

The practical contribution is a reusable measurement audit. The Price equation,
mutation bias, and boundary-induced structural bias are established theory, and
Price accounting has already been used in Avida. This is a modest methods-note
application; publication-level significance or novelty is not certified. The
selection term is a realized association, including drift and linked-trait effects,
not an identified causal effect of the focal trait. One simulator, one scale, and
one finite horizon do not establish a universal failure or a stationary equilibrium.

## Evidence and reproduction

- Pre-run design commit: `32f2c19`,
  `docs/superpowers/plans/2026-10-27-boundary-transmission-paper.md`.
- `ecology/transmission.py`: pure drift and census accounting functions.
- `experiments/exp283_boundary_transmission.py`: fixed conditions, seeds, criteria,
  observer and output writer. The default output paths refuse existing evidence.
- `tests/test_transmission.py`: pedigree, quadrature, missing-parent, extinction,
  nonperturbation, criterion and overwrite-guard checks.
- `experiments/outputs/exp283.json` and `.txt`: full metadata, configurations,
  scientific predeclaration, preflight, source hashes and per-seed summaries.
- `experiments/outputs/exp283_trajectories/*.json.gz`: all 144,000 per-step records
  in 48 files. Each has a `columns` array and full-precision numeric row arrays.
- `experiments/outputs/exp283_smoke*`: excluded pilot seed 9499, not confirmation.
- `experiments/exp283_analyze.py` and `experiments/outputs/exp283_analysis.*`:
  exact rerun comparison, frozen counts, aggregates and explicitly post-hoc contrasts.
- `experiments/outputs/exp283_verification.md`: detached numeric verification,
  with its input restrictions and limits. The verifier packet contains local
  absolute paths; rerun the analysis to regenerate it in another checkout.

Run from the repository root with the established `.venv`:

```sh
uv run --python .venv python experiments/exp283_boundary_transmission.py \
  --full --output-prefix experiments/outputs/exp283-reproduction
uv run --python .venv python experiments/exp283_analyze.py
uv run --python .venv pytest -q tests/test_transmission.py
```

The analysis checks the committed canonical run against its archived first run;
it does not automatically substitute reproduction-prefix results. Set
`UV_CACHE_DIR` to a writable directory if the sandbox cannot access the uv cache.
The manuscript is self-contained LaTeX with inline pgfplots coordinates;
`scatter-data.tex` is the generated figure fragment for inspection. The built-in
Codex compiler and Tectonic both compile it. References count toward the five pages.

## Run provenance and review

The first full run took 35.608310 seconds. Source review then found that a repeat
invocation could overwrite its evidence. We archived that complete run and its
original runner/test sources under `experiments/outputs/exp283_initial_archive/`,
added refusal checks for existing files/directories/symlinks, and reran the same
fixed seeds without any scientific change. The final run took 34.832755 seconds.
All numerical rows and all 48 event hashes match exactly. Runtime, the output-guard
source hashes, and gzip metadata can differ. This is a reproducibility check, not
an independent confirmation cohort. Canonical source hashes match current source;
original runner/test hashes resolve to the retained source archive.

Concurrent auto-sync commits (`c7cdf5c`, `56ecb8a`) captured working files and a partial
output while this task was active. These are not completed experiments or registered
results. The final experiment commit reconciles the complete source, raw records,
append-only log and site entry. No earlier evidence is rewritten to change its
scientific verdict. This process exception is disclosed rather than presented as
atomic first publication of the experiment artifacts.

Source-informed mathematical/code review and detached numeric verification have
different scopes. The latter independently recomputes the frozen counts from
aggregate raw rows; it cannot reconstruct individual pedigrees or independently
certify the simulation implementation. Source review, meaningful unit tests, and
excluded-pilot observed/unobserved event-hash checks address those separate issues.

Final validation: all tests in `test_transmission.py`, `test_site_data.py`,
`test_meta_monkey.py` and `test_check_iteration.py` passed. The iteration rubric
passed with three confirmed derived-value warnings: the fraction 1/16 and the
rounded first/final runtimes. The whole repository suite was started but interrupted
while unrelated simulation tests were still running; it is not reported as passed.
Both LaTeX compilers succeeded, the exported PDF has five pages, and all pages were
visually inspected. The final compile reports no overfull/underfull boxes.

Claudeception review: the reusable knowledge is the census ancestry invariant,
endpoint separation, and immutable evidence workflow described below. The helper,
regression tests and this guide already preserve it; an additional skill would
duplicate those project instructions. No new user-wide skill was installed.

## Applying the audit to future experiments

Freeze the living endpoint and any newborn diagnostic before running. Record each
before/after census, map survivors to themselves and births to before-census parents,
and reject missing ancestry. Include parents that reproduce and die in one step.
Check whether the engine lets a newborn act or die before the next census; the
one-step map here depends on delayed newborn activation. Do not replace extinction
with a valid zero-change ledger. Compare total trait change with the accumulated
selection and transmission terms, then use a separate causal assay for fitness
claims. A common mutation rule across conditions does not make its accumulated
transmission contribution cancel: parental distributions and birth rates differ.

For operator comparisons, preserve the original run and predeclare a new test.
Reflection or rejection is not a mean-unbiased escape at an exact nonnegative
boundary if positive innovation remains possible. Source hashes, immutable output
paths and before/after event hashes protect this observational interpretation.

## Primary literature

- Price (1970), [Selection and covariance](https://doi.org/10.1038/227520a0).
- Frank (2012), [Natural selection. IV. The Price equation](https://arxiv.org/abs/1204.1515).
- Caraffini, Kononova & Corne (2019), [Infeasibility and structural bias in differential evolution](https://arxiv.org/abs/1901.06153).
- Boks, Kononova & Wang (2021), [Quantifying the impact of boundary constraint handling methods on differential evolution](https://arxiv.org/abs/2105.06757).
- Mirmomeni, Punch & Adami (2014), [Is information a selectable trait?](https://arxiv.org/abs/1408.3651), a close digital-life precedent.
- Okasha & Otsuka (2020), [The Price equation and the causal analysis of evolutionary change](https://doi.org/10.1098/rstb.2019.0365).
