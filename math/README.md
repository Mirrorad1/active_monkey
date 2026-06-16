# `math/` — the math behind the experiments

A reference that explains **all the math the experiments talk about**, so you can read any
entry in [`EXPERIMENTS.md`](../EXPERIMENTS.md) (or the [journey](../journey.html)) and
understand the numbers.

Every concept is explained four ways, in this order:

> **Glossary** (the math, defined, with the real equation) → **Example** (worked by hand) →
> **Data** (where it shows up in the experiments, with real `Exp NN` numbers) →
> **▸ In programmer terms** (a short, correct code snippet that maps the math to things a
> programmer already knows).

**Two ways to read it:**
- **On the web** — open [`math.html`](../math.html) (same look as the journey "stuff page":
  collapsible concept cards, topic filter, a global *programmer-terms* toggle, search, and
  every `Exp NN` links into the journey). Start at [`00-orientation.md`](00-orientation.md).
- **As Markdown** — the eight files below, browsable right here on GitHub.

This whole reference grew out of a simple goal: a curious reader (or a programmer) should be
able to look at the data and understand what it means.

---

## Start here

**[`00-orientation.md`](00-orientation.md)** — the map. There are **two** research lines with
**two** families of math: **(A) Bayesian active inference** for the single creature (currency:
probability; unit: *bits/char*), and **(B) evolutionary population dynamics** for the ecology
(currency: fitness; unit: *births/step*), plus **(C)** a metacognition/control layer on top.
It also explains the honest units and **how to read an experiment entry** (predeclared
falsifiers, blind verification, the verdict taxonomy).

## The eight files

| # | File | What it covers | Mostly maps to |
|---|---|---|---|
| 00 | [orientation](00-orientation.md) | the two lines, the currencies, units, how to read an entry | Exp 1–207 |
| 01 | [free energy & active inference](01-free-energy-and-active-inference.md) | the Free Energy Principle, variational & expected free energy, perception as Bayesian filtering, precision, valence = −F, planning, the RECIPE | Line A · Exp 1–35, 141–154 |
| 02 | [Bayesian inference & learning](02-bayesian-inference-and-learning.md) | Bayes' rule, the categorical/Dirichlet conjugate pair, the Bayesian HMM (`A`/`B`/`D`), forward filtering, the mean-field collapse, Dirichlet parameter learning, place fields, Bayesian model reduction & structure growth | Line A · Exp 1–31, 152–154 |
| 03 | [information theory](03-information-theory.md) | surprise, entropy, cross-entropy, KL divergence, bits/char, conditional entropy & the first-order Markov wall, the pair-state / `O(Vᵈ)` memory-depth ceiling | Line A units · Exp 1, 3–8 |
| 04 | [probability & distributions](04-probability-and-distributions.md) | the simplex, categorical/Bernoulli, Dirichlet/Beta, Gaussian & precision, sigmoid/softmax, the EMA tracker | foundations · Exp 201–207 |
| 05 | [evolutionary dynamics](05-evolutionary-dynamics.md) | fitness, the selection gradient `∂w/∂h`, the cross-partial & 2-D valley, fitness valleys & benefit saturation, heritability vs survivor bias, drift, the forced-vs-evolvable gap | Line B · Exp 194–207 |
| 06 | [control & dynamical systems](06-control-and-dynamical-systems.md) | the symmetric-saddle/collapse finding, attractors & non-identifiability, precision as control, the N3/N4 monitors & controllers (type-2 AUROC), hysteresis, the universal-constant law | Line C · Exp 31, 155–193 |
| 07 | [statistics & experimental method](07-statistics-and-experimental-method.md) | seeds & reproducibility, mean/variance, correlation (Pearson/Spearman), effect size, nulls & matched controls, predeclared falsifiers, blind verification, confounds, the verdict taxonomy | method · Exp 195–207 |

## All 78 concepts

<details><summary><b>01 · Free energy & active inference</b> (9)</summary>

Free Energy Principle · Variational free energy · Surprise and the bits/char metric ·
Perception as inference (fixed-point filtering) · Expected free energy (action) · Precision ·
Valence (−F and −dF/dt) · Planning (value iteration) · The RECIPE
</details>

<details><summary><b>02 · Bayesian inference & learning</b> (10)</summary>

Bayes' rule · Categorical distribution & the simplex Δ^K · The Bayesian HMM generative model ·
Dirichlet conjugate prior (learning = counting) · Parameter learning (Dirichlet count updates) ·
Forward filtering · Mean-field approximation & the collapse finding · MAP vs posterior-mean ·
Place fields (Gaussian receptive fields) · Bayesian model reduction & structure growth
</details>

<details><summary><b>03 · Information theory</b> (9)</summary>

Self-information / surprise · Entropy · Cross-entropy · Conditional entropy & the entropy rate ·
The order-1 floor argument · Mutual information · KL divergence · bits/char as the project metric ·
Memory depth: the pair-state construction & the O(Vᵈ) wall
</details>

<details><summary><b>04 · Probability & distributions</b> (8)</summary>

Random variables, pmf vs pdf, the simplex · Categorical & Bernoulli · Dirichlet & Beta
(the concentration α) · Gaussian & precision τ = 1/σ² · Sigmoid, logit, softmax,
inverse-temperature β · Expectation & variance · Exponential moving average (EMA) ·
Stochasticity, RNG seeds, reproducibility
</details>

<details><summary><b>05 · Evolutionary dynamics</b> (13)</summary>

Genotype, phenotype & the gene pool · Fitness as realized reproduction · Mutation ·
Heritability vs survivor bias · The selection gradient · The cross-partial & the falsified
"2-D fitness valley" · The fitness valley & benefit saturation · Genetic drift vs selection ·
The monomorphic optimum h* & convergence to an attractor · The forced-vs-evolvable gap (L22) ·
Per-capita growth rate r, N* & Tilman's R* · Invasion fitness, adaptive dynamics & the ESS ·
Frequency-dependent selection
</details>

<details><summary><b>06 · Control & dynamical systems</b> (10)</summary>

The collapse as a symmetric saddle · Fixed points, attractors & convergence · Non-identifiability ·
Precision-weighting as control · Metacognitive controllers (N3) · N4 identity monitor + commitment
controller (the stopwatch) · Hysteresis · Timescale overlap (the flicker tax) · Regulation vs a
constant (the universal-constant law) · Signal detection & type-2 AUROC
</details>

<details><summary><b>07 · Statistics & experimental method</b> (12)</summary>

Random seeds & reproducibility · Sample mean/variance/std · Correlation (Pearson r vs Spearman ρ) ·
Effect size (the gap vs a matched control) · Null hypotheses & matched controls · Predeclared
falsifiers · Blind verification (PROTOCOL 4.5) · Confounds & how they are neutralized · The verdict
taxonomy · Fitting a trend (OLS & the log-growth slope) · Standard error & the look-elsewhere effect ·
Cryptographic hashing as a byte-identical control
</details>

---

## Honesty

Every numeric claim and every `Exp NN` citation in these files was grep-checked against
[`EXPERIMENTS.md`](../EXPERIMENTS.md) / [`RESEARCH.md`](../RESEARCH.md) by an independent
fact-checking pass (the binding rule: [`loop/VALIDATION.md`](../loop/VALIDATION.md)). Negative
and MIXED results are cited as such. Where a logged narrative figure and the reproducible
committed output disagree (e.g. Exp 1's `4.81 → 4.00` narrated vs `4.007 → 3.424` reproduced),
**both are shown and the committed output is named as ground truth.**

## The web page is generated from this Markdown

[`math.html`](../math.html) renders from [`../site/data/math-data.js`](../site/data/math-data.js), which is
**generated** from these `.md` files — the Markdown is the single source of truth. After editing
any file, regenerate:

```bash
python3 math/build_math_data.py     # stdlib only; writes ../site/data/math-data.js
```

`math.html` reuses the site's shared chrome under `site/` (`site/styles/am.css`,
`site/components/am-shared.jsx`, `site/components/tweaks-panel.jsx`, all at `?v=5`).
The **"The Math"** nav link lives in the shared `site/components/am-shared.jsx`, so it appears on
every page automatically. Root HTML deploy files are generated from `site/pages/` by
`tools/site/build_static.py`.
