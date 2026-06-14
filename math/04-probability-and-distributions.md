# Probability & distributions

> This file collects the probability objects the experiments actually compute on: the discrete
> distributions that live on a simplex (the categorical emission/transition tensors of the
> character HMM), their conjugate Dirichlet priors (the thing that lets the model *learn*), the
> continuous Gaussians used for place fields and word-substrates, and the squashing/averaging
> functions (sigmoid, softmax, EMA) the ecology uses to turn a noisy signal into an action or a
> running estimate. Everything here is plain probability — the active-inference machinery
> (free energy, EFE) is built *on top* of these primitives.

## Random variables, pmf vs pdf, the simplex & normalization

**Glossary.** A *random variable* `X` is a quantity whose value is uncertain; its distribution
says how probability mass/density is spread over the outcomes. A **discrete** RV has a
*probability mass function* (pmf) `p(x)`; a **continuous** RV has a *probability density
function* (pdf) `f(x)`. Both must be non-negative and sum/integrate to 1. A pmf over `n`
outcomes lives on the **probability simplex** `Δⁿ⁻¹` — the set of non-negative vectors that sum
to one — and the universal way to *make* a vector into a valid pmf is to divide by its sum
("normalize").

```
discrete (pmf):     p(x) ≥ 0,   Σ_x p(x) = 1
continuous (pdf):   f(x) ≥ 0,   ∫ f(x) dx = 1
simplex:            Δⁿ⁻¹ = { p ∈ ℝⁿ : p_i ≥ 0,  Σ_i p_i = 1 }
normalize a weight vector w:   p_i = w_i / Σ_j w_j
```

- `p(x)` — probability of outcome `x` (a number in [0,1]); `f(x)` — a density (can exceed 1).
- `Σ_x` — sum over all discrete outcomes; `∫` — integral over the continuous range.
- `w` — any non-negative "weight" / count vector; dividing by `Σ w` projects it onto `Δⁿ⁻¹`.

**Example.** Take counts of three letters seen in a tiny stream: `w = (3, 1, 0)`. The sum is
`3 + 1 + 0 = 4`, so the normalized pmf is `p = (3/4, 1/4, 0) = (0.75, 0.25, 0.0)`. Check:
`0.75 + 0.25 + 0 = 1` ✓. The point `(0.75, 0.25, 0)` sits on the 2-simplex (a triangle).

**Data.** The whole character HMM is built from simplex-valued objects. RESEARCH.md §1.1
defines the emission `A ∈ Δ^{V×K}` and transition `B ∈ Δ^{K×K×1}` as **column-stochastic**:
each column is a pmf over the `V = 28` characters (`a–z`, space, period) or `K = 14` states. The
predictive distribution after a belief is itself a point in `Δ^K` compressed to `Δ^V` — RESEARCH
§1.5 stresses that "the entire history is compressed into the belief vector
`P(s_{t-1} | o_{<t}) ∈ Δ^K`," which is why a first-order model has a *fixed* prediction per
state (Exp 6's "1 char of context can't distinguish 'm mid-word' from 'm starting the answer'").

**▸ In programmer terms.** A pmf is a `dict`/array of floats that sums to 1; normalizing is one
line. A pdf is a function you call, not a lookup table, and its values are not probabilities
until you integrate.

```python
def normalize(weights):
    total = sum(weights)
    return [w / total for w in weights]   # project counts onto the simplex

p = normalize([3, 1, 0])      # -> [0.75, 0.25, 0.0]
assert abs(sum(p) - 1.0) < 1e-12
```

---

## Categorical & Bernoulli distributions

**Glossary.** A **Categorical** RV takes one of `K` labels with probabilities `p = (p_1,…,p_K)`,
a point on `Δ^{K-1}`. The **Bernoulli** is the `K = 2` special case: a single biased coin with
success probability `θ`. (Its repeated-trials cousin, the Binomial, counts successes in `n`
flips.)

```
Categorical:   P(X = k) = p_k,        Σ_k p_k = 1,   k ∈ {1,…,K}
Bernoulli(θ):  P(X = 1) = θ,  P(X = 0) = 1 − θ,   θ ∈ [0,1]
E[Bernoulli] = θ,    Var[Bernoulli] = θ(1 − θ)
```

- `p_k` — probability of class `k`; `θ` — Bernoulli success probability.

**Example.** A fair 3-color sensor is `Categorical(p = (1/3, 1/3, 1/3))`. A coin that lands
heads 70% of the time is `Bernoulli(0.7)`: mean `0.7`, variance `0.7 × 0.3 = 0.21`.

**Data.** Each column `A[:,s]` of the emission matrix is a Categorical over the `V = 28`
characters — `A[o,s] = P(o_t = o | s_t = s)` (RESEARCH §1.1). The place-field experiments make
this concrete and *exact*: Exp 20 learned per-state color tuning `[0,0,1,0,1,1]` — degenerate
Categoricals collapsed onto single colors — recovering the true colormap, with localization
`0.00 bits`. Exp 21 scaled it to a 3×3 grid, learning the colormap `[0,1,2,1,2,0,2,0,1]`
exactly. A Bernoulli appears whenever an outcome is binary: Exp 72's "kidnapped twin" ends with
a **coin-flip winner** — which of two creatures owns a contested cell is `Bernoulli`-like (B won
6/8, A won 2/8, "coin-flip-compatible").

**▸ In programmer terms.** Categorical sampling = "pick an index by cumulative probability";
Bernoulli = "one `rng.random() < θ` test."

```python
import numpy as np
rng = np.random.default_rng(0)

def categorical(p, rng):            # p is a pmf (list summing to 1)
    return int(rng.choice(len(p), p=p))

def bernoulli(theta, rng):
    return int(rng.random() < theta)   # True ~ theta of the time
```

---

## Dirichlet & Beta: the concentration parameter α

**Glossary.** The **Dirichlet** is a distribution *over* categorical pmfs — a prior on the
simplex `Δ^{K-1}`, parameterized by a concentration vector `α = (α_1,…,α_K)`, all `α_k > 0`. The
**Beta** is its 2-D special case (a prior over a Bernoulli's `θ`). The Dirichlet is **conjugate**
to the Categorical: observing counts `c` updates `α → α + c`, so learning is just *adding
pseudo-counts*. The magnitude of `α` controls peakedness:

```
Dir(p ; α)  ∝  ∏_k p_k^(α_k − 1)
E[p_k] = α_k / Σ_j α_j           (the point estimate used downstream)
Beta(θ ; a, b) = Dir over (θ, 1−θ) with α = (a, b)
symmetric concentration:  α_k = c for all k  →  c < 1 sparse/peaked,  c = 1 uniform,  c > 1 smooth
posterior update:  α* = α + counts
```

- `α_k` — pseudo-count / concentration for class `k`; larger total `Σα` = more confident prior.
- `c < 1` pushes mass to the *corners* of the simplex (sparse: "a state emits few characters");
  `c > 1` pushes mass to the *center* (smooth: "every character a little").

**Example.** Start from a symmetric prior `α = (0.1, 0.1, 0.1)` (the `A_CONC = 0.1` regime).
Observe the letter-1 outcome 5 times: `α* = (0.1+5, 0.1, 0.1) = (5.1, 0.1, 0.1)`, sum `5.3`.
Point estimate `E[p] = (5.1/5.3, 0.1/5.3, 0.1/5.3) ≈ (0.962, 0.019, 0.019)` — five counts on a
sparse prior already give a near-certain emission. A *smooth* prior `α = (5, 5, 5)` plus the
same five counts gives `(10/25, 5/25, 5/25) = (0.4, 0.2, 0.2)` — far less committed.

**Data.** RESEARCH.md §1.1 sets the emission prior as `p(A) = ∏_s Dir(a_{·s} ; α^A_{·s})` with
`pA = A_CONC · 1` and **`A_CONC = 0.1`** (sparse: each state should emit a *few* characters, not
all 28). §1.4 gives the learning rule exactly — `α^A* = α^A_0 + κ · Σ_t o_t ⊗ s_t` — and the
point estimate `A = E[a] = α/Σα`. RESEARCH calls this "exact conjugate Bayesian counting in the
latent space — the active-inference analogue of smoothed n-gram counting." The same `infer_parameters`
machinery is what Exp 8's pair-state trigram (`K = V² = 784`, frozen `A`, Dirichlet-learned `B`)
relies on. (For how the resulting predictive distribution lowers surprise, see *Exponential
moving average* and the free-energy file.)

**▸ In programmer terms.** A Dirichlet posterior *is* a count vector with a prior offset; the
"distribution" you use is just `counts / counts.sum()`. `α < 1` ≈ add-`α` (Laplace/Lidstone)
smoothing that favors sparsity.

```python
import numpy as np

A_CONC = 0.1
alpha = np.full(28, A_CONC)        # symmetric Dirichlet prior over 28 chars (sparse)

def observe(alpha, char_idx, kappa=1.0):
    alpha[char_idx] += kappa       # conjugate update: add a pseudo-count
    return alpha

p_emit = alpha / alpha.sum()       # E[p] = alpha / sum(alpha): the point estimate
```

---

## Gaussian N(μ, σ²) and precision τ = 1/σ²

**Glossary.** The **Gaussian** (normal) is the canonical continuous distribution, a bell curve
fixed by its mean `μ` and variance `σ²`. Its inverse variance `τ = 1/σ²` is the **precision** —
high precision = sharp, confident; low precision = wide, vague. In multiple dimensions the
covariance `Σ` generalizes `σ²` and the precision matrix is `Λ = Σ⁻¹`.

```
N(x ; μ, σ²) = (1 / √(2π σ²)) · exp( −(x − μ)² / (2 σ²) )
precision:   τ = 1 / σ²                 (1-D)
            Λ = Σ⁻¹                     (multivariate)
product of Gaussians (conjugate fusion):  precisions add:  Λ_post = Λ_1 + Λ_2
```

- `μ` — mean (center); `σ²` — variance (spread); `τ` / `Λ` — precision (confidence).
- Key fact the substrate exploits: **fusing two Gaussian observations adds their precisions** —
  belief gets monotonically sharper as evidence accrues.

**Example.** Combine a prior `N(0, 4)` (precision `τ = 1/4 = 0.25`) with one observation
`N(2, 1)` (precision `τ = 1`). Posterior precision `0.25 + 1 = 1.25`, so `σ²_post = 1/1.25 =
0.8`; posterior mean is the precision-weighted average
`μ_post = (0.25·0 + 1·2) / 1.25 = 2/1.25 = 1.6`. One observation pulled the belief from 0 toward
the data and tightened it from `σ² = 4` to `0.8`.

**Data.** `active_loop/continuous.py` implements exactly this in **natural-parameter (precision)
form**: a `GaussianBelief` stores `Λ = Σ⁻¹` and updates by *adding* likelihood precision (its
docstring: "the conjugate update for a Gaussian likelihood with known precision `Λ_k`"). The
word-Gaussian ladder (**Exp 133**, EXPERIMENTS.md "6 word-Gaussians at hexagon vertices") used
`Σ_k = 0.35² I` emission footprints with prior `N(0, 4I)`; the result was POSITIVE on all
conjuncts (8/8), final localization error `0.0088–0.0401` and the belief trace contracting
`tr(Σ) 8.0 → 0.001225` — precision accumulating as predicted. The ecology's drifting-band
sensor also reads through a Gaussian: a creature observes the true center plus
`rng.normal(0, noise_sd)` with `noise_sd = noise_base·(1 − intensity)` (`ecology/creature.py`),
so a precise organ has a *tighter* sensory Gaussian (see the EMA section). (For the discrete vs.
continuous trade-off see the entropy file: Exp 143's aliasing rung notes a wide Gaussian per
color keeps neighbor-overlap entropy a categorical would not.)

**▸ In programmer terms.** Track precision, not variance — fusion is then just addition, and you
never invert until you need a mean.

```python
import numpy as np

class GaussianBelief1D:
    def __init__(self, mu0, var0):
        self.lam = 1.0 / var0          # precision = 1 / variance
        self.eta = self.lam * mu0      # natural param: precision * mean

    def observe(self, x, var_obs):
        lam_obs = 1.0 / var_obs
        self.lam += lam_obs            # precisions ADD
        self.eta += lam_obs * x

    @property
    def mu(self):  return self.eta / self.lam
    @property
    def var(self): return 1.0 / self.lam

b = GaussianBelief1D(0.0, 4.0); b.observe(2.0, 1.0)
print(round(b.mu, 4), round(b.var, 4))   # 1.6 0.8
```

---

## Sigmoid, logit, softmax, inverse-temperature β

**Glossary.** The **sigmoid** (logistic) `σ(x)` squashes any real number into `(0,1)` — turning a
"score" into a probability. Its inverse is the **logit** `log(p/(1−p))`. The **softmax**
generalizes it to `K` options (a vector → a pmf), and an **inverse-temperature** `β`
(Boltzmann form) sharpens or softens the result: `β → ∞` is hard argmax, `β → 0` is uniform.

```
sigmoid:   σ(x) = 1 / (1 + e^(−x))           (σ(0) = 0.5, monotone, σ(−x) = 1 − σ(x))
logit:     x = log( p / (1 − p) )            (inverse of σ)
softmax:   softmax(z)_k = e^(β z_k) / Σ_j e^(β z_j)
Boltzmann/inverse-temperature β  (= 1/τ_temp):  large β → peaked, small β → flat
```

- `x` / `z_k` — a real-valued score ("utility", "logit"); `β` — inverse temperature (sharpness).
- A softmax with `K = 2` and `β = 1` reduces to a sigmoid of the score *difference*.

**Example.** `σ(0) = 1/(1+1) = 0.5`; `σ(2) = 1/(1+e^(−2)) = 1/(1+0.1353) ≈ 0.881`. Softmax of
`z = (1, 2)` with `β = 1`: weights `(e¹, e²) = (2.718, 7.389)`, sum `10.107`, pmf `≈ (0.269,
0.731)`. Raise `β` to 3: `(e³, e⁶) = (20.1, 403.4)`, pmf `≈ (0.047, 0.953)` — sharper.

**Data.** The closing entry of the sense-evolution sub-arc, **Exp 207**, states the hypothesized
fitness as a sigmoid gate: `w(h,θ) = R·σ(k·h·θ − d) − C_h(h) − C_θ(θ)` (EXPERIMENTS.md), where `h`
is sensor precision, `θ` the controller gain, `k` a coupling, `d` a threshold/cost offset, and
`C_·` the metabolic costs. The whole experiment tests whether the `h·θ` interaction inside `σ(·)`
creates a 2-D fitness valley — and **FALSIFIES it**: the cross-partial `∂²B/∂h∂θ ≈ 0` (±0.0046),
`∂B/∂θ = +0.147` at low `h` (the controller pays *alone*), and `∂B/∂h = −0.041 / −0.046` (the
sensor is pure cost at both `θ`). Verdict: DESIGN-STAGE NEGATIVE. Softmax/Boltzmann action
selection appears in the navigation harness: Exp 59 plans with **softmax `τ = 0.3`** (i.e.
`β ≈ 3.3`), and Exp 147's value-share planner used a sharper `τ = 0.05` one-step lookahead on
`V(s) = Σ value_share·p(color|s)`. The ecology's niche router (Exp 206) uses the sigmoid's
*cousin* — read noise
`σ = niche_confusion·(1 − h)` shrinks with precision `h` (`ecology/creature.py`), so a sharper
sensor resolves the rotating class.

**▸ In programmer terms.** Sigmoid is the binary-classifier output layer; softmax is the
multi-class one; `β` is the temperature knob you tune between greedy and exploratory.

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(z, beta=1.0):
    z = beta * np.asarray(z, float)
    z -= z.max()                      # numerical-stability shift
    e = np.exp(z)
    return e / e.sum()

# Exp 207's gate (illustrative): benefit only when BOTH h and theta are high.
def w(h, theta, R=1.0, k=4.0, d=1.0, C_h=0.2, C_t=0.2):
    return R * sigmoid(k * h * theta - d) - C_h * h - C_t * theta
```

---

## Expectation E[X] and variance Var[X]

**Glossary.** The **expectation** `E[X]` is the probability-weighted average outcome (the
distribution's center of mass); the **variance** `Var[X]` is the expected squared deviation from
the mean (spread). From a sample you *estimate* them with the sample mean and the (unbiased)
sample variance.

```
E[X]   = Σ_x x · p(x)            (discrete)   /   ∫ x f(x) dx   (continuous)
Var[X] = E[(X − E[X])²] = E[X²] − (E[X])²
sample mean:      x̄ = (1/n) Σ_i x_i
sample variance:  s² = (1/(n−1)) Σ_i (x_i − x̄)²     (Bessel-corrected, unbiased)
```

- `p(x)` — the pmf; `x̄` — sample mean; `s²` — sample variance; `n−1` — Bessel's correction.

**Example.** Data `(2, 4, 9)`: mean `x̄ = (2+4+9)/3 = 5`. Deviations `(−3, −1, 4)`, squares
`(9, 1, 16)`, sum `26`; sample variance `s² = 26 / (3−1) = 13`. For a `Bernoulli(0.7)`:
`E[X] = 0.7`, `Var[X] = 0.7 × 0.3 = 0.21` (matches the Bernoulli section).

**Data.** Every per-arm headline number in the ecology is a **sample mean over seeds**. Exp 201
reports newborn mean thermosense intensity FAST `0.1279` from the five seed values
`(0.1491, 0.1644, 0.1122, 0.1453, 0.0683)` — average those:
`(0.1491+0.1644+0.1122+0.1453+0.0683)/5 = 0.6393/5 = 0.12786 ≈ 0.1279` ✓. The spread across
those seeds (variance) is exactly why the verdict rests on a *majority* falsifier ("FAST < 0.15
in a majority" fires 4/5) rather than a single point. The Dirichlet/Gaussian point estimates used
downstream are themselves expectations: `A = E[a] = α/Σα` (RESEARCH §1.4) and the Gaussian
posterior mean `μ_post` (Exp 133).

**▸ In programmer terms.** `E[X]` is `np.average(x, weights=p)`; sample mean/variance are
`x.mean()` and `x.var(ddof=1)` — note `ddof=1` for the unbiased `n−1` denominator.

```python
import numpy as np

seeds = np.array([0.1491, 0.1644, 0.1122, 0.1453, 0.0683])  # Exp 201 FAST
print(round(seeds.mean(), 4))         # 0.1279  (matches the logged arm mean)
print(round(seeds.var(ddof=1), 5))    # unbiased sample variance across seeds

# expectation under a pmf:
p = [0.75, 0.25]; x = [0, 1]
print(np.average(x, weights=p))       # 0.25
```

---

## Exponential moving average (EMA)

**Glossary.** An **EMA** is a running estimate that blends the previous estimate with the newest
observation, weighting recent data exponentially more. The single parameter `α ∈ [0,1]` is the
*responsiveness*: `α → 1` tracks fast (forgets old data), `α → 0` is sluggish (heavily smoothed).
It is a one-state online estimator — no history buffer needed.

```
x̄_t = (1 − α)·x̄_{t−1} + α·x_t          (equivalently  x̄_t = x̄_{t−1} + α·(x_t − x̄_{t−1}))
weight on the observation k steps ago:  α·(1 − α)^k   (geometric decay)
effective window ≈ 1/α steps
```

- `x̄_t` — the running estimate at step `t`; `x_t` — the new observation; `α` — learning
  rate / responsiveness.
- The bracketed second form shows it is "old estimate **plus** `α` times the prediction error."

**Example.** `α = 0.1`, start `x̄_0 = 1.0`, observe `x_1 = 0.5`:
`x̄_1 = 0.9·1.0 + 0.1·0.5 = 0.95`. Another `x_2 = 0.5`:
`x̄_2 = 0.9·0.95 + 0.1·0.5 = 0.905`. It eases toward 0.5 over roughly `1/α = 10` steps.

**Data.** The EMA is the per-creature **band tracker** at the heart of Exp 201/206/207. In
`ecology/creature.py` the drifting food-band center is tracked as
`self.band_estimate += alpha * (noisy_center − self.band_estimate)` with
`alpha = clamp(intensity · band_responsiveness, 0, 1)` — so a precise organ (high
`thermosense_intensity`) gets a *more responsive* tracker. Liveness check: the precise tracker
(intensity 0.80) had mean error `0.070` to the true center vs the crude tracker (0.10) at `0.120`
(Exp 201). The same EMA shape appears earlier as comfort-gated approach: Exp 69 uses "EMA
`alpha=0.1` of comfort experienced at the source, init 1.0." And the cell-value belief map is an
EMA too — `self.m[pos] += self.learning_rate * (observed − self.m[pos])` in `update_belief`.
Crucially, the EMA tracker was **not enough**: Exp 201's verdict was NEGATIVE (FAST `0.1279`,
SLOW null `0.1081`, USELESS `0.0529`) — precision helps tracking but the benefit is concave and
cost-dominated, so a costed sense never becomes a functional organ. (EMAs also relate to
keep-mean decay: Exp 137 notes a decaying mean is "analytically an EMA with rate `λ = 1 − 1/N_eff`.")

**▸ In programmer terms.** An EMA is leaky integration — one accumulator, no buffer; identical to
the `momentum`/`beta` term in SGD optimizers and the smoothing in a low-pass filter.

```python
class EMA:
    def __init__(self, x0, alpha):
        self.x = x0          # running estimate
        self.alpha = alpha   # responsiveness in [0, 1]

    def update(self, obs):
        self.x += self.alpha * (obs - self.x)   # old + alpha * error
        return self.x

# Exp 201 band tracker: alpha keyed to the creature's organ precision
def tracker_alpha(intensity, responsiveness):
    return min(1.0, max(0.0, intensity * responsiveness))
```

---

## Stochasticity, RNG seeds, and reproducibility

**Glossary.** A simulation is **stochastic** when its outcomes depend on random draws. A
*pseudo-random number generator* (PRNG) is deterministic given its **seed**: same seed →
identical stream of draws → identical run. Fixing seeds turns an experiment from "a story" into
a **falsifiable, byte-reproducible measurement** — and lets you average over seeds to estimate
the true effect (see *Expectation and variance*) while paired seeds cancel common noise.

```
stream = PRNG(seed)             # deterministic given seed
draw_t = stream.next()          # reproducible sequence
report  = mean over seeds {s_1, …, s_n}   (an E[·] estimate; spread = sampling noise)
paired comparison: arm_A and arm_B share the SAME seed stream → noise cancels
```

- `seed` — the integer that pins the stream; `PRNG` — e.g. NumPy's `default_rng` (PCG64).
- More seeds → a tighter estimate of the *population* effect, not a one-off lucky/unlucky run.

**Example.** `np.random.default_rng(42)` always yields the same first draw on a given NumPy
build. To keep sub-systems independent yet reproducible, derive child seeds from a master:
`master = default_rng(seed); world_seed = master.integers(0, 2³¹); main_seed =
master.integers(0, 2³¹)` — exactly the engine's pattern.

**Data.** `ecology/engine.py` routes *all* randomness through `np.random.default_rng` with a
master→child split (`master_rng → world_seed, main_seed → self.rng, world_rng`); its docstring
says the `Ecology` is "Deterministic given cfg + seed." This discipline is what makes the
verdicts trustworthy: Exp 201 ran 6 arms × 5 fresh seeds `{33,34,35,36,37}` (P1 determinism
PASS, 30/30 arm-seeds measurable); Exp 207's corner-grid used seeds `{90–94}`, and the
anti-cheat guards are literally **byte-identical** checks across `h` when the mechanic is off
(`eaten 79368.597 = 79368.597`; `8310.901 = 8310.901`). The honesty audit (2026-06-09) verified
"37/37 runnable scripts" reproduce byte-for-byte. Paired-seed comparisons appear throughout
(Exp 70: "ADAPT vs FIXED … paired rngs"). A seed-related *honesty* note worth carrying: Exp 1's
logged narrative says `4.81 → 4.00 bits/char`, but the original transcript and the re-run both
show `4.007 → 3.424`; the discrepancy is in the *logged text*, not in reproducibility — the run
itself reproduces exactly.

**▸ In programmer terms.** Never use a global/un-seeded RNG in an experiment; instantiate a
seeded generator, derive independent children, and make "same seed ⇒ same bytes" a test.

```python
import numpy as np

def make_streams(seed):
    master = np.random.default_rng(seed)            # the only entropy source
    world_seed = int(master.integers(0, 2**31))
    main_seed  = int(master.integers(0, 2**31))
    return np.random.default_rng(main_seed), np.random.default_rng(world_seed)

# reproducibility guard: identical seed -> identical draw
a = np.random.default_rng(42).random()
b = np.random.default_rng(42).random()
assert a == b                                       # deterministic given the seed
```

---
