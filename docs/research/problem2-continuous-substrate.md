# Problem 2: getting off the enumerated-state substrate — the steering research program

> **Provenance (repo annotation, 2026-06-10):** this program was supplied by the human as
> the steer for `loop/directions/continuous-substrate.md`, and is recorded here verbatim.
> It is the direction's source of truth for the math, the tests, the metrics, and the
> honest gaps. It does NOT replace per-experiment discipline: every rung still predeclares
> its falsifiers per `loop/VALIDATION.md` before running, and entries land in
> `EXPERIMENTS.md` as ever. Repo-added notes are confined to the final
> "Repo annotations" section and are marked as such.

---

## What is the problem, stated generally?

An agent that represents the world as a finite list of discrete states — `S ∈ {s₁, s₂, ..., sₙ}` — cannot scale. The number of possible states in any real environment is combinatorially vast or genuinely continuous. The tabular assumption works in gridworlds and toy language games but imposes a hard ceiling: every distinction the agent can make must be pre-enumerated by the designer. You cannot interpolate between known states, you cannot represent uncertainty about *which distinctions are the right ones*, and you cannot smoothly track a continuously changing world.

The problem is: **how do you replace `S ∈ {1..N}` with `s ∈ ℝᵈ` — a continuous latent space — while keeping inference tractable and online, without backpropagation, replay buffers, or large batch sizes?**

This is the representational fork in active inference. The deep active inference camp answers with neural networks: parameterize the generative model and the inference network, amortize with an encoder, train with gradients. The closed-form camp (AXIOM, VBGS) answers with conjugate Bayesian updates: keep everything in exponential families where Bayes rule has a closed-form solution, update parameters online with simple accumulation of sufficient statistics. Both camps agree the tabular substrate must go. They disagree on whether you need gradients to replace it.

---

## Why this is a relevant open problem

**1. It's the scaling bottleneck.** Every other active inference advance — hierarchy, structure learning, language grounding — assumes you've solved representation. You can't build a multi-level model if the bottom level is hand-enumerated pixels. You can't learn structure if "adding a state" means adding a row to a matrix rather than splitting a continuous mixture component.

**2. It's the live argument in the field.** VERSES claims closed-form continuous inference beats DreamerV3 on Atari with 39× less data. The deep active inference camp says amortization is necessary for high-dimensional inputs. Both cannot be right in all regimes. The scaling properties of conjugate vs. amortized inference as a function of dimensionality, data sparsity, and environment non-stationarity are *not characterized*. That's a genuine gap.

**3. It determines what "biological plausibility" means.** If closed-form updates work at scale, you have evidence that brains might do something similar — local, online, synapse-level updates without global error signals. If they fail past some dimensionality, you have a constraint on what biological systems must solve differently.

**4. It's where your toy repo has leverage.** You've already found that tabular representations collapse when learning structure from noise (Exp 31). The continuous version of that experiment directly tests whether conjugate priors provide the representational inertia that discrete probability vectors lack. That's a crisp, testable hypothesis at exactly the scale you can build.

---

## The mathematical system

### The tabular setup (what you have)

Your agent's generative model:

- **States:** `s ∈ {1, ..., N}` (discrete, N concepts)
- **Observations:** `o ∈ {1, ..., M}` (discrete, M utterance types)
- **Likelihood (A matrix):** `P(o|s) = Cat(Aₛ)` — each state emits observations according to a categorical distribution
- **Prior over states:** `P(s) = Cat(D)` — a categorical with parameter vector D
- **Inference:** Given observation `o`, the posterior is `P(s|o) ∝ A[s, o] · D[s]` — element-wise multiplication and normalization

The problems with this system:
- The number of states N is fixed and must be chosen by the designer
- Each state is equally dissimilar to every other state — there is no geometry
- The posterior is a probability vector of length N, with N-1 degrees of freedom
- When learning A from data, every observation reallocates probability mass across a fixed set of bins — there is zero inertia

### The continuous conjugate setup (what you'd build)

Replace the discrete state with a continuous latent variable.

- **States:** `s ∈ ℝᵈ` (continuous, d-dimensional — start with d=2)
- **Observations:** `o ∈ {1, ..., M}` (keep discrete for now — this is a "continuous latent, discrete emission" model)
- **Likelihood:** Each observation type k is associated with a Gaussian in state space:

```
P(o=k | s) ∝ exp(-½(s - μₖ)ᵀ Σₖ⁻¹ (s - μₖ))
```

This is a Gaussian mixture model where each mixture component corresponds to an observation type, and the latent state s is the mixture assignment variable.

Equivalently, if we think of the state generating the observation:

```
s ~ N(μ₀, Σ₀)           # prior over continuous state
o | s ~ Cat(softmax(f(s)))  # discrete emission, where f(s) gives logits
```

But the first formulation — each observation type has a Gaussian footprint in state space — is easier to work with for conjugate updates.

**Prior:** `P(s) = N(μ₀, Σ₀)` — a multivariate Gaussian with mean μ₀ and covariance Σ₀

**Posterior after observing a *single* word o=k:**

```
P(s | o=k) ∝ P(o=k | s) · P(s)
          ∝ exp(-½(s - μₖ)ᵀ Σₖ⁻¹ (s - μₖ)) · exp(-½(s - μ₀)ᵀ Σ₀⁻¹ (s - μ₀))
```

This is a product of Gaussians, which is itself a Gaussian:

```
P(s | o=k) = N(μ_post, Σ_post)
```

where:

```
Σ_post⁻¹ = Σₖ⁻¹ + Σ₀⁻¹                    # precision addition
μ_post = Σ_post (Σₖ⁻¹ μₖ + Σ₀⁻¹ μ₀)      # precision-weighted mean
```

**Posterior after observing a *sequence* of words o₁, o₂, ..., oₜ:**

Since all observations share the same latent state s, we accumulate precision-weighted evidence:

```
Σ_post⁻¹ = Σ₀⁻¹ + Σ_{t=1}^T Σ_{o_t}⁻¹
μ_post = Σ_post ( Σ₀⁻¹ μ₀ + Σ_{t=1}^T Σ_{o_t}⁻¹ μ_{o_t} )
```

This is the key computational property: **inference is accumulation of sufficient statistics.** You maintain a running precision matrix and precision-weighted mean. Each new observation adds its contribution. No iteration, no gradient steps, no variational bound optimization. The posterior is exact (under the Gaussian assumption) and computed in O(d²) per observation for the precision matrix update.

**Learning the emission parameters:**

If the Gaussian footprints `(μₖ, Σₖ)` are not known and must be learned, you place a conjugate Normal-Inverse-Wishart (NIW) prior over them:

```
(μₖ, Σₖ) ~ NIW(ν₀, κ₀, m₀, S₀)
```

After observing T instances where the agent inferred state sₜ and observation was o=k, the posterior over emission parameters for word k updates:

```
κ_T = κ₀ + T
ν_T = ν₀ + T
m_T = (κ₀ m₀ + T s̄) / κ_T          # s̄ is empirical mean of states when word k was observed
S_T = S₀ + Σ(sₜ - s̄)(sₜ - s̄)ᵀ + (κ₀ T / κ_T)(s̄ - m₀)(s̄ - m₀)ᵀ
```

Again: closed-form accumulation. No gradients. This is the VBGS/AXIOM bet at its purest.

**Action selection:**

Actions change the state. In the discrete model, `B[s'|s, a]` is a transition matrix. In the continuous model:

```
s' | s, a ~ N(s + Δ(a), Σ_action)
```

where `Δ(a)` is the expected displacement for action a (e.g., "move up" → Δ = (0, +1)) and `Σ_action` is the motor noise.

Expected free energy for a candidate action:

```
G(a) = E_{q(s'|a)}[ -log P(o | s') ] + D_KL[ q(s'|a) || P(s') ]
```

The first term is epistemic value / expected information gain — the agent prefers actions that take it to states where observations are predictable (or, in the ambiguity-resolving variant, where observations resolve uncertainty). The second term is risk / pragmatic value — the agent prefers to stay near its prior.

For Gaussian q and Gaussian-mixture likelihood, this is computable in closed form (it involves Gaussian integrals and entropies).

---

## What to test

### Test 1: Convergence to a point

**Setup:** Place 6 word-Gaussians in 2D space (hand-specified, your innate anchor). Agent starts at a random position with broad prior. Feed words from one concept repeatedly.

**What to measure:**
- Distance `||μ_post - μ_true||` as a function of number of observations
- Trace of posterior covariance `tr(Σ_post)` — should decrease monotonically
- Compare to tabular: does the continuous model settle on the correct region in fewer observations?

**What would be surprising:** If the continuous model is *slower* to converge than tabular. The precision-accumulation math says it should be at least as fast, because each observation contributes independently and there's no normalization step competing across bins.

**What would be genuinely novel:** If you find a regime (e.g., very high observation noise, or very broad priors) where tabular converges faster. That would be a boundary condition on the conjugate approach worth reporting.

### Test 2: Interpolation to unseen blends

**Setup:** Define 4 concepts at corners of a square: (0,0), (0,1), (1,0), (1,1). Train the agent on these. Then present a sequence of words drawn from a *blend* — e.g., alternate words from (0,0) and (1,1).

**What to measure:**
- Does the posterior mean land near (0.5, 0.5)?
- Does the posterior covariance increase (the agent knows it's between known concepts)?
- What happens if you then present a word from (0,1)? Does the posterior smoothly shift, or jump?

**What would be surprising:** If the posterior *doesn't* interpolate — if it snaps to one of the four corners. This would indicate that the Gaussian likelihoods are too concentrated (Σ too small) relative to the distance between concepts, making the posterior multimodal. The conjugate update assumes unimodal Gaussian posterior, which is an approximation. If the true posterior is bimodal, the Gaussian approximation will pick one mode. Finding the parameter regime where this happens is useful.

**What would be genuinely novel:** A systematic characterization of when the unimodal Gaussian approximation breaks down as a function of concept separation and observation noise. The literature hand-waves this; a clean phase diagram at toy scale would be publishable.

### Test 3: Structure learning with drifting means (the Exp 31 rematch)

**Setup:** Initialize 6 word-Gaussians with random means and broad covariances. Feed structured data (words that actually cluster in 6 regions of 2D space) for T steps, then feed noise (uniform random words) for T steps.

**What to measure:**
- During structured phase: distance between estimated μₖ and true cluster centers
- During noise phase: do the estimated μₖ drift toward the global centroid, or stay put?
- Quantify: between-cluster variance / within-cluster variance over time
- Compare to Exp 31's tabular collapse

**What would be surprising:** If the continuous model *also* collapses under noise. This would mean the representational inertia from conjugate priors isn't enough, and the collapse you found is a deeper property of online Bayesian inference, not an artifact of tabular representations. That would be a negative result worth documenting.

**What would be genuinely novel:** If you can quantify the *critical noise level* at which collapse occurs as a function of the NIW prior hyperparameters (ν₀, κ₀ — the "virtual counts" that determine how much data it takes to move a Gaussian). Finding that you can dial collapse resistance up and down with these parameters, and characterizing the phase transition, would be a direct demonstration that conjugate priors solve a problem the field has only gestured at.

### Test 4: Dimensionality scaling

**Setup:** Run the same inference task (6 concepts, fixed number of observations) with d = 2, 4, 8, 16, 32 dimensions. The true concept centers are placed randomly on the unit hypersphere.

**What to measure:**
- Posterior convergence time as a function of d
- Computational cost: precision matrix update is O(d²), matrix inversion for μ_post is O(d³) — measure wall clock
- Does inference quality degrade with d? (Distance to true center after fixed observations)

**What would be surprising:** If performance *improves* with dimensionality due to the blessing of dimensionality (concepts become more separable in high-d space, and the Gaussian posterior remains well-behaved).

**What would be genuinely novel:** Finding the cross-over point where O(d³) matrix inversion makes the closed-form approach slower than a few gradient steps of amortized inference. This is the scaling boundary between the two camps. Nobody has characterized this at a parameter level.

### Test 5: Non-stationary tracking

**Setup:** Define a single concept whose true center drifts slowly over time: μ_true(t) = μ_true(0) + vt. The agent receives words from this moving concept.

**What to measure:**
- Tracking error `||μ_post - μ_true(t)||` as a function of drift velocity v
- Does the agent's posterior mean track the moving concept, or lag behind?
- What happens if you add a forgetting factor (exponential decay on accumulated precision)?

**What would be surprising:** If the conjugate model tracks *better* than a gradient-based approach, due to the exact posterior updating rather than a learning rate hyperparameter.

**What would be genuinely novel:** Using the tracking error vs. drift velocity curve to estimate the effective time constant of the conjugate update, and showing how to set NIW priors to match the timescale of environmental change. This is adaptive online learning without learning rates.

---

## What to measure, and how

### Primary metrics

| Metric | Formula | What it captures |
|--------|---------|------------------|
| **Reconstruction NLL** | `-log P(o_test \| s_inferred)` | Predictive accuracy on held-out words |
| **Localization error** | `\|μ_post - s_true\|` | How well the agent knows where it is |
| **Posterior entropy** | `½ log det(2πe Σ_post)` | Uncertainty calibration |
| **Collapse index** | `tr(Σ_between) / tr(Σ_within)` | Representational distinctness of concepts |

### Derived diagnostics

| Diagnostic | How to compute | What to watch for |
|------------|----------------|-------------------|
| **Precision trace** | `tr(Σ_post⁻¹)` over time | Should increase monotonically with observations |
| **Effective sample size** | `κ_T - κ₀` for each word Gaussian | How many observations each concept has absorbed |
| **KL from prior** | `D_KL[ q(s) \|\| P(s) ]` | How much the agent has learned from its observations |
| **Posterior multimodality** | Fit 2-component GMM to posterior samples, check weight ratio | Detecting when unimodal approximation fails |

---

## Where things are lacking — the honest gaps

### Gap 1: The unimodal approximation

The conjugate update gives you a Gaussian posterior. But the true posterior when you observe words from multiple distinct concepts is a *mixture* of Gaussians. The single Gaussian approximation will:
- Place its mean at the precision-weighted average of the component centers
- Underestimate uncertainty if the components are far apart
- Snap to one mode if one observation type dominates

The AXIOM literature acknowledges this and sometimes uses mixture-of-Gaussians posteriors (which remain conjugate to Gaussian likelihoods). But then the number of mixture components grows exponentially with observations unless you prune. This is the "mixture explosion" problem — it's why particle filters exist. You'll hit it immediately.

**What to do:** Start with unimodal and document when it breaks. The phase boundary where it breaks is itself a finding.

### Gap 2: The discrete observation assumption

I've kept observations discrete to match your current setup. But the whole point is to handle continuous observations (pixels, audio). With continuous observations, the likelihood `P(o|s)` becomes a Gaussian with mean that's a function of s — e.g., `o | s ~ N(f(s), Σ_o)` where f is a nonlinear mapping. This breaks conjugacy unless f is linear. The AXIOM approach handles this by chunking continuous observations into discrete tokens (object-centric parsing), which is a different can of worms.

**What to do:** Stay with discrete observations for the first round. The transition to continuous observations is Problem 2b.

### Gap 3: The action-observation loop

In active inference, actions generate observations that update beliefs. Your Exp 128 showed that credit assignment breaks when actions and their consequences aren't co-presented in inference. The continuous model inherits this — if the agent moves and then hears a word, it needs to correctly attribute the word to the *new* state, not the old one. This is a state estimation problem (forward prediction + correction) that the precision accumulation handles cleanly *if* the transition model is known, but becomes messy if B must also be learned.

**What to do:** First experiments should fix the action model (known Δ(a), known Σ_action) and only learn the emission model. Then relax.

### Gap 4: The NIW prior sensitivity

The NIW prior has four hyperparameters (ν₀, κ₀, m₀, S₀) per concept. These control:
- ν₀: how many virtual observations the prior is worth (inertia)
- κ₀: how strongly the prior mean is weighted
- m₀: prior guess at the Gaussian center
- S₀: prior guess at the covariance scale

The model's behavior is sensitive to these, and the field doesn't have clear guidance on how to set them beyond "use weakly informative priors." Your collapse resistance experiment (Test 3) directly probes this sensitivity. Characterizing it is a contribution.

### Gap 5: Comparison to amortized inference

The claim "closed-form is better" only means something relative to "amortized with gradients." You need a minimal neural baseline:
- Same generative model (Gaussian emissions, continuous state)
- Replace exact posterior with an encoder network `q(s|o) = N(μ_enc(o), Σ_enc(o))`
- Train with the ELBO: `E_q[log P(o|s)] - D_KL[q(s|o) || P(s)]`
- Compare convergence speed, final accuracy, and robustness to noise

Without this, you're not evaluating the closed-form claim — you're just building a continuous model and calling it done.

---

## What would be genuinely surprising and of real interest

1. **The collapse boundary is characterizable.** If you can draw a phase diagram showing what noise level / prior strength combinations cause representational collapse in continuous vs. tabular models, you've quantified a phenomenon the field only gestures at. The Bruineberg critique ("FEP is unfalsifiable") is disarmed by exactly this kind of parameter-level prediction.

2. **Closed-form beats amortized at small scale but loses at some dimensionality.** Finding the crossover — "conjugate is better for d < ~10, amortized wins above" — would be directly useful to both camps and would clarify the scaling argument.

3. **Interpolation fails in a predictable way.** If you can show that Gaussian posteriors fail to interpolate when concepts are more than X standard deviations apart, and that this threshold depends on observation noise in a specific way, you've found a quantitative limitation of the conjugate approach that motivates either mixture posteriors or amortization.

4. **The NIW hyperparameters act as a "learning rate" without a learning rate.** If you can show that varying κ₀ smoothly controls the speed-stability tradeoff without the oscillations that plague gradient-based learning rates, you've demonstrated something elegant about conjugate Bayesian online learning.

5. **Your Exp 31 collapse is *worse* in tabular than continuous.** This would be the cleanest demonstration that the representational substrate matters, and that the field's move toward continuous states is empirically justified even at toy scale.

---

## Summary card

| Element | Answer |
|---------|--------|
| **Problem** | Replace enumerated discrete states with continuous latent space while keeping inference tractable, online, and gradient-free |
| **Why open** | Two camps disagree on whether closed-form or amortized inference scales; no clean comparison exists at parameter level |
| **Mathematical core** | Precision accumulation: `Σ_post⁻¹ = Σ₀⁻¹ + Σ Σₖ⁻¹`, `μ_post = Σ_post(Σ₀⁻¹μ₀ + Σ Σₖ⁻¹μₖ)` |
| **Key tests** | Convergence speed, interpolation, collapse resistance under noise, dimensionality scaling, non-stationary tracking |
| **Key metrics** | Localization error, posterior entropy, collapse index, reconstruction NLL |
| **Honest gaps** | Unimodal approximation, discrete obs only, NIW sensitivity, no amortized baseline |
| **What's surprising** | Phase diagram of collapse, crossover dimensionality vs. amortized, NIW-as-adaptive-learning-rate |
| **Your advantage** | You have the tabular baselines, the Exp 31 finding, and the measurement discipline. This is the right next increment. |

---

## Repo annotations (added at capture, 2026-06-10 — NOT part of the supplied program)

1. **The conjugacy/normalization wrinkle (rung 1's first declared choice).** The emission as
   written — `P(o=k|s) ∝ exp(-½(s-μₖ)ᵀΣₖ⁻¹(s-μₖ))` — is unnormalized over k. The properly
   normalized categorical emission, `P(o=k|s) = wₖ(s)/Σⱼ wⱼ(s)`, has an s-dependent
   normalizer that breaks the product-of-Gaussians conjugacy (even with shared Σₖ). Using
   the unnormalized form is precisely the modeling choice that buys closed-form updates;
   the implementation must DECLARE it in the rung-1 predeclaration, and its cost is part
   of what rung 2 (interpolation / multimodality) measures. The program's Gap 1 is the
   adjacent issue on the posterior side.
2. **EFE term naming.** In `G(a)` as written, the first term is the AMBIGUITY (expected
   surprise) term; the information-gain ("epistemic value") decomposition is a different
   rearrangement — the program itself hedges this ("or, in the ambiguity-resolving
   variant"). Any rung that uses `G(a)` must state which decomposition it computes.
3. **Mapping to the RECIPE (PREMISE.md).** The hand-placed word-Gaussians of Tests 1–2 ARE
   the "one innate anchor," in continuous form; Test 3 (NIW-learned means) is the
   anchor-relaxation probe — the continuous rematch of Exp 31. The RECIPE is not amended
   by this direction; what's under test is whether its collapse clause is
   substrate-dependent.
4. **Thread isolation.** This direction touches neither the halted M4a thread (increment
   1c still awaits its own explicit human word — Exp 128 / loop/IDEAS.md) nor mirro's and
   vela's spines. Rung scripts are self-contained `experiments/expNNN_*.py` agents; the
   creature substrate is not migrated until the direction's stop condition is reached
   with a positive verdict.
