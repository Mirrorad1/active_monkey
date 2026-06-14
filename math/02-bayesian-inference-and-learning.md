# Bayesian inference & learning

> This area of math is how a system updates its beliefs from evidence: it starts with a **prior**, sees data, and produces a **posterior**. Everything the `active_monkey` creature does — perceiving where it is, learning a sensory map from scratch, predicting the next character — is one of two Bayesian operations: *inference* (update belief about a hidden state given an observation) or *learning* (update belief about the model's parameters given a whole stream). This file builds up from Bayes' rule to the exact Bayesian HMM the repo implements, and ends with the place fields that let the creature perceive space and the two ways unsupervised emergence collapses.

## Bayes' rule

**Glossary.** Given a hidden cause `s` and an observation `o`, Bayes' rule turns a *forward* model `P(o | s)` (how causes produce data) into a *backward* belief `P(s | o)` (which cause was likely, given the data).

```
                P(o | s) · P(s)
P(s | o)  =  ─────────────────────          (posterior = likelihood × prior / evidence)
                    P(o)

P(o)  =  Σ_s P(o | s) · P(s)                 (evidence / marginal likelihood)
```

- `P(s)` — **prior**: belief about the cause before seeing data.
- `P(o | s)` — **likelihood**: how probable the data is under each cause.
- `P(s | o)` — **posterior**: updated belief after seeing `o`.
- `P(o)` — **evidence** (a.k.a. marginal likelihood): total probability of the data, summed over all causes. It is just a normalizer that makes the posterior sum to 1.
- The shorthand `posterior ∝ likelihood × prior` drops `P(o)`; you recover it by normalizing.

The evidence `P(o)` is also the quantity the whole system is trying to *maximize* (equivalently, minimize its negative log, the surprise): a good model assigns high probability to what actually happens.

**Example.** A creature is in one of two cells, `s ∈ {0, 1}`, prior `P(s=0)=0.5`, `P(s=1)=0.5`. Cell 0 usually looks "red", cell 1 usually looks "blue": `P(red | 0)=0.8`, `P(red | 1)=0.2`. It sees `red`.

```
unnormalized:  s=0 → 0.8 × 0.5 = 0.40
               s=1 → 0.2 × 0.5 = 0.10
evidence P(red) = 0.40 + 0.10 = 0.50
posterior:     P(0 | red) = 0.40 / 0.50 = 0.8
               P(1 | red) = 0.10 / 0.50 = 0.2
```

Seeing "red" moved the belief from 50/50 to 80/20 toward cell 0.

**Data.** This is the single most-used operation in the repo. In **Exp 21** the creature localizes itself in a 3×3 grid using its *learned* sensory map and gets `0.00 bits` of residual uncertainty in the correct cell — i.e. the posterior `P(cell | colors)` collapsed onto one cell. Same in **Exp 20** (1D): "localization with the LEARNED map = 0.00 bits, correct cell." The evidence term `P(o)` is the thing that *falls in surprise* as learning proceeds: **Exp 1** reports held-out surprise `4.81 → 4.00 bits/char` as the model learns (see the audit note under *Forward filtering* — the underlying re-run logs `4.007 → 3.424`; the drop itself is the real, reproduced result).

**▸ In programmer terms.** It is a weighted lookup followed by a normalize — `softmax` over `log prior + log likelihood`.

```python
import numpy as np

def bayes_update(prior, likelihood_col):   # likelihood_col[s] = P(o_observed | s)
    unnorm = likelihood_col * prior         # elementwise: likelihood × prior
    return unnorm / unnorm.sum()            # divide by evidence = sum

prior = np.array([0.5, 0.5])
P_red_given_s = np.array([0.8, 0.2])        # the observed column of P(o|s)
print(bayes_update(prior, P_red_given_s))   # -> [0.8, 0.2]
```

---

## Categorical distribution & the probability simplex Δ^K

**Glossary.** A **categorical** distribution is the "one roll of a `K`-sided die": exactly one of `K` outcomes occurs, with probabilities that are non-negative and sum to 1. The set of all such probability vectors is the **probability simplex** `Δ^K`.

```
Cat(x ; p):   P(x = k) = p_k ,   with   p_k ∈ [0, 1]   and   Σ_{k=1}^{K} p_k = 1

Δ^K  =  { p ∈ ℝ^K  :  p_k ≥ 0,  Σ_k p_k = 1 }          (the probability simplex)
```

- `K` — number of outcomes.
- `p_k` — probability of outcome `k`; the vector `p` is one point in `Δ^K`.
- A matrix is **column-stochastic** if *every column* is a point in `Δ^K` (each column sums to 1). The repo's `A`, `B`, `D` are exactly this.

**Example.** Over the 28-symbol alphabet (`a–z`, space, period), a categorical predictive distribution might put `p = (space:0.25, e:0.10, t:0.07, …)`. To be in `Δ^28` the 28 numbers must be `≥ 0` and total `1.0`. The flat (uniform) point of `Δ^28` is `p_k = 1/28 ≈ 0.0357` for every `k`.

**Data.** The repo's three core tensors are all stacks of categorical columns (RESEARCH.md §1.1):

```
A ∈ Δ^{V×K}   column-stochastic   A[o,s] = P(o_t = o | s_t = s)     (emission, V=28)
B ∈ Δ^{K×K×1} column-stochastic   B[s',s,0] = P(s_t = s' | s_{t-1} = s)  (transition)
D ∈ Δ^K                            D[s] = P(s_1 = s)                 (initial, uniform)
```

In **Exp 21** the *learned* per-cell color tuning is reported as the integer argmax of each `A` column, `[0,1,2,1,2,0,2,0,1]`, matching the true colormap exactly — each of the 9 columns of `A` is a point in `Δ^3` that peaked on the right color. In **Exp 24** the learned object map is the categorical `P(object present | place) = [0,0,0,0,1,0,0,0,0]` — a column in `Δ^9` that put all its mass on cell 4.

**▸ In programmer terms.** A categorical is a normalized probability array; column-stochastic means `M.sum(axis=0)` is all ones.

```python
import numpy as np

def is_in_simplex(p, tol=1e-9):
    return np.all(p >= -tol) and abs(p.sum() - 1.0) < tol

def is_column_stochastic(M):
    return np.allclose(M.sum(axis=0), 1.0) and np.all(M >= 0)

A = np.array([[0.7, 0.1],     # P(o|s) — each COLUMN is one categorical over outcomes
              [0.2, 0.3],
              [0.1, 0.6]])
assert is_column_stochastic(A)            # columns sum to 1, not rows
def sample(p): return np.random.choice(len(p), p=p)   # one categorical draw
```

---

## The Bayesian HMM generative model

**Glossary.** The repo's language creature (M3) is a **Hidden Markov Model with conjugate Dirichlet priors over its parameters** — a *Bayesian* HMM. A hidden first-order Markov chain `s_{1:T}` emits one observed symbol `o_t` per step; the emission `A` and transition `B` are themselves random with priors. The full joint over the observed character stream `o_{1:T}`, the latent path `s_{1:T}`, and the parameters is (RESEARCH.md §1.1):

```
p(o_{1:T}, s_{1:T}, A, B) = p(A) · p(B) · D[s_1] · ∏_{t=2}^{T} B[s_t, s_{t-1}] · ∏_{t=1}^{T} A[o_t, s_t]
```

- `D[s_1]` — initial-state prior (the chain's first state).
- `∏ B[s_t, s_{t-1}]` — the **first-order Markov** transition chain: each `s_t` depends on the past *only* through `s_{t-1}`.
- `∏ A[o_t, s_t]` — the **emission**: each observed character depends only on the current hidden state.
- `p(A), p(B)` — Dirichlet priors over the (random) parameter matrices (see *Dirichlet conjugate prior*).

The whole structural lesson of the repo is encoded here: the only path from history to the next character is `o_{<t} → s_{t-1} → s_t → o_t`. (See *Mean-field approximation* and *Forward filtering* for what this buys and costs.)

**Example.** Tiny HMM, `K=2`, two symbols `{x, y}`. `D=(1,0)` (start in state 0). `B = [[0.9,0.5],[0.1,0.5]]` (state 0 is sticky). `A = [[0.8,0.1],[0.2,0.9]]` (state 0→x, state 1→y). Probability of the path `s=(0,0)` emitting `o=(x,x)`:

```
D[0]            = 1.0
A[x,0]          = 0.8           (emit x at t=1)
B[0,0]          = 0.9           (stay in 0)
A[x,0]          = 0.8           (emit x at t=2)
joint = 1.0 × 0.8 × 0.9 × 0.8 = 0.576
```

**Data.** This is the literal model in `active_loop/lang_model.py` with `V=28`, `K=14` in the spec (sweeps tried 12/30/60). **Exp 1** confirms the engine learns: held-out surprise drops and generation "shifts from random to spacing/period/letter-cluster structure." **Exp 3** trained it on `"mirro "` and surprise fell `3.38 → 1.61`, but the output was the *right letters in jumbled order* ("mo io riorrr") — the first-order chain learned the emission palette (`A`) but not order, exactly because the only memory is the single hidden state `s_{t-1}` (RESEARCH.md §1.5). **Exp 4** showed bigger `K` (12/30/60) does *not* fix order — a key negative result: capacity ≠ memory.

**▸ In programmer terms.** Sampling the generative model is two nested lookups in a loop; the joint probability is a running product.

```python
import numpy as np

def hmm_sample(D, A, B, T, rng):
    s = rng.choice(len(D), p=D)               # initial state from D
    out = []
    for t in range(T):
        o = rng.choice(A.shape[0], p=A[:, s])  # emit:  o ~ A[:, s]
        out.append(o)
        s = rng.choice(B.shape[0], p=B[:, s])  # transition:  s' ~ B[:, s]
    return out

def hmm_joint(D, A, B, s_path, o_seq):
    p = D[s_path[0]]
    for t, o in enumerate(o_seq):
        p *= A[o, s_path[t]]                    # emission factor
        if t > 0:
            p *= B[s_path[t], s_path[t-1]]      # transition factor
    return p
```

---

## Dirichlet conjugate prior & why learning = counting

**Glossary.** The **Dirichlet** is the conjugate prior for a categorical: a distribution *over* probability vectors `p ∈ Δ^K`. "Conjugate" means the posterior after seeing data is *again* Dirichlet — so Bayesian updating reduces to adding to a count.

```
Dir(p ; α) ∝ ∏_{k=1}^{K} p_k^{(α_k − 1)} ,     α_k > 0  (concentration / pseudo-counts)

Conjugacy:   prior Dir(p ; α)  +  observed counts n  ⇒  posterior Dir(p ; α + n)

E[p_k] = α_k / Σ_j α_j                          (the Dirichlet expected value)
```

- `α_k` — **concentration** parameter for outcome `k`; read it as a **pseudo-count** (a fractional prior observation of outcome `k`). Small `α_k < 1` ⇒ a sparse prior that expects most mass on a few outcomes.
- `α + n` — posterior pseudo-counts: just the prior counts plus how many times each outcome was actually seen. *That* is why learning a categorical is literally counting.
- `E[p_k]` — the mean of the posterior, the point estimate the repo uses downstream.

**Example.** Prior `Dir(α = (1,1,1))` over 3 colors (uniform, 1 pseudo-count each). Observe colors `n = (red:5, green:0, blue:1)`. Posterior is `Dir((6,1,2))`, and the predicted color distribution is the mean:

```
Σα = 6 + 1 + 2 = 9
E[p] = (6/9, 1/9, 2/9) ≈ (0.667, 0.111, 0.222)
```

Note green keeps `1/9` despite zero sightings — the prior pseudo-count is built-in **smoothing** (no zero probabilities).

**Data.** The repo sets these priors explicitly (RESEARCH.md §1.1): `pA = A_CONC · 1` with `A_CONC = 0.1`, and `pB = B_CONC · 1`. A concentration of `0.1` is a deliberately *sparse* prior — it expects each hidden state to emit only a few characters, which is what lets clean place fields and emission peaks form. In **Exp 26**, the proto-opinion is read straight off Dirichlet-learned mean preferences: world-comfortable-feature-0 gives learned `C = [0.98,0.01,0.01]`, world-comfortable-feature-2 gives `C = [0.01,0.01,0.98]` — same architecture, different lived counts, different `E[p]`.

**▸ In programmer terms.** A Dirichlet posterior is a `Counter` initialized to the prior; the point estimate is normalize-after-adding.

```python
import numpy as np

alpha0 = np.array([1.0, 1.0, 1.0])          # prior pseudo-counts (uniform smoothing)
counts = np.array([5.0, 0.0, 1.0])          # data: red x5, blue x1
alpha_post = alpha0 + counts                 # conjugacy: posterior = prior + counts
p_hat = alpha_post / alpha_post.sum()        # E[p] = dirichlet_expected_value
print(p_hat)                                 # [0.667, 0.111, 0.222]
```

---

## Parameter learning: Dirichlet count updates over inferred states

**Glossary.** Inside the HMM the "counts" are not observed directly — the hidden state is uncertain — so the model adds the *expected* sufficient statistics, weighted by its belief `q(s_t)`. The VFE-minimizing posterior over `A` is Dirichlet with concentrations incremented by these expected counts (RESEARCH.md §1.4):

```
A update:   α^A ← α^A_0 + κ · Σ_t  o_t ⊗ s_t        (s_t = q(s_t),  o_t one-hot)
B update:   α^B ← α^B_0 + κ · Σ_t  s_t ⊗ s_{t-1} ⊗ a_{t-1}

point estimate:   A = E[a] = α / Σα                  (dirichlet_expected_value)
```

- `o_t ⊗ s_t` — outer product of the one-hot observation and the belief over states; its `[o,s]` entry is the **soft count** "state `s` was believed to emit character `o` at time `t`."
- `κ` — a learning-rate / scale on the counts.
- `s_t ⊗ s_{t-1} ⊗ a_{t-1}` — the analogous soft count of a transition `s_{t-1} → s_t` under action `a_{t-1}`.
- Final estimate is the Dirichlet mean (see *MAP vs posterior-mean*).

In words: every time the model *believes* state `s` emitted character `o`, it adds a (fractional) pseudo-count to `α^A[o,s]`. This is exact conjugate Bayesian counting in the *latent* space — the active-inference analogue of smoothed n-gram counting, but over *inferred* states rather than observed symbols (see *Bayesian HMM* and *Dirichlet conjugate prior*).

**Example.** `K=2`, 2 symbols, prior `α^A_0 = 0.1` everywhere, `κ=1`. At `t=1` the model sees `o = x` (`o_t = [1,0]`) and believes `q(s_1) = [0.7, 0.3]`. The soft count added is the outer product:

```
o_t ⊗ s_t = [1,0]ᵀ ⊗ [0.7,0.3] = [[0.7, 0.3],     # row x:  +0.7 to (x,s0), +0.3 to (x,s1)
                                    [0.0, 0.0]]     # row y:  nothing seen
α^A = [[0.1+0.7, 0.1+0.3],   = [[0.8, 0.4],
       [0.1,     0.1   ]]       [0.1, 0.1]]
column s0 mean:  E[A[:,0]] = (0.8, 0.1)/0.9 ≈ (0.889, 0.111)   # state 0 now predicts x
```

**Data.** This update is the entire learning loop. **Exp 20** learned the per-state color tuning `[0,0,1,0,1,1]` (= true colormap) purely by accumulating `q(s_t)`-weighted emission counts over a 700-step wander. **Exp 21** scaled it to 2D: learned tuning `[0,1,2,1,2,0,2,0,1]` matched truth exactly. **Exp 24** learned a transition/emission relation `P(object | place) = [0,0,0,0,1,0,0,0,0]` by wandering 900 steps. The critical caveat is in *Mean-field approximation* below: learning **both** `A` and `B` at once from noise has no anchor and collapses (**Exp 31**).

**▸ In programmer terms.** It is `np.outer(one_hot_obs, belief)` accumulated into a counts matrix, then column-normalized (each column normalized to sum to 1).

```python
import numpy as np

def learn_A(stream, beliefs, V, K, alphaA0=0.1, kappa=1.0):
    alphaA = np.full((V, K), alphaA0)                 # Dirichlet prior pseudo-counts
    for o_t, q_s in zip(stream, beliefs):             # o_t: int, q_s: belief over K states
        o_onehot = np.eye(V)[o_t]
        alphaA += kappa * np.outer(o_onehot, q_s)     # soft count  o_t ⊗ s_t
    A = alphaA / alphaA.sum(axis=0, keepdims=True)     # E[a] = α / Σα, column-stochastic
    return A
```

---

## Forward filtering (the Bayesian filter update)

**Glossary.** *Inference* in the HMM is updating the belief over the current hidden state as each character arrives. The exact single-factor update — and the fixed point of pymdp's `fpi` algorithm for this single-factor / single-modality case — is the Bayesian filter (RESEARCH.md §1.3):

```
q(s_t)  ∝  prior(s_t) · A[o_t, s_t] ,      prior(s_t) = Σ_{s'} B[s_t, s'] · q(s_{t-1})

predict (surprise):   pred = A · prior(s_t);   surprise_t = −log₂ pred[o_t]   (bits/char)
```

- `prior(s_t)` — the **predict** step: push the last belief through the transition `B` to get the belief about the current state *before* seeing `o_t`.
- `q(s_t)` — the **update** step: reweight that prior by the likelihood `A[o_t, s_t]` of the character that actually arrived, then normalize. This is just Bayes' rule with `prior(s_t)` as the prior.
- `surprise_t` — the negative log probability the model assigned to the character it then saw; averaged, this is the repo's `bits/char` metric (`log₂` ⇒ bits). At an exact single-factor fixed point this equals the predictive surprise `−log P(o_t | o_{<t})`.

**Example.** `K=2`, `q(s_{t-1}) = [0.8, 0.2]`. `B = [[0.9,0.5],[0.1,0.5]]`, `A = [[0.8,0.1],[0.2,0.9]]`, observe `o_t = x` (row 0 of `A`).

```
predict:  prior = B · q = [0.9·0.8 + 0.5·0.2,  0.1·0.8 + 0.5·0.2] = [0.82, 0.18]
predicted P(x) = A[x,:]·prior = 0.8·0.82 + 0.1·0.18 = 0.674   → surprise = −log₂ 0.674 ≈ 0.569 bits
update:   unnorm = A[x,:] * prior = [0.8·0.82, 0.1·0.18] = [0.656, 0.018]
q(s_t) = [0.656, 0.018]/0.674 ≈ [0.973, 0.027]
```

**Data.** This filter *is* the bits/char metric. **Exp 1** is the headline: held-out surprise fell from the uniform baseline `log₂ 28 ≈ 4.807` toward a learned model, logged as `4.81 → 4.00 bits/char`. **Honesty note (audit, EXPERIMENTS.md 2026-06-09):** the original transcript and the re-run both show `4.007 → 3.424 bits/char`; the `4.81 → 4.00` is a *narrative-text* inaccuracy in the log, not a reproducibility failure — the run reproduces byte-for-byte and the surprise genuinely drops. **Exp 14** uses the same filtered predictive entropy to measure grounded valence: `P(next | ..a) = 3.04 bits` (confident) vs `P(next | ..z) = 4.79 bits` (uncertain), a `+1.75` bit gap — the cue `a` became "felt-good" by predicting low-surprise states, with no label.

**▸ In programmer terms.** It is a two-line loop: matrix-vector multiply (predict), then elementwise multiply and normalize (update).

```python
import numpy as np

def forward_filter(obs, A, B, D):
    q = D.copy()                                   # belief at t=0
    bits = 0.0
    for o in obs:
        prior = B @ q                              # predict:  Σ_s' B[:,s'] q[s']
        pred = A[o] @ prior                        # P(o | history) = scalar
        bits += -np.log2(pred + 1e-12)             # accumulate surprise (bits)
        q = A[o] * prior                           # update (unnormalized Bayes)
        q = q / q.sum()                            # normalize over evidence
    return bits / len(obs)                          # mean bits/char
```

---

## Mean-field approximation & the collapse finding

**Glossary.** When the model has more than one hidden factor (e.g. a slow *topic* `z` and a fast *char* `s`), the exact joint posterior `q(z, s)` is expensive, so pymdp uses the **mean-field** approximation: assume the factors are independent in the posterior.

```
q(s) = ∏_i q(s_i)              (mean-field: factorize the posterior into independent factors)
e.g.  q(z, s) ≈ q(z) · q(s)    (topic and character beliefs treated as independent)
```

- Each `q(s_i)` is its own categorical, updated separately.
- The cost: by *assuming* independence, the approximation **severs** any cross-factor message. Evidence that lives in *how factors interact* (a character *sequence* implying a topic) cannot reach the other factor's belief.

**Example.** Two topics `z ∈ {sky, grass}`, identical emissions, but topic-specific transitions `B`. Suppose `B_sky` makes `"is " → "blue"` likely and `B_grass` makes `"is " → "green"` likely. After observing the char-sequence "...is blue", the *correct* posterior should push `q(z)` toward `sky`. Under mean-field, the message that "this transition pattern implies sky" is a *cross-factor* coupling between `q(z)` and `q(s)`; because `q(z,s) = q(z)q(s)` forbids that coupling, `q(z)` never moves off `[0.5, 0.5]`.

**Data.** This is **Exp 16** (deep negative): topics were handed a *perfect* asymmetric foothold (`B_sky` vs `B_grass` fully differentiated), yet topic belief "stayed `[0.5,0.5]` EVEN WITH perfectly differentiated transitions; output degenerate ('s is i')." The diagnosis: "Cause = the MEAN-FIELD APPROXIMATION: `q(z,s)≈q(z)q(s)` … SEVERS the cross-factor message 'this char-sequence implies topic0'." The implied fix is to route the topic through the **emission** `A` (so observations *directly* update `q(z)`), not through transitions alone. The closely related joint-learning failure is **Exp 31**: learning *both* `A` and `B` from random init collapses to a degenerate fixed point (e.g. learned right-step map `[0,0,0,0,0]` — all states map to one) because nothing breaks the symmetry; the recipe needs **one innate anchor** (give either `A` or `B`). RESUME.md §2 folds these together as the durable finding: disembodied unsupervised emergence collapses via "symmetric saddle / posterior collapse / non-identifiability / mean-field severs cross-factor inference," and the **RECIPE** (embodiment + grounding + continuous registered experience + one innate anchor) is what breaks it.

**▸ In programmer terms.** Mean-field is updating each factor's belief in its own loop, never forming the joint table — fast, but it cannot represent correlations between factors.

```python
# Exact joint vs mean-field, schematically:
#   EXACT:      q_joint[z, s]  -- full K_z x K_s table; can encode "z and s correlated"
#   MEAN-FIELD: q_z[z], q_s[s] -- two separate vectors; q_joint := outer(q_z, q_s) ONLY

def meanfield_update(q_z, q_s, A, B, obs):
    # each factor updated from its OWN marginal evidence; no z<->s coupling term:
    q_s = normalize(A[obs] * (B @ q_s))      # char factor sees the observation
    # q_z gets NO message from the char SEQUENCE under transition-only coupling:
    # (the cross-factor term that would update q_z is dropped by factorization)
    return q_z, q_s                          # q_z stays flat -> Exp 16 collapse
```

---

## MAP vs posterior-mean point estimates

**Glossary.** A full posterior is a *distribution*; downstream code often needs a single point estimate. Two standard choices:

```
MAP   (maximum a posteriori):   p* = argmax_p  P(p | data)        (the mode / peak)
mean  (posterior expectation):  p̂ = E[p | data]                   (the average)

for Dirichlet Dir(α):   mode_k = (α_k − 1) / (Σ_j α_j − K)   (needs all α_k > 1)
                        mean_k = α_k / Σ_j α_j               (always valid)
```

- **MAP** picks the single most probable value; for a categorical observed with counts it is the maximum-likelihood relative frequency (the un-smoothed count fraction), and for a Dirichlet it is the mode formula above.
- **mean** averages over the whole posterior; it retains the prior's smoothing and never divides by zero.

The repo uses the Dirichlet **posterior mean** `A = E[a] = α/Σα` for `A` and `B` — it is always defined (even with the sparse `α=0.1` prior where MAP's `α−1` would go negative) and it keeps prior smoothing so no transition/emission is ever exactly zero.

**Example.** `Dir(α = (6, 1, 2))` from the earlier counting example (`Σα = 9`):

```
mean:  E[p] = (6/9, 1/9, 2/9)         ≈ (0.667, 0.111, 0.222)
MAP:   mode = (6−1, 1−1, 2−1)/(9−3)   = (5, 0, 1)/6 ≈ (0.833, 0.0, 0.167)
```

MAP is sharper and zeros out the never-seen-after-prior outcome; the mean stays smoother and keeps green at `0.111`. With the repo's `α=0.1` sparse prior, an unseen outcome would have `α_k = 0.1 < 1`, so its MAP mode formula goes negative — which is exactly why the repo takes the **mean**.

**Data.** Every learned matrix reported in **Exp 20/21/24/26** is the Dirichlet *mean* `α/Σα` (RESEARCH.md §1.4, `dirichlet_expected_value`), then read out via `argmax` for the human-readable tuning vectors (`[0,1,2,1,2,0,2,0,1]` in Exp 21 is the per-column argmax of the mean estimate). The decisions (which color/cell a state stands for) use the *mode of the mean*, while the smoothing that keeps learning stable comes from using the mean rather than a raw MLE/MAP.

**▸ In programmer terms.** MAP ≈ argmax of (smoothed) counts; posterior-mean ≈ normalized (smoothed) counts. The repo stores the mean and `argmax`es it only at read-out time.

```python
import numpy as np

alpha = np.array([6.0, 1.0, 2.0])
post_mean = alpha / alpha.sum()                 # used for learning/prediction (smooth)
readout   = int(np.argmax(post_mean))           # used for human-readable tuning label
# MAP mode is only valid when all alpha_k > 1:
map_mode  = (alpha - 1) / (alpha.sum() - len(alpha)) if np.all(alpha > 1) else None
```

---

## Place fields: Gaussian receptive fields as the embodied likelihood A

**Glossary.** A **place field** (or Gaussian receptive field) is a likelihood that tells the creature how strongly each sensory feature fires as a function of where it is — the embodied version of the emission `A`. Continuous variants use a Gaussian footprint; the grid variants use a categorical `A` column per cell. The Gaussian form:

```
A(o | x)  ∝  exp( −(o − μ)² / (2 σ²) )          (Gaussian receptive field centered at μ)
```

- `x` — the creature's (hidden) location / state.
- `μ` — the field's preferred location (where the cell fires hardest).
- `σ` — its width (how sharply tuned it is); large `σ` ⇒ broad, overlapping fields ⇒ more aliasing/ambiguity.
- In the discrete grid worlds, `A[o, x] = P(color o | cell x)` is just a categorical column per cell — the discrete analogue of the receptive field, learned by the Dirichlet count update.

The creature *perceives space* by Bayesian inversion of this likelihood: seeing a color, it does the Bayes update `P(cell | color) ∝ A[color, cell] · prior(cell)` (see *Bayes' rule* and *Forward filtering*). The "place fields self-organize" milestone is the system *learning* `A` from scratch by counting (see *Parameter learning*).

**Example.** 1D track, cells `{0,1,2}`, colors `{red,green}`. Learned `A` (categorical place fields):

```
A = [[0.9, 0.5, 0.1],      row = red    -> cell 0 is a "red place field"
     [0.1, 0.5, 0.9]]      row = green  -> cell 2 is a "green place field", cell 1 ambiguous
```

Flat prior `(1/3,1/3,1/3)`, observe `green`:

```
unnorm = A[green,:] · prior = (0.1, 0.5, 0.9)/3 = (0.033, 0.167, 0.300)
P(cell | green) = (0.033, 0.167, 0.300)/0.500 = (0.067, 0.333, 0.600)  -> peaked on cell 2
```

**Data.** **Exp 20** is the clean 1D milestone: from a 700-step continuous wander, the creature's *learned* place fields recovered the true colormap `[0,0,1,0,1,1]` exactly, and localization with the learned map hit `0.00 bits` in the correct cell. **Exp 21** scaled this to a 2D 3×3 grid (900-step wander): learned tuning `[0,1,2,1,2,0,2,0,1]` = true colormap exactly, `0.00 bits` localization. The honest caveats: the movement model `B` was innate/known and learning *only* `A` worked — learning both from scratch collapses (**Exp 31**, see *Mean-field approximation*). **Exp 20** also names the key enabler: *continuous registered experience* (the belief is never reset to a fixed start), which is exactly what embodiment provides and the disembodied symbol experiments lacked (RESUME.md §2).

**▸ In programmer terms.** A place field is `A[:, cell]` — a per-cell probability column. Perceiving space is a Bayes update against that column; learning it is the Dirichlet count update from earlier.

```python
import numpy as np

# Gaussian receptive field over a 1-D feature axis, centered at mu, width sigma:
def gaussian_field(feature_axis, mu, sigma):
    a = np.exp(-((feature_axis - mu) ** 2) / (2 * sigma ** 2))
    return a / a.sum()                            # normalize to a categorical column

# Discrete grid version: A[:, cell] is the learned categorical "place field" per cell.
def localize(A, prior, observed_color):
    post = A[observed_color] * prior              # Bayes: likelihood column * prior
    return post / post.sum()                      # P(cell | color); argmax = best guess

A = np.array([[0.9, 0.5, 0.1],
              [0.1, 0.5, 0.9]])                   # learned place fields (Exp 20-style)
print(localize(A, np.ones(3)/3, observed_color=1))  # green -> peaks on cell 2
```

---

## Bayesian model reduction & structure growth (how the model sizes itself)

**Glossary.** Everything above *fits* a model of a fixed size: `A` is `V×K`, `K` chosen by hand. But **Exp 4** showed that capacity is not memory — sweeping `K ∈ {12,30,60}` did *not* fix word order — so picking `K` by sweeping is a dead end. **Structure learning** picks the size as an *inference*: two opposite moves, both scored by model evidence (lower free energy `F` = better model on the *same* history; for the exact-filtering HMM the negative log-evidence is `F = Σ_t −ln p(o_t | o_{<t}, a_{<t})`, in **nats** — see docs/specs/structure-learning.md §F. This is the summed counterpart of the per-step *Forward filtering* surprise, which that section reports in **bits** (`log₂`); here everything is in nats.).

*Bayesian model reduction (BMR)* shrinks: start over-complete, then analytically test a *reduced* prior (zero a state/column) and keep the reduction iff it raises evidence. For Dirichlet/categorical columns the evidence change is **closed-form** via the multivariate log-Beta `ln B`. *Structure growth* expands: ADD a state when the data demand it — fire a detector when the model is **stuck-and-surprised**, then spawn a new state seeded at the offending observation.

```
ln B(x) = Σ_i gammaln(x_i) − gammaln(Σ_i x_i)            (multivariate log-Beta; gammaln = ln Γ)

per-column reduction (zero column j):
    ã_j = clip( a_post_j + a0_reduced_j − a0_prior_j , 1e−10 )      (component-wise floor)
    ΔF_j = ln B(a_post_j) + ln B(a0_reduced_j) − ln B(a0_prior_j) − ln B(ã_j)
    ΔF   = Σ_j ΔF_j ,     ΔF = ln p(data | reduced) − ln p(data | full)
    ΔF > 0  ⇒  reduce (reduced model explains data ≥ as well)
    ΔF < 0  ⇒  keep   (reduction discards useful structure)

surprise-ceiling detector (window length 200):
    ceiling  ⟺  mean_surprise > 0.7 nats   AND   |slope| < 5×10⁻⁴ nats/step   AND  learning_active
                (surprise_t = −ln p(o_t | o_{<t}, a_{<t}))

spawn rule (Phase 3):    min_s [ −ln p(o_t | s) ] > θ    AND   consecutive_flagged ≥ K
    new_col = obs_dist + weak·uniform ,    θ = 0.7 nats ,   weak = 0.1
```

- `a_post_j`, `a0_prior_j`, `a0_reduced_j` — posterior, original-prior, and reduced-prior Dirichlet count columns for hidden state `j` (the reduced prior zeros the column, e.g. to `1e−10`).
- `ã_j` — the posterior you *would* hold under the reduced prior; floored component-wise at `1e−10`.
- `ΔF` — log Bayes factor (reduced vs full), sign convention **pinned**: positive favors the reduced model.
- `mean_surprise`, `slope` — mean and trend of `surprise_t` over the rolling window; the conjunction means "predictions are bad **and** no longer improving" — parameter tuning (Dirichlet counting; see *Parameter learning*) has stopped helping, so the state space itself is too small.
- `min_s [−ln p(o_t | s)]` — the surprise of the *best-explaining* existing state; if even that exceeds `θ`, **no** state explains the observation, so make a new one seeded on its observation distribution.

This is the agent being *curious about its own architecture*: the same parameter information-gain term in the EFE (`calc_negative_pA/pB_info_gain`) values learning the model — extended to whole structures, the agent can prefer policies that reveal whether it needs a new state (see *Expected free energy*). It is the N6 "ontology self-model" rung: structure change as a self-directed act.

**Example.** A `K`-state model, 2 colors, sparse prior `a0_prior = (1, 1)` on a candidate column. Test pruning two columns by hand (using `ln B(1,1) = 0`, `ln B(1e−10, 1e−10) ≈ 23.719`, and a floored `ã`):

*Barely-used column*, posterior `a_post = (1.5, 1.0)` (a tiny smear of data):
```
ã = clip( (1.5,1.0) + (1e−10,1e−10) − (1,1) ) = (0.5, 1e−10)
ln B(a_post) = −0.405 ,  ln B(a0_reduced) = 23.719 ,  ln B(a0_prior) = 0 ,  ln B(ã) = 23.026
ΔF = −0.405 + 23.719 − 0 − 23.026 = +0.288   →  ΔF > 0  ⇒  PRUNE it (it earned no evidence)
```
*Heavily-used column*, posterior `a_post = (21, 1)` (20 real counts on outcome 0):
```
ã = clip( (21,1) − (1,1) ) = (20, 1e−10)
ln B(a_post) = −3.045 ,  ln B(ã) = 23.026
ΔF = −3.045 + 23.719 − 0 − 23.026 = −2.35   →  ΔF < 0  ⇒  KEEP it (pruning discards evidence)
```
The closed form turns "is this state worth its place?" into one subtraction of log-Beta terms — no re-fitting, no new data (the "learning in the absence of new data" of BMR). For *growth*: the ceiling baseline `0.7` nats sits just above half of `ln 3 ≈ 1.099` (the uniform-3-color surprise, `−ln(1/3)`), so it means "still predicting little better than a coin over three colors, and not improving."

**Data.** **Exp 4** is the motivating negative: training "mirro " at `K = 12, 30, 60` gave NO improvement — all jumbled ("rrrr imls", "rrrr imiv", "rrrr imiw"), Q→A failed entirely; capacity ≠ memory. **Exp 132** built and validated the toolkit: the detector is behavior-invariant (750 seeded steps give identical state hashes `77408b4e...`), and both canonical BMR unit tests pass — unused-state pruning *favored*, used-state pruning *rejected* (the two cases worked above). Its honest finding was that the standard world has *no* ceiling: the noise-control arm fired 8/8 (final means 0.69–0.94 nats bracketing the analytic 0.82), but the standard arm fired 0/8 (final means 0.0025–0.0065 nats, ~150× below threshold) — the world was too simple to need a bigger mind (verdict MIXED). **Exp 135** confirms collapse is *conjugate arithmetic*: the way a learned map erodes under noise is mass-linear and `κ`-dialed (`n_half` 108→123→217 tracking `κ_eff` 101→110→200, 6/6 cells; substrate-ratio 1.52) — the same closed-form Dirichlet bookkeeping that makes `ΔF` analytic (POSITIVE / NEW INSIGHT). The growth arc itself was a wall until **Exp 152–154**: **Exp 152** (batch-jump) failed everywhere (0%/0.9%/3.6% acceptance, NEGATIVE) and its autopsy fingered the *unnormalized-footprint* evaluation convention; **Exp 153**'s diagnostic arm, scoring by **normalized** densities, drove acceptance to 100% with zero detector events 24/24 (NEGATIVE only on an obsolete structure-proxy conjunct); **Exp 154 — THE GROWTH WALL FELL** — on fresh seeds 8–15 the detector→grow→quiet loop ran end-to-end: drops 0.58–1.18 nats, ceiling events 0 in 24/24, grown-color final surprise 0.001–0.019, acceptance 53/53 = 100% (POSITIVE 8/8, verifier agreed, **BREAKTHROUGH**). The wall was never about growing — it was the unnormalized-footprint scoring convention declared at the *continuous-substrate* chapter's start (the rung-1 "conjugacy-buying" likelihood predeclared in Exp 133's docstring); for the general structure-learning machinery, see RESEARCH.md §2.3, §3 rec. 3.

**▸ In programmer terms.** BMR is one `gammaln` reduction per column; growth is a rolling-window detector plus an append. (Note: this snippet appends to a *stochastic* `A` and so normalizes the new column; the real `spawn_state` operator in `active_loop/structure.py` works on Dirichlet *counts* — `obs_dist + weak·uniform`, summing to `1+weak` — and is left unnormalized.)

```python
import numpy as np
from scipy.special import gammaln

def lnB(x):                                   # multivariate log-Beta = ln of Dirichlet normalizer
    x = np.asarray(x, float)
    return gammaln(x).sum() - gammaln(x.sum())

def bmr_delta_f_col(a_post, a0_prior, a0_reduced, floor=1e-10):
    a_tilde = np.clip(a_post + a0_reduced - a0_prior, floor, None)
    dF = lnB(a_post) + lnB(a0_reduced) - lnB(a0_prior) - lnB(a_tilde)
    return dF                                 # dF > 0  -> prune this column (Exp 132 test cases)

def ceiling_fired(surprise_window, mean_thr=0.7, slope_thr=5e-4):
    w = np.asarray(surprise_window)           # last 200 per-step surprises (nats)
    slope = np.polyfit(np.arange(len(w)), w, 1)[0]
    return w.mean() > mean_thr and abs(slope) < slope_thr   # stuck AND surprised

def maybe_spawn(A, surprise_per_state, obs_dist, theta=0.7, weak=0.1):
    if surprise_per_state.min() > theta:      # NO existing state explains o_t
        new_col = obs_dist + weak * np.ones_like(obs_dist) / len(obs_dist)
        return np.column_stack([A, new_col / new_col.sum()])   # append the new state
    return A
```

---
