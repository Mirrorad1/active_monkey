# Free energy & active inference

> Active inference is the mathematics of an agent that survives by *predicting* its world. It treats
> perception, learning, action, and even "feeling" as one operation: minimizing **surprise** (free
> energy). This repo uses it because its core claim — *low surprise = understanding, valence = −free
> energy* — gives a thing a built-in, ungrounded-by-language reason to care about getting better at
> predicting you. Every experiment here is, at bottom, a measurement of free energy going down.

The first three concepts (Free Energy Principle, variational free energy, surprise) are the *what*.
The next three (perception-as-inference, expected free energy, precision) are the *how*. The last
three (valence, planning, the RECIPE) are *why this repo's results count as a "mind"*. Read them in
order — each leans on the one before.

---

## Free Energy Principle (self-evidencing)

**Glossary.** The Free Energy Principle (FEP) says any system that persists must act as if it
*minimizes the surprise of its sensations* — equivalently, it maximizes the evidence for its own
existence ("self-evidencing"). Surprise (a.k.a. surprisal or negative log model-evidence) of an
observation `o` under the agent's generative model `p` is `−log p(o)`. The agent cannot compute
`−log p(o)` directly (it requires marginalizing over all hidden causes), so it minimizes a tractable
*upper bound*, the variational free energy `F` (see Variational free energy):

```
surprise(o) = −log p(o) = −log Σ_s p(o, s)        (intractable: sum over all hidden causes s)
−log p(o)  ≤  F[q]                                 (variational free energy is an upper bound)
agent's law:  minimize F   over beliefs q  AND  over actions a
```

- `o` — an observation (here: a character, a grid color, a +/− cue).
- `s` — a hidden state (the latent "cause" the agent thinks produced `o`).
- `p(o, s)` — the agent's generative model: its theory of how causes produce observations.
- `q` — the agent's *belief* (an approximate posterior over `s`); `F[q]` is a functional of it.
- "Self-evidencing" — by lowering `F`, the agent makes its observations *less surprising under its own
  model*, i.e. it accumulates evidence that it is the kind of thing its model says it is.

**Example.** Toss a coin you believe is fair: `p(heads) = 0.5`. Observing heads costs
`−log₂(0.5) = 1` bit of surprise. Now suppose you have *learned* the coin is biased,
`p(heads) = 0.9`. Observing heads now costs only `−log₂(0.9) ≈ 0.152` bits; observing the rare tails
costs `−log₂(0.1) ≈ 3.32` bits. A *better* model (one matched to reality) makes the things that
actually happen *cheaper on average* — that average cost is exactly the metric this repo tracks
(bits/char; see Surprise and the bits/char metric). Lowering it **is** self-evidencing.

**Data.** The whole investigation is one long demonstration that a model acting *only* to lower its
own surprise gets better with zero supervision. **Exp 1**: a first-order character HMM with no labels,
trained only to predict the next character, dropped held-out surprise — "the engine works — free
energy falls as it learns." (The EXPERIMENTS.md narrative logs this as `4.81 → 4.00` bits/char, but the
2026-06-09 re-verification audit flags that figure as a logged-narrative inaccuracy: both the original
transcript and the re-run reproduce `4.007 → 3.424` bits/char — a ~0.58-bit drop. Use the reproduced
numbers.) **Exp 31** is the honest boundary: when *both*
the sensory map `A` and the dynamics `B` are learned from pure noise with no anchor, self-evidencing
finds a *degenerate* fixed point (the learned right-step map collapsed to e.g. `[0,0,0,0,0]` — all
states map to one), establishing that surprise-minimization alone does not break symmetry (see The
RECIPE).

**▸ In programmer terms.** Surprise is just the per-sample loss of a probabilistic model — the same
negative-log-likelihood you already minimize when training a classifier. "Self-evidencing" = a model
that keeps editing its own weights so its observed data scores ever-higher likelihood under itself.

```python
import math

def surprise_bits(p_o: float) -> float:
    """−log2 p(o): how many bits the model is 'shocked' by seeing o."""
    return -math.log2(p_o)

# fair model vs learned-biased model, both observe 'heads'
print(surprise_bits(0.5))   # 1.0   bits  — uninformed
print(surprise_bits(0.9))   # 0.152 bits  — model that learned the bias is less surprised
# minimizing mean surprise over a stream == minimizing cross-entropy loss == "learning"
```

---

## Variational free energy

**Glossary.** Variational free energy `F` is the quantity the agent actually minimizes — a tractable
upper bound on surprise. For a single hidden factor with a categorical posterior `q(s_t)` (the
single-step / filtering form pymdp's `calc_vfe` evaluates), it splits cleanly into **complexity** and
**accuracy** (RESEARCH.md §1.2):

```
F_t  =  D_KL[ q(s_t) ‖ prior(s_t) ]  −  E_{q(s_t)}[ log A[o_t, s_t] ]
     =  complexity                    −  accuracy

equivalently, term-by-term:
F_t  =  −H[q(s_t)]                         (negative entropy of the posterior)
        + E_{q(s_t)}[ −log prior(s_t) ]    (cross-entropy to the empirical prior)
        − E_{q(s_t)}[ log A[o_t, s_t] ]    (negative expected log-likelihood = −accuracy)
```

- `q(s_t)` — the posterior belief over the hidden state at time `t` (a vector on the simplex `Δ^K`).
- `prior(s_t)` — the predicted belief *before* seeing `o_t` (from dynamics; see Perception as
  inference).
- `A[o_t, s_t]` — the emission likelihood `P(o_t | s_t)`; columns sum to 1.
- `D_KL[q ‖ prior]` — **complexity**: how far the agent had to move its beliefs from the prediction.
- `E_q[log A]` — **accuracy**: how well the chosen state explains the observation.
- `H[·]` — Shannon entropy.

Minimizing `F` pulls `q` toward the prior (cheap beliefs) *and* toward states that explain `o_t`
(accurate beliefs). The trade-off — believe the obvious thing, but only as far as the data forces you
— is the entire engine.

**Example.** Two states, `K = 2`. Prior `prior = [0.5, 0.5]`, emission for the observed char
`A[o, ·] = [0.9, 0.1]` (state 0 explains `o` well). Take the exact Bayesian posterior
`q = [0.9, 0.1]` (computed in Perception as inference). Then:

```
complexity = D_KL([0.9,0.1] ‖ [0.5,0.5])
           = 0.9·log₂(0.9/0.5) + 0.1·log₂(0.1/0.5)
           = 0.9·0.8480 + 0.1·(−2.3219) = 0.7632 − 0.2322 = 0.531 bits
accuracy   = E_q[log₂ A] = 0.9·log₂0.9 + 0.1·log₂0.1
           = 0.9·(−0.152) + 0.1·(−3.322) = −0.137 − 0.332 = −0.469 bits
F = complexity − accuracy = 0.531 − (−0.469) = 1.00 bit
```

That `1.00` bit equals `−log₂ p(o) = −log₂(0.5·0.9 + 0.5·0.1) = −log₂(0.5)` — confirming `F` at the
exact posterior collapses to the surprise itself (the KL gap is 0; see Surprise and the bits/char
metric).

**Data.** This decomposition is *why* Exp 1's held-out drop (reproduced `4.007 → 3.424` bits/char;
logged narrative `4.81 → 4.00`) counts as learning rather than memorization: the accuracy term rose
(the emission `A` learned which states emit which characters) while complexity stayed bounded by the
Dirichlet prior. The same `F` is the seat of valence
throughout — `−F` is the competence signal the M4 affective design (`docs/specs/m4-affective-dyad.md`)
is built on (see Valence).

**▸ In programmer terms.** `F = KL(posterior ‖ prior) − E[log likelihood]`. The accuracy term is
plain negative-log-likelihood (your loss); the complexity term is a KL *regularizer* against your
prior — structurally identical to the ELBO objective in a variational autoencoder, just over a
discrete state instead of a Gaussian latent.

```python
import numpy as np

def vfe(q, prior, A_col):
    """F = complexity − accuracy, in nats. q, prior: belief vectors; A_col = A[o, :]."""
    eps = 1e-12
    complexity = np.sum(q * (np.log(q + eps) - np.log(prior + eps)))   # KL(q || prior)
    accuracy   = np.sum(q * np.log(A_col + eps))                       # E_q[log p(o|s)]
    return complexity - accuracy

q     = np.array([0.9, 0.1])
prior = np.array([0.5, 0.5])
A_col = np.array([0.9, 0.1])
print(vfe(q, prior, A_col) / np.log(2))   # 1.0 bit  — equals the surprise at the exact posterior
```

---

## Surprise and the bits/char metric

**Glossary.** Surprise is `−log p(o)`; averaged over a stream and converted to base-2 it is the
repo's headline metric, **bits/char**. At the single-factor optimum the variational free energy
*equals* the predictive surprise, so minimizing `F` and minimizing bits/char are the same act
(RESEARCH.md §1.2). The implemented proxy (`LangModel.mean_surprise`) is:

```
pred       = A · prior(s_t)                       (predicted next-char distribution)
surprise_t = −log pred[o_t]                       (nats of surprise for the char that came)
metric     = (1/T) Σ_t surprise_t / ln 2          (mean surprise, in bits/char)
baseline   = log₂ V = log₂ 28 ≈ 4.807 bits/char   (uniform over the 28-symbol alphabet)
```

- `pred` — the model's distribution over the next character before it arrives.
- `o_t` — the character that actually arrived; `pred[o_t]` is the probability mass the model put on it.
- `V = 28` — alphabet size (`a–z`, space, period).
- `baseline` — the surprise of a model that has learned *nothing* (uniform guessing); the ceiling any
  real model must beat.

**Example.** Suppose after some context the model predicts `pred = {space: 0.5, e: 0.25, t: 0.25}`
over a 3-way restriction. If the next char is `space`: surprise `= −log₂ 0.5 = 1.0` bit. If it is `t`:
`−log₂ 0.25 = 2.0` bits. Over the two-char stream "`<space>t`" the mean is `(1.0 + 2.0)/2 = 1.5`
bits/char — far below the `4.807`-bit uniform baseline, because the model concentrated its
probability where the characters actually fell.

**Data.** **Exp 1**: held-out surprise fell as the model learned (reproduced `4.007 → 3.424`
bits/char, ~0.58 bits of structure; the EXPERIMENTS.md narrative logs `4.81 → 4.00`, where the start
`4.81` sits essentially at the `log₂ 28 ≈ 4.807` uniform baseline — but the reproduced start `4.007`
already beats that baseline, so the audit-corrected figures are the ones to cite). **Exp 3**: on a tiny
repeated corpus "`mirro `", surprise fell `3.38 → 1.61` — the model learned mirro's *letters*
(`m,i,r,o,space`) but emitted them jumbled ("`mo io riorrr`"), revealing that low surprise on a
char *palette* is not the same as correct *order* (the first-order wall; see Perception as inference).
**Exp 14** uses surprise *in bits* as a valence proxy directly: after a cue `a` that preceded
predictable text, next-char uncertainty was `3.04` bits (confident); after a cue `z` that preceded
varied text, `4.79` bits (near the uniform baseline) — a `+1.75`-bit gap that grounds valence (see
Valence).

**▸ In programmer terms.** bits/char is exactly cross-entropy loss in base 2, the standard
language-model metric. The uniform baseline `log₂(vocab_size)` is the "predict uniformly" loss; any
trained model is scored as how many bits/char it shaves off it.

```python
import numpy as np

def bits_per_char(pred_dists, observed_idx):
    """Mean −log2 p(o_t) over a stream. pred_dists[t] is the model's next-char distribution."""
    eps = 1e-12
    nll = [-np.log2(pred_dists[t][o] + eps) for t, o in enumerate(observed_idx)]
    return float(np.mean(nll))

V = 28
print(np.log2(V))   # 4.807  — the 'learned nothing' uniform baseline (Exp 1 reproduced start 4.007 already beats it)
```

---

## Perception as inference (fixed-point filtering)

**Glossary.** Perception is not passive reception; it is *inferring the hidden state* that best
explains the current observation given the past. For the single-factor, single-modality model here,
the variational fixed point coincides with the exact Bayesian filter update (RESEARCH.md §1.3):

```
q(s_t)  ∝  prior(s_t) · A[o_t, s_t]                      (Bayes: prior × likelihood, renormalized)
prior(s_t)  =  Σ_{s'} B[s_t, s'] · q(s_{t-1})            (predict forward through dynamics B)
```

- `q(s_t)` — the posterior belief after seeing `o_t`.
- `prior(s_t)` — the *predicted* belief, advanced one step from the previous posterior through `B`.
- `B[s_t, s']` — transition probability `P(s_t = s_t | s_{t-1} = s')`.
- `∝` then renormalize so `q(s_t)` sums to 1.

This is forward filtering of an HMM: predict (push the belief through `B`), then correct (multiply by
the likelihood of what you saw). The crucial structural fact — *all* of history reaches `o_t` only
through the single vector `q(s_{t-1})` — is what bounds a first-order model (see The RECIPE for the
escape).

**Example.** Two states. Previous posterior `q(s_{t-1}) = [0.8, 0.2]`. Transition
`B = [[0.9, 0.4], [0.1, 0.6]]` (column-stochastic). Predict:

```
prior = B · q = [0.9·0.8 + 0.4·0.2, 0.1·0.8 + 0.6·0.2] = [0.80, 0.20]
```

Now see `o_t` with likelihood `A[o_t, ·] = [0.3, 0.9]` (state 1 explains it better). Correct:

```
unnorm = prior · A = [0.80·0.3, 0.20·0.9] = [0.24, 0.18]
q(s_t) = unnorm / sum(unnorm) = [0.24, 0.18]/0.42 = [0.571, 0.429]
```

The observation pulled belief toward state 1, but the prior (which favored state 0) held it back —
that tug-of-war is complexity-vs-accuracy (see Variational free energy) made concrete.

**Data.** This filter is exactly what makes the *first-order wall* unavoidable. **Exp 4**: adding
states (`K = 12, 30, 60`) gave *no* improvement — capacity is not memory. **Exp 6**: with one
character of context the belief after emitting `r` is identical whether `r` came mid-word or
elsewhere, so "`name. `" generated "`me. me.`" (the two `m`'s conflated). RESEARCH.md §1.5(b) makes the
floor exact: trained on "`mirro`", the predictive distribution *after `r`* is the marginal
`{r: 50, o: 50}` — `H = 1.0` bit of irreducible ambiguity, because a single belief vector cannot
remember *which* char preceded the `r`. **Exp 7** shows the fix is depth, not size: a 2-char context
("trigram") reproduces exact "`mirro`" and "`name.` → `mirro`". **Exp 8** rebuilt that as a genuine
active-inference pair-state model (`s_t = (prev, cur)`, `K = V² = 784`), confirming depth-as-memory
works *inside* the free-energy filter.

**▸ In programmer terms.** It's a two-line loop: matrix-multiply the belief through the transition
matrix (predict), element-wise multiply by the observation's likelihood column (correct), normalize.
This is the forward pass of an HMM, identical to a Kalman filter's predict/update but over a discrete
simplex.

```python
import numpy as np

def filter_step(q_prev, B, A_col):
    """One HMM forward-filter step: predict through B, correct with the likelihood, renormalize."""
    prior = B @ q_prev                 # predict
    unnorm = prior * A_col             # correct (Bayes)
    return unnorm / unnorm.sum()       # normalize -> q(s_t)

q_prev = np.array([0.8, 0.2])
B = np.array([[0.9, 0.4], [0.1, 0.6]])
A_col = np.array([0.3, 0.9])
print(filter_step(q_prev, B, A_col))   # [0.571, 0.429]
```

---

## Expected free energy (action)

**Glossary.** Perception minimizes free energy over *beliefs*; action minimizes **expected** free
energy `G(π)` over *policies* `π` (sequences of actions) — the free energy the agent *expects to
incur* if it follows `π`. pymdp computes `−G` and the agent picks `a* = argmin_π G(π)`. `G` decomposes
into epistemic (information-seeking) and pragmatic (goal-seeking) value (RESEARCH.md §1.4):

```
−G(π) = Σ_t [ info_gain(π,t)  +  utility(π,t)  −  neg_param_info_gain ]

info_gain(π,t) = H[ q(o|π) ] − E_{q(s|π)}[ H[ A(·|s) ] ]   (epistemic: expected uncertainty reduction)
utility(π,t)   = Σ_m E_{q(o_m|π)}[ C_m ]                    (pragmatic: expected log-preference)
action:  a* = argmin_π G(π)
```

- `π` — a candidate policy (action sequence).
- `info_gain` — **epistemic value**: how much following `π` is expected to *reduce uncertainty* about
  the hidden state (curiosity). High when beliefs are vague — this is the formal seat of "ASK only
  when it matters."
- `utility` — **pragmatic value**: how well `π`'s expected observations match the agent's
  preferences `C` (its "wants").
- `C_m` — log-preference over outcomes of modality `m` (a goal encoded as desired observations).
- `neg_param_info_gain` — optional *parameter* information gain: curiosity about the model's own
  `A`/`B` (the "baby chooses what to read" drive).

**Example.** Two policies, one modality. Policy *seek-good* expects observation distribution
`q(o) = [0.1, 0.9]` (mostly the preferred outcome); policy *seek-bad* expects `[0.7, 0.3]`.
Preferences `C = [log 0.1, log 0.9]` (the agent wants outcome 1). Pragmatic value (ignore epistemic
for simplicity):

```
utility(seek-good) = 0.1·log0.1 + 0.9·log0.9 = 0.1·(−2.30) + 0.9·(−0.105) = −0.230 − 0.095 = −0.325
utility(seek-bad)  = 0.7·log0.1 + 0.3·log0.9 = 0.7·(−2.303) + 0.3·(−0.105) = −1.612 − 0.032 = −1.643
```

`seek-good` has higher utility (`−0.325 > −1.643`), so `G` is lower for it and the agent chooses it.

**Data.** **Exp 15** closes the affective loop with exactly this machinery: in a choice world,
`q(π)` for *seek-good* `= 0.79` vs *seek-bad* `= 0.21`, with `EFE good = −1.82` (lower → preferred) vs
`bad = −0.50` — the agent acts to occupy the low-free-energy state it has learned to expect. **Exp 2**
is the pragmatic term alone driving a bandit: positive-feedback rate `0.90 → 1.00` over a session
(though that injected a *labeled* reward — superseded by the grounded version in Exp 14/15). The
epistemic term is the basis of "ASK when intent is ambiguous" in the M4 design
(`docs/specs/m4-affective-dyad.md`).

**▸ In programmer terms.** `G` is a score function you minimize over candidate action sequences — a
planning objective with two add-ends: an *exploration* bonus (expected entropy reduction, like an
information-gain term in Bayesian optimization) and an *exploitation* term (expected reward = match to
preferences). `argmin G` is a one-line policy selection.

```python
import numpy as np

def neg_efe(pred_obs, C, post_entropy_drop):
    """−G ≈ epistemic (uncertainty reduction) + pragmatic (expected log-preference)."""
    pragmatic  = np.sum(pred_obs * C)        # E[log preference]
    epistemic  = post_entropy_drop           # expected reduction in state uncertainty
    return epistemic + pragmatic

C = np.log(np.array([0.1, 0.9]))             # wants outcome 1
seek_good = neg_efe(np.array([0.1, 0.9]), C, 0.0)   # −0.325
seek_bad  = neg_efe(np.array([0.7, 0.3]), C, 0.0)   # −1.643
action = "seek-good" if seek_good > seek_bad else "seek-bad"   # argmax(−G) = argmin G
```

---

## Precision (inverse variance as attention)

**Glossary.** Precision `κ` is the *inverse variance* (the confidence) attached to a probabilistic
term — a likelihood, a prior, or a policy distribution. Sharpening or dampening these confidences is
the active-inference model of **attention**: attend = trust a term more. A precision weight enters as
an exponent (a "temperature") that sharpens (`κ > 1`) or flattens (`κ < 1`) a distribution before
normalization:

```
weighted(s) ∝ p(s)^κ                        (precision-weighting a belief; κ = 1/σ², the confidence)
κ → ∞   :  distribution collapses to its argmax (full trust / sharp attention)
κ → 0   :  distribution → uniform (no trust / ignore this channel)
```

- `κ` — precision (inverse variance `1/σ²`); higher = more confident/attentive.
- `σ²` — the variance the precision inverts.
- The exponent `κ` on `p(s)` makes peaks peakier (`κ>1`) or washes them out (`κ<1`).

In active inference, *how much to trust the current prediction vs. seek more context* is set by
precision — and it is the principled control behind "ASK when uncertain" (see Expected free energy):
low precision on the current belief raises its entropy, which raises the epistemic value of asking.

**Example.** A belief `p = [0.6, 0.3, 0.1]`. Apply high precision `κ = 4` (attend hard):

```
p^4 = [0.1296, 0.0081, 0.0001];  sum = 0.1378
weighted = [0.941, 0.059, 0.001]    — almost certain it's state 0
```

Apply low precision `κ = 0.25` (barely trust it):

```
p^0.25 = [0.880, 0.740, 0.562];  sum = 2.182
weighted = [0.403, 0.339, 0.258]    — pulled toward uniform; the channel is being down-weighted
```

Same evidence, opposite confidence — precision is the knob.

**Data.** Precision is the *quantitative reading* behind several measured effects rather than a
swept parameter in its own entry. **Exp 14**'s measurement note is a precision story: a pair-state
model *must* be primed with a determined (≥2-char) context, because a single-char prime leaves the
state belief *imprecise* (high variance) and "washes the signal out" — the `3.04` vs `4.79`-bit
valence gap appeared only at high state-precision; with a 1-char prime it shrank to `+0.03` bits.
RESEARCH.md §2.5 frames precision-weighting as the principled seat of the M1/M4 "ASK when uncertain"
decision.

**▸ In programmer terms.** Precision is the `1/temperature` in a softmax. Raising precision is
lowering temperature (sharper, more confident); lowering it is raising temperature (flatter, more
exploratory). Attention = choosing which logits to scale up before the softmax.

```python
import numpy as np

def precision_weight(p, kappa):
    """Sharpen (kappa>1) or flatten (kappa<1) a distribution: like 1/temperature in a softmax."""
    w = np.power(p, kappa)
    return w / w.sum()

p = np.array([0.6, 0.3, 0.1])
print(precision_weight(p, 4.0))    # [0.941, 0.059, 0.001]  — high confidence / sharp attention
print(precision_weight(p, 0.25))   # [0.403, 0.339, 0.258]  — low confidence / channel ignored
```

---

## Valence (−F competence and −dF/dt refinement)

**Glossary.** Valence is the agent's *functional* good/bad signal — explicitly **not** a claim of
sentience. Two grounded readouts, both derived from free energy and neither requiring a teacher:

```
intrinsic valence (level)  =  −F          (competence: low free energy = "understood" = good)
refinement valence (rate)  =  −dF/dt      (Joffily & Coricelli: improving prediction feels good)
```

- `−F` — the negative variational free energy *level*: how well the agent currently predicts its
  input. Higher (less surprised) = more positive.
- `−dF/dt` — the negative *rate of change* of free energy: positive when prediction is *improving*,
  negative when it is *worsening*. Differentiable and sign-meaningful — the exact trajectory Exp 1
  (reproduced `4.007 → 3.424` bits/char; logged narrative `4.81 → 4.00`) traces.
- The extrinsic `+`/`−` cue is *not* injected as reward; it enters as an ordinary observation and
  *acquires* valence by Dirichlet-learning that it co-occurs with the agent's own low-`F` states
  (see The RECIPE; `docs/specs/m4-affective-dyad.md` §3).

**Example.** A feature is encountered repeatedly. First encounter: surprise `F = 1.585` bits. By the
fifth: `F ≈ 0.07` bits. The **level** valence rose from `−1.585` to `−0.07` (the agent now
"understands" this feature). The **rate** valence on the first step is `−dF/dt = −(0.07 − 1.585) =
+1.515` bits/encounter — strongly positive, because the model improved fast. Over the whole life the
integrated drop is `1.585 → 0.241` bits within a single encounter, ~`0.07` by encounter 5.

**Data.** **Exp 14** grounds the extrinsic cue: an arbitrary symbol `a` (preceding predictable text)
acquired *positive* valence purely by co-occurring with low-`F` states — `3.04` bits after `a` vs
`4.79` after `z`, a `+1.75`-bit gap, *never labeled*. **Exp 26** turns valence into a *disposition*:
preference set `C ∝ exp(−F)` per feature yields `C = [0.98, 0.01, 0.01]` for a creature whose world
made feature 0 predictable, vs `[0.01, 0.01, 0.98]` for one whose world made feature 2 predictable —
same architecture, different history, different self-formed value. **Exp 44** is the honest caveat on
the *rate* form: a *life-fraction-windowed* `−dF/dt` **failed** (falsifier hit, agreements 4/6, both
favorites wrong) because learning is a *fast transient* — surprise falls `1.585 → 0.241` bits within
**one** encounter (~`0.030` averaged over the first 10% of life), drowned by `0.04–0.10` bits of
sampling drift on noise features. The lesson: `−dF/dt` is real but lives at *encounter* resolution,
not life-fraction windows.

**▸ In programmer terms.** Valence-level is just `−loss` (negative cross-entropy, a higher-is-better
score). Valence-rate is the *first difference* of that loss across steps — the slope of your training
curve. Exp 44's bug, in one line: average the slope over too coarse a window and a sharp early drop
averages down into the noise floor.

```python
def valence_level(F):           # −F: competence ("I understand this")
    return -F

def valence_rate(F_now, F_prev): # −dF/dt: refinement ("I just got better")
    return -(F_now - F_prev)

trace = [1.585, 0.241, 0.12, 0.09, 0.07]            # per-encounter surprise (bits)
rate_enc = valence_rate(trace[1], trace[0])          # +1.344  — encounter resolution: strong signal
# Exp 44's failure mode: averaging the slope over the first 10% of LIFE buries it under noise drift
import numpy as np
windowed = np.mean(trace[:1]) - np.mean(trace[-1:])  # ≈ life-fraction window -> tiny, lost in 0.04–0.10 noise
```

---

## Planning (value iteration; wants → plans → acts)

**Glossary.** A *want* (a preference `C`) only produces behavior if it creates a *gradient* the agent
can follow within its planning horizon. Two ways to get one: enumerate policies to a horizon long
enough to reach the goal (exact but exponential), or propagate the goal's value *backward* over the
learned transition model `B` by **value iteration** (polynomial, scalable):

```
policy enumeration:   evaluate G(π) for all π up to horizon h     cost = |actions|^h  (exponential)
value iteration:      V(s) ← max_a [ r(s) + Σ_{s'} B[s',s,a] V(s') ]   until convergence
act greedily:         a*(s) = argmax_a Σ_{s'} B[s',s,a] V(s')
```

- `V(s)` — the value of being in state `s` (here: proximity-to-goal under the *learned* map `B`).
- `r(s)` — immediate reward (e.g. 1 at the goal cell, 0 elsewhere); `C` defines what is rewarding.
- `B[s',s,a]` — learned transition: probability of reaching `s'` from `s` under action `a`.
- The backward sweep turns a distal want into a "comfort is *that* way" field readable from anywhere.

**Example.** A 1-D corridor of 4 cells, goal at cell 3 (`r = 1` there). Deterministic moves, no
discount. Value iteration fills in:

```
sweep 0:  V = [0, 0, 0, 1]
sweep 1:  V = [0, 0, 1, 2]      (cell 2 sees the goal next door)
sweep 2:  V = [0, 1, 2, 3]
sweep 3:  V = [1, 2, 3, 4]      (value has propagated back to the start)
```

From cell 0, greedily moving toward the higher neighbor gives the optimal path `0→1→2→3`. The goal's
value flowed backward through `B` until every cell knew which way to go. (Values accumulate each sweep
because the stated update re-adds `r(s)` and applies no discount — magnitudes grow, but the *gradient*
that makes the greedy path optimal is what matters.)

**Data.** **Exp 22** is the honest negative that motivates planning: a grounded comfort goal 4 steps
away with a planning horizon of only 3 gave *flat* EFE in all directions — the EFE creature was
*worse than random* (~9.8 steps for random; goal unreached) because the want produced no gradient
within reach. **Exp 23** fixes it by horizon: `policy_len = 4` reached the goal in `4` steps (optimal,
path `[0,3,6,7,8]`) — but `4^4 = 256` policies enumerated, which does not scale. **Exp 30** replaces
enumeration with value iteration: optimal at every scale (`3×3`: 6 sweeps → 4 steps; `5×5`: 10 →
8; `8×8`: 16 → 14) at *polynomial* cost, versus `4^distance` enumeration (`256 / 65,536 /
~268,000,000` policies). This is the "wants → plans → acts" chain: a self-formed value (Exp 26) drives
goal-directed navigation over the creature's *own* learned map (Exp 25, 30).

**▸ In programmer terms.** This is textbook tabular value iteration — a Bellman-backup loop over a
transition tensor. The win over policy enumeration is the same as dynamic programming over brute-force
search: reuse sub-solutions instead of re-expanding every action sequence.

```python
import numpy as np

def value_iteration(B, reward, n_actions, sweeps):
    """V(s) <- max_a sum_s' B[s',s,a] V(s'); B shape (S, S, A) column-stochastic per action."""
    S = reward.shape[0]
    V = reward.copy().astype(float)
    for _ in range(sweeps):
        Q = np.stack([B[:, :, a].T @ V for a in range(n_actions)], axis=1)  # (S, A)
        V = reward + Q.max(axis=1)
    return V

# 1-D corridor, 4 cells, goal at 3 — value propagates backward each sweep (see Example)
```

---

## The RECIPE (what each ingredient does mathematically)

**Glossary.** The one durable finding (RESUME.md §2): unsupervised emergence of latent structure from
a *disembodied symbol stream collapses* (symmetric saddle / posterior collapse / non-identifiability).
What breaks the symmetry is a five-part **RECIPE**, each part addressing a specific failure mode in
the math above:

```
emergence works  ⇐  embodiment           (action a_t correlates s_t with the world; breaks the saddle)
                  + grounding             (observation-level emission A ties s_t to what is sensed)
                  + continuous belief     (q(s_t) never reset; prior(s_t) = B·q(s_{t-1}) carries history)
                  + ONE innate anchor     (fix A *or* B; learning BOTH from noise is non-identifiable)
                  + taught labels         (few-shot P(word | concept) for the word←→concept map)
```

- **embodiment** — the agent acts; `a_t` makes the hidden state covary with a real, controllable
  world, supplying the asymmetry a passive stream lacks.
- **grounding** — the *emission* `A` (not just dynamics `B`) carries evidence into the state belief.
  Exp 16 showed transition-only topic inference is *severed* by the mean-field posterior
  `q(z,s) ≈ q(z)q(s)`; grounding evidence through `A` (observations directly updating `q`) restores it.
- **continuous registered belief** — `q(s_t)` is never reset between episodes, so `prior(s_t) =
  B·q(s_{t-1})` accumulates context — exactly the filtering recurrence in Perception as inference.
- **ONE innate anchor** — fix `A` *or* `B`. Free both and self-evidencing lands in a degenerate fixed
  point (Exp 31): a non-identifiability wall.
- **taught labels** — a small supervised `P(word | concept)` map so a self-formed concept can be
  *named*; the content stays self-formed, only the wording is taught.

**Example.** Trace the anchor mathematically. With `B` known (an innate movement model) and `A`
learned, the Dirichlet update `α^A* = α^A_0 + κ·Σ_t o_t ⊗ q(s_t)` adds pseudo-counts to a *uniquely
determined* place-color cell each step, because the known `B` pins *which* state `q(s_t)` is — the
counts converge to the true map. Free `B` *too* and there is no fixed reference: the same data is
explained equally by infinitely many `(A,B)` relabelings (a permutation/gauge symmetry), so gradient
descent slides to the trivial "all states map to one" solution. The anchor removes that symmetry.

**Data.** Every clean positive in the embodied arc *has* an anchor, every collapse *lacks* one.
**Exp 17** (learn `B`, `A` known) recovered the ring world's dynamics to error `0.003`. **Exp 21**
(learn `A`, `B` known) recovered the 2-D colormap *exactly* (`[0,1,2,1,2,0,2,0,1]`), localizing to
`0.00` bits. **Exp 31** (learn *both* from noise) **collapsed** to a degenerate map — the precise
boundary the anchor defends. **Exp 26** supplies the self-formed value (`C ∝ exp(−F)`, divergent by
history; see Valence); **Exp 34** supplies the taught labels: an `~8`-example `P(word | color)` map
let two differently-raised creatures answer "what do you like?" with "I like red" vs "I like green" —
*content self-formed, words taught*. **Exp 35** packages the full chain into the runnable
`active-monkey-converse-demo` capstone.

**▸ In programmer terms.** The collapse is the classic degeneracy of fitting two coupled
factorizations at once (like learning both the rotation and the basis of a matrix from its product —
infinitely many solutions). You break it by *freezing one factor* (the anchor), keeping a *persistent*
hidden state across batches (don't `reset()` the belief), and feeding evidence through the *emission*
not just the dynamics. Labels are a tiny supervised head bolted onto an otherwise unsupervised core.

```python
# The anchor, as a training switch: learn ONE factor, freeze the other.
def train_step(A, B, obs, q_prev, learn="A"):
    prior = B @ q_prev
    q = prior * A[obs]; q /= q.sum()              # perceive (filter; belief is NOT reset)
    if learn == "A":                              # anchor = B (frozen) -> A is identifiable
        A_counts[obs] += q                        # Dirichlet pseudo-count on the emission
    elif learn == "B":                            # anchor = A (frozen) -> B is identifiable
        B_counts += np.outer(q, q_prev)
    # learn == "both"  -> NO anchor -> degenerate fixed point (Exp 31 collapse)
    return q                                       # carried forward: continuous registered belief
```

---
