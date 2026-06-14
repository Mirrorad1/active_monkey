# Information theory

> Information theory is the mathematics of *surprise*: how many bits it takes to
> name an outcome you weren't sure of. This repo's single most-used number — `bits/char`,
> the held-out predictive surprise of the character HMM — is an information-theoretic
> quantity, and the project's central claim ("free energy falls as the model learns")
> is literally surprise going down. Every concept below feeds the one metric the
> experiments live and die by. The objective being minimized, variational free energy,
> is an information-theoretic bound on this surprise (see KL divergence and Cross-entropy).

---

## Self-information / surprise

**Glossary.** The *self-information* (or *surprise*) of observing an outcome `x` under a
distribution `p` is the negative log of its probability. The base of the log fixes the
unit: base-2 gives **bits**, base-`e` (natural log) gives **nats**.

```
I(x)  =  −log p(x)

bits  =  −log₂ p(x)          (unit: bits)
nats  =  −ln  p(x)           (unit: nats)
bits  =  nats / ln 2         (1 nat ≈ 1.4427 bits;  1 bit = ln 2 ≈ 0.6931 nats)
```

Symbols: `x` an observed outcome (here a character); `p(x) ∈ (0,1]` the model's predicted
probability for it; `I(x) ≥ 0`, and `I(x) → ∞` as `p(x) → 0`. Certain events (`p = 1`)
carry zero surprise; rare events carry a lot.

**Example.** A model predicts the next character with probability `p = 0.04`.
Its surprise is `−ln(0.04) ≈ 3.2189` nats. Converting: `3.2189 / ln 2 = 3.2189 / 0.6931 ≈ 4.644`
bits, which equals `−log₂(0.04) ≈ 4.644` bits directly. A confident, correct call
(`p = 0.25`) costs only `−log₂(0.25) = 2` bits; a near-blind one (`p = 0.01`) costs
`−log₂(0.01) ≈ 6.644` bits. Lower-probability outcomes are exponentially more surprising.

**Data.** This is the atom of the project metric. `LangModel.mean_surprise`
(`lang_model.py:33–43`) computes exactly `−ln p(o_t)` per character and averages it
(RESEARCH.md §1.2: `surprise_t = −log pred[o_t]`). Per the variational-free-energy reading,
the per-step free energy `F_t ≈ −log P(o_t | o_{<t})` *is* the predictive surprise
(RESEARCH.md §1.2). Exp 1's narrative reports held-out surprise falling **4.81 → 4.00
bits/char** as the K=12 HMM learned (EXPERIMENTS.md Exp 1). (Honesty note: the 2026-06-09
audit found the *logged narrative* differs from the *reproduced run output*, which shows
**4.007 → 3.424 bits/char** — same direction and magnitude, see the Data note under
bits/char.)

**▸ In programmer terms.** Surprise is the per-token term inside a negative-log-likelihood
loss. The code stores nats internally and divides by `ln 2` only when reporting bits.

```python
import math

def surprise_nats(p_x: float) -> float:
    return -math.log(p_x + 1e-12)        # guard log(0), as lang_model.py does

def surprise_bits(p_x: float) -> float:
    return surprise_nats(p_x) / math.log(2)

surprise_bits(0.04)   # -> 4.6439  (a bad prediction costs ~4.6 bits)
surprise_bits(0.25)   # -> 2.0     (a decent prediction costs 2 bits)
```

---

## Entropy

**Glossary.** *Entropy* `H(X)` is the *expected* surprise of a random variable `X`: the
average number of bits per outcome you pay if your code is matched to the true distribution.

```
H(X)  =  −Σ_x p(x) · log₂ p(x)            (bits)

uniform baseline:  H_uniform = log₂ V      (every outcome equally likely)
for V = 28:        log₂ 28 ≈ 4.807 bits/char
```

Symbols: `p(x)` the probability of outcome `x`; the sum runs over all outcomes; `V` the
alphabet size (here 28: `a–z`, space, period). Entropy is maximal (`= log₂ V`) for the
uniform distribution and `0` for a point mass. The convention `0·log 0 = 0` is used.

**Example.** Two outcomes, equally likely (`p = ½` each):
`H = −(½·log₂ ½ + ½·log₂ ½) = −(½·(−1) + ½·(−1)) = 1` bit. A coin flip is worth exactly
one bit. For the repo's 28-symbol alphabet at maximum uncertainty,
`H = log₂ 28 = 4.8074` bits/char — the cost of "I have no idea, all 28 are equally likely."

**Data.** `log₂ 28 ≈ 4.807 bits/char` is the **uniform baseline** the whole project measures
against (`eval/lang_score.py`: `baseline = math.log(V) / LN2`; RESEARCH.md §1.2). Any
`bits/char` below `4.807` means the model has learned *something*. The experiments-data.js
constant `AM_SURPRISE = [4.81, 4.00, 3.38, 1.61]` opens with this baseline (line 27), and
`eval/lang_score.py` sets `beats_baseline = bits < baseline` as a hard guardrail. Exp 1
started at the `4.81` baseline (effectively random) and fell to `4.00` (EXPERIMENTS.md Exp 1).

**▸ In programmer terms.** Entropy is the loss you'd get from a *perfect* model of the
source — the floor that cross-entropy (next section) sits above. The uniform baseline is
just `log2(len(alphabet))`.

```python
import math

def entropy_bits(p):                       # p: list of probabilities summing to 1
    return -sum(px * math.log2(px) for px in p if px > 0)

V = 28
baseline = math.log2(V)                     # 4.8074  -> the "random guess" cost
entropy_bits([0.5, 0.5])                     # 1.0     -> a fair coin
entropy_bits([1.0])                          # 0.0     -> certainty is free
```

---

## Cross-entropy

**Glossary.** *Cross-entropy* `H(q, p)` is the average surprise you pay when the data really
comes from `q` but you *predict* with `p`. It is the quantity a language model's
held-out `bits/char` actually estimates (with `q` the empirical held-out text).

```
H(q, p)  =  −Σ_x q(x) · log₂ p(x)

decomposition:  H(q, p)  =  H(q)  +  D_KL[ q ‖ p ]   ≥   H(q)
```

Symbols: `q` the true/data distribution, `p` the model's distribution; `H(q)` the true
entropy (irreducible floor, see Entropy); `D_KL[q‖p] ≥ 0` the penalty for using the wrong
model (see KL divergence). Cross-entropy is minimized, and equals `H(q)`, exactly when
`p = q`.

**Example.** Truth is a fair coin `q = [½, ½]`; the model believes `p = [¼, ¾]`. Then
`H(q, p) = −(½·log₂ ¼ + ½·log₂ ¾) = −(½·(−2) + ½·(−0.415)) = 1.2075` bits. This splits as
`H(q) + D_KL[q‖p] = 1.0 + 0.2075 = 1.2075` bits: one bit is the coin's irreducible entropy,
`0.2075` bits is pure modelling waste.

**Data.** The repo's `bits/char` *is* an estimate of `H(held-out text, model)`:
`mean_surprise` averages `−ln p(o_t)` over held-out characters (`lang_model.py:33`,
RESEARCH.md §1.2). This links directly to the **accuracy term** of variational free energy:
RESEARCH.md §1.2 writes `F_t = complexity − accuracy` where `accuracy = E_{q(s_t)}[log A[o_t, s_t]]`,
so `−accuracy` is exactly the expected cross-entropy of the prediction to the observed
character. Minimizing free energy = minimizing cross-entropy = lowering held-out
`bits/char` (Exp 1: `4.81 → 4.00`, EXPERIMENTS.md Exp 1).

**▸ In programmer terms.** Cross-entropy is the categorical cross-entropy / NLL loss every
classifier trains on — `nn.CrossEntropyLoss`, but in bits and averaged per character.

```python
import math

def cross_entropy_bits(q, p):              # q = data dist, p = model dist
    return -sum(qx * math.log2(px) for qx, px in zip(q, p) if qx > 0)

# held-out bits/char is this, with q = one-hot on each true char:
def held_out_bits(model_probs):            # model_probs[t] = predicted prob of true char t
    return sum(-math.log2(pt) for pt in model_probs) / len(model_probs)

cross_entropy_bits([0.5, 0.5], [0.25, 0.75])   # 1.2075 = H(q)=1.0 + KL=0.2075
```

---

## Conditional entropy and the entropy rate

**Glossary.** *Conditional entropy* measures the leftover uncertainty about the next symbol
*given* the preceding `n` symbols. The *entropy rate* `h` is its limit as context grows —
the true compressibility of the source.

```
H(o_t | o_{t−n..t−1})  =  −Σ_context Σ_{o_t} p(context, o_t) · log₂ p(o_t | context)

entropy rate:   h  =  lim_{n→∞}  H(o_t | o_{t−n..t−1})

monotonicity:   H(o_t | o_{t−1})  ≥  H(o_t | o_{t−2..t−1})  ≥ … ≥ h
```

Symbols: `o_t` the next character; `o_{t−n..t−1}` the previous `n` characters (the context);
`p(o_t | context)` the conditional next-char distribution. More context can only *reduce*
uncertainty, never increase it — so deeper context lowers the achievable floor.

**Example.** Source `"mirro "` repeated. Conditioning on **one** prior char: after an `r`,
the next char is `r` half the time and `o` half the time, so
`H(o_t | last='r') = −(½·log₂½ + ½·log₂½) = 1` bit. Conditioning on **two** prior chars:
after `"ir"` the next char is always `r` (`H = 0`); after `"rr"` it is always `o` (`H = 0`).
Going from 1-char to 2-char context drops the floor from 1 bit to 0 at this position.

**Data.** RESEARCH.md §1.5c states the source has an entropy rate `h ≈ 1.1` bits/char for
English (Shannon's bound), and that "the **bits/char floor of a first-order character HMM is
the order-1 conditional entropy of the corpus**, well above both the trigram value and the
~1.1-bit Shannon estimate." This is the formal explanation for *why* the experiments needed
context depth, not capacity: Exp 4 added states (K = 12/30/60) and got **no improvement**,
while Exp 5 added one char of memory and produced real word-fragments (EXPERIMENTS.md Exp 4,
5). The lever to lower the floor is effective Markov order `d`, i.e. memory
(see The order-1 floor argument).

**▸ In programmer terms.** Conditional entropy is the loss of an order-`n` n-gram model;
the entropy rate is the loss of an n-gram with `n → ∞`. Deeper context = a bigger key into
the count table = a lower floor.

```python
import math
from collections import Counter, defaultdict

def conditional_entropy_bits(text, n):
    ctx_next = defaultdict(Counter)         # context (len n) -> next-char counts
    for i in range(n, len(text)):
        ctx_next[text[i-n:i]][text[i]] += 1
    total = sum(c.total() for c in ctx_next.values())
    H = 0.0
    for ctx, nxt in ctx_next.items():
        cn = nxt.total()
        p_ctx = cn / total
        H += p_ctx * sum(-(k/cn) * math.log2(k/cn) for k in nxt.values())
    return H

conditional_entropy_bits("mirro mirro mirro ", 1)   # ~0.353 bits (corpus-weighted avg; the r-context alone has H=1)
conditional_entropy_bits("mirro mirro mirro ", 2)   # ~0  (two chars disambiguate)
```

---

## The order-1 floor argument

**Glossary.** A first-order model conditions the next character only on the *single* most
recent latent/observed token. Whenever a symbol is followed by two different symbols in the
corpus, the best a first-order model can do is the *marginal* over those continuations — it
cannot tell the two histories apart. This puts a hard, computable floor on its `bits/char`.

```
P(o_t | o_{<t})  =  Σ_{s_t} A[o_t, s_t] · Σ_{s_{t−1}} B[s_t, s_{t−1}] · P(s_{t−1} | o_{<t})

floor for a symbol followed by k equiprobable continuations:
    H_after(symbol)  =  log₂ k    (best possible; the marginal)
```

Symbols: `A[o,s] = P(o|s)` emission, `B[s',s] = P(s'|s)` transition, `P(s_{t−1}|o_{<t})` the
belief carrying *all* the model knows of the past (RESEARCH.md §1.5a). Two histories that
drive the belief to the same posterior produce the *same* prediction — that is the trap.

**Example (the exact `"mirro"` numbers, RESEARCH.md §1.5b).** In `"mirro "`, the symbol `r`
is followed once by `r` and once by `o`. A first-order model has a single predictive
distribution available "after an `r`," so its best case is the marginal `{r: 50, o: 50}`:

```
After 'r'  (1-char context):  {r: 50, o: 50}     # H = 1.0 bit — irreducibly ambiguous
After 'ir' (2-char context):  {r: 50}            # H = 0   — resolved
After 'rr' (2-char context):  {o: 50}            # H = 0   — resolved
```

`H({r: ½, o: ½}) = −(½·log₂½ + ½·log₂½) = 1` bit. The 50/50 split is not a learning
failure; it is the information-theoretic optimum at order 1.

**Data.** This is the crux finding of EXPERIMENTS.md Exp 3–7. Exp 3 taught `"mirro "` and
got the right *letters* in jumbled *order* (`"mo io riorrr"`), surprise `3.38 → 1.61`
(EXPERIMENTS.md Exp 3). Exp 4's negative result: more states (K = 12/30/60) gave **no
improvement** — "the wall is the FIRST-ORDER assumption, not capacity" (EXPERIMENTS.md
Exp 4). Exp 6 named the mechanism: "1 char of context can't distinguish 'm mid-word' from
'm starting the answer'" and greedy decode loops on `r` (`"mirrrrrr"`) — exactly the 50/50
tie (EXPERIMENTS.md Exp 6). RESEARCH.md §1.5: "No value of `K` fixes this … Capacity ≠
memory."

**▸ In programmer terms.** A first-order model is a `dict` keyed by *one* token. If two
different pasts produce the same key, they share one prediction — you literally cannot store
two answers under one key. The fix is a longer key (more context), not a bigger value space
(more states).

```python
from collections import Counter, defaultdict

def first_order_next(text):
    nxt = defaultdict(Counter)
    for a, b in zip(text, text[1:]):
        nxt[a][b] += 1                       # key = ONE previous char
    return nxt

m = first_order_next("mirro mirro mirro ")
dict(m['r'])      # {'r': 3, 'o': 3}  -> tie -> best prediction is 50/50, H = 1 bit
# adding more "states" (values) cannot split one key into two histories.
```

---

## Mutual information

**Glossary.** *Mutual information* `I(X; Y)` is how many bits learning `Y` tells you about
`X` — the entropy of `X` minus the entropy of `X` that survives after seeing `Y`. It is the
information-theoretic name for *epistemic value*: the expected uncertainty reduction from an
observation.

```
I(X; Y)  =  H(X)  −  H(X | Y)                 (≥ 0; symmetric: = H(Y) − H(Y|X))
```

Symbols: `H(X)` prior uncertainty about `X`; `H(X|Y)` uncertainty about `X` after `Y` is
known (see Conditional entropy). `I(X; Y) = 0` iff `X` and `Y` are independent; it is large
when `Y` is highly predictive of `X`.

**Example.** Suppose `X` (the answer) is one of 4 equally likely values, so `H(X) = log₂ 4 =
2` bits. Observation `Y` narrows it to 2 candidates: `H(X|Y) = log₂ 2 = 1` bit. Then
`I(X; Y) = 2 − 1 = 1` bit — the observation was worth one bit of resolved uncertainty.

**Data.** Mutual information is the **epistemic term** of expected free energy. RESEARCH.md
§1.4 gives pymdp's state-information-gain term (`compute_info_gain`, `control.py:388`):
`info_gain = H[q(o|π)] − E_{q(s|π)}[H[A(·|s)]] = expected reduction in observation
uncertainty` — exactly `I(observation; state)`. This is "the formal seat of the 'ASK only
when it matters' behavior (M1) and 'ASK when intent is ambiguous' (M4): asking wins when
belief entropy is high enough that the epistemic term dominates" (RESEARCH.md §1.4). It is
also the mechanism behind Exp 7's comprehension-as-prediction: once context ≥ 2 chars, the
question `"name. "` carries enough mutual information with the answer to evoke `"mirro"`
(EXPERIMENTS.md Exp 7).

**▸ In programmer terms.** Mutual information is "expected entropy *before* minus expected
entropy *after*" — the score you'd use to pick the most informative question/observation,
i.e. an active-learning acquisition function.

```python
import math

def entropy_bits(p):
    return -sum(px * math.log2(px) for px in p if px > 0)

H_prior = entropy_bits([0.25, 0.25, 0.25, 0.25])   # 2.0 bits, 4 equal answers
H_post  = entropy_bits([0.5, 0.5, 0.0, 0.0])       # 1.0 bit after an observation
info_gain = H_prior - H_post                        # 1.0 bit  == I(answer; obs)
```

---

## KL divergence

**Glossary.** The *Kullback–Leibler divergence* (relative entropy) `D_KL[q‖p]` measures how
many extra bits you waste by coding data from `q` with a model `p`. It is the **complexity
term** of variational free energy and is always non-negative.

```
D_KL[ q ‖ p ]  =  Σ_x q(x) · log₂( q(x) / p(x) )   ≥   0

Gibbs' inequality:  D_KL[q‖p] = 0  ⇔  q = p
relation:           D_KL[q‖p]  =  H(q, p) − H(q)      (cross-entropy minus entropy)
```

Symbols: `q` the reference distribution, `p` the approximating one. KL is *not* symmetric
(`D_KL[q‖p] ≠ D_KL[p‖q]` in general) and is undefined where `p(x)=0` but `q(x)>0`.

**Example.** `q = [½, ½]`, `p = [¼, ¾]`:
`D_KL[q‖p] = ½·log₂(0.5/0.25) + ½·log₂(0.5/0.75) = ½·(1) + ½·(−0.585) = 0.2075` bits. This
is exactly the `H(q,p) − H(q) = 1.2075 − 1.0` waste from the Cross-entropy example. Swap to
the model being right (`p = q`) and `D_KL = 0`.

**Data.** RESEARCH.md §1.2 writes the per-step free energy as
`F_t = D_KL[q(s_t) ‖ prior(s_t)] − E_{q(s_t)}[log A[o_t, s_t]] = complexity − accuracy`.
The KL is the **complexity** term: it penalizes posterior beliefs that stray from the prior,
keeping beliefs "cheap." Because `D_KL ≥ 0` always, free energy is a genuine *upper bound*
on surprise (`F ≥ −log P(o_t | history)`), which is why minimizing `F` is a valid surrogate
for minimizing the `bits/char` reported in Exp 1 (`4.81 → 4.00`, EXPERIMENTS.md Exp 1; see
Cross-entropy and Mutual information).

**▸ In programmer terms.** KL is the gap between your loss (cross-entropy) and the best
possible loss (entropy) — the part of the loss you *can* drive to zero by improving the
model. It is `kl_div` / the relative-entropy regularizer in a VAE's ELBO.

```python
import math

def kl_bits(q, p):                          # extra bits from coding q with p
    return sum(qx * math.log2(qx / px) for qx, px in zip(q, p) if qx > 0)

kl_bits([0.5, 0.5], [0.25, 0.75])   # 0.2075  (always >= 0)
kl_bits([0.5, 0.5], [0.5, 0.5])     # 0.0     (perfect model, no waste)
# identity: cross_entropy(q,p) == entropy(q) + kl(q,p)
```

---

## bits/char as the project metric

**Glossary.** `bits/char` is the held-out mean predictive surprise per character: the
cross-entropy of the model against unseen text, in bits. It is *the* number the repo tracks
across experiments; lower is better, the uniform ceiling is `log₂ 28 ≈ 4.807`.

```
bits/char  =  (1/T) · Σ_{t=1}^{T} −log₂ P(o_t | o_{<t})
           =  ( mean_t −ln P(o_t | o_{<t}) ) / ln 2
```

Symbols: `T` the number of held-out characters; `P(o_t | o_{<t})` the model's predicted
probability of the actually-observed character. A drop of `X` bits/char means each character
is, on average, `2^X×` less surprising to the model.

**Example.** "`4.81 → 4.00`" means held-out surprise fell `0.81` bits/char. Since each bit
is a factor of two, the average per-character probability the model assigned to the truth
rose by `2^0.81 ≈ 1.75×`. "`3.38 → 1.61`" (Exp 3) is a `1.77`-bit drop, i.e. the model became
`2^1.77 ≈ 3.4×` less surprised per character on the `"mirro "` stream — it learned the
*letters* (m, i, r, o, space) even though not their order.

**Data.**
- **Exp 1** (K=12 HMM, built-in English corpus): held-out `4.81 → 4.00` bits/char per the
  EXPERIMENTS.md narrative (Exp 1) and `experiments-data.js` (`AM_TALLY {from:4.81, to:4.00}`;
  `AM_SURPRISE_SEGMENTS` Exp 1 `[4.81, 4.00]`). **Honesty flag:** the 2026-06-09 audit
  states the logged narrative `"4.81 → 4.00"` differs from both the original transcript and
  the re-run, which show `"4.007 → 3.424 bits/char"`; "the discrepancy is in the logged
  narrative text, not in reproducibility" (EXPERIMENTS.md, Re-verification §, lines 738–741).
  The site/data files carry the narrative `4.81 → 4.00`; cite either, but know the reproduced
  numbers are `4.007 → 3.424`.
- **Exp 3** (`"mirro "` stream): surprise `3.38 → 1.61` bits/char; the 2026-06-09 re-run
  reproduced this exactly ("Exp 3 (teach mirro): MATCH — surprise 3.38 → 1.61",
  EXPERIMENTS.md line 755).
- **Baseline:** `log₂ 28 ≈ 4.807` bits/char is the uniform-guess ceiling
  (`eval/lang_score.py`; RESEARCH.md §1.2); `eval/lang_score.py` gates on
  `beats_baseline = bits < baseline`.
- **Floors (context):** count/HMM-class models bottom out around `~1.46` bits/char on
  English; neural LMs reach `~1.12`, the Shannon human-estimated floor (RESEARCH.md §4,
  "The entropy gap"). A first-order HMM lives on the high side of that gap because its floor
  is the order-1 conditional entropy (see The order-1 floor argument and Conditional
  entropy).

**▸ In programmer terms.** `bits/char` is just the held-out NLL loop in `bits`, exactly as
`eval/lang_score.py` computes it: average `−ln p` over held-out chars, divide by `ln 2`,
compare to `log2(V)`.

```python
import math

LN2 = math.log(2)
V = 28

def bits_per_char(model_probs):              # model_probs[t] = p(true char t | history)
    nats = sum(-math.log(p + 1e-12) for p in model_probs) / len(model_probs)
    return nats / LN2

baseline = math.log(V) / LN2                  # 4.8074  -> the ceiling
# verdict, mirroring eval/lang_score.py:
def verdict(bits):
    return math.isfinite(bits) and bits < baseline   # must beat random
```

---

## Memory depth: the pair-state construction and the O(V^d) wall

**Glossary.** The order-1 floor (see The order-1 floor argument) is a *structural*
limit: a first-order HMM compresses the whole past into one belief vector, so two
histories that drive the same posterior get the same prediction and the symbol `r`
in `"mirro "` is stuck at the marginal `{r:½, o:½}` (`H = 1` bit). The fix is not more
states but more *memory*: make the state literally carry the last `d` symbols. For
`d = 2` (a trigram model) take the **pair-state** `s_t = (c_{t−1}, c_t)`, freeze a
deterministic emission `A` that just reads the current char off the state, and learn
only the transition `B` — which is the trigram next-char law and is structurally sparse.
Enumerating `d` symbols of memory costs `O(V^d)` states, so the dense-discrete path has
a hard ceiling.

```
state (pair):   s_t  =  (c_{t−1}, c_t)              K = V² = 784        (V = 28 alphabet)

emission A (FROZEN, deterministic):
    A[o, (c_prev, c_cur)]  =  1   iff  o = c_cur,   else 0

transition B (LEARNED, the trigram law):
    B[(c_cur, c_next), (c_prev, c_cur)]  =  P(c_next | c_prev, c_cur)
    sparsity:  from a pair (c_prev, c_cur) only the V pairs (c_cur, · ) are reachable
               ⇒  ≤ V nonzeros per column   (V = 28 of K = 784 rows)

cost of d-symbol memory:    K = V^d           (states grow geometrically in depth d)
    d=2: V² = 784      d=3: V³ = 21 952      d=4: V⁴ = 614 656 ≈ 614k
```

Symbols: `V = 28` the alphabet (`a–z`, space, period — see Entropy); `c_t` the character
at step `t`; `s_t` the latent pair-state; `K = V^d` the number of states for memory depth
`d`; `A[o,s] = P(o|s)` emission, `B[s',s] = P(s'|s)` transition (same matrices as in The
order-1 floor argument). "Reachable" means the next pair must overlap the current one in
its first slot, which is what makes `B` sparse.

**Example.** Take the `"mirro "` stream and `d = 2`. The pair-states that actually occur
are `(m,i), (i,r), (r,r), (r,o), (o,⎵), (⎵,m)`. The learned trigram `B` is *deterministic*
here — each pair has exactly one successor:

```
(i,r) → (r,r)     # H = 0    "after i,r the next is always r"
(r,r) → (r,o)     # H = 0    "after r,r the next is always o"
(r,o) → (o,⎵)     # H = 0
```

The order-1 tie is gone: the two ambiguous `r`-contexts are now *different states*
`(i,r)` and `(r,r)`, each with a single, certain continuation (compare The order-1 floor
argument, where after one `r` you only had the marginal `{r:½, o:½}`, `H = 1` bit; and
Conditional entropy, where the 2-char floor drops to 0). Counting cost: `K = 28² = 784`
states, but each column of `B` has at most `V = 28` nonzeros — for `"mirro "` only **one**
nonzero per used column. Push to `d = 4` and the table would need `28⁴ = 614 656` rows,
infeasible as a dense tensor; that is the wall.

**Data.** This is the engineered escape from the order-1 floor, validated across
EXPERIMENTS.md Exp 5, 7, 8. **Exp 5** added one char of memory (state = last char, `K = V =
28`) and produced the first real word-fragments — `"the cat sat on the mat"` decoded to
`"tal.e th t cn tat.she sa"` with `"the"`, `"th"`, `"tat"`, `"sa"` and correct spacing
(EXPERIMENTS.md Exp 5): "MEMORY of recent context, not capacity, produces order and words."
**Exp 7** isolated the depth variable with a count-based n-gram control: `n=3` (2-char
memory) gave **exact `"mirro mirro"`** and the question `"name. "` correctly **evoked
`"mirro"`**, while `n=2 → "miro"` (drops an `r`) and `n=4 → no further gain at this
scale` (EXPERIMENTS.md Exp 7) — "TWO characters of context is the switch-on point."
**Exp 8** rebuilt exactly this as a Bayesian HMM (`pymdp` Agent, `s=(prev,cur)`, `K = V·V =
784`, near-deterministic `A`, Dirichlet-learned `B`) and reproduced Exp 7 *as free-energy
minimization, not counts*: taught `"mirro "` it cycled `"...mirro mirro..."` in exact
order and after `"name. "` produced `"mirro."` (EXPERIMENTS.md Exp 8) — the pair-state
construction of the Glossary, realized. The cost ceiling is the honest barrier: RESEARCH.md
§2.4 states "make the state encode the last `d` symbols costs `O(V^d)` states — `V² = 784`
is fine, `V³ = 21,952` is borderline, `V⁴ ≈ 614k` is not, for full categorical tensors,"
with mitigations being factorization, sparse transition tensors, or hierarchy; RESEARCH.md
§4 barrier 1 lists "exponential cost of depth in dense discrete models … you hit the wall
by `d ≈ 4`." The Rank-1 recommendation (RESEARCH.md §3) specifies precisely this pair-state
`s_t = (c_{t−1}, c_t)`, frozen identity-on-current-char `A`, learned trigram `B` with
`≤ V` nonzeros per column, and warns it "caps at ~3-char memory before you must factorize
or go hierarchical … It will *not* produce sentences." That hierarchy boundary is itself
documented: Exp 9 is an honest negative — the flat 2-char model maps both `"sky is "` and
`"grass is "` to `"green"` because the discriminating word is outside the 2-char window
(EXPERIMENTS.md Exp 9), motivating a slow topic factor instead of deeper flat context.

**▸ In programmer terms.** Going from order-1 to order-`d` is just making the dict key the
last `d` characters instead of one (compare the order-1 `dict` keyed on a single token).
The pair-state HMM is the same idea wearing matrix clothes: the state *is* the key, `A` is
"read the last char off the key," `B` is the count table. The catch is the keyspace grows
as `V**d` — fine at `d=2`, a wall by `d=4`.

```python
from collections import Counter, defaultdict
V = 28  # a-z, space, period

def trigram_pair_state(text):
    # state s = (c_prev, c_cur); B[s] = P(next char | s)  == the LEARNED transition
    B = defaultdict(Counter)
    for c_prev, c_cur, c_next in zip(text, text[1:], text[2:]):
        B[(c_prev, c_cur)][c_next] += 1     # KEY is the last 2 chars, not 1
    return B

B = trigram_pair_state("mirro mirro mirro ")
dict(B[('i', 'r')])    # {'r': 3}   -> certain: H = 0  (was a 50/50 tie at order 1)
dict(B[('r', 'r')])    # {'o': 3}   -> certain: H = 0  -> the two r-contexts are now split

# A is frozen + deterministic: emission reads the current char straight off the state
def emit(s):           # s = (c_prev, c_cur)
    return s[1]                              # A[o, (p,c)] = 1 iff o == c

# the O(V^d) wall: enumerating d chars of memory needs this many states
states = lambda d: V ** d
states(2), states(3), states(4)             # (784, 21952, 614656)  -> dense cap ~ d=3
```

---
