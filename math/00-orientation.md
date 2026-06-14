# Orientation — the two kinds of math in this lab

> This lab runs two parallel research lines, and they speak two different mathematical
> languages. One is **Bayesian active inference** — a single creature that perceives,
> learns, wants, plans, acts, and answers by *minimizing free energy*; its currency is
> **probability** and its unit of progress is **bits**. The other is **evolutionary
> population dynamics** — a whole population where the environment, not a scorer, decides
> who reproduces; its currency is **fitness** and its unit is **births per step**. This
> page is the map: what the two currencies are, what the honest units mean, how to read
> an experiment entry, and which file in this directory covers which math.

---

## The big picture: two research lines, two families of math

**Glossary.** The repository (`active-loop` / `active_monkey`) is one moonshot —
*"build an agent you can talk to that formed its own opinions from lived experience,
never pretrained"* — pursued across two distinct substrates with two distinct
mathematical objectives.

```
Line A  (single creature, Exp ~1–151):   minimize   F   = variational free energy   [bits]
Line C  (metacognition layer, Exp 155–193): a controller ON TOP of Line A's F-stream
Line B  (population, Exp 194–207):        maximize   w   = fitness (lifetime offspring) [births]
```

- `F` — variational free energy, an upper bound on a creature's *surprise* about its
  own sensory stream. Lower `F` = "understanding". (see Variational free energy.)
- `w` — Darwinian fitness, the expected number of surviving offspring a genotype leaves.
  Nothing scores it; the environment realizes it through births and deaths.
- The **two loops** of `RESUME.md §4` are an orthogonal distinction (who *runs* the
  experiments — an autopilot vs. Claude). The **two math families here** are about what
  the experiments *measure*. Don't conflate them.

**Example.** Ask "is this creature getting better?" In Line A the answer is a number of
bits: Exp 1's held-out surprise fell from the uniform baseline toward a learned value
(below). In Line B the same question has no per-individual answer — "better" only exists
as a *gradient*: does a mutant that senses more sharply leave more offspring than its
neighbor? Exp 207 measured exactly that local gradient and found it negative
(`dB/dh = −0.041` births/step) — the sharper sensor is pure cost.

**Data.**
- Line A spans the journey's Acts I–VI (`experiments-data.js`: Language, Valence,
  Embodiment, Opinion, Frontier, Growth). The durable finding is the **RECIPE**
  (`RESUME.md §2`): emergence needs embodiment + grounding + continuous registered
  experience + **one** innate anchor + taught labels. Learning *both* the sensory map
  `A` and motor model `B` from noise collapses (Exp 31).
- Line B is the ecology arc (Exp 194–207). Its durable finding is a **wall**: across
  seven structurally-distinct regimes — avoidance (Exp 199), foraging (Exp 200),
  increasing-returns (Exp 201), interference-competition (Exp 202), residue/false-positive
  (Exp 203–205), rotating niche (Exp 206), sensor–controller coupling (Exp 207) — *a
  costed sense never becomes a functional organ at this toy scale* (`RESUME.md §3b`).
- The whole corpus, as tallied in `experiments-data.js`: 207 experiments, 9 breakthrough,
  99 positive, 38 wall, 61 partial.

**▸ In programmer terms.** Two codebases with two objective functions. Line A is a
training loop minimizing a loss; Line B is a genetic algorithm with *no* explicit fitness
function — fitness is whatever the simulation's birth/death rules emit.

```python
# Line A: gradient on a loss you can read every step
loss = free_energy(belief, observation)      # bits; you minimize this

# Line B: no loss object at all — "fitness" is realized, not computed
for t in range(horizon):
    world.regrow_food()
    for c in list(world.alive):              # O(alive), never O(ever_born)
        c.act(); c.pay_metabolism()
        if c.energy <= 0: world.kill(c)      # environment selects
        elif c.can_reproduce(): world.birth(c.mutated_child())
# you only ever MEASURE w afterward: births_per_step, trait means in the gene pool
```

---

## (A) Active inference — the single-creature line

**Glossary.** A creature is a **Bayesian generative model** of its own sensations. It
holds a probabilistic model `p(o, s)` linking hidden states `s` (meaning) to observations
`o` (sensations), and at every step it does Bayesian *perception* (infer `s` from `o`),
*learning* (update the model's parameters), and *action* (pick what to do next). Every one
of these is a single optimization: **minimize variational free energy** `F`. The built
language creature is a Bayesian HMM (`RESEARCH.md §1.1`):

```
p(o₁:T, s₁:T, A, B) = p(A) p(B) · D[s₁] · ∏ₜ₌₂ᵀ B[sₜ, sₜ₋₁] · ∏ₜ₌₁ᵀ A[oₜ, sₜ]
```

- `oₜ ∈ {1,…,V}` — the observation at step `t`; here a character, `V = 28` (a–z, space,
  period).
- `sₜ ∈ {1,…,K}` — the hidden "meaning" state; `K` is the number of latent states.
- `A ∈ Δ^{V×K}`, `A[o,s] = P(oₜ = o | sₜ = s)` — the **emission** (meaning→sensation).
- `B ∈ Δ^{K×K}`, `B[s′,s] = P(sₜ = s′ | sₜ₋₁ = s)` — the **transition** (how meaning moves).
- `D ∈ Δ^K` — the initial state prior; `Δ` = the probability simplex (entries ≥ 0, sum 1).
- `p(A), p(B)` — Dirichlet priors that make the model *learnable* by counting.

**Example.** Take `K = 2` states and `V = 3` characters. Suppose the model believes
`P(sₜ₋₁) = [0.5, 0.5]` and the transition is `B = [[0.9, 0.2], [0.1, 0.8]]` (columns sum
to 1). The one-step prediction for `sₜ` is `B · P(sₜ₋₁) = [0.9·0.5 + 0.2·0.5, 0.1·0.5 +
0.8·0.5] = [0.55, 0.45]`. If state 1 emits character `a` with probability 0.8 and state 2
emits `a` with 0.1, then the predicted probability of seeing `a` next is
`0.55·0.8 + 0.45·0.1 = 0.44 + 0.045 = 0.485`. The surprise if `a` actually appears is
`−log₂(0.485) ≈ 1.04` bits. That number *is* the creature's momentary free energy
(see Variational free energy, Bits and surprise).

**Data.**
- Exp 1 (the first rung): the creature learns at all — held-out surprise **fell as it
  trained**. The EXPERIMENTS.md narrative records this as `4.81 → 4.00 bits/char`, and the
  journey hero plot (`experiments-data.js` `AM_SURPRISE`) uses the same `4.81 → 4.00`.
  **Honesty note:** the 2026-06-09 reproduction audit (EXPERIMENTS.md, "Re-verification
  follow-up") found the *reproducible* committed output is `4.007 → 3.424 bits/char`; the
  `4.81 → 4.00` figure is the logged narrative, the `4.007 → 3.424` is what the script
  actually prints. Both show free energy genuinely falling; cite the audited pair when the
  exact number is load-bearing.
- Exp 31 (the anchor law): learning `A` **and** `B` from random noise collapses to a
  degenerate fixed point — "all states map to one" — even with unambiguous sensing. One of
  `{A, B}` must be innate. This is why the RECIPE insists on *one* anchor.
- Exp 26/28 (the moonshot core): two architecturally identical creatures raised in
  different worlds form **different** preferences — `C = [0.98, 0.01, 0.01]` vs.
  `C = [0.01, 0.01, 0.98]` — and answer the same query differently. The *content* is
  self-formed; the *wording* is a hand-mapped template (the honest caveat).

**▸ In programmer terms.** A creature is an object with three methods, each a small
optimization over probability vectors. Perception is a Bayes update; learning is smoothed
counting; surprise is just cross-entropy loss.

```python
import numpy as np

def perceive(prior, A, o):                 # Bayesian filter step
    post = prior * A[o]                     # likelihood × prior
    return post / post.sum()               # renormalize to the simplex

def predict_next(B, post):
    return B @ post                         # push belief through the transition

def surprise_bits(A, prior, o):            # = free energy proxy ≈ −log P(o | history)
    pred = A @ prior                        # P(next char) under current belief
    return -np.log2(pred[o])               # cross-entropy loss, in bits

# learning = add a pseudo-count wherever you believe state s emitted char o:
#   alpha_A[o, s] += kappa * post[s]   (Dirichlet update; A = alpha_A / alpha_A.sum(0))
```

---

## (B) Evolutionary population dynamics — the ecology line

**Glossary.** Line B drops the single creature and runs a *population*. Each creature has
a **genotype** (heritable traits) realized as a **phenotype** (its actual costs and
behavior). It burns energy to live and move, eats finite regrowing food, and — only if old
and rich enough by its *own* inherited thresholds — pays energy to spawn a mutated child.
Death is `energy ≤ 0`. No external scorer exists; **the environment is the selector**.
The quantity that matters is **fitness** `w`, and selection follows the standard
replicator/selection-gradient logic:

```
selection on trait h  ∝  ∂w/∂h           (the local fitness gradient at the resident)
w(h)  =  R · benefit(h)  −  C(h)          (a costed trait pays iff its marginal benefit > its marginal cost)
genotypeₜ₊₁  =  mutate(genotype of survivors who reproduced)
```

- `w` — fitness, operationalized as **births/step** (the realized reproduction rate) or as
  the trait's mean in the newborn gene pool over time.
- `h` — a costed, evolvable trait (e.g. `thermosense_intensity`, the sharpness of a heat
  sensor); `θ` — a controller knob that uses `h`.
- `benefit(h)` — what sharper sensing buys; `C(h)` — its metabolic cost.
- `∂w/∂h` — the **selection gradient**: positive ⇒ the trait spreads, ≤ 0 ⇒ it decays.
- The killer subtlety this arc discovered: a trait can be *enormously useful when handed to
  you for free* yet have `∂w/∂h ≤ 0` at the resident, so evolution never builds it
  (Exp 200, lesson L22). A forced/behavioral benefit test does **not** predict evolvability.

**Example.** Compute a discrete selection gradient by hand from Exp 207's corner grid.
Births/step `B` was measured at two sensor sharpnesses `h ∈ {0.10, 0.45}` and two
controller levels `θ ∈ {0.6, 6.0}`:

```
              θ = 0.6     θ = 6.0
h = 0.10      0.1134      0.2606
h = 0.45      0.0678      0.2196
```

Sensor change at high θ: `ΔB = 0.2196 − 0.2606 = −0.0410` births/step as `h` goes
0.10 → 0.45 — *negative*, so sharper sensing loses births (the entry reports this raw
drop as `dB/dh = −0.041`; dividing by the 0.35 span gives the per-unit slope −0.117, same
sign). Controller change at low h: `ΔB = 0.2606 − 0.1134 = +0.1472` as `θ` goes 0.6 → 6.0
— *positive*, the controller pays on its own. The cross-partial (does the pair help where
neither does alone?) is ≈ 0. No 2-D valley exists; selection pushes the two traits apart.
(see Selection gradient, Fitness valley.)

**Data.**
- Exp 194 (the substrate, MIXED): a 12×12 regenerating grid; in a balanced world the
  population persists to horizon across 8–12 generations (`births 628/622/509`), scarce
  worlds crash 95–100% and go extinct in 2/3 seeds. Byte-reproducible under fixed seeds.
- Exp 199 (the avoidance valley, NEGATIVE): even with the deck stacked for it, a primitive
  heat sensor never climbs — newborn intensity stays `~0.05` (`V1 0.0619, V2 0.0559,
  V3 0.0505, V4 0.0558`) across the whole noise sweep, and a seeded-functional organ (0.50)
  decays back to primitive or drives extinction (2/5). The signature of a real *fitness
  valley*: high-intensity mutants keep arising but are always culled.
- Exp 205 (the cleanest statement, MIXED): at survivable losses the functional optimum
  truly *is* the bulk-fittest (`h* = 0.60`) and the population survives — yet evolution
  *still* stays primitive (0/5 functional, local pairwise gradient ≤ 0). So the valley
  (not demographic collapse) is the sole binding barrier.
- Exp 207 (the closer, design-stage NEGATIVE): the corner grid above falsified the
  co-adaptation premise without a full batch, closing the sub-arc.

**▸ In programmer terms.** No `loss.backward()`. Fitness is *emergent* from a discrete-event
loop; you estimate gradients by finite differences over a parameter grid, exactly as Exp 207
did. The recurring trap — "useful when forced ≠ evolvable" — is the gap between calling a
function directly and seeing whether selection ever calls it.

```python
def births_per_step(h, theta, seeds, horizon=3500, tail=1000):
    rates = []
    for s in seeds:                                  # >= 3 seeds, report ALL (VALIDATION.md)
        world = Ecology(thermosense_intensity=h, niche_weight=theta, seed=s)
        births = world.run(horizon)                  # realized, not computed
        rates.append(births_in_last(tail) / tail)
    return mean(rates)

# selection gradient = finite difference, NOT autograd:
dB_dh = (births_per_step(0.45, 6.0, seeds) -
         births_per_step(0.10, 6.0, seeds)) / (0.45 - 0.10)   # ≈ -0.041  -> trait decays
```

---

## (C) The metacognition / control layer

**Glossary.** Line C sits *on top of* a Line-A creature: it does not change how the
creature minimizes `F`, it **monitors and regulates** that process. Two rungs were built.
**N3** is a *trust monitor* — it watches the creature's own forecast errors and decides
when to distrust the model enough to hand over parameter authority. **N4** is an *identity
controller* — it watches whether a creature's learned identity is being overwritten and
decides whether to defend it. Both are graded by signal-detection math:

```
type-2 AUROC  =  P( monitor-confidence(correct) > monitor-confidence(error) )
controller passes  iff  it both DEFENDS (resists overwrite) and REVISES (updates when it should)
```

- **type-2 AUROC** — area under the ROC curve for the monitor's confidence *as a predictor
  of the creature's own correctness*. 0.5 = chance (the monitor knows nothing); higher =
  the monitor's "I'm unsure" tracks real errors. (see Signal detection / AUROC.)
- **DEFEND vs REVISE** — the two bars an identity controller must clear simultaneously;
  the N4 arc's law is that one fixed constant rarely clears both (the "universal-constant
  law").

**Example.** Suppose on 4 trials the monitor reports confidence `[0.9, 0.4, 0.8, 0.3]`
and the creature was actually correct on trials 1 and 3. The correct-set confidences are
`{0.9, 0.8}`, the error-set are `{0.4, 0.3}`. Every correct beats every error
(`0.9>0.4, 0.9>0.3, 0.8>0.4, 0.8>0.3` — 4 of 4 pairs), so type-2 AUROC = 4/4 = 1.0: a
perfect monitor here. If instead the correct-set were `{0.4, 0.8}` and error-set
`{0.9, 0.3}`, the winning pairs would be `0.4>0.3, 0.8>0.3` = 2 of 4 = 0.5 — chance.

**Data.**
- Exp 157 (N2 prerequisite, POSITIVE): a per-place expected-uncertainty channel read off
  the creature's *own* lived outcomes tracks its accuracy at **type-2 AUROC 0.80 pooled**
  vs. 0.56 for the natural channel, 8/8 forks. The confidence half of metacognition exists
  on this body.
- The N3 chapter (Exp 155–168): rungs 1–3 passed — the forecast-scoring trust monitor and
  the lock-on-label-consistency controller convert metacognitive distrust into stable,
  regime-adaptive parameter authority. **"Agency over metacognition" is SUPPORTED at toy
  richness** (`RESUME.md §3b`).
- The N4 chapter (Exp 174–193, CLOSED-NEGATIVE then partial cracks): monitoring of identity
  is real (a sensitive, specific read-only monitor was built), but *commitment control is
  CONFIG, not agency* — defense, where achievable, needs only a stopwatch.

**▸ In programmer terms.** Line C is a wrapper/decorator around the Line-A step. The monitor
is a binary classifier scored by AUROC; the controller is a policy whose pass condition is a
*conjunction* of two opposing tests — which is why a single tuned constant usually fails.

```python
from sklearn.metrics import roc_auc_score

# type-2 AUROC: does the monitor's confidence predict the creature's own correctness?
auroc = roc_auc_score(y_true=correct_flags, y_score=monitor_confidence)   # 0.5 = useless

def controller_passes(arm):
    return defends(arm) and revises(arm)     # must clear BOTH — the universal-constant law
```

---

## The universal currencies and the honest units

**Glossary.** Three quantities recur; two units are the lab's load-bearing honesty anchors.

```
probability   p ∈ Δ      (a normalized belief; perception lives here)
surprise      −log₂ p    [bits]            (how shocked the model is by an observation)
free energy   F          [bits/nats]       (the upper bound on average surprise it minimizes)
fitness       w          [births]          (realized lifetime reproduction; selection acts on it)

bits/char  =  (mean over a stream of  −log₂ P(charₜ | history))     # Line A's honest unit
births/step =  total births / horizon                              # Line B's honest unit
```

- **bits/char** — the average surprise per character on *held-out* text. The uniform
  baseline is `log₂ V = log₂ 28 ≈ 4.807` bits/char (`RESEARCH.md §1.2`); a model that has
  learned *anything* scores below it. This is the single number Line A reports as progress.
- **births/step** — the realized reproduction rate, the only honest readout of fitness in a
  world with no scorer. Compared across arms, its differences are the selection gradient.

**Example.** A uniform guesser over 28 symbols spends `log₂ 28 ≈ 4.807` bits/char — pure
ignorance. If a learned model assigns the *actual* next character probability 0.25 on
average, it spends `−log₂(0.25) = 2.0` bits/char: it has saved `4.807 − 2.0 ≈ 2.8` bits of
surprise per character. That saved surprise *is* the learning. On the ecology side
(illustrative arithmetic), a run whose last 1000 steps produced 261 births reports
`261/1000 = 0.261` births/step for that tail — the very readout Exp 207 used (its `h0.10,
θ6.0` cell was 0.2606 births/step), directly comparable across arms.

**Data.** bits/char is the spine of Acts I–VI: Exp 1 (`4.007 → 3.424` reproduced, narrated
`4.81 → 4.00`), Exp 3 (`3.38 → 1.61` on the `"mirro "` stream — a *different* corpus, never
drawn as one curve with Exp 1; see `experiments-data.js` `AM_SURPRISE_SEGMENTS`). births/step
is the spine of the ecology arc: Exp 207's grid (`0.1134 … 0.2606`) and every selection
gradient in Exp 199–207.

**▸ In programmer terms.** bits/char is just mean cross-entropy in base 2 over a held-out
split; births/step is a counter divided by a horizon. Keeping the two corpora separate
(Exp 1 vs Exp 3) is the same discipline as never averaging val-loss across two different
datasets.

```python
import numpy as np

def bits_per_char(model, held_out):                 # Line A unit
    return np.mean([-np.log2(model.prob(c, ctx))    # cross-entropy, base 2
                    for ctx, c in held_out])
# baseline to beat: log2(28) ≈ 4.807

def births_per_step(total_births, horizon):         # Line B unit
    return total_births / horizon
```

---

## How to read an experiment entry

**Glossary.** Every entry in `EXPERIMENTS.md` (and its mirror in `experiments-data.js`)
follows a fixed, falsifiable shape enforced by `loop/PROTOCOL.md` and graded by the binding
honesty contract `loop/VALIDATION.md`. The skeleton:

```
Plain      — one jargon-free sentence: what we're really testing
Hypothesis — one sentence, PREDECLARED before any code
Prediction — what we'd see if the hypothesis is TRUE   (property-level, explicit thresholds)
Falsifier  — what result would make it FALSE           (declared BEFORE the run)
Setup      — the smallest test; script at experiments/expNN_<slug>.py
Result     — raw numbers, quoted from the committed output
Verifier   — an INDEPENDENT blinded recompute of the verdict (agree / disagreed)
Verdict    — POSITIVE / NEGATIVE / MIXED  +  CONSOLIDATION / NEW INSIGHT  (+ BREAKTHROUGH?)
```

- **Predeclared falsifier** — the falsifier is written *before* seeing data; if the result
  matches it, the verdict is NEGATIVE, full stop (no post-hoc reinterpretation).
- **Blind verification** (`PROTOCOL.md §4.5`) — a separate worker, given only the
  predeclared docstring and the raw output (never the author's reading), recomputes the
  verdict conjunct-by-conjunct. Every Exp 152+ verdict is blind-verified.
- **Verdict taxonomy** — POSITIVE / NEGATIVE / MIXED is the outcome; CONSOLIDATION (predictable
  from prior work) vs. NEW INSIGHT is the novelty tag; **BREAKTHROUGH** is reserved for a
  first-of-its-kind capability and must survive the hostile-reviewer test (`VALIDATION.md`:
  "the same as before, but bigger/cleaner/again" is POSITIVE-SINGLE, not breakthrough).

**Example.** Read Exp 199 as a falsification: its predeclared falsifier was *"any primitive
arm climbs > 0.30, or the functional arm retains > 0.30."* The observed newborn intensities
(`0.0505–0.0619`, functional arm decays to `< 0.20`) did **not** trigger it, and the
deliberately-favorable deck makes the NEGATIVE *stronger*, not weaker — a structural ceiling,
logged as NEW INSIGHT. Contrast Exp 154, graded BREAKTHROUGH because the creature did
something it provably *could not do before*: notice its own model was too small and grow it,
24/24 runs, once scoring switched to normalized densities.

**Data.** The taxonomy is visible in the tally (`experiments-data.js`): of 207 experiments,
only 9 are breakthrough; 38 are walls and 61 partial — negatives and partials outnumber the
clean wins, which is the point. Walls cited above: Exp 4 (more states ≠ memory), Exp 31
(both-anchors collapse), Exp 199/200/207 (the sense-evolution ceiling). Breakthroughs
include Exp 5 (one char of memory yields word fragments) and Exp 154 (the growth wall fell).

**▸ In programmer terms — how to read the data files.** `EXPERIMENTS.md` is the
append-only source of truth (newest last, never edited); `experiments-data.js` is its
structured mirror that the journey site renders. A few practical rules when you parse them:

```python
# experiments-data.js exposes browser globals; the structured fields you want:
#   window.AM_EXPERIMENTS  -> list of {n, kind, chapter, plain, setup, result,
#                                       implication, caveat, trace:{script, output}}
#   window.AM_TALLY        -> {total, breakthrough, positive, wall, partial}
#   window.AM_SURPRISE_SEGMENTS -> per-experiment bits/char points (kept SEPARATE on purpose)
#
# kind ∈ {"breakthrough","positive","wall","partial"}  ->  the verdict, honestly graded
# every entry's trace.{script,output} points at the committed, reproducible artifact.
#
# To trust a number: grep EXPERIMENTS.md for the Exp ID, read Result, then open the
# committed trace.output — the entry quotes the FINAL committed output (the re-run/re-quote
# rule). If a narrative figure and the raw output disagree (Exp 1: 4.81→4.00 narrated vs
# 4.007→3.424 reproduced), the committed output is ground truth.
def is_trustworthy(entry):
    return (entry["trace"]["output"]            # raw output committed
            and entry["kind"] in VERDICTS       # honestly graded
            and "Verifier" in raw_md_entry(entry["n"]))   # blind-verified (Exp 152+)
```

---

## Roadmap — which file covers which math

**Glossary.** This `math/` reference is split so each file owns a cluster of concepts and
points back at the experiments that exercise it.

| File | Math it covers | Maps to |
|---|---|---|
| `00-orientation.md` (this file) | the two lines, the currencies (probability, bits, fitness), units (bits/char, births/step), how to read an entry | the whole corpus, Exp 1–207 |
| `01-free-energy-and-active-inference.md` | the Free Energy Principle, variational & expected free energy, perception as Bayesian filtering, precision, valence = −F, planning, the RECIPE | Line A: Exp 1–35, 141–154 |
| `02-bayesian-inference-and-learning.md` | Bayes' rule, the categorical/Dirichlet conjugate pair, the Bayesian HMM (`A`/`B`/`D`), forward filtering, the mean-field collapse, Dirichlet parameter learning, place fields | Line A: Exp 1–31 |
| `03-information-theory.md` | surprise, entropy, cross-entropy, KL divergence, bits/char, conditional entropy & the first-order Markov wall | Line A units; Exp 1, 3–7 |
| `04-probability-and-distributions.md` | the probability simplex, categorical/Bernoulli, Dirichlet/Beta, Gaussian & precision, sigmoid/softmax, the EMA tracker | foundations; Exp 201–207 (EMA, `σ(k·h·θ−d)`) |
| `05-evolutionary-dynamics.md` | fitness, the selection gradient `∂w/∂h`, the cross-partial & 2-D valley, fitness valleys & benefit saturation, heritability vs survivor bias, drift, the forced-vs-evolvable gap | Line B: Exp 194–207 |
| `06-control-and-dynamical-systems.md` | the symmetric-saddle/collapse finding, attractors & non-identifiability, precision as control, the N3/N4 monitors & controllers (type-2 AUROC), hysteresis, the universal-constant law | Line C: Exp 31, 155–193 |
| `07-statistics-and-experimental-method.md` | seeds & reproducibility, mean/variance, correlation (Pearson/Spearman), effect size, nulls & matched controls, predeclared falsifiers, blind verification, confounds, the verdict taxonomy | method; Exp 195–207 |

**Example.** A reader who sees "Exp 7: n=3 → exact `'mirro'`" and wants the math should open
`03-information-theory.md` (the conditional-entropy argument for *why* depth lowers the
floor) and `02-bayesian-inference-and-learning.md` (the HMM and its first-order limit). A
reader puzzling over "Exp 207: cross-partial ≈ 0, no 2-D valley" should open
`05-evolutionary-dynamics.md` (selection gradients and fitness valleys).

**Data.** The split mirrors the journey's own acts: Acts I (Language) and VI (Growth) are
Line A / structure-learning math; Acts II–IV (Valence, Embodiment, Opinion) are active
inference + the RECIPE; Act V (Frontier) is where Line A hits its honest ceiling
(`open_problem.html`); the ecology arc (Exp 194–207) is Line B math, not yet a journey act
at the time of writing.

**▸ In programmer terms.** Treat this table as the import map: when an experiment cites a
concept, jump to the file that owns it rather than re-deriving it.

```python
CONCEPT_TO_FILE = {
    "free_energy":        "01-free-energy-and-active-inference.md",
    "bayesian_hmm":       "02-bayesian-inference-and-learning.md",
    "bits_per_char":      "03-information-theory.md",
    "distributions":      "04-probability-and-distributions.md",
    "selection_gradient": "05-evolutionary-dynamics.md",
    "type2_auroc":        "06-control-and-dynamical-systems.md",
    "blind_verification": "07-statistics-and-experimental-method.md",
}
# every Exp NN cited here is grep-able in EXPERIMENTS.md; verify before you trust.
```

---
