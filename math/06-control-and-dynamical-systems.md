# Control, attractors & dynamical systems

> This area of math treats the agent's belief and value updates as a **dynamical system**: state vectors that flow over time toward fixed points (attractors), and **controllers** that steer that flow. This repo uses it for two jobs: to state *why* unsupervised learning from a disembodied stream **collapses** (it falls into a symmetric saddle / degenerate attractor), and to ask, layer by layer, whether a higher-order *controller* — a metacognitive monitor (N3), an identity-defense (N4) — is ever **load-bearing**, or whether a fixed constant does the same job (the anti-regress / universal-constant kill-test).

---

## The collapse as a dynamical-systems statement: a symmetric saddle

**Glossary.** Learning is gradient/coordinate descent on the variational free energy `F` (see Variational free energy, math/01) over model parameters `θ` (e.g. the sensory map `A` and motor model `B`). A **fixed point** is a `θ*` where the update stops; a **minimum** attracts from every direction; a **saddle** is flat or repelling along some directions and attracting along others. The collapse finding says: when `θ` is learned *jointly from a disembodied/noise stream with no anchor*, the only attractor reachable is a **degenerate** one — every latent state maps to the same output — sitting in a **symmetric saddle** of the loss, where relabeling the latents leaves `F` unchanged.

```
θ_{k+1} = θ_k − η ∂F/∂θ                          (descent dynamics)
∂F/∂θ |_{θ*} = 0                                  (fixed point)
F(θ*) = F(P · θ*)  for any latent permutation P   (label-switching symmetry ⇒ saddle)
θ* degenerate:  A[o, s] ≈ A[o, s′]  ∀ s, s′      (all states emit the same thing)
```

Symbols: `θ` = model parameters; `F` = variational free energy (surprise upper bound); `η` = learning rate; `∂F/∂θ` = gradient; `P` = a permutation of latent-state labels; `A[o,s]` = probability latent `s` emits observation `o`. The **RECIPE** breaks the symmetry by *pinning one factor* (give `A` or `B` innately) so the permutation `P` is no longer free — exactly one labeling survives.

**Example.** Two latent states, each "should" learn one of two characters. Start symmetric: both states emit `½ a + ½ b`. The gradient of `F` toward "state 0 → a, state 1 → b" has the **same magnitude** as the gradient toward "state 0 → b, state 1 → a" — the two are mirror images. Their vector sum is zero along the symmetry axis, so the parameters do not move off the tie. The system slides down the one remaining direction it *can* move: both states collapse onto the **same** distribution (the marginal), the degenerate fixed point. Net: `F` decreased a little, but no structure was learned. Pin state 0 to emit `a` (an anchor): now the tie is broken, the gradient points uniquely, and state 1 is free to specialize on `b`.

**Data.** **Exp 31** is the embodied collapse: learning BOTH `A` and `B` from random init on a 1200-step wander lands in a degenerate fixed point — even with *unique* (non-aliased) sensing, the learned right-step map became `[0,0,0,0,0]` / `[3,3,3,3,3]`, i.e. **all states map to one**. Clean recoveries happened only with an anchor: Exp 17 learned `B` with `A` known; Exp 20/21 learned `A` with `B` known. **Exp 16** pinned the mechanism one level deeper for the symbolic case: even handing the topics perfectly differentiated transitions, the topic posterior stayed `[0.5, 0.5]` before AND after 6 epochs — the **mean-field** factorization `q(z,s) ≈ q(z)q(s)` *severs* the cross-factor message (see Non-identifiability). **Exp 135** later showed the collapse is conjugate arithmetic, linear in accumulated mass and dialed by `κ` alone (end-structure error `≈ κ₀/(κ₀+100)`), common to both the tabular and continuous substrates.

**▸ In programmer terms.** A loss with a permutation symmetry has *tied* optima; if you init symmetric, the optimizer can't pick one and you fall to the trivial solution that satisfies the tie — like k-means with all centroids initialized to the same point.

```python
import numpy as np

def free_energy(A):              # toy: penalize how much states differ from the marginal (a degenerate basin)
    marginal = A.mean(axis=1, keepdims=True)
    return ((A - marginal) ** 2).sum() * -0.0  # flat: F doesn't reward specialization without an anchor

# symmetric start: 2 states, 2 chars, both emit [0.5, 0.5]
A = np.array([[0.5, 0.5],
              [0.5, 0.5]])       # rows=chars, cols=states
# no gradient signal distinguishes "state0->a" from "state1->a": stuck.
anchor = None                    # learn BOTH -> collapse (Exp 31)
# RECIPE: pin one column -> symmetry broken, the other can specialize
anchor = np.array([1.0, 0.0])    # state 0 := emits char 'a' innately
A[:, 0] = anchor                 # now descent on state 1 has a unique target
```

The "give either `A` or `B`" rule is literally: **freeze one tensor's gradient** so the optimizer has a non-degenerate problem.

---

## Fixed points, attractors & convergence (saddle vs minimum)

**Glossary.** A discrete dynamical system `x_{t+1} = f(x_t)` has a **fixed point** at `x*` where `f(x*) = x*`. Its character is set by the Jacobian `J = ∂f/∂x` at `x*` (eigenvalues `μ_i`): all `|μ_i| < 1` ⇒ **attractor** (stable, converges); some `|μ_i| > 1` ⇒ **saddle/repeller**. The value vector in this repo is an explicit such system.

```
x_{t+1} = f(x_t)                          (the map)
x* :  f(x*) = x*                           (fixed point)
J = ∂f/∂x |_{x*},   spectral radius ρ(J)   (stability test)
ρ(J) < 1  ⇒ attractor (converges)
ρ(J) > 1  ⇒ saddle / repeller
```

The value update (the Exp 26 rule) is an affine contraction toward an equilibrium:

```
v_{t+1} = λ · v_t + w_t · e_{o_t}
v*      = (mean input) / (1 − λ)           (equilibrium mass)
```

Symbols: `v_t` = value vector over colors; `λ ∈ (0,1)` = forgetting/decay (here `λ = 0.9997`); `w_t = exp(−H(predictive)) ∈ (0,1]` = per-step predictability weight (see Precision-weighting); `e_{o_t}` = one-hot of the observed color. Because `λ < 1`, the map is a contraction (`ρ = λ < 1`): it has a **single attractor** `v*`.

**Example.** With `λ = 0.9997` and a steady input of `w·e = 1` unit/step into one color, the equilibrium mass is `1 / (1 − 0.9997) = 1 / 0.0003 ≈ 3333`. Start at `v = 0`: after one step `v = 0.9997·0 + 1 = 1`; the gap to `v*` shrinks by a factor `λ` each step, so to close 99% of the distance takes `ln(0.01)/ln(0.9997) ≈ 15,350` steps — slow forgetting, durable memory. Change the input color and the attractor *moves*; `v` slides toward the new `v*` at the same rate.

**Data.** **Exp 176** (rung 1) shows the system being driven from one attractor to another: an 800-step captivity permanently re-makes the baseline's favorite — **8/8 forks, flips 24/24, recoveries 0/24** (96/96 flips across Exp 174–177). The displaced identity is **world-determined**: it lands on the occupancy equilibrium (color 2 at 9/25 cells) in 7/8 forks — i.e. the new attractor is set by the environment's stationary distribution, not by biography. **Exp 174–175** show the precondition can *fail*: at `λ = 0.999` the baseline has no stable favorite even unpressured (the attractor basin is too shallow given the spine's inherited near-tie, a 3.8% gap), which is why the chapter raised `λ` to 0.9997 and ran a 6000-step settle to deepen the basin before testing.

**▸ In programmer terms.** This is an exponential moving average (EWMA) — the canonical contraction map. Its fixed point is `input/(1−λ)`, and the half-life is `−ln 2 / ln λ`.

```python
lam = 0.9997
v = 0.0
for o in stream:                         # o = 1.0 unit into the tracked color each step
    v = lam * v + o                       # affine contraction toward v*
v_star = 1.0 / (1 - lam)                  # ~3333: the attractor mass
half_life = -np.log(2) / np.log(lam)      # ~2310 steps to forget half
```

A *minimum* basin pulls back from any push (recovers); a *saddle* (the collapse case) lets one direction run away. Raising `lam` deepens the basin — the Exp 174→176 move.

---

## Non-identifiability (two settings explain the data equally)

**Glossary.** A model is **non-identifiable** when distinct parameter settings produce the *same* likelihood, so the data cannot choose between them. Two flavors appear here. **Label-switching:** permuting latent labels leaves the likelihood invariant (`P(data | θ) = P(data | Pθ)`), the source of the saddle above. **Mean-field severing:** the variational posterior is *assumed* factorized, `q(z,s) ≈ q(z)q(s)`, which deletes the channel by which evidence about `s` could update belief about `z`.

```
P(data | θ) = P(data | P·θ)    for permutation P   (label-switching non-identifiability)
q(z, s) ≈ q(z) · q(s)                                (mean-field factorization)
⇒ ∂ELBO/∂q(z) carries no term in the s-transitions   (the cross-factor message is severed)
```

Symbols: `θ` = parameters; `P` = label permutation; `q(z,s)` = joint posterior over topic `z` and char-state `s`; `ELBO = −F` = evidence lower bound. Under the mean-field cut, the gradient that should push `q(z)` from transition evidence is exactly zero — information that lives in the *correlation* between `z` and `s` is unreachable.

**Example.** Suppose topic 0 generates "sky" transitions and topic 1 generates "grass" transitions, emissions identical. A *joint* posterior `q(z,s)` could read a run of sky-like transitions and conclude `z = 0`. But factorize it: `q(z)q(s)` says "the topic and the current char-state are independent." Then `q(z)`'s update integrates over `q(s)` and the transition-specificity averages out — the message "this transition pattern implies topic 0" never lands. So `q(z)` stays at its prior `[0.5, 0.5]` no matter how clean the transitions are. Two topic assignments are *observationally equal* under this approximation: non-identifiable by construction.

**Data.** **Exp 16** is the canonical instance: handed perfectly differentiated transitions, the topic belief stayed `[0.5, 0.5]` **before AND after 6 unsupervised epochs**, output degenerate ("s is i"). The diagnosis is explicit: the mean-field approximation `q(z,s) ≈ q(z)q(s)` *severs* the cross-factor message, so transition-only topic inference is blocked by the variational approximation itself, not merely by symmetry. The named fix — route the topic through **emission** (`A_dependencies = [[0,1]]`) so observations directly update `q(z)` — is the same "grounding" ingredient as the RECIPE (see The collapse as a symmetric saddle). At the value-vector level the same non-identifiability appears as a **near-tie**: the spine's inherited 0↔2 gap of 3.8% (Exp 174–175) means *which* color wins is barely determined by the data, so the attractor is fragile until the basin is deepened.

**▸ In programmer terms.** Mean-field is the assumption you make when you optimize each factor with the other held fixed (coordinate ascent on independent variational params). If two factors are *correlated* in the true posterior, that loop can't see it.

```python
# mean-field: optimize q_z and q_s independently -> no cross term
def update_q_z(q_s, transitions):
    # marginalizes over q_s, so transition->topic evidence averages out
    return prior_z                        # stays [0.5, 0.5]  (Exp 16)

# the fix: let observations depend on BOTH factors (A_dependencies=[[0,1]])
def update_q_z_grounded(obs, q_s):
    return normalize(prior_z * P_obs_given_z[obs])   # now evidence reaches q(z)
```

Label-switching is the classic "your two clusters swapped IDs between runs but the loss is identical" problem.

---

## Precision-weighting as a control signal

**Glossary.** Precision is the *confidence* (inverse variance) the agent assigns to a prediction; **precision-weighting** scales how much a prediction error is allowed to move belief or value. In this repo the per-step weight is `w_t = exp(−H(predictive)) ∈ (0,1]`: when the next-symbol prediction is sharp (low entropy `H`), the observation is *trusted* and writes strongly; when the prediction is uncertain (high `H`), the write is damped. It is the system's "trust the model vs. seek context" knob.

```
H_t = − Σ_i p_i · log p_i                  (entropy of the predictive distribution p)
w_t = exp(−H_t) ∈ (0, 1]                    (precision weight: sharp prediction ⇒ w≈1)
v_{t+1} = λ · v_t + w_t · e_{o_t}           (precision-gated value write)
```

Symbols: `p_i` = predicted probability of symbol `i`; `H_t` = predictive entropy; `w_t` = precision weight; the rest as in Fixed points & attractors. Note `w` enters the value dynamics as a **time-varying input gain** — it modulates the attractor's pull, not the decay.

**Example.** Three colors, prediction `p = [0.9, 0.05, 0.05]` (sharp): `H = −(0.9·ln 0.9 + 2·0.05·ln 0.05) ≈ 0.394` nats, so `w = exp(−0.394) ≈ 0.674` — a strong write. Flat prediction `p = [⅓, ⅓, ⅓]`: `H = ln 3 ≈ 1.099`, `w = exp(−1.099) = ⅓` — a weak write. So a confident agent updates its values ~2× harder per observation than a confused one; "I understand what I'm seeing" is *literally* the gain on learning from it.

**Data.** Precision-weighting is the predictability signal `w_t = exp(−H(predictive))` carried in the Exp 26 value rule used throughout the N4 chapters. It connects directly to valence (`−F` = competence; see Variational free energy, math/01): **Exp 1** measured held-out surprise falling **4.007 → 3.424 bits/char** on a `V = 16` alphabet (uniform baseline `log₂ 16 = 4.000` bits/char, so it began essentially at chance and learned below it) — as `F` falls, predictions sharpen, `w` rises, and "the agent feels better as it understands." In the N4 freeze controllers, precision-gating is *insufficient on its own* to defend identity (see Metacognitive controllers / N4 identity): **Exp 181** showed the write-gate `g_k = min(1, (m̄/m_k)²)` engages (`g = 0.22` at first in-burst snapshot) but the monitor's absorption re-opens it mid-burst (`g` rising 0.22→0.55 within 4 snapshots), leaking 256–498 writes per 800-step burst — gain-control alone defends 0/8.

**▸ In programmer terms.** It's an attention/learning-rate gate computed from prediction entropy — a confidence-weighted update, like inverse-variance weighting or a softmax temperature on how much each sample counts.

```python
import numpy as np
def precision_weight(p):                 # p = predictive distribution (sums to 1)
    H = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return np.exp(-H)                     # in (0, 1]: sharp p -> ~1, flat p -> small

p_sharp = np.array([0.9, 0.05, 0.05]);  print(precision_weight(p_sharp))  # ~0.674
p_flat  = np.array([1/3, 1/3, 1/3]);    print(precision_weight(p_flat))   # ~0.333
v = lam * v + precision_weight(p) * onehot(observed_color)
```

High confidence ⇒ high learning rate on that step; uncertainty automatically throttles the write.

---

## Metacognitive controllers (N3: distrust → parameter authority)

**Glossary.** An N3 controller sits *above* the agent's metacognition (N2): it scores whether N2's own diagnoses can be trusted, and if not, rewrites N2's policy parameter `θ_N2` (here a detector **window**). The governing **anti-regress law**: a layer is real iff a constructible perturbation degrades the lower layer AND the higher layer detects + corrects it via a control surface the lower layer lacks. The trust monitor scores N2's diagnoses against the agent's *own future record*; the load-bearing controller is **lock-on-label-consistency**.

```
trust = forecast-score of N2's labels vs the next 100 steps of the creature's record
trigger:  trust < τ            (distrust ⇒ act)
control:  θ_N2 (window) climbs 200 → 400 → 800 → 1600 while trust is low
LOCK:     freeze θ_N2 while the last K = 8 labels under the current dial are one class
```

Symbols: `θ_N2` = the N2 detector's window length; `trust` = forecast-consistency score in `[0,1]`; `τ` = distrust threshold (0.7 in Exp 164); `K = 8` = the lock horizon (a regime statistic over the diagnosis stream — the minimal "N1 inside N3"). The lock is what converts transient distrust into *stable* parameter authority instead of an oscillating dial.

**Example.** N2 labels a stretch "OK" (promise: errors stay rare). Over the next 100 steps the creature logs 31% errors — the promise is broken, so the forecast-score for that cell is low. With a valid label the score is 1.0; in the broken regime it is 0.6897 — a gap of 0.3103 (Exp 163). When the score sits below 0.7, the controller widens the window; once the last 8 labels under the current window are all the same class, it **locks** — a fixed-horizon checker can't validate the regime-conditional STRUCTURAL claim, but "8 labels in a row agree" is a cheap regime statistic that can.

**Data.** **Exp 157** built the confidence half of N2 (per-place expected-uncertainty channel: type-2 AUROC **0.80** pooled vs 0.56 for the natural channel). **Exp 162** (rung 1) found the window blind spot: hidden-context alternation slower than the fixed 200-step window makes STRUCTURAL drop to **0.23** with majority OK, while the *same* stream at `W′ = 800` reads **100% STRUCTURAL** — a sustained, dial-pinned failure. **Exp 163** (rung 2): the forecast-scoring monitor reaches trust **1.0000** in valid cells and **0.6897** broken (gap **0.3103**); the Brier-style alternative *fails* (it tracks world difficulty, not diagnostic brokenness). **Exp 167** (rung 3, BREAKTHROUGH): with lock-on-label-consistency the controller climbs 200→400→800→1600 and locks on STRUCTURAL at `t = 5799` in **8/8** forks, combined score **1.0 vs 0.7** for the best constant — the first conversion of metacognitive distrust into stable regime-adaptive authority. **Exp 168** (rung 4, NEGATIVE on no-harm): overrides concentrate in broken segments (**0.7500**) and deliver **+0.5034** where broken, but valid segments suffer **−0.1688** pooled — the **RATCHET LAW**: climbing has a driver (forecast violations) but descending has no motor, because an oversized dial in an honest world produces consistent (indistinguishable-from-correct) diagnoses.

**▸ In programmer terms.** A supervisor watching a detector's hit-rate, retuning its window, with a hysteresis lock so it doesn't thrash.

```python
WINDOWS = [200, 400, 800, 1600]; i = 0; recent = []
def step(label, future_record):
    global i, recent
    trust = forecast_score(label, future_record)     # 1.0 valid, ~0.69 broken (Exp 163)
    recent = (recent + [label])[-8:]                  # K = 8
    if len(set(recent)) == 1:                         # LOCK: last 8 agree
        return WINDOWS[i]                             # freeze the dial (Exp 167)
    if trust < 0.7 and i < len(WINDOWS) - 1:
        i += 1                                        # widen the window (climb)
    return WINDOWS[i]
```

The ratchet bug (Exp 168): there is no `i -= 1` branch with a valid trigger — descent has no signal.

---

## N4 identity: a read-only monitor + a commitment controller (the stopwatch)

**Glossary.** N4 asks whether *identity defense* needs agency or just config. It has two parts. A **read-only monitor** predicts its own value drift and flags "I am not who I predicted." A **commitment controller** decides when to *freeze* the value dynamics (resist) vs let them revise. The crack chapter's working controller is a **stopwatch on the continuous-pressure clock** (INT-C2900): it times how long pressure has been *unbroken*, resetting on every gap; it concedes only past horizon `C`. The governing **interval law** says a working `C` lives in a band.

```
v_hat_k = v_{k−1} + (v_{k−1} − v_{k−11}) / 10        (monitor's linear drift forecast)
m_k     = ‖ v_hat_k − v_k ‖₂                          (mismatch: "not who I predicted")
trigger: r_k = m_k / m_bar ≥ θ = 3.5                  (enter RESIST; m_bar frozen on entry)
RESIST:  v_{t+1} = v_t                                 (freeze: no decay, no writes)
INT-C:   concede when continuous-pressure stretch > C; every gap RESETS the clock

INTERVAL LAW:  C ∈ ( single-burst stretch , tolerance ]
            ≈ ( L_max + ~window , revision_tolerance ]
```

Symbols: `v_hat_k` = predicted value vector from own history; `m_k` = L2 mismatch; `m_bar` = median trailing mismatch; `r_k` = mismatch ratio; `θ = 3.5` = trigger; `C` = concession horizon (2900 steps); `L_max` = longest single attack burst; `window ≈ 200` steps. A working `C` must **exceed the longest unbroken pressure stretch a train can produce** yet stay **within the revision tolerance**.

**Example.** Attacks come in bursts of `L = 2400` steps with gaps between them; a genuinely changed world applies *unbroken* pressure. A frozen-*time* stopwatch (how long have I been frozen) can't tell these apart. The continuity stopwatch can: inside a train, each gap resets the clock, so the continuous-pressure stretch never exceeds `~L + window ≈ 2400 + 200 = 2600` — pick `C = 2900 > 2600` and the controller *never* concedes to a train. A changed world has no gaps, so the clock runs past 2900 and the controller revises. The C-sweep confirms the band: `C = 2000` fails exactly the three `L = 2400` cells (2000 < 2600); `C = 2900` passes all 9.

**Data.** **Exp 180** (rung 2): the read-only monitor separates identity displacement from quiet life at median **AUROC 0.894** (8/8 ≥ 0.8) and from a value-neutral scramble (Δ **0.536–0.719** in 8/8 pairs) — argmax-independently, with measured habituation (onset spike decays **3.3–5.2×** within a burst, 24/24). **Exp 187** (rung 5, MIXED): the stopwatch **INT-C2900** passes both bars in **9/9** cells (defense 7/8 each, Phase-R latency ~2703 ≤ baseline + 3000); the C-sweep traces the law exactly — **C1500 4/9, C2000 6/9 (fails the three L=2400 cells), C2900 9/9** — quoting the interval law `C ∈ (L_max + ~window, tolerance]`. **Exp 188** (rung 5 retest, POSITIVE): one *regulated* controller REG-TB (no oracle, no cell-specific tuning) also passes **9/9** at **exact defense parity** (identical 6/8 per cell, same two failing seeds) — its online tempo learning **never bound a single decision** (the *single-stretch ambiguity bound*: at fixed-`L` geometry judgment cannot beat the right clock). The honest counterpoint — **Exp 183** showed the *evidence-integral* concession surrenders mid-attack (`τ_concede ≈ 75 + 600/0.956 ≈ 703 < 800`, the surrender-schedule law), while fixed horizons H1200–H2400 pass both bars at 7/8 + 8/8 and H3000 at 7/8 + 6/8 (all four pass both bars) — the **commitment-as-config law**: at this richness, defense, where achievable, needs only a stopwatch.

**▸ In programmer terms.** A watchdog timer with a reset-on-idle: it fires only when *one* uninterrupted activity stretch exceeds a threshold, so bursty traffic (which always pauses) never trips it.

```python
class ContinuityStopwatch:               # INT-C2900 (Exp 187)
    def __init__(self, C=2900): self.C = C; self.stretch = 0; self.frozen = True
    def step(self, pressure):            # pressure: is the world pushing this step?
        if pressure:
            self.stretch += 1
        else:
            self.stretch = 0             # EVERY gap resets the clock
        if self.stretch > self.C:        # only an UNBROKEN run > C concedes
            self.frozen = False          # revise: a genuinely changed world
        return self.frozen               # frozen ⇒ v_{t+1} = v_t (RESIST)
```

`C` must beat the longest burst (so trains stay defended) but stay under the revision deadline (so real change is accepted) — the interval law is just `burst < C ≤ tolerance`.

---

## Hysteresis (de-assert thresholds; revision floor vs defense ceiling)

**Glossary.** **Hysteresis** is path-dependence: the threshold to *exit* a state differs from the threshold to *enter* it, so brief blips don't flip the system. Here the proposed repair is a **de-assert hysteresis** on the continuity clock: a pressure-off run of `h` snapshots or fewer does *not* reset the stretch. The repair is buildable only if a single `h` both keeps the defense (the gaps inside attack trains must reset the clock) and permits revision (changed-world flicker must *not* reset it). Exp 190 shows the two requirements pull `h` apart.

```
de-assert run ≤ h snaps  ⇒  do NOT reset the continuity stretch   (hysteresis)
DEFENSE needs:  attack-gap runs DO reset ⇒ gap_run > h
REVISION needs: onset-flicker runs do NOT reset ⇒ flicker_run ≤ h
BUILDABLE iff   ceiling_defense  ≥  floor_revision
MEASURED:       h ≤ 20  (defense)   vs   h ≥ 36  (revision)   ⇒  EMPTY
```

Symbols: `h` = hysteresis tolerance in 25-step snapshots; `gap_run` = de-assert run length inside an attack train; `flicker_run` = de-assert run during a changed-world onset. The interval `[floor_revision, ceiling_defense]` is empty when the floor exceeds the ceiling.

**Example.** Revision needs `h` large enough to bridge onset flicker: the per-seed rescue minima are `36 / 58 / 76`, so the smallest `h` giving even 6/8 revision is **36**. Defense needs `h` small enough that real attack-gaps still reset: E1's defense dies at `h = 21` (two mid-train concessions per session), C1 at 21–24, E3 at 46 — so the defense **ceiling** is `h ≤ 20`. Since `36 > 20`, no `h` satisfies both: the revision floor sits **16 snaps = 400 steps** above the defense ceiling. A complete sweep of 19 values returns **0/19 buildable**.

**Data.** **Exp 190** (the one authorized repair, NEGATIVE at design time): the counterfactual sweep through the real runner gives revision pass-count climbing 5/8 (h≤32) → **6/8 (h=36)** → 7/8 (h=58) → 8/8 (h=76), while defense decays through the *same* parameter — E1 LOST at **h≥21**, C1 at 21–24, E3 at 46. Buildability (rev ≥ 6/8 AND E1+E3 intact): **0/19** `h` values. The other candidate, total-active-time, died by arithmetic (defending E1 needs ~7,175 active steps; revision allows ~2,900 — empty by >4,000). This is the squeeze that names the **timescale-overlap law** (see Timescale overlap / the flicker tax).

**▸ In programmer terms.** A debounce/Schmitt-trigger interval. You want one debounce window that filters out *its* noise without swallowing *real* events — and here the noise and the events have overlapping durations, so no window works.

```python
def buildable(h, defense_ok, revision_ok):
    return revision_ok(h) and defense_ok(h)        # need BOTH at one h

# measured (Exp 190): revision_ok(h) iff h >= 36 ; defense_ok(h) iff h <= 20
hs = range(2, 21)                                    # the swept range
print(any(h >= 36 and h <= 20 for h in hs))         # False: 0/19 buildable
```

Hysteresis only helps when enter/exit thresholds can be separated; here the gap between "blip" and "real change" is gone.

---

## Timescale overlap (the flicker tax)

**Glossary.** Two control jobs share one event class — a pressure de-assert (a gap). Inside an attack train, a gap *should* reset the clock (so the defense never mistakes a train for permanent change). At a changed-world onset, the pressure *flickers* before settling, and each flicker also resets the clock — the **flicker tax**, which delays concession and breaks revision. The **timescale-overlap law**: changed-world onset de-assert runs *span and exceed* the attack-train gap scale, so any reset surface that classifies transient-vs-sustained by de-assert length faces a structural confound — bridging the flickers bridges the gaps.

```
flicker tax:  delay_concede ≈ Σ (discarded pre-stretches before the T₀ crossing)
TIMESCALE-OVERLAP:  onset de-assert runs ∈ [25, 2600] steps
                    attack-train gaps    ∈ [525, 1175] steps
   [525, 1175] ⊂ [25, 2600]   ⇒   FULL OVERLAP ⇒ no separating threshold
```

Symbols: a de-assert "run" is consecutive pressure-off snapshots × 25 steps; `T₀` = the regulated controller's adaptation budget; "discarded pre-stretches" = continuous-pressure runs thrown away by each reset before the budget is crossed.

**Example.** A changed world settles in over an onset that stutters: de-assert runs of, e.g., `[4, 2, 84, 23, 76]` snaps (seed 286, max 84) — up to **104 snaps = 2,600 steps across seeds** (the global cap, seed 287). The attack-train gaps are `21–47` snaps (**525–1,175 steps**). Because the gap range sits *inside* the onset-flicker range, any `h` that ignores a 1,000-step onset flicker (to revise on time) also ignores a 1,000-step attack gap (and so concedes mid-train). The continuity stopwatch's reset — which *buys* train defense — is the same event that *taxes* revision.

**Data.** **Exp 189** (rung 6, NEGATIVE / NEW INSIGHT): REG-TB defends the escalating trains no constant survives (E1/E3 **8/8**, the **κ-reach law** confirmed 6/6) but loses its own revision **5/8** — seeds 281/283/286 over the +3000 bar by **7/39/67 steps** — via the flicker tax (1–5 small completed pre-stretches discarded per seed; the committed max is seed 286's 4). The same flicker hit the constant C2900's revision on this seed block too (5/8, vs 7/8 on the prior block). **Exp 190** measured the overlap directly: onset de-assert runs **25–2,600 steps** vs attack-train gaps **525–1,175 steps** — FULL OVERLAP — establishing the **timescale-overlap law**: the flicker tax is structural to the reset surface, not tunable away (see Hysteresis). The chapter's final shape: at fixed-`L` geometry regulation *ties* the constant (the ambiguity bound), at variable-`L` it *uniquely defends* (κ-reach) but *cannot revise in time* (the flicker tax).

**▸ In programmer terms.** Two signals you must distinguish by burst-gap length have *overlapping* gap distributions — so no debounce threshold separates them, and the same reset that filters one corrupts the other.

```python
onset_flicker_runs = range(25, 2601)     # changed-world onset (Exp 190)
attack_gap_runs    = range(525, 1176)    # gaps inside attack trains
overlap = set(attack_gap_runs) & set(onset_flicker_runs)
print(bool(overlap))                     # True: every gap length is also a possible flicker
# => no threshold h cleanly separates "transient flicker" from "real attack gap"
```

When two regimes overlap in the feature you'd threshold on, no threshold classifier works — the failure is in the data, not the controller.

---

## Regulation vs a constant (the universal-constant law)

**Glossary.** The **universal-constant law** is the anti-regress principle applied to the program's own design constants: **regulation (a self-tuning controller) is only *necessary* where no feasible constant covers all regimes.** It is run as an explicit *kill-test* against every claimed layer: if a fixed constant matches the regulator on *all* declared bars, the "layer" is **config**, not agency. A layer earns the name only when some regime exists that demands adaptation a constant cannot supply.

```
LAYER is load-bearing only if  ∄ constant c :  R(c) ≥ R_layer  AND  Q(c) ≥ Q_layer
   R = defense (resist transient pressure) ;  Q = revision (accept sustained evidence)
UNIVERSAL-CONSTANT LAW:
   regulation is NECESSARY  ⟺  no feasible constant covers all regimes
```

Symbols: `c` = a fixed constant (window `K`, freeze horizon `H`, concession `C`); `R(c)`, `Q(c)` = the constant's defense / revision pass-counts; `R_layer`, `Q_layer` = the regulator's. If a constant dominates on both, the regulator is redundant.

**Example.** Test a regulator vs the best constant on (defense, revision). N3: the regulator scores combined **1.0**; the best fixed window scores **0.7** — the constant *loses*, so the layer is load-bearing (Exp 167). N4 identity: the regulated concession scores defense **2/8** while fixed `H1200–H2400` score **7/8** + revision **8/8** (and `H3000` 7/8 + 6/8, still passing both bars) — the constant *dominates*, so commitment is config (Exp 183). The decision is mechanical: compare the two columns; a constant that ties or wins on both kills the layer claim.

**Data.** **Exp 173** (NEGATIVE) names **THE UNIVERSAL-CONSTANT LAW**: in a multi-tempo world built to force the regulator to matter, `K = 16` covers BOTH tempos and the time-decay regulator converges onto it *bit-identically* — "horizon regulation is only necessary where no feasible constant covers all regimes, and on this body such a world is not constructible." It lands at the identity level in **Exp 183** (**commitment-as-config law**: fixed-H freeze passes the resist/revise tradeoff, regulated commitment measured strictly worse, 2/8). It is then **refined, not broken** in **Exp 187/188**: on the *right* surface (continuous-pressure stretch) the constant **INT-C2900** and the regulator **REG-TB** *tie* at 9/9 (identical 6/8 per-cell defense, same two failing seeds) — "constants cover everything tried, on the right surface" (the single-stretch ambiguity bound). The one place regulation *uniquely* wins (variable-`L` defense, **Exp 189** κ-reach law) it then *fails* revision (the flicker tax) — so even there no controller beats the dilemma, and the law's boundary is exactly mapped.

**▸ In programmer terms.** Before shipping an adaptive/auto-tuning component, A/B it against a well-chosen constant on *every* requirement. If a constant matches it everywhere, the adaptivity is dead code — delete it (or admit it's config).

```python
def layer_is_load_bearing(regulator, constants, regimes):
    R_reg = defense(regulator, regimes); Q_reg = revision(regulator, regimes)
    for c in constants:                                  # try every feasible constant
        if defense(c, regimes) >= R_reg and revision(c, regimes) >= Q_reg:
            return False        # a constant ties/wins both -> it's CONFIG (Exp 173/183)
    return True                 # no constant matches -> regulation is necessary (Exp 167)
```

The kill-test: a capability is only a capability where a regime exists that *demands* it.

---

## Signal detection & type-2 AUROC (does confidence track correctness?)

**Glossary.** An **ROC curve** is traced by sweeping a decision threshold over a continuous **score** and plotting true-positive rate against false-positive rate; the **AUROC** (area under it) has a rank interpretation — it equals the probability that a randomly chosen positive outranks a randomly chosen negative (the Mann–Whitney U statistic, normalized). 0.5 = chance, 1.0 = perfect ranking. **Type-1** asks *can the system detect the signal?* (score = the decision evidence; positive = signal present). **Type-2** (metacognition) asks *does the system's own confidence predict its own correctness?* — the score is the agent's **confidence**, the "positive" class is its **correct** trials and the "negative" class its **errors**. `meta-d′` is the signal-detection cousin: the type-1 sensitivity `d′` that the confidence ratings *imply* — `meta-d′ > 0` means confidence carries information about correctness (the N2 test in docs/specs/n-order-self-modeling.md). The repo's headline metacognition metric is type-2 AUROC.

```
AUROC = P( score(positive) > score(negative) )                 (rank / Mann–Whitney form)
      = (1/(n₊·n₋)) · Σ_{i∈pos} Σ_{j∈neg} [ s_i > s_j ]        (count winning pairs / all pairs)

type-1 AUROC :  positive = signal-present ,  score = decision evidence
type-2 AUROC :  positive = CORRECT trial ,  score = own CONFIDENCE c
            = P( c_correct > c_error )                          (metacognition: does c track correctness?)
meta-d′ > 0  ⟺  confidence is informative about correctness    (the N2 prereq)
```

Symbols: `s_i` = score on trial `i`; `n₊`, `n₋` = counts of positive / negative trials; `[·]` = 1 if true else 0 (ties count ½); `c` = the agent's confidence; `meta-d′` = the type-1 `d′` implied by the confidence ratings. Type-2 AUROC is just the type-1 rank statistic applied to (confidence, was-it-correct) instead of (evidence, was-signal-present).

**Example.** Six trials. Confidences on the **correct** trials: `{0.9, 0.7, 0.6}`; on the **error** trials: `{0.8, 0.4, 0.3}`. There are `n₊·n₋ = 3×3 = 9` correct/error pairs. Count how many a correct trial wins: 0.9 beats all three errors (3); 0.7 beats {0.4,0.3} (2); 0.6 beats {0.4,0.3} (2). Winning pairs = `3+2+2 = 7`, so type-2 AUROC = `7/9 ≈ 0.778` — confidence outranks correctness about 78% of the time, well above the 0.5 chance line. If instead confidence were *anti*-calibrated (high on errors), the same count would fall below 0.5 — an **inversion**, exactly what the value-neutral scramble produces (`AUROC_B ≈ 0.262`, Exp 180).

**Data.** **Exp 157** built the confidence half of N2: the per-place expected-uncertainty channel (a per-cell EWMA of the creature's own hits/misses) hit type-2 AUROC **0.8025 pooled** (per fork 0.7915–0.8062), versus **0.5594 pooled** (per fork 0.5237–0.5787) for the natural channel (max predictive probability, C-old) — new > old in **8/8** forks, calibration `r` 0.9544–0.9882 (POSITIVE; "0.80 pooled vs 0.56 for the natural channel" is the entry's own headline). The 0.80 is near the world's **analytic ideal ≈ 0.81**. **Exp 159** re-confirmed under per-fork randomization at AUROC **0.7989–0.8202 in 8/8 (pooled 0.8091)** — note the per-fork values and the pooled value here barely differ (pooling over trials moves only the third decimal, Exp 161's observation), so pooled-vs-per-fork is a reporting choice, not a result; the natural channel sat at chance on uniform noise (0.476–0.513), since uniform noise has no per-cell differential to detect. **Exp 180** is the N4 identity monitor's type-2/separation AUROC: a read-only linear-drift self-predictor separates identity displacement from quiet life at median AUROC **0.894 (range 0.859–0.911, all 8/8 ≥ 0.8)**, with the value-neutral scramble *inverted* below chance (`AUROC_B` median **0.262**, Δ 0.536–0.719 in 8/8 pairs). This is the headline metric of the N2/N3/N4 metacognition line (see Metacognitive controllers and N4 identity).

**▸ In programmer terms.** Type-2 AUROC = run a rank/AUC over (confidence, correct?) instead of (score, label) — `sklearn.metrics.roc_auc_score(y_correct, confidence)`. The rank form is a double loop counting winning pairs.

```python
def auroc(scores_pos, scores_neg):                 # rank / Mann–Whitney form
    wins = sum(1.0 if sp > sn else 0.5 if sp == sn else 0.0
               for sp in scores_pos for sn in scores_neg)
    return wins / (len(scores_pos) * len(scores_neg))

# type-1: positives = signal-present trials, score = decision evidence
# type-2 (metacognition): positives = CORRECT trials, score = own confidence
correct_conf = [0.9, 0.7, 0.6]; error_conf = [0.8, 0.4, 0.3]
print(auroc(correct_conf, error_conf))             # 0.7778  -> confidence tracks correctness

# equivalently, with labels (1 = was correct), score = confidence:
#   from sklearn.metrics import roc_auc_score
#   roc_auc_score([1,1,1,0,0,0], [0.9,0.7,0.6,0.8,0.4,0.3])  # 0.7778
# 0.80 vs 0.56 (Exp 157) ; 0.894 (Exp 180) ; below 0.5 == anti-calibrated inversion (0.262, Exp 180)
```

AUROC below 0.5 is not "bad detection" — it is a *flipped* detector (the scramble's `AUROC_B ≈ 0.26` means confidence is reliably *higher* on errors); flip the sign and you recover a 0.74 detector.

---
