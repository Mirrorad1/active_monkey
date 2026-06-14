# Evolutionary & population dynamics

> This is the math of the `ecology/` substrate — a reproducing, multi-generation toy world where the **environment**, not a scorer, decides who breeds. The experiments (Exp 194–207) ask one falsifiable question: *under what conditions does a costed sense-like trait become increasingly selected — an organ — rather than merely useful when gifted?* The tools below are the standard machinery of evolutionary theory (genotype/phenotype, fitness, mutation, selection gradients, fitness valleys, drift) applied to a single evolvable trait, the **thermosense intensity** `h ∈ [0,1]`. The arc's durable answer is **negative**: across seven distinct levers the local selection gradient at the resident never turns sufficiently positive, so a costed sense stays primitive even when a strong sensor is genuinely fitter in bulk.

## Genotype, phenotype, and the gene pool

**Glossary.** A **genotype** is the heritable record of a creature's parameters; a **phenotype** is the realized, time-varying state those parameters produce in the world (position, energy, age). The headline heritable trait of this arc is the **thermosense intensity** `h`, the precision/sharpness of the heat (or, by re-skin, foraging/niche) sensor. The **gene pool** is the multiset of genotypes carried by the living — and, for an honesty-critical distinction, by the *newborns* (see Heritability vs survivor bias).

```
g = (movement_cost, baseline_metabolic_cost, energy_capacity, …, h, inefficiency)   the genotype (a frozen record)
h = thermosense_intensity ∈ [0, 1]        the evolvable sensor trait (0 = organ absent)
phenotype:  state_t = (pos_t, energy_t, age_t, learned_map_t)   derived by living, NOT inherited
gene pool:  the multiset { g_i : creature i alive }      (or { g_i : creature i born in window } for the newborn pool)
```

- `g` — genotype, a `frozen` dataclass of ~15 trait fields, each clamped to `[lo, hi]`.
- `h` — `thermosense_intensity`, the trait under study; active only when `h > threshold ≈ 0.05`.
- `inefficiency` — `thermosense_inefficiency ∈ [0.2, 1]`, the upkeep multiplier (see Fitness).
- phenotype state is *recomputed each step from the world*; only `g` crosses generations.

**Example.** A founder has `h = 0.00` (no organ). It reproduces; the child inherits a copy of `g`, then each trait is jittered (next section). Suppose the child draws `h = 0.04`. That `0.04` is now part of the gene pool. If a sensor is "active" only above `h = 0.05`, this child still expresses **no** organ — it carries a silent allele. The phenotype (where it walks, how much it eats) depends on `0.04` *only* if the trait crosses the activation threshold and changes its routing.

**Data.** The founder genotype is `h = 0.0`, `inefficiency = 1.0` (`ecology/genotype.py: founder()`). Exp 194 established the substrate: a balanced world persists to horizon 600 across **8–12 generations** with **628/622/509 births** across three seeds, and 60 trait-shift items moved ≥1σ (e.g. `baseline_metabolic_cost 0.5000 → 0.0236`) — but Exp 194 was MIXED, because no zero-mutation control was run, so *drift vs selection was not distinguished* (see Genetic drift vs selection).

**▸ In programmer terms.** Genotype is the serialized config you copy into a child object; phenotype is the mutable runtime state you never serialize. The gene pool is just a list comprehension over the population.

```python
from dataclasses import dataclass, replace

@dataclass(frozen=True)            # genotype = immutable, heritable record
class Genotype:
    h: float = 0.0                 # thermosense_intensity ∈ [0,1]
    inefficiency: float = 1.0
    # ... ~13 more trait fields ...

class Creature:                    # phenotype = mutable runtime state (NOT inherited)
    def __init__(self, g: Genotype):
        self.g = g                 # the only thing that crosses generations
        self.energy, self.age, self.pos = 10.0, 0, (0, 0)
        self.learned_map = {}      # built by living; dies with the creature

gene_pool = [c.g.h for c in population if c.alive]      # the living pool
newborn_pool = [c.g.h for c in population if c.born_in_window]  # the honest pool
```

---

## Fitness as realized reproduction; the environment as selector

**Glossary.** There is **no fitness function written down** — fitness is *emergent*. A creature reproduces only when, by its own inherited rules, it is old enough and rich enough; it dies when energy hits zero. Realized fitness is therefore the **births per step** a strategy actually achieves under metabolic and competitive load.

```
w_i = B(a_i, z, h_i, θ_i, ρ) − C_h(h_i) − C_θ(θ_i)      realized fitness (§2 of the sense-axis note)
B   = gross intake / births from acting in the world      the installed benefit
C_h(h) = floor + inefficiency × h        sensor upkeep, charged every tick the organ is active (floor>0)
death:  energy ≤ 0  ⇒  "starvation"      the ONLY death cause (Exp 194); aging added in Exp 195
reproduce iff:  age ≥ maturity_age  AND  energy ≥ reproduction_energy_threshold
child energy = parent energy × transfer_fraction;  parent pays transfer + complexity overhead
```

- `w_i` — realized per-capita fitness of creature `i` (it is *measured*, never assigned).
- `z` — hidden world state (fresh vs depleted food, the drifting comfort band, residue, niche class).
- `θ` — the controller's ability to *use* the sensor (Exp 207); `ρ` — ecological state (density/depletion/crowding).
- `C_h(h) = floor + inefficiency·h` — the organ's upkeep; with `floor = 0` and `inefficiency = 0.2`, a sensor of `h = 0.45` still burns `0.09` energy/tick. In every cited config `cost_floor = 0`, so the **never-free** guarantee rests on `inefficiency ≥ 0.2` alone (cost `= 0.2·h > 0` for any active `h`); the engine docstring's `floor > 0` phrasing is the more general case, not the regime these experiments run.

**Example.** A creature has `h = 0.45`, `inefficiency = 0.2`, `floor = 0`. Each tick it pays `C_h = 0 + 0.2 × 0.45 = 0.09` energy just to run the sensor, on top of `baseline_metabolic_cost`. If sharper sensing lets it intake an extra `0.04`/tick (`B` gain) but costs `0.09`/tick, its net is `0.04 − 0.09 = −0.05`/tick: it starves faster than a crude rival and leaves fewer offspring. Nobody penalizes its sensor — the *energy budget* does.

**Data.** Exp 194: 1279 balanced deaths were **100% starvation, 0 ranking deaths** — confirming the environment, not an evaluator, selects (a verified design invariant). Scarcity bites overwhelmingly: scarce-vs-balanced final-population reduction was **100.00% / 95.48% / 100.00%**, extinct in 2/3 seeds by step 32. The "forced benefit is real" anchor: Exp 200's engine test confirmed a forced strong forager (`h = 0.8`) reproduces **~4× more** than a no-organ creature — yet that organ still does not evolve (see Forced-vs-evolvable gap).

**▸ In programmer terms.** No `reward = f(h)` line exists anywhere — that is the binding anti-cheat (`assert_no_direct_h_reward`). Fitness is whatever falls out of the loop. The upkeep is a subtraction inside the per-step metabolism.

```python
def step(creature, world):
    creature.energy -= creature.g.baseline_metabolic_cost          # base metabolism
    if active(creature.g):                                          # organ on?
        creature.energy -= FLOOR + creature.g.inefficiency * creature.g.h   # C_h: never free
    creature.energy += eat(creature, world)                        # B: emergent, NOT f(h)
    if creature.energy <= 0:
        creature.alive = False                                     # death = starvation
    elif creature.age >= creature.g.maturity_age and \
         creature.energy >= creature.g.reproduction_energy_threshold:
        spawn_child(creature)                                      # realized fitness = a birth
```

---

## Mutation: heritable transmission plus noise

**Glossary.** Reproduction is asexual: the child copies the parent's genotype, then each trait is independently perturbed by Gaussian noise scaled to the trait's range, and clamped back into bounds. The **founder value** is the explicit starting allele (`h = 0` for de-novo emergence, `h = 0.10` or `0.20` for "is it retained?" tests).

```
g_child = clamp( g_parent + ε ),     ε_k ~ 𝒩(0, σ_k),   σ_k = rate × (hi_k − lo_k)
clamp(v) = max(lo, min(hi, v))       (and round for integer traits)
rate ≈ 0.05      (per-trait mutation rate)
```

- `ε_k` — the mutation increment for trait `k`, mean 0.
- `σ_k = rate × (hi_k − lo_k)` — step size proportional to the trait's span. For `h ∈ [0,1]`, `σ_h = 0.05 × 1 = 0.05`.
- `clamp` — projects back into `[lo_k, hi_k]`; integer traits are rounded.

**Example.** Parent has `h = 0.10`. The child draws `ε_h ~ 𝒩(0, 0.05)`. Say `ε_h = +0.12`, giving a raw `0.22`, clamped to `0.22` (still in range). Most draws are small: about 68% land in `±0.05`, so a single mutation rarely jumps far. This is why **small steps must pay** for a trait to climb — a typical child is `±0.05` from its parent, not a leap to `0.45` (this is the crux of the fitness valley, below).

**Data.** Exp 199 verified the mutation supply is *not* the bottleneck: high-intensity mutants arise transiently (individual `h` up to **0.41–0.55** in the climb arms, **1.00** in the seeded arm) but **never fix** in the gene pool — "the variance is actively CULLED, the signature of a genuine selective fitness valley (not a mutation-supply problem)." Exp 200 likewise: max individual `h` reached **0.56–0.66** in every arm including the control, but the gradient could not sweep them.

**▸ In programmer terms.** Mutation is `parent_config + np.random.normal(0, sigma)` per field, then a clip. The determinism contract requires the rng draws happen in a fixed field order; gated traits (`thermosense`) skip their draw entirely when disabled, to keep the byte-identical `events_hash`.

```python
import numpy as np

def mutate(g, rng, rate=0.05):
    child = {}
    for k, v in g.items():
        lo, hi = TRAIT_BOUNDS[k]
        sigma = rate * (hi - lo)                  # σ scales with the trait's span
        child[k] = float(np.clip(v + rng.normal(0.0, sigma), lo, hi))
    return Genotype(**child)

# founder value: the explicit starting allele
founder_de_novo  = Genotype(h=0.00)   # Exp 198 arm A — survivor bias impossible
founder_seeded   = Genotype(h=0.20)   # Exp 197 — tests RETENTION, not emergence
```

---

## Heritability vs survivor bias

**Glossary.** A trait can look selected-for in the **living** population for two different reasons: (a) genuine **heritable selection** — sensing lineages out-breed, raising the trait in the *gene actually passed on*; or (b) **survivor bias** — sensing creatures merely live longer, so they pile up among the living without breeding more. The discriminating instrument is the **newborn-intensity tracker**: a trait nobody is *born* with cannot be survivor-biased.

```
liv_gap = ⟨h⟩_living(treatment) − ⟨h⟩_living(control)          inflated by survivor bias
new_gap = ⟨h⟩_newborn(treatment) − ⟨h⟩_newborn(control)        the heritable signal only
⟨h⟩_newborn = mean over creatures born in [t_start, t_end]      the gene-pool measure
de-novo control:  start ALL arms at h = 0  ⇒  any newborn h is PURE inheritance
```

- `⟨h⟩_living` — mean intensity over the standing (alive) population — conflates selection, survivor bias, and density.
- `⟨h⟩_newborn` — mean intensity over recently-*born* creatures — isolates what the gene pool transmits.

**Example.** Suppose under heat, living creatures average `h = 0.50` but newborns average only `h = 0.045`. The huge living value is mostly old survivors who happened to carry the organ; the genes flowing to the next generation barely moved. Now start everyone at `h = 0`: any newborn with `h > 0` *had to inherit it via a selected mutation*, so a newborn gap is unambiguous heritability.

**Data.** Exp 197 (seeded founder `h = 0.20`): living thermosense active-fraction **0.49–0.53** (treatment) vs **0.03–0.10** (control), `liv_gap` **0.4252/0.4732/0.4714/0.4638/0.4266` — dramatic. But newborn intensity was only **0.042–0.050** (treatment) vs **0.028–0.033** (control), `new_gap` **0.0088–0.0222** — *weak*. The age-stratified fingerprint: old thermosense `0.0682 > young 0.0383`. Exp 198 closed the confound by starting from zero: arm A (temp-ON, from 0) newborn `h = 0.0488/0.0416/0.0472/0.0456/0.0516` vs arm C (temp-OFF, from 0) `0.0303/0.0286/0.0299/0.0290/0.0280`, A−C gaps `0.0130–0.0236` — **pure heritable**, ~`0.045 vs 0.029`. So Exp 197's weak heritable signal was confirmed real but survivor-bias-*dominated*. (Exp 196's standing-complexity decline was POSITIVE but explicitly *left open* between selection, survivor bias, and density for the same reason.)

**▸ In programmer terms.** Track two means, not one. The living mean is a snapshot bug magnet (older = over-represented); the newborn mean is the clean signal.

```python
def heritability_gap(pop, t, window=2000):
    living  = [c.g.h for c in pop if c.alive]
    newborn = [c.g.h for c in pop if t - window <= c.birth_t <= t]
    return {
        "living_mean":  sum(living)  / len(living),    # contaminated by survivor bias
        "newborn_mean": sum(newborn) / len(newborn),   # the heritable signal
    }
# Exp 198 trick: if EVERY founder starts at h=0, newborn_mean>0 ⇒ inheritance, full stop.
```

---

## The selection gradient

**Glossary.** A population's mean trait can rise only when the **local selection gradient** is positive at the current resident value — the marginal benefit of a small heritable step must beat its marginal cost. The arc measures this *directly* (Exp 203, 205, 207) with a **pairwise** head-to-head assay: clone equal numbers of a resident (`h_res`) and a slightly-mutated invader (`h_inv`) into one world and see which out-multiplies the other.

```
g(h) = dE[w | h] / dh > 0   ⇔   dB/dh > dC/dh      (w = B − C; the deterministic gradient)
to actually climb:  g(h) must ALSO beat drift + noise + transmission-erosion   (separate stochastic forces)
pairwise selection coefficient:  s = d ln(N_inv / N_res) / dt
gradient sign read from:  invader_won = (# seeds where invader out-multiplies resident) / (# seeds)
the strict POSITIVE bar:  invader_won ≥ 7/8  (and survives the learning-rate-freeze control)
```

- `g(h)` — the local realized gradient; its **sign** is what decides evolvability.
- `s` — the pairwise selection coefficient: the log-growth-rate difference of invader vs resident, with the founder cold-start "differenced out." It is not an analytic derivative — it is *estimated* as an OLS slope of `ln(N_inv/N_res)` vs `t` (see *Invasion fitness, adaptive dynamics & the ESS* below, and *Fitting a trend: ordinary least squares and the log-growth slope* in `07-statistics-and-experimental-method.md`).
- `invader_won` — the win fraction over seeds; the founder-lottery is averaged away.

**Example.** Stand a resident at `h_res = 0.10` and an invader at `h_inv = 0.15`. Seed 8 equal-founder worlds. If the invader ends with more lineage in 7 of 8, `invader_won = 7/8 ≈ 0.875` and the step *up* pays — a positive local gradient. If it wins only 2/8, the step is disfavoured: `g(0.10) ≤ 0`.

**Data.** Exp 203 (band-staleness FORAGE) found the **first positive local gradient of the arc**: `0.10 → 0.15` won **7/8** (auc 0.872, `s_mean +0.01035`); `0.10 → 0.06` (a step down) won only **2/8**; `0.10 → 0.45` won 7/8. It survived the LR-freeze control at **6/8** — one seed short of the strict 7/8 bar, hence MIXED. Exp 205 (survivable residue): pairwise `0.10 → 0.15` won only **1–3/8** across losses (≤0, no climb). Exp 206 (rotating niche): `0.10 → 0.15` won **3/8** (auc 0.509, neutral), `0.10 → 0.30` won **1/8** (against). Exp 207: `dB/dh = −0.041` (high θ) and `−0.046` (low θ) — pure cost, negative gradient at both controller levels.

**▸ In programmer terms.** Don't measure the trait at equilibrium — measure the *slope* with an A/B race of two clones in the same world, repeated over seeds to average the lottery out.

```python
def pairwise_gradient(h_res, h_inv, seeds):
    wins = 0
    for seed in seeds:
        world = seed_equal_founders(h_res, h_inv, seed)   # equal counts, one shared world
        world.run(horizon=H)
        n_res = sum(lineage_of(c) == "res" for c in world.alive)
        n_inv = sum(lineage_of(c) == "inv" for c in world.alive)
        wins += (n_inv > n_res)
    return wins / len(seeds)        # ≥ 7/8 ⇒ strict positive local gradient
# Exp 203 FORAGE: pairwise_gradient(0.10, 0.15, seeds_50_57) == 7/8  (first positive in the arc)
```

---

## The cross-partial and the falsified "2-D fitness valley"

**Glossary.** Exp 207's *premise* was a **two-dimensional fitness valley**: maybe the sensor `h` never paid because the **controller** `θ` (the policy that uses the sensor) couldn't exploit it, so co-evolving both might climb a ridge neither reaches alone. The signature of such synergy is a **positive cross-partial** `∂²B/∂h∂θ` — the two traits must be *complements*. A cheap monomorphic **corner-grid** pre-flight (births at low/high `h` × low/high `θ`) measures the cross-partial before committing a full batch. The premise was **FALSIFIED**.

```
hypothesized:  w(h, θ) = R · σ(κ·h·θ − d) − C_h(h) − C_θ(θ)    a 2-D valley (synergy)
synergy test:  ∂²B/∂h∂θ > 0   AND   θ does not pay alone   AND   h pays at high θ      (gate G0)
discrete cross-partial:  Δ²B = [B(h₁,θ₁) − B(h₁,θ₀)] − [B(h₀,θ₁) − B(h₀,θ₀)]
```

- `σ(·)` — a logistic squash; `κ` the coupling gain, `d` an offset; `R` the reward scale.
- `∂²B/∂h∂θ` — the cross-partial: positive means a sharper sensor is *worth more* when the controller acts harder (true complementarity).
- `Δ²B` — its discrete corner-grid estimate.

**Example.** Take the corner grid `B(h, θ)`. With `h₀=0.10, h₁=0.45, θ₀=0.6, θ₁=6.0`, following the Glossary formula (`h₁` block minus `h₀` block):
`Δ²B = [B(0.45,6.0) − B(0.45,0.6)] − [B(0.10,6.0) − B(0.10,0.6)]`
`    = [0.2196 − 0.0678] − [0.2606 − 0.1134] = 0.1518 − 0.1472 = +0.0046 ≈ 0.`
(The sign flips with convention — reverse the two blocks and it reads `−0.0046`; only the near-zero magnitude matters.) No synergy: turning `θ` up helps about the same whether the sensor is crude or sharp. And `θ` already pays alone — `dB/dθ` at low `h` = `0.2606 − 0.1134 = +0.147` — while `h` *loses* births at both `θ` levels.

**Data.** Exp 207 corner-grid `B` (births/step): `[h0.10: θ0.6→0.1134, θ6.0→0.2606 | h0.45: θ0.6→0.0678, θ6.0→0.2196]`. Result: **cross-partial ≈ 0** (±0.0046, ~32× smaller than the θ-alone effect); **dB/dθ @ low h = +0.147** (controller pays alone via herd-escape); **dB/dh = −0.041 (high θ), −0.046 (low θ)** (sensor pure cost at every θ). Anti-cheat guards passed: at `confusion=0` cost-OFF, intake was byte-identical across `h` (`eaten 79368.597 = 79368.597`); at `niche_weight=0`, `8310.901 = 8310.901`. G0 failed all three conjuncts → **DESIGN-STAGE NEGATIVE**, blind-verified AGREE. A ~40-run pre-flight killed a would-be 6-arm × 5-seed × 8000-step batch (methodological lesson L28: pre-flight the *premise*, not just the runtime).

**▸ In programmer terms.** Before launching a 2-D co-evolution sweep, compute the discrete mixed second difference on four corner cells. If it's ~0, the two knobs are not complements and the big experiment can only reproduce "θ climbs, h decays."

```python
def cross_partial(B):                 # B is a {(h, θ): births_per_step} corner grid
    h0, h1, t0, t1 = 0.10, 0.45, 0.6, 6.0
    return (B[h1, t1] - B[h1, t0]) - (B[h0, t1] - B[h0, t0])   # Δ²B

B = {(0.10,0.6):0.1134, (0.10,6.0):0.2606, (0.45,0.6):0.0678, (0.45,6.0):0.2196}
assert abs(cross_partial(B)) < 0.01          # ≈ 0  ⇒  NO 2-D valley; do NOT run the batch
dB_dtheta_lowh = B[0.10,6.0] - B[0.10,0.6]   # +0.147  → θ pays ALONE (herd-escape)
dB_dh_highth   = B[0.45,6.0] - B[0.10,6.0]   # < 0     → sensor pure cost even acted on hard
```

---

## The fitness valley and benefit saturation

**Glossary.** A **fitness valley** is a region where the trait must cross a *dip* in fitness to reach a higher peak — the resident sits on the near rim, a small step *down* into the valley, and only a leap reaches the far peak. In this arc the valley arises from **benefit saturation**: a crude sensor already grabs the easy part of the benefit, so the **marginal** return to precision is small and concave while the cost rises **linearly**.

```
fitness valley:   B(0.60) − B(0.00) ≫ 0       a gifted strong sensor helps a lot
but               B′(0.08) − C′(0.08) ≤ 0       a small heritable step does not pay
saturation:       B(h) concave  ⇒  dB/dh falls as h rises;   C_h(h) = floor + inefficiency·h  linear
net fitness:      argmax_h [ B(h) − inefficiency·h ]  =  the primitive founder, not the bulk optimum
```

- `B(h)` concave — diminishing returns to precision (the easy benefit is captured cheaply).
- `C_h(h)` linear — `inefficiency·h` grows at a constant rate, so eventually `dC/dh > dB/dh`.

**Example.** Exp 201's returns-curve probe: across frozen precision `p = 0.10 → 0.60`, gross intake rose `0.068 → 0.089` — precision *does* help tracking. But the marginal benefit fell `+0.043 → +0.018` (concave), while upkeep costs a flat `0.20/unit p`. Net fitness `intake − 0.2p` is therefore maximised at the **primitive founder** and declines with precision. A reader can check the endpoints: the *total* extra intake from `p = 0.10` to `p = 0.60` is `0.089 − 0.068 = 0.021`, but the extra *cost* over that same `Δp = 0.50` is `0.20 × 0.50 = 0.10` — nearly 5× larger. Climbing makes you worse off.

**Data.** Exp 199 (avoidance): the deck stacked *for* the organ (cheap, near-perfect info), yet primitive arms stayed at **0.0505–0.0619**, the seeded `h = 0.50` arm **decayed** to 0.0562–0.1119, and **2/5 went extinct** carrying it; the noise sweep was *flat* (`V4 0.0558 ≈ V1 0.0619`) — not a noise problem, a structural valley. Exp 200 (foraging): a benefit **real-in-isolation** (4× reproduction when gifted) was **invisible in evolution** — gene pool stuck at `~0.08` vs control `~0.06`; "no Goldilocks gradient." Exp 204/205 (residue false-positive): a crude sensor *avoiding a costly mistake* finally made a strong sensor genuinely fitter in bulk — the first functional monomorphic optimum `h* = 0.60` — yet pairwise `0.10 → 0.15` won only **2/8** (Exp 204) / **1–3/8** (Exp 205); only the big leap `0.10 → 0.45` paid (5/8). The valley, isolated.

**▸ In programmer terms.** Two opposing curves: a concave benefit and a linear cost. Their difference peaks at the *founder*, so small steps lose. The "gift" (endpoint difference) and the "gradient" (local slope) are different quantities — measuring the gift mis-predicts evolvability.

```python
def net_fitness(h, B, inefficiency=0.2, floor=0.0):
    return B(h) - (floor + inefficiency * h)      # concave B minus linear cost

# Exp 201 probe shape (illustrative): B saturates, cost is linear
B = lambda h: 0.068 + 0.07 * (1 - 2.718 ** (-4 * h))   # concave, diminishing returns
import numpy as np
grid = np.linspace(0, 0.6, 13)
star = grid[np.argmax([net_fitness(h, B) for h in grid])]
# star ≈ the primitive founder — NOT 0.60 — because dB/dh < dC/dh past the rim: a valley.
```

---

## Genetic drift vs selection

**Glossary.** **Drift** is random change in trait frequency from finite-population sampling noise, with no fitness cause; **selection** is systematic change driven by `g(h) ≠ 0`. At small or collapsed populations, drift dominates and can manufacture spurious trait–population correlations. The arc's drift diagnostic is `corr(pop, h)`: a *negative* correlation (high `h` only when the population is small) is a **drift artifact**, not a Red-Queen selection signal.

```
selection:  Δ⟨h⟩ ∝ g(h) · Var(h)      systematic, gradient-driven (Price/breeder's-eq. flavour)
drift:      Var(Δ⟨h⟩) ∝ 1 / N_eff      sampling noise; grows as the population shrinks
diagnostic: corr(pop, h) < 0  at collapsed N  ⇒  the high-h values are a small-N drift artifact
drift floor: a minimum N_eff below which a reading is NOT a clean selection datum
```

- `N_eff` — effective population size; small `N_eff` ⇒ large drift variance.
- `corr(pop, h)` — Pearson correlation between standing population and mean intensity, across arms/seeds.

**Example.** An arm shows mean `h = 0.13` — looks like selection for the sensor. But its populations are only **214–461**, and `corr(pop, h) = −0.82`: the high `h` appears *only* where the population is smallest. That is the signature of drift at low `N`, not a real advantage — a large population with the same gradient would not show it.

**Data.** Exp 202 (interference competition): the COMPETE arm decayed to `h = 0.0285` at **healthy** populations (**1033–1080**, well above the ~300 drift-floor) — genuine suppression below the 0.10 founder. The NO_SHUFFLE arm reported a *higher* `h = 0.1303`, but at **smaller, drift-prone** pops (214–461), with **corr(pop, intensity) = −0.82** — "high intensity only at low pop = DRIFT," predeclared, *not* selection. Contrast Exp 206, where the niche arm decayed to `h = 0.0271` at a **healthy** pop (min 586) with **corr(pop, h) = −0.146** — i.e. *not* drift, so the failure is cleanly the gradient (≤0), not demography.

**▸ In programmer terms.** A trait value is only trustworthy as a selection datum above a minimum population. Always log `corr(pop, h)`; a strong negative correlation means your "signal" is small-sample noise.

```python
import numpy as np

def is_drift_artifact(pop_series, h_series, drift_floor=300):
    r = np.corrcoef(pop_series, h_series)[0, 1]
    healthy = np.mean(pop_series) > drift_floor
    return (r < -0.5) and not healthy          # high h only at low N ⇒ drift, not selection

# Exp 202 NO_SHUFFLE: corr = -0.82, pops 214-461  → True  (drift, discard as selection signal)
# Exp 202 COMPETE:    pops 1033-1080             → suppression is REAL (below founder, healthy N)
```

---

## The monomorphic optimum h* and convergence to an attractor

**Glossary.** The **monomorphic optimum** `h*` is the single trait value that, when an *entire* population is fixed at it, maximises the population's carrying capacity `N*(h)` — the bulk/installed optimum. It is found by clamping a pure population at each grid value and reading `argmax_h N*(h)` (a Tilman-style `R*` reading uses `argmin_h R*(h)`, the strategy that draws the limiting resource lowest). An **attractor** is a value the evolving mean *converges to from both directions* — from above and from below.

```
N*(h)  =  monomorphic carrying capacity of a pure-h population
h*     =  argmax_h N*(h)         the bulk optimum (clamp grid {0, 0.03, …, 0.45, 0.60})
attractor h_eq:  ⟨h⟩(t) → h_eq  from a low founder (rising) AND a high founder (falling)
convergence test:  | ⟨h⟩_from0 − ⟨h⟩_from0.20 | ≤ ε
```

- `N*(h)` — steady-state population a monomorphic strategy supports; pure cost ⇒ monotone decreasing.
- `h_eq` — the convergent equilibrium of the evolving mean (the relaxation point).

**Example.** Clamp pure populations at `h = 0.00, 0.06, …, 0.60` and record `N*`. If `N*` rises with `h` (more precise = more carrying capacity), `h* = argmax` lands high; if it falls, `h* = 0`. Separately, run evolution from `h = 0.20` (it falls) and from `h = 0.00` (it rises): if both settle near the same value, that value is the attractor.

**Data.** Exp 204/205: under the residue false-positive payoff, `N*(h)` **rises** `22.3 → 55.7`, giving the **first functional monomorphic optimum h* = 0.60** (vs the no-residue control's pure-cost decline `595.7 → 133.4`, `h* = 0.06`). Exp 198: the evolving mean **converges from both directions** — arm A (rose from 0) ≈ arm B (fell from 0.20), `|A−B| = 0.0009/0.0022/0.0051/0.0038`, to a **LOW attractor ~0.045** near the 0.05 activation threshold, never a functional organ. In the cost-only ecologies `h* = 0.0` (Exp 203 `argmax_h N* = 0.0` in every ecology; Exp 206 `h*N = 0.0`). Crucially, `h* = 0.60` (bulk-optimal) and the attractor `~0.045` (evolved) *disagree* — the setup for the next concept.

**▸ In programmer terms.** Two different sweeps: one clamps and measures steady-state population (`h*`); the other lets the trait evolve from two starting points and checks they meet (the attractor).

```python
def monomorphic_optimum(grid, run_clamped):
    N_star = {h: run_clamped(h).equilibrium_pop for h in grid}   # pure-h carrying capacity
    return max(N_star, key=N_star.get)                           # argmax_h N*(h)

def is_attractor(run_evo, eps=0.02):
    hi = run_evo(founder_h=0.20).newborn_mean   # relaxes DOWN
    lo = run_evo(founder_h=0.00).newborn_mean   # climbs UP
    return abs(hi - lo) <= eps                   # converge from both sides ⇒ attractor
# Exp 205: monomorphic_optimum == 0.60  BUT  evolved attractor ≈ 0.045  → they DISAGREE.
```

---

## The forced-vs-evolvable gap (L22)

**Glossary.** The arc's central lesson: a **forced/behavioral benefit test** ("organ X helps a lot when imposed") does **not** predict that X will be **selected for**. A trait can be bulk-fitter, demographically affordable, and functional-when-gifted, yet still be **un-evolvable**, because the only quantity that governs climbing is the *sign of the local gradient at the resident* — and small steps may not pay. This is the generalised **L22**.

```
forced/installed benefit:   B(h*) − B(0) ≫ 0          (gift; measured by clamping or a 4× probe)
evolvability:               sign of g(h_res)          (local; measured by the pairwise assay)
the gap (L22):   B(h*) − B(0) ≫ 0   AND   N*(h*) high   AND   population survives
                  YET   g(h_res) ≤ 0   ⇒   trait does NOT evolve
necessary but not sufficient:  a positive g(h_res) is required to climb, but must also beat drift + cost
```

- `B(h*) − B(0)` — the installed gift (Forced-vs-evolvable's first term).
- `g(h_res)` — the local gradient at the resident (the *only* term that decides — see Selection gradient).

**Example.** Hand a population a strong sensor: it reproduces 4× more (Exp 200) — overwhelming "evidence" the organ is good. Now let the *same* trait evolve from the resident `h = 0.10`: a `0.10 → 0.15` step loses head-to-head, so the gene pool decays to `~0.08`. The gift was real; the gradient was ≤0; the trait does not evolve. Measuring the gift mis-predicted evolvability — that is the gap.

**Data.** Exp 200 named it: a forced forager reproduces **~4×** more, yet the organ stays primitive (L22 coined here). Exp 204 sharpened it: a genuinely **functional monomorphic optimum h* = 0.60** with pairwise `0.10 → 0.15` at **2/8** — "a genuinely functional optimum, still un-reachable by small steps." Exp 205 isolated it in its **purest form**: at survivable losses (0.8/1.2/1.5) the bulk optimum is functional (`h* = 0.60`) **and** the population persists (≥4/5 valid), yet evolution keeps the sensor primitive (0/5 functional, mean 0.037–0.093, pairwise ≤0) — "a bulk/installed/functional-when-gifted/demographically-affordable optimum is STILL not sufficient for evolvability; only the sign of g(h_res) decides." The whole sub-arc (Exp 199–207, seven levers) converges: a costed sense is un-evolvable here because `g(h_res) ≤ 0`.

**▸ In programmer terms.** Don't validate a trait by gifting it and measuring the lift — that's testing the wrong quantity. Test whether a one-mutation-sized step out-breeds the resident. The two answers can be opposite.

```python
def gift_benefit(run_clamped):
    return run_clamped(h=0.60).intake - run_clamped(h=0.00).intake   # forced/installed: looks great

def evolvable(pairwise_gradient, h_res=0.10):
    return pairwise_gradient(h_res, h_res + 0.05) >= 7/8              # the ONLY decisive test

# L22 — the gap: these disagree.
assert gift_benefit(run_clamped) > 0          # Exp 204/205: bulk-fitter, h* = 0.60
assert not evolvable(pairwise_gradient)       # Exp 205: pairwise ≤ 3/8  →  does NOT evolve
# Lesson: a forced/behavioral "X helps" does NOT imply "X is selected for."
```

---

## Per-capita growth rate r, carrying capacity N*, and Tilman R*

**Glossary.** Three classical readouts of the same evolvable trait `h`, separating *density-independent* from *density-dependent* fitness. The **per-capita (Malthusian, intrinsic) growth rate** `r` governs a small population not yet limited by resources: `dN/dt = rN`, whose solution is exponential, so `ln N` is *linear* in `t` with slope `r` — `r` is estimated as the OLS slope of `ln(count)` vs `t`. It is the *density-INDEPENDENT* per-capita rate: it ignores crowding. `r(h)` is the basis of the selection gradient (see The selection gradient) — the clamp whose sub-population grows fastest per capita is the one selection favours (`ecology/sense_axis.py: run_intrinsic_growth`). Once crowding bites, growth is *density-DEPENDENT* — **logistic** — and settles at the **carrying capacity** `N*`, the standing population a pure-`h` strategy supports (see The monomorphic optimum h*). The **Tilman R\* rule** gives a *third*, often counter-intuitive criterion: under competition for one limiting resource the winner is **argmin R\***, the strategy that draws the resource down to the lowest standing level `R*` — it competitively excludes the others, regardless of who has the higher `N*`.

```
exponential / log-linear:   dN/dt = r·N        ⇒   N(t) = N₀ · e^(r·t)       ⇒   ln N(t) = ln N₀ + r·t   (slope r)
estimator:                  r = OLS slope of ln(count) vs t   over an early, density-free window
logistic (density-dep.):    dN/dt = r·N·(1 − N/K)             ⇒   N(t) → N* = K   (the carrying capacity)
monomorphic carrying cap.:  N*(h) = mean standing pop of a pure-h population, late-window
Tilman R*:                  R*(h) = standing level of the limiting resource a pure-h pop draws down to
competitive winner:         argmin_h R*(h)        (lowest R* excludes the rest)   ≠   argmax_h N*(h)  in general
```

- `r` — intrinsic, *density-independent* per-capita growth rate; sign of `r(h₁) − r(h₀)` is the density-independent gradient.
- `N₀`, `N(t)` — population at time 0 and `t`; `K = N*` — the logistic plateau / carrying capacity.
- `N*(h)` — equilibrium standing pop of a clamped pure-`h` population (cost ON, breeds true).
- `R*(h)` — equilibrium standing *resource* under that population; **lower is more competitive**.

**Example.** A clamped seed of `N₀ = 12` creatures is measured at `t = 100, 150, …, 700`. Suppose at `t = 100` there are `12` alive and at `t = 300` there are `33`. Then `ln 12 = 2.485`, `ln 33 = 3.497`, and over `Δt = 200` the slope is `r ≈ (3.497 − 2.485)/200 = +0.00506`/step — a positive intrinsic growth rate. To compare two sensors you fit one such slope per clamp and difference: if `r(0.15) ≈ r(0.10)`, the *density-independent* gradient is ≈ 0 (flat) even if the contested-resource pairwise race (see The selection gradient) says a step up pays. Separately, clamp a pure population and read its plateau: if `N*(0.10) = 1518` but `N*(0.60) = 279`, the carrying capacity *falls* with precision — `argmax_h N*(h) = 0.0`, the organ is pure cost at the population level — yet whichever clamp leaves the *least* food standing (lowest `R*`) is the one that would competitively exclude the rest.

**Data.** Exp 203 is the crux: the density-independent intrinsic growth `B(h) = r_costOFF` is **FLAT/negative** — gift `B(0.60) − B(0.00) = −0.00159`, `dB/dh|₀.₁₀ = −0.00511`, realized `r_on` slope `@0.10 = −0.00662` — a lone strong forager grows **no faster** per capita. Monomorphic `N*(h)` is **monotone decreasing** in FORAGE (`1518.2 → 279.0`, `argmax h*N = 0.0`) and in every ecology (`h*N = 0.0/0.06/0.06/0.0`) — pure cost at the population level. *Yet* the pairwise gradient is positive (`0.10 → 0.15` won 7/8, `s_mean +0.01035`; see The selection gradient). That contradiction — flat `r(h)` and falling `N*(h)` but a positive contested-resource gradient — is *exactly* why Exp 203 read **MIXED**: the benefit is **purely competitive/relational**, with no density-independent or population-level component. Exp 204 shows the opposite regime: under the residue false-positive payoff `N*(h)` **RISES** (`22.3 → 55.7`, also reported `24 → 52 → 76`), giving the first functional monomorphic optimum `h* = 0.60`, vs the no-residue control's pure-cost decline `595.7 → 133.4` (`h* = 0.06`) — yet small steps still don't pay (pairwise `0.10 → 0.15` only 2/8; see The fitness valley). Exp 205 confirms it at survivable losses: `h* = 0.60` functional and the population persists, but evolution keeps the sensor primitive — the density-dependent `N*` optimum and the local gradient disagree (the L22 gap). The `argmin_h R*(h)` Tilman criterion is the engine's competitive-dominance diagnostic (`run_carrying_capacity` returns `R_star`; doc §3 of `docs/research/sense-axis-organ-evolution.md`).

**▸ In programmer terms.** `r` is the slope of a log-linear regression on an early, uncrowded window; `N*` is the late-window plateau; `R*` is the late-window standing resource (lower = better competitor). The density-independent `r(h)` and the density-dependent `N*(h)` are *different measurements of the same trait* and can point opposite ways — that is the Exp 203 crux.

```python
import numpy as np

def intrinsic_r(t, count):                      # density-INDEPENDENT per-capita rate
    t = np.asarray(t, float)
    ln_n = np.log(np.asarray(count, float))     # ln N is linear in t with slope r
    A = np.vstack([t, np.ones_like(t)]).T
    return float(np.linalg.lstsq(A, ln_n, rcond=None)[0][0])   # OLS slope = r

def carrying_capacity(pop_series):              # density-DEPENDENT plateau
    return float(np.mean(pop_series))           # N* = late-window mean standing pop

def tilman_winner(R_star_by_h):                 # competitive dominance, NOT argmax N*
    return min(R_star_by_h, key=R_star_by_h.get)   # argmin_h R*(h): draws resource lowest

# Exp 203 crux: intrinsic r(h) FLAT (gift B(0.60)-B(0.00) = -0.00159) and N*(h) FALLS
# (1518.2 -> 279.0, argmax N* = 0.0) — yet the pairwise gradient is POSITIVE (7/8). -> MIXED.
```

---

## Invasion fitness, adaptive dynamics & the ESS

**Glossary.** **Adaptive dynamics** asks of a **resident** population fixed at trait `h_res`: can a *rare* **mutant** `h_inv` invade? The deciding quantity is the **invasion fitness** `s(h_inv ; h_res)` — the mutant's initial per-capita growth rate *while still rare* against the resident-set background. It is the time-derivative of the log mutant-to-resident ratio, estimated in this engine as the OLS slope of `ln(count)` against `t` over the early window (see The selection gradient). `s > 0` means the mutant spreads, `s ≤ 0` means it dies out. The **selection gradient** is `s` differentiated with respect to the mutant trait and read *on the diagonal* `h_inv = h_res` (see The selection gradient). An **attractor** is a resident the evolving mean converges to from both sides; an **ESS** (evolutionarily stable strategy) is a resident where the gradient is `≤ 0` in every direction, so no nearby mutant can invade — the formal statement of "un-evolvable at this resident."

```
invasion fitness:   s(h_inv ; h_res) = d/dt ln( N_inv(t) / N_res(t) )   (mutant per-capita growth rate when rare)
estimator:          s ≈ OLS-slope of ln N_inv(t) vs t  −  OLS-slope of ln N_res(t) vs t   (early window, cold-start differenced out)
verdict:            s > 0 ⇒ invade/spread ;   s ≤ 0 ⇒ die out
selection gradient: D(h_res) = ∂ s(h_inv ; h_res) / ∂ h_inv  |_(h_inv = h_res)
ESS (un-evolvable): D(h_res) ≤ 0 in ALL directions  ⇒  no nearby mutant invades  ⇒  s(h_inv ; h_res) ≤ 0 for h_inv ≈ h_res
                    (the discrete two-sided uninvadability reading: no neighbour either side can invade —
                     not the interior singular-point ESS, where D(h*) = 0 with ∂²s/∂h_inv² < 0)
attractor h_eq:     ⟨h⟩(t) → h_eq  from a low founder (rising) AND a high founder (falling)
```

- `N_inv, N_res` — head-counts of the rare-invader and resident clamp sub-populations in one shared world (the common garden); both breed true under `freeze_thermosense`.
- `s` — the invasion fitness / pairwise selection coefficient; the shared founder cold-start cancels because resident and invader feel it equally.
- `D(h_res)` — the local selection gradient at the resident; its *sign* alone decides evolvability (see Selection gradient, The forced-vs-evolvable gap).
- `h_eq` — the convergent equilibrium; an `h_eq` that is also an ESS is a *terminal* attractor (you arrive and cannot leave).

**Example.** Stand the resident at `h_res = 0.10` and a rare invader at `h_inv = 0.15`. Run one shared world; sample both counts at checkpoints. Suppose the invader cohort goes `4 → 6 → 9` over `t = 100, 350, 600` and the resident background goes `48 → 50 → 49`. Then `ln N_inv` ≈ `1.386, 1.792, 2.197` (slope over `Δt = 500` ≈ `(2.197 − 1.386)/500 = +0.00162/step`) while `ln N_res` is roughly flat (slope ≈ 0), so `s ≈ +0.00162 − 0 > 0`: the mutant invades, the gradient at `0.10` points up. Repeat over 8 seeds and the founder lottery averages out into a **win fraction** `invader_won = (# seeds invader out-grows resident)/8`; the strict positive bar is `≥ 7/8`. If instead the invader's slope were below the resident's, `s ≤ 0` and `0.10` is locally an ESS.

**Data.** The assay is exactly `ecology/sense_axis.py: founder_mix_resident`, whose docstring states it reads "the **adaptive-dynamics invasion fitness** near the resident — the gradient sign evolution actually faces" (resident `h = 0.10` as dominant background, every other grid value a rare invader cohort; `_growth_rate` returns the OLS log-growth slope). **Exp 198** found the resident `0.10` sits just *above* a convergent **attractor ≈ 0.045**: the evolving mean reaches the same low equilibrium from above (seeded `0.20` falls) AND below (de-novo `0` rises), `|A−B| = 0.0009/0.0022/0.0051/0.0038` (4/5; seed-17 B NaN), with A `0.045` vs control C `0.029` — a low attractor near the `0.05` activation threshold, never a functional organ (MIXED / NEW INSIGHT). **Exp 203** found the **first positive local gradient of the arc**: in the band-staleness FORAGE regime `0.10 → 0.15` invaded **7/8** (auc `0.872`, `s_mean +0.01035`), `0.10 → 0.06` (a step down) only **2/8**, `0.10 → 0.45` **7/8** — but the learning-rate-freeze control came in at **6/8**, one seed short of the strict `7/8` bar, so MIXED, and the gradient is *purely competitive* (`N*(h)` falls with `h`; gift `B(0.60)−B(0.00) = −0.00159`). **Exp 205** showed the resident is locally an ESS despite a bulk-fitter optimum: at survivable residue losses the monomorphic optimum is functional (`h* = 0.60`) yet the pairwise invasion at `0.10 → 0.15` won only **1/8, 3/8, 1/8, 2/8, 2/8** across losses `0.5/0.8/1.0/1.2/1.5` (all `≤ 3/8`, `s ≤ 0`), so the **fitness valley — the local gradient `g(0.10) ≤ 0` — is the SOLE binding barrier**, not demographic collapse (MIXED / NEW INSIGHT). **Exp 207**'s corner-grid pre-flight confirmed the gradient is negative at both controller levels: `dB/dh = −0.041` (high θ) and `−0.046` (low θ) — the sensor is pure cost even when acted on hard (DESIGN-STAGE NEGATIVE, closing the sub-arc).

**▸ In programmer terms.** Don't equilibrate and read the mean — seed one resident-dominant world with a rare invader cohort, fit a line to each side's `ln(count)`, and take the slope difference. A positive difference (over enough seeds) means the mutant invades; if no nearby mutant invades, the resident is an ESS.

```python
import numpy as np

def invasion_fitness(world, h_res, h_inv, w_lo=100, w_hi=700):
    # one shared world: resident dominant, invader rare (founder_mix_resident)
    t, n_inv, n_res = world.log_counts(h_inv), world.log_counts(h_res)  # checkpoints in [w_lo,w_hi]
    def ols_logslope(ts, counts):
        ts = np.asarray(ts, float); ys = np.log(np.asarray(counts, float))
        A = np.vstack([ts, np.ones_like(ts)]).T
        return float(np.linalg.lstsq(A, ys, rcond=None)[0][0])   # d ln N / dt
    return ols_logslope(t, n_inv) - ols_logslope(t, n_res)       # s(h_inv ; h_res)

def is_ess(invasion, h_res, nbrs=(0.06, 0.15), seeds=range(50, 58), bar=7/8):
    # ESS: no nearby mutant invades in >= bar of seeds, in ANY direction.
    # (illustrative: per-seed win = sign of the s-slope. The real assay,
    #  run_pairwise_competition, scores a win as inv_frac_final > 0.5 — head-count majority.)
    for h_inv in nbrs:
        won = np.mean([invasion(seed_world(h_res, h_inv, s), h_res, h_inv) > 0 for s in seeds])
        if won >= bar:                # a mutant DID invade  ->  not an ESS
            return False
    return True
# Exp 205 at h_res=0.10: 0.10->0.15 won <=3/8  =>  s<=0  =>  is_ess == True (un-evolvable),
#                        even though the bulk monomorphic optimum is h*=0.60.
# Exp 203 FORAGE: 0.10->0.15 won 7/8 (s_mean +0.01035) => NOT an ESS there (first positive in the arc).
```

---

## Frequency-dependent selection (negative & positive)

**Glossary.** A trait is under **frequency-dependent selection** when its fitness depends on how *common it already is* in the population, not only on its own value. Two signs matter. **Negative** frequency-dependence: a trait is advantageous while **rare** and the advantage **erodes as it spreads** — this tends to *stabilise a polymorphism* (no single value fixes). **Positive** frequency-dependence: a trait is advantageous while **common** — winner-take-all, the majority gets fitter and minorities are squeezed out. This is the key to the arc's central puzzle: a benefit can be **invisible at equilibrium** (the gift `B(h*) − B(0)` is ~0 and `N*(h)` falls with `h` — see Forced-vs-evolvable gap) yet **positive in a head-to-head** pairwise race (see Selection gradient), precisely because the advantage is purely relational — it exists only over rivals, so it erodes as the trait spreads.

```
w_i = w(h_i, p)            fitness depends on the resident frequency p, not h_i alone
negative freq-dep:   ∂w/∂p < 0     advantage of a rare type DECAYS as its frequency p rises  ⇒ stabilises a mix
positive freq-dep:   ∂w/∂p > 0     advantage of a type GROWS with its frequency p            ⇒ winner-take-all
pairwise selection coeff:  s = OLS slope of ln(N_inv / N_res) over window, from an EQUAL-FOUNDER 50/50 start
neg-freq-dep signature:   measured edge is positive at parity  →  INTERPRETED to decay toward s → 0 as it spreads
engine mechanism (crowding):   kept = intake / (1 + niche_crowding × occ_prev[j_true])   (a rare-CLASS advantage)
```

- `p` — the frequency (relative abundance) of the focal type in the resident population.
- `∂w/∂p` — the sign of frequency-dependence; `< 0` is negative (rare-advantage), `> 0` is positive (common-advantage).
- `s` — the pairwise selection coefficient: the invader-vs-resident per-capita log-growth edge, fit by OLS over the coexistence window from an **equal-founder 50/50** head-to-head (`sense_axis.py: run_pairwise_competition`, `count_each = 50` → 50 residents + 50 invaders; `0.5 = neutral`). It is *not* measured while `N_inv ≪ N_res`. The rare-advantage/erosion character (positive while rare, decaying as the trait spreads) is the **inferred** negative-frequency-dependent interpretation of that parity-start number (EXPERIMENTS.md: "negative-frequency-dependent in spirit [the advantage is only over rivals, so it erodes as the trait spreads]"), not the condition under which it was taken.
- `occ_prev[j_true]` — the frozen previous-step **count** of creatures sharing your true niche class `j_true`; `niche_crowding` scales the discount. A class crowded by rivals returns *less* intake — so your reward depends on how many others share your class, the definitional frequency-dependence (`ecology/engine.py` line 618). Note the level: a crowded class paying less is a rare-*class* advantage (negative frequency-dependence at the class level); the *positive* frequency-dependence label below applies at the **trait** level — precision is designed to pay more as sensing becomes common (the Red-Queen cross-term `d²B/dh_i·d(mean h_rivals) > 0`, exp206 design dossier).

**Example.** *Negative.* Stand a resident at `h_res = 0.10` and an invader at `h_inv = 0.15`, then run them as an **equal-founder 50/50 head-to-head** (50 residents + 50 invaders in one shared world; both breed true, cost on, placement shuffled per seed). The founder lottery is differenced out by the parity start, so the win-fraction over seeds reads the selection *sign*: the invader's lineage out-grows the resident in 7 of 8 seeded worlds — `invader_won = 7/8`, a positive edge (`s_mean +0.01035`, Exp 203). Now read what *kind* of edge it is. The advantage is "be sharper than the crowd at the contested drifting food" — it is **purely relational** (a lone strong sensor grows no faster; a whole population of them supports *fewer* creatures). So we *interpret* it as negative-frequency-dependent: were the trait to sweep so the crowd is equally sharp, the relative edge would vanish (`∂w/∂p < 0`) and the mix would be held back from fixing. The gift, measured at equilibrium where the trait is common, reads ~0 (`B(0.60) − B(0.00) = −0.00159`). *Positive (crowding).* Take the niche discount `kept = intake / (1 + niche_crowding × occ)` with `niche_crowding = 1.5`. If rivals herd onto your class so `occ = 4`, you keep `intake / (1 + 1.5×4) = intake / 7 ≈ 0.143·intake`; if you instead read the signature precisely and slip to an under-crowded class with `occ = 0`, you keep `intake / 1 = intake` — 7× more. The payoff to precision *rises* the more rivals pile into the common class: common-class fitness drops as the class fills, a positive-frequency-dependent pressure *at the trait level* that *should* reward discriminating away from the herd.

**Data.** **Exp 203** (band-staleness FORAGE, MIXED / NEW INSIGHT) is the arc's negative-frequency-dependent gradient: the equal-founder pairwise race `0.10 → 0.15` won **7/8** (auc 0.872, `s_mean +0.01035`), `0.10 → 0.06` won only **2/8**, `0.10 → 0.45` won **7/8** — yet `N*(h)` is **monotone-decreasing** (FORAGE `1518.2 → 279.0`, `argmax h*N = 0.0`, pure cost at the population level) and the density-independent gift is **flat/negative** (`B(0.60) − B(0.00) = −0.00159`, `dB/dh@0.10 = −0.00511`). EXPERIMENTS.md states it directly: "negative-frequency-dependent in spirit [the advantage is only over rivals, so it erodes as the trait spreads]" — so the gradient is "purely relational" and "decays as it spreads" (the interpretation; the number itself comes from a 50/50 start). **Exp 206** (rotating-class niche, NEGATIVE / NEW INSIGHT, blind-verified 3/3) installed the *positive*-frequency-dependent crowding mechanism (`kept = consume(deficit)/(1 + niche_crowding·occ_prev[j_true])`, `niche_crowding = 1.5`): the design intent was that "crowding should make precision pay MORE as rivals herd into the common class (Red-Queen sign `d²B/dh_i·d(mean h_rivals) > 0`)," a trait-level positive frequency-dependence. It still **failed** — precision decayed `0.10 → 0.0271` (0/5 functional) at a **healthy** pop (min 586, `corr(pop,h) = −0.146`, not drift), `h*N = 0.0`, equal-founder pairwise `0.10 → 0.15` won only **3/8** (auc 0.509, neutral), `0.10 → 0.30` won **1/8**; the cost-off gift was real but tiny (`+0.00141`). Honest caveat: because `h` stayed primitive, `I(h;niche) = 0.0003 bits` is ~0 *circularly* (no variation to measure), so the niche's non-inertness rests on control deltas, not on mutual information.

**▸ In programmer terms.** Frequency-dependence means fitness is a function of the population mix, so you cannot read it from a single isolated clone — you read its *slope* against the resident with an A/B race (see Selection gradient). Crucially, that race is an **equal-founder 50/50** start, not a rare-invader injection: the parity start differences out the cold-start lottery, and the win-fraction over seeds gives the selection sign. The rare-advantage/erosion property is then *inferred* (the advantage is purely relational), not measured at `N_inv ≪ N_res`. The engine bakes a rare-CLASS advantage into one line: a crowded class pays less (negative frequency-dependence at the *class* level; the intended *trait*-level positive frequency-dependence is "precision pays more as sensing becomes common").

```python
def kept_intake(want, occ_prev_jtrue, niche_crowding=1.5):
    # ecology/engine.py:618 — h-BLIND crowding discount.
    # CLASS-level negative freq-dep: the MORE rivals share your true class, the LESS you keep
    # (a rare-CLASS advantage). The intended TRAIT-level positive freq-dep is the downstream
    # consequence: precision pays MORE as sensing becomes common (d²B/dh_i·d(mean h_rivals) > 0).
    return want / (1.0 + niche_crowding * occ_prev_jtrue)

assert kept_intake(1.0, occ_prev_jtrue=4) < kept_intake(1.0, occ_prev_jtrue=0)   # crowded class pays less

def neg_freq_dep_edge(pairwise_gradient, h_res=0.10, h_inv=0.15):
    # The measured edge is an EQUAL-FOUNDER 50/50 head-to-head (count_each=50), NOT a rare-invader
    # assay: the parity start differences out the founder lottery; the win-fraction reads the sign.
    # Exp 203 FORAGE: == 7/8 (positive at parity). It is purely RELATIONAL, so we INTERPRET it as
    # negative-frequency-dependent: it WOULD erode toward s -> 0 as h_inv spreads and the resident matched it.
    return pairwise_gradient(h_res, h_inv)            # >= 7/8 ⇒ strict positive local gradient (at parity)
# The trap: at EQUILIBRIUM (trait common) the gift reads ~0 (B(0.60)-B(0.00) = -0.00159);
# the head-to-head (equal-founder parity) reads +. Same trait, opposite numbers — that's frequency-dependence.
```

---
