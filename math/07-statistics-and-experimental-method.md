# Statistics & experimental method

> This file is about how the repo *knows* anything. Every experiment in `EXPERIMENTS.md`
> produces numbers; this is the machinery that turns numbers into *trustworthy claims* —
> seeds, means, correlations, effect sizes against controls, predeclared falsifiers,
> independent verification, confound-killing, and the verdict taxonomy. The binding rules
> live in `loop/VALIDATION.md` and `loop/PROTOCOL.md`; this page explains the math behind
> them with worked numbers from the population-ecology arc (Exp 195–207).

## Random seeds & reproducibility

**Glossary.** A *seed* is the integer that initializes the pseudo-random number generator
(PRNG). Under a deterministic PRNG, the entire trajectory is a pure function of the seed and
the code. So a result is a *distribution over seeds*, and a *control* is the same code on the
*same seeds* with one switch flipped. Byte-identical hashing is the strongest reproducibility
control: hash the recorded event stream and compare.

```
trajectory      = f(code, seed)            (deterministic: same inputs → same bytes)
result          = { f(code, seedᵢ) : seedᵢ ∈ S }    (a sample over a seed set S)
control(switch) = f(code with switch=OFF, seed)     (matched on the SAME seed)
hash_match      ⇔  SHA(events_OFF) = SHA(events_baseline)   (byte-identical)
```

`S` is the seed set, e.g. `S = {13,14,15,16,17}`. `events` is the recorded per-step event
log. `SHA(·)` is any fixed cryptographic hash. Two runs are *byte-identical* iff their hashes
match — proof that flipping the switch changed *nothing* on that path.

**Example.** Suppose `f(code, 3) = 0.284` and `f(code, 4) = 0.087`. Re-running seed 3 must
return `0.284` exactly, every time — that is determinism. To test a feature, build it behind a
flag whose OFF path is supposed to be inert; run both arms on seed 3; hash each arm's event
log. If `SHA(OFF) = a3f9…` and `SHA(prior_baseline) = a3f9…`, the new code provably did not
perturb the old behavior — any difference in the ON arm is attributable to the feature alone.

**Data.** Results are reported across *fresh* seed sets so a pattern noticed on one set is
re-tested out-of-sample (VALIDATION.md: "patterns noticed post-hoc must be tested on FRESH
seeds" — re-running the *same* seeds reproduces byte-identical trajectories, so it is circular).
Exp 198 used fresh seeds {13,14,15,16,17}; Exp 201 used {33,34,35,36,37}; Exp 202 used
{38,39,40,41,42}. The byte-identical control is everywhere: Exp 195's senescence-OFF arm
"reproduces Exp 194 170/628/458 with events_hash MATCH"; Exp 201 reproduced "the full
12000-step exp200 WIDE seed23 hash 502e0539 … exactly"; Exp 200/202 are stated "byte-identical
(hash-verified)" against earlier experiments. The honest counter-example: Exp 1's logged
"held-out 4.81 → 4.00 bits/char" did *not* match its own committed output, which a 2026-06-09
re-run showed to be "4.007 → 3.424 bits/char" — a narrative-text error caught precisely
*because* the script reproduced byte-for-byte.

**▸ In programmer terms.** A seeded run is a pure function; the "control" is the same function
with one boolean changed, called with the identical seed; "byte-identical" is just an equality
check on a hash of the output.

```python
import hashlib

def run(code_flag: bool, seed: int) -> "Events":
    rng = Random(seed)            # deterministic PRNG
    return simulate(rng, feature_enabled=code_flag)

def sha(events) -> str:
    return hashlib.sha256(repr(events).encode()).hexdigest()

# reproducibility: same seed → same bytes
assert run(False, 3) == run(False, 3)

# matched byte-identical control: OFF path must be inert vs the prior baseline
assert sha(run(False, 3)) == PRIOR_BASELINE_HASH   # "events_hash MATCH"

# report across a fresh seed SET, not one lucky seed
S = [13, 14, 15, 16, 17]
results = [metric(run(True, s)) for s in S]
```

---

## Sample mean, variance, standard deviation

**Glossary.** Given a per-seed metric `x₁…xₙ`, summarize the sample with its mean, (sample)
variance, and standard deviation.

```
mean      x̄  = (1/n) · Σᵢ xᵢ
variance  s² = (1/(n−1)) · Σᵢ (xᵢ − x̄)²          (Bessel-corrected, sample)
std dev   s  = √(s²)
```

`n` = number of seeds; `xᵢ` = the metric on seed `i`; `x̄` = mean; `s²` = variance; `s` =
standard deviation (same units as `x`). The `n−1` divisor (Bessel's correction) makes `s²` an
unbiased estimate of the population variance from a finite sample.

**Example.** Take Exp 202's five COMPETE per-seed values
`0.0297, 0.0276, 0.0283, 0.0290, 0.0281`.
- `x̄ = (0.0297+0.0276+0.0283+0.0290+0.0281)/5 = 0.14270/5 = 0.02854 ≈ 0.0285`.
- deviations: `+0.00116, −0.00094, −0.00024, +0.00046, −0.00044`; squared:
  `1.346, 0.884, 0.058, 0.212, 0.194` (×10⁻⁶), sum `2.694×10⁻⁶`.
- `s² = 2.694×10⁻⁶ / 4 = 0.674×10⁻⁶`; `s = √(0.674×10⁻⁶) ≈ 0.00082`.
So the COMPETE arm is `0.0285 ± 0.0008` — a *tight* cluster, which is what licenses reading it
as a real signal rather than noise.

**Data.** Exp 202 reports COMPETE "mean 0.0285" across {38–42} (the worked example above), well
below the 0.10 founder. Exp 196's core metric is a per-seed gap with mean character ~0.19
(values 0.2844/0.0869/0.1863/0.1920/0.1860). Exp 195's spread is reported *as* a coefficient of
variation `CV = s/x̄` (0.1134/0.0703/0.0788) — variance turned dimensionless to ask "is
age-at-death fixed or variable?" (small CV ⇒ near-fixed; the spread `CV ≈ 0.07–0.11` confirms
lifespan is variable, not a constant cap).

**▸ In programmer terms.** Mean is `sum/len`; sample variance divides by `len−1`; std is its
square root. Reporting `mean ± std` over a seed list is the whole game.

```python
from statistics import mean, stdev   # stdev() uses the n-1 (sample) divisor

compete = [0.0297, 0.0276, 0.0283, 0.0290, 0.0281]
m  = mean(compete)        # 0.02854  → reported as 0.0285
sd = stdev(compete)       # ~0.00082
cv = sd / m               # ~0.029  (coefficient of variation, unitless spread)
print(f"{m:.4f} ± {sd:.4f}  (CV {cv:.3f})")
```

---

## Correlation: Pearson r vs Spearman ρ

**Glossary.** Correlation measures whether two variables move together. *Pearson r* measures
*linear* association of the raw values; *Spearman ρ* is Pearson r computed on the *ranks*, so
it captures any *monotonic* relationship (and is robust to non-linearity and outliers).

```
Pearson   r = Σᵢ (xᵢ − x̄)(yᵢ − ȳ) / √( Σᵢ (xᵢ − x̄)² · Σᵢ (yᵢ − ȳ)² )

Spearman  ρ = Pearson r computed on rank(x), rank(y)
            = 1 − 6 · Σᵢ dᵢ² / ( n · (n² − 1) )    (dᵢ = rank(xᵢ) − rank(yᵢ), no ties)
```

`r, ρ ∈ [−1, +1]`: `+1` perfect increasing, `−1` perfect decreasing, `0` none. `dᵢ` is the
per-item rank difference; `n` the number of items.

**Example.** Spearman on five creatures, `x` = complexity, `y` = age-at-death:

| creature | x (compl.) | y (age) | rank(x) | rank(y) | d | d² |
|---|---|---|---|---|---|---|
| A | 0.10 | 150 | 1 | 5 | −4 | 16 |
| B | 0.20 | 140 | 2 | 4 | −2 | 4 |
| C | 0.30 | 120 | 3 | 3 | 0 | 0 |
| D | 0.40 | 110 | 4 | 2 | 2 | 4 |
| E | 0.50 | 90 | 5 | 1 | 4 | 16 |

`Σd² = 40`, `n = 5`. `ρ = 1 − 6·40 / (5·24) = 1 − 240/120 = 1 − 2.0 = −1.0` — perfect inverse
ranking: more complex ⇒ dies younger. Real data is never this clean; a noisy version with
`Σd² = 28` gives `ρ = 1 − 168/120 = −0.40`.

**Data.** Exp 195 (lifespan ~ complexity): Spearman `ρ(complexity, age-at-death) =
−0.8647/−0.6774/−0.6704` across three seeds — a real, graded inverse relationship. Crucially,
the *first* tuning gave `ρ ≈ −0.997`, and the project flagged that near-perfect correlation on
a *built-in* relation as a **degeneracy red flag, not a win** (the model had collapsed to a
fixed linear cap); it was rejected and re-tuned to the noisy `ρ ≈ −0.7` regime (L19). Exp 202
uses Pearson-style `corr(pop, intensity) = −0.82`: high sensor intensity appears *only* at low
population, which is the signature of genetic *drift* (small-population sampling noise), not
selection — so the higher NO_SHUFFLE mean is discounted (see Effect size and Confounds below).

**▸ In programmer terms.** Pearson is cosine-similarity of the two mean-centered vectors;
Spearman is the same thing after replacing values with their argsort-ranks.

```python
import numpy as np
def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xc, yc = x - x.mean(), y - y.mean()
    return (xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc))

def spearman(x, y):
    rx = np.argsort(np.argsort(x))      # ranks (0-based), assumes no ties
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)

compl = [0.10, 0.20, 0.30, 0.40, 0.50]
age   = [150, 140, 120, 110, 90]
spearman(compl, age)   # -1.0  (Exp 195 measured ~ -0.67 to -0.86 on real data)
```

---

## Effect size: the gap against a matched control

**Glossary.** An *effect size* is how big a difference is — here, the **gap** between a
treatment arm and a *matched* control arm (same seeds, one switch flipped). A raw number alone
("intensity 0.08") is meaningless without the control; the gap, ideally normalized by spread
(Cohen's d), is the claim.

```
gap   Δ  = x̄_treatment − x̄_control                         (raw effect size)
Cohen d  = Δ / s_pooled,   s_pooled = √( (s²_T + s²_C) / 2 )  (standardized)
```

`x̄_T, x̄_C` are the treatment/control means; `s_pooled` is the pooled standard deviation.
`d` says how many "noise widths" apart the arms are. The reason a gap beats a raw number: the
control absorbs every confound the two arms share, so only the manipulated variable can explain
`Δ`.

**Example.** Exp 197 living active-fraction: treatment `≈ 0.52`, control `≈ 0.06`, so the raw
gap `Δ = 0.52 − 0.06 = 0.46`. If both arms have within-arm `s ≈ 0.02`, then
`s_pooled = √((0.02² + 0.02²)/2) = 0.02`, and `d = 0.46 / 0.02 = 23` — astronomically separated.
Compare a *raw* reading of `0.52`: alone it could mean anything; against the matched OFF arm at
`0.06`, the temperature switch is unambiguously responsible.

**Data.** Exp 196: complexity *gap* of `0.09–0.28` (per-seed 0.2844/0.0869/0.1863/0.1920/0.1860)
between aging-ON and a matched senescence-OFF control whose complexity stays flat, 5/5 fresh
seeds — and the gap is *progressive* (near 0 through ~t=2000, then opens), which is itself an
effect-size-over-time story. Exp 197: living thermosense active-fraction `~0.52` (range
0.49–0.53) vs control `~0.06` (0.03–0.10), liv_gap `≈ 0.43–0.47` — a huge *standing*-population
effect. But the matched newborn gap is only `0.0088–0.0222`: the same comparison at the
*heritable* level is ~30× smaller (mean liv_gap ≈ 0.45 vs mean new_gap ≈ 0.014), which is exactly why the entry is graded survivor-bias-
dominated, not a clean heritable win. The gap framing is what exposes that distinction.

**▸ In programmer terms.** Compute the metric per seed for both arms, subtract the means; the
control is your baseline, the gap is your "lift." Normalizing by pooled std turns it into an
effect size you can compare across experiments.

```python
treat   = [0.49, 0.53, 0.47, 0.46, 0.43]   # Exp 197 living active-fraction (illustrative)
control = [0.10, 0.06, 0.04, 0.05, 0.03]
gap = mean(treat) - mean(control)          # ~0.43  (the claim)
import statistics as st
s_pool = ((st.pvariance(treat) + st.pvariance(control)) / 2) ** 0.5
cohen_d = gap / s_pool                      # large → arms are far apart in noise units
# A raw `mean(treat)` would hide everything the control reveals.
```

---

## Null hypotheses & matched controls

**Glossary.** The *null hypothesis* H₀ is "the manipulation did nothing"; the experiment looks
for evidence to *reject* it. The repo's nulls are *constructive*: a concrete matched control
arm that *should* reproduce H₀, so the gap against it is the test.

```
H₀ : x̄_treatment = x̄_control       (the switch has no effect)
H₁ : x̄_treatment ≠ x̄_control       (it does)
test statistic : Δ = x̄_treatment − x̄_control,  judged against predeclared threshold τ
reject H₀ ⇔ |Δ| ≥ τ  in ≥ k of n seeds
```

`τ` is the predeclared effect-size threshold; `k/n` the predeclared seed-fraction. A *good*
null arm differs from treatment in exactly one variable and is byte-identical otherwise.

**Example.** Predeclare `τ = 0.05` and `k/n = 4/5`. Exp 196's gaps are
`0.2844, 0.0869, 0.1863, 0.1920, 0.1860`; all five are `≥ 0.05`, so 5/5 ≥ 4/5 and H₀ ("aging
does not depress complexity") is rejected. A clean *non*-rejection: Exp 195 at the 600-step
horizon found treatment-vs-control final complexity diffs `0.000–0.023`, all `< 0.03` — an
honest null (the effect simply had not emerged yet; Exp 196 later showed it is a horizon limit).

**Data.** The controls catalog: senescence-OFF is a *byte-identical* null (Exp 195: OFF == Exp
194 exactly). The **CLAMPED-LR control** freezes the learning-rate trait so any climb cannot be
"resource-memory substitution" — Exp 201 CLAMPED_LR `0.1271 ≈ FAST 0.1279`, Exp 202 CLAMPED_LR
`0.0284 ≈ COMPETE 0.0285`, both confirming the (weak) signal is genuine thermosense. The
**useless-thermosense control** (organ on, but does nothing) gives the pure-cost baseline: Exp
200 useless `~0.06` vs foraging `~0.08`; Exp 201 USELESS `0.0529`. The **"selection null" /
horizon limit**: Exp 195's null at 600 steps was not absence but undersampling, resolved by Exp
196's 5000-step run.

**▸ In programmer terms.** H₀ is "the OFF flag and the ON flag give the same metric." You don't
compute a p-value; you compute the gap and check it clears a *predeclared* threshold in enough
seeds.

```python
TAU, K, N = 0.05, 4, 5                       # predeclared BEFORE running
gaps = [metric(run(True, s)) - metric(run(False, s)) for s in seeds]   # matched control
passed = sum(1 for g in gaps if abs(g) >= TAU)
reject_H0 = passed >= K                       # Exp 196: 5/5 ≥ 4/5 → reject "aging does nothing"
```

---

## Predeclared falsifiers

**Glossary.** Before any data exists, the script's docstring states the *prediction* (what a
true hypothesis would show) and the **falsifier** (what result would prove it FALSE). The
verdict is read by applying that rule conjunct-by-conjunct to the raw output — never by a
narrative made up after seeing the numbers.

```
predeclare, time t₀ (before data):
  prediction  P : H true ⇒ metric ≥ θ_P  in ≥ k seeds
  falsifier   F : metric ≤ θ_F  in ≥ k seeds   ⇒  verdict = NEGATIVE
rule (binding): if raw output matches F → NEGATIVE, regardless of story
```

`θ_P, θ_F` are explicit thresholds fixed at `t₀`; "property-level with explicit thresholds" is
required for stochastic runs (VALIDATION.md), because post-hoc exact numbers are inadmissible.

**Example.** Exp 200 predeclared P3 = "some foraging arm gene-pool mean `> 0.30` in ≥4/5 seeds"
and the falsifier "both foraging arms primitive `< 0.15` in a majority." WIDE seeds came in
`0.0783/0.0947/0.0971/0.0348/0.1072` — every one `< 0.15`, so the falsifier *fired 5/5* and the
verdict is NEGATIVE. No amount of "but the gradient trends upward" can rescue it: the rule was
fixed first.

**Data.** Exp 199 predeclared the POSITIVE falsifiers up front: "any primitive arm climbs
`> 0.30`, or the functional arm retains `> 0.30`" — neither fired, so NEGATIVE. Exp 201's
predeclared *resolution order* settled an ambiguous case honestly: one seed (0.1644) landed in
the MIXED band [0.15, 0.30], but "the majority-primitive falsifier precedes the MIXED clause,"
so NEGATIVE-over-MIXED was forced by predeclared *order*, not by weight of evidence. Exp 203 and
205 are the strongest demonstrations: each *predicted the wrong thing* and said so — Exp 205's
"predicting NEGATIVE; the data REFUTED the mechanism — an honest surprise."

**▸ In programmer terms.** Write the grading function *before* the experiment and never edit it
after seeing output. The printed "verdict" line in the output is the coder's claim, not the
result — re-derive it.

```python
def grade(metric_by_seed, *, P_thresh=0.30, F_thresh=0.15, k=4, n=5):
    """Predeclared at t0, frozen. Applied conjunct-by-conjunct to raw output."""
    above_P = sum(v >  P_thresh for v in metric_by_seed)
    below_F = sum(v <  F_thresh for v in metric_by_seed)
    if below_F > n // 2:           # falsifier fires in a majority (≥3 of 5)
        return "NEGATIVE"
    if above_P >= k:
        return "POSITIVE"
    return "MIXED"

grade([0.0783,0.0947,0.0971,0.0348,0.1072])   # Exp 200 WIDE → "NEGATIVE"
```

---

## Blind verification (PROTOCOL 4.5)

**Glossary.** After running but *before* logging, an independent agent recomputes the verdict
from only (a) the predeclared docstring and (b) the committed raw output — blinded to the main
session's interpretation. It must ignore any verdict printed *inside* the output and re-apply
the predeclared rule. Agreement is recorded; disagreement triggers investigation and the
*stricter* verdict.

```
verdict_main      = grade(raw_output)              # the experimenter's reading
verdict_blind     = grade(raw_output)              # independent agent, predeclaration + raw only
record:
  agree    : verdict_main = verdict_blind  → log "Verifier: agree"
  disagree : take stricter(verdict_main, verdict_blind), record both
ordering of strictness: NEGATIVE / NO_VERDICT  >  MIXED  >  POSITIVE
```

The rationale is statistical, not ceremonial: "the same mind that designed an experiment grades
it leniently," so an independent recompute beats self-critique on the exact failure class
(over-calling POSITIVE).

**Example.** Exp 203's blinded grader recomputed conjunct-by-conjunct and took the *stricter*
reading: POSITIVE failed on conjunct C3 (the CLAMPED_LR confound control was 6/8, below the
strict 7/8 bar), NEGATIVE failed on conjuncts D1+D2 (the gift was not `> 0` *and* the resident
slope was positive 7/8) — leaving MIXED. The main session and the verifier agreed; the entry
records "Verifier: AGREE (MIXED)."

**Data.** Every Exp 152+ verdict is blind-verified (RESUME.md §3b). Exp 201 and Exp 206 were
"unanimously blind-verified 3/3 AGREE"; Exp 202 and Exp 207 "blind-verified AGREE." Exp 204's
verifier *independently* derived NO_VERDICT/MIXED by noticing F2 fired (only 3/5 evolution seeds
valid) — the validity gate precedes POSITIVE/NEGATIVE in the predeclared precedence. The
verifier line is mandatory and enforced by `loop/check_iteration.py`.

**▸ In programmer terms.** Run the *same* grading function in a process that never saw your
narrative, feeding it only the docstring rule and the raw file; assert the two verdicts match,
or escalate to the stricter one.

```python
STRICTNESS = {"POSITIVE": 0, "MIXED": 1, "NO_VERDICT": 2, "NEGATIVE": 2}

def blind_verify(raw_output_path, predeclared_rule):
    raw = load_numbers(raw_output_path)          # ignore any printed "verdict" line
    return predeclared_rule(raw)                  # recompute from scratch

v_main  = grade(my_numbers)
v_blind = blind_verify("experiments/outputs/exp203.txt", grade)
verdict = v_main if v_main == v_blind else max(v_main, v_blind, key=STRICTNESS.get)
```

---

## Confounds and how they are neutralized

**Glossary.** A *confound* is a third variable that offers an alternative explanation for the
gap, so the manipulation gets undeserved credit. Each confound is named in advance and killed by
a *control arm* or an *engine change* whose OFF path is byte-identical.

```
observed effect  Δ_obs = Δ_true + Δ_confound
neutralize: add control C such that Δ_confound(C) = 0,
            then  Δ_true ≈ x̄_treatment − x̄_C
verify the control is inert: SHA(events_OFF) = SHA(baseline)   (byte-identical)
```

**Example.** The *creature-id-order* confound: cells are eaten in ascending creature-id order,
so a low-id creature "eats first" regardless of skill — that birth-order luck masquerades as
sensing skill. Neutralizer (Exp 202): `shuffle_creature_order` randomizes per-step processing
order (ON-branch only), so a contested cell is won by navigation, not id. The `consume()` logic
is unchanged and the OFF path stays byte-identical, so the gap between COMPETE (shuffled) and
NO_SHUFFLE isolates exactly the id-order effect.

**Data.** The Exp 201–207 confound ledger:
- *resource-memory free-ride* — a lifetime map lets a crude sensor revisit known cells; killed
  by the CLAMPED-LR control (Exp 201 CLAMPED_LR 0.1271 ≈ FAST 0.1279; Exp 202 0.0284 ≈ 0.0285).
- *creature-id-order* — birth-order eat-first; killed by `shuffle_creature_order` (Exp 202).
- *survivor bias* — sensing creatures live longer and pile up among the *living*, inflating the
  standing metric; exposed by the *newborn tracker* (Exp 197 living gap ~0.43 vs newborn gap
  ~0.01) and addressed by *starting the organ at intensity 0* (Exp 198: "a trait nobody has
  can't be survivor-biased").
- *gating* — every new mechanic sits behind a flag whose OFF path is byte-identical/hash-verified
  (Exp 200/201/202: exp194-prior byte-identical; full exp200 hash 502e0539 reproduces).
- *drift* (small-population sampling noise) — Exp 202 discounts NO_SHUFFLE's higher mean 0.1303
  because it occurs only at collapsed pops 214–461 with `corr(pop, intensity) = −0.82`, a
  predeclared drift artifact, not selection.
- *validity gate / strip-gate* — Exp 202 certified the test genuinely interference-competitive
  with `strip_frac = 1.00` (vs Exp 201's `strip ≈ 0`, which left the question untested).

**▸ In programmer terms.** For each "what else could explain this?" you build a control arm
that zeroes that one factor while staying byte-identical elsewhere, then re-measure the gap.

```python
arms = {
    "COMPETE":    dict(deplete=True,  shuffle=True),   # treatment
    "NO_SHUFFLE": dict(deplete=True,  shuffle=False),  # isolates id-order confound
    "CLAMPED_LR": dict(deplete=True,  shuffle=True, freeze_lr=True),  # kills memory free-ride
    "USELESS":    dict(deplete=False, shuffle=True),   # organ does nothing → pure-cost baseline
}
out = {name: [metric(run(cfg, s)) for s in seeds] for name, cfg in arms.items()}
assert sha(run(arms["COMPETE"], 38, off_path=True)) == BASELINE_HASH   # gating is inert
# discount drift: if corr(pop, intensity) is strongly negative, high values are small-N noise
```

---

## The verdict taxonomy

**Glossary.** Every entry is graded with a *result* tag and an *insight* tag. The result tag is
the outcome of the predeclared rule; the insight tag says whether the finding was new.

```
RESULT ∈ { POSITIVE, NEGATIVE, MIXED, NO_VERDICT, BREAKTHROUGH }
INSIGHT ∈ { CONSOLIDATION, NEW INSIGHT }
POSITIVE → self-grade { BREAKTHROUGH, POSITIVE-SINGLE }
strictness for tie-break: NEGATIVE / NO_VERDICT  >  MIXED  >  POSITIVE
```
- **POSITIVE** — the prediction held; default sub-grade POSITIVE-SINGLE.
- **NEGATIVE** — the falsifier fired; a finding, often the most valuable kind.
- **MIXED** — some conjuncts pass, some fail (e.g. the headline holds but a confound control
  falls short).
- **NO_VERDICT** — a *validity gate* failed (the metric was unmeasurable, e.g. a collapsed arm),
  so neither POSITIVE nor NEGATIVE can be licensed.
- **BREAKTHROUGH** — first demonstration of a capability the system did not have before, or a
  result that redirected the program; must survive the hostile-reviewer test ("what could it
  *demonstrably do for the first time*?"). Confirmations, scale tests, and "the thing we built
  works" default to POSITIVE-SINGLE.
- **NEW INSIGHT** vs **CONSOLIDATION** — was the result predictable from prior experiments?

**Example.** Exp 204 graded **MIXED / NO_VERDICT**: a *validity gate* fired first (only 3/5
evolution seeds valid — two populations collapsed to NaN), so under the predeclared precedence
NO_VERDICT/MIXED is reached *before* POSITIVE/NEGATIVE are even evaluated; the stricter of
MIXED-vs-NEGATIVE is MIXED. Contrast Exp 196, graded **POSITIVE / NEW INSIGHT, POSITIVE-SINGLE**
— the controlled 5/5 gap held, but the direction was imposed and probe-foreshadowed, so it is
explicitly *not* a BREAKTHROUGH.

**Data.** The sense-evolution arc reads cleanly in the taxonomy: Exp 196 POSITIVE; Exp 197
POSITIVE (fine-grained POSITIVE-MIXED); Exp 195/198/203/204/205 MIXED / NEW INSIGHT; Exp
199/200/201/202/206/207 NEGATIVE / NEW INSIGHT — "seven structurally-distinct levers … all
fail," with Exp 207 a *design-stage* NEGATIVE that closed the sub-arc without a full batch. The
discipline is enforced: VALIDATION.md bans "BREAKTHROUGH" for results that are "the same as
before, but bigger / cleaner / again" (those are POSITIVE-SINGLE) — "if everything is a
breakthrough, nothing is."

**▸ In programmer terms.** The verdict is a small state machine: a validity gate short-circuits
to NO_VERDICT/MIXED; otherwise the predeclared rule returns POSITIVE/NEGATIVE/MIXED; POSITIVE
then asks the breakthrough question; an orthogonal flag records new-vs-consolidation.

```python
def verdict(raw, *, valid_seeds, n, rule, is_first_of_kind):
    if valid_seeds < required_min(n):          # validity gate fires first (Exp 204)
        return "NO_VERDICT / MIXED"
    tag = rule(raw)                            # "POSITIVE" | "NEGATIVE" | "MIXED"
    if tag == "POSITIVE":
        tag = "BREAKTHROUGH" if is_first_of_kind else "POSITIVE-SINGLE"
    return tag

insight = "NEW INSIGHT" if not predictable_from_prior else "CONSOLIDATION"
```

---

## Fitting a trend: ordinary least squares and the log-growth slope

**Glossary.** *Ordinary least squares* (OLS) fits a line `y = a + b·x` by choosing the
intercept `a` and slope `b` that minimize the sum of squared residuals. The minimizer has a
closed form. Two specializations carry the ecology arc: the **log-growth slope** (fit `ln(count)`
vs `t`, the slope *is* the per-capita growth rate `r`; see Fitness as realized reproduction in
05-evolutionary-dynamics.md) and the **selection coefficient** (the OLS slope of
`ln(N_inv/N_res)`; see The selection gradient in 05-evolutionary-dynamics.md). A *polynomial fit with
fixed effects* adds a curvature term `h²` and per-seed intercepts to measure concavity.

```
OLS line       minimize  Σᵢ (yᵢ − a − b·xᵢ)²   over (a, b)
               b = cov(x,y) / var(x) = Σᵢ (xᵢ − x̄)(yᵢ − ȳ) / Σᵢ (xᵢ − x̄)²
               a = ȳ − b·x̄

log-growth     r = OLS slope of ( t ,  ln Nₜ )            (Nₜ = count at step t)
selection s    s = OLS slope of ( t ,  ln(N_inv / N_res) )   (invader vs resident)

poly + FE      wᵢ = αₛₑₑ𝒹 + b₁·hᵢ + b₂·hᵢ² + b₃·(hᵢ·densityᵢ) + εᵢ
               b₂ < 0  ⇔  concave (diminishing returns)
goodness       R² = 1 − SS_res / SS_tot,   SS_res = Σᵢ (yᵢ − ŷᵢ)²,  SS_tot = Σᵢ (yᵢ − ȳ)²
```

`x̄, ȳ` are the means; `cov, var` the sample covariance and variance; `b` the slope, `a` the
intercept. `Nₜ` is a per-step head-count; `r` a Malthusian growth rate (units 1/step). `αₛₑₑ𝒹`
is a per-seed intercept (a *fixed effect* that absorbs between-seed level differences so
`b₁, b₂, b₃` are estimated *within* seed); `ŷᵢ` is the fitted value. `R² ∈ [0,1]` is the
fraction of variance the fit explains. Fitting `ln Nₜ` rather than `Nₜ` is **fixation-robust**:
it reads the *rate* early, before one clamp sweeps and head-counts saturate.

**Example.** Three log-count checkpoints from one clamp's sub-population:
`(t, N) = (100, 8), (300, 11), (500, 15)`. Fit `ln N` vs `t`.
- `y = ln N = 2.0794, 2.3979, 2.7081`; `x = t = 100, 300, 500`.
- `x̄ = 300`, `ȳ = 2.3951`; deviations `x−x̄ = −200, 0, +200`, `y−ȳ = −0.3157, +0.0028, +0.3130`.
- `cov numerator = Σ(x−x̄)(y−ȳ) = (−200)(−0.3157) + 0 + (200)(0.3130) = 63.14 + 62.60 = 125.74`.
- `var numerator = Σ(x−x̄)² = 40000 + 0 + 40000 = 80000`.
- `b = 125.74 / 80000 ≈ 0.00157` per step → that clamp grows at `r ≈ 0.00157`. `a = ȳ − b·x̄ =
  2.3951 − 0.00157·300 ≈ 1.924`. The clamp with the largest `b` is the one selection favours.

**Data.** This is the workhorse of the sense-evolution sub-arc. `ecology/sense_axis.py` computes
`r(h)` in `run_intrinsic_growth` and `_growth_rate` as exactly `np.linalg.lstsq(A, ys)` on
`ys = ln(count)` over the early window, and the selection coefficient `s` in
`run_pairwise_competition` as the lstsq slope of `ln(n_inv/n_res)` over coexistence points.
The concave benefit of precision is a *measured* `b₂ < 0`, not an assertion: **Exp 201**'s
returns-curve probe found gross intake rising with precision (`0.068 → 0.089` across `h 0.10 →
0.60`) but with marginal returns *falling* `+0.043 → +0.018` (`convex_above_0.30=False`) — a
concave `B(h)` whose `~0.02–0.04`/unit marginal benefit is dwarfed by the `0.20`/unit linear
upkeep, so net fitness peaks at the primitive founder. The pairwise/regression slopes drive the
**Exp 203–207** verdicts: **Exp 203** read the resident slope from pairwise wins (`0.10 vs 0.15`
won 7/8) plus a density-independent slope `dB/dh@0.10 = −0.00511` and `realized r_on slope@0.10 =
−0.00662` → MIXED. **Exp 205** swept the local pairwise gradient `s(0.10 vs 0.15)` across loss
levels (won `1/8, 3/8, 1/8, 2/8, 2/8`) with the evolved mean `h` growing `0.042 → 0.093` yet
never functional → MIXED, isolating the fitness-valley barrier. **Exp 207**'s design-stage
pre-flight read finite-difference slopes off a corner grid (`dB/dθ@low h = +0.147`, `dB/dh =
−0.041 / −0.046`) → DESIGN-STAGE NEGATIVE, closing the sub-arc.

**▸ In programmer terms.** Stack `[t, 1]` into a design matrix and call `lstsq`; the first
coefficient is the slope. For the growth rate, feed it `log(counts)` — exactly what
`sense_axis.py` does.

```python
import numpy as np

def ols_slope(x, y):
    A = np.vstack([np.asarray(x, float), np.ones(len(x))]).T   # columns: [x, 1]
    (slope, intercept), *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return slope            # == cov(x, y) / var(x)

# log-growth rate r(h): slope of ln(count) vs t  (fixation-robust)
r = ols_slope(ts, np.log(counts))                 # Exp 203: realized r_on slope@0.10 < 0

# selection coefficient s: invader vs resident  (run_pairwise_competition)
s = ols_slope(ts, np.log(n_inv / n_res))          # Exp 205 pairwise gradient ≤ 0

# concavity via polynomial + per-seed fixed effects (fit_selection_regression)
# design row: [seed dummies..., h, h**2, h*density];  b2 < 0  ⇒  diminishing returns
beta, *_ = np.linalg.lstsq(X, w, rcond=None)
r2 = 1 - ((w - X @ beta) ** 2).sum() / ((w - w.mean()) ** 2).sum()
```

---

## Standard error and the look-elsewhere effect

**Glossary.** The *standard deviation* `s` measures the spread of *individual* seeds; the
*standard error of the mean* `SE` measures the precision of the *per-arm mean* over `n` seeds —
it shrinks only as `√n`. With this lab's `n = 5–8` seeds `SE` is large, so the defense is to
report ALL seeds and read a *k/n majority bar* (e.g. "won 7/8") rather than trust one run. A
second hazard is *multiple comparisons* (the **look-elsewhere effect**): test enough
levers/regimes and a spurious "7/8" appears by chance — the family-wise error grows with the
number of tests. The lab's reply (loop/VALIDATION.md, LESSONS.md L7) is not a p-value but a
*predeclared threshold* + a *fresh-seed re-test*: the data that suggested a pattern may not also
confirm it.

```
std      s   = √( (1/(n−1)) · Σᵢ (xᵢ − x̄)² )         (spread of individuals)
std-err  SE  = s / √n                                 (precision of the MEAN)
shrinkage    : SE ∝ 1/√n   ⇒ halving SE needs 4× the seeds

family-wise error over m independent tests at per-test rate α:
  P(≥1 false "hit")  =  1 − (1 − α)ᵐ   ≈  m · α      (small α)

lab substitute for a p-value (binding, no CIs/p-values used):
  predeclare θ_P, θ_F and a k/n bar at t₀   →   verdict = rule(raw)
  any POST-HOC pattern ⇒ re-test on FRESH seeds (never the same seeds)
```

`n` = seed count; `xᵢ` = per-seed metric; `x̄` = arm mean; `s` = sample std (Bessel `n−1`, see
Sample mean, variance, standard deviation); `SE` = standard error of the mean; `m` = number of
distinct tests run; `α` = per-test false-positive rate; `k/n` = the predeclared seed-fraction
bar (see Null hypotheses & matched controls).

**Example.** Take Exp 202's five COMPETE per-seed values
`0.0297, 0.0276, 0.0283, 0.0290, 0.0281` (mean `0.0285`, `s ≈ 0.00082` — worked in *Sample
mean, variance, standard deviation*). The standard error of that mean is
`SE = 0.00082 / √5 = 0.00082 / 2.236 ≈ 0.00037` — about *half* the std, because `√5 ≈ 2.24`.
To halve `SE` again you would need `n = 20`, not `10` (the `√n` law). Now the look-elsewhere
arithmetic: suppose each lever has a `α = 1/16` chance of a fluke "≥ 4/5 pass". Run one lever
and `P(false hit) ≈ 0.06`; run `m = 7` levers and `P(≥1 false hit) = 1 − (15/16)⁷ ≈ 1 − 0.637
= 0.36` — better than a one-in-three chance of *some* spurious "win" across the arc, which is
exactly why a single lucky bar cannot be trusted and the threshold must be fixed in advance.

**Data.** The sense-evolution sub-arc reports k/n bars throughout, with two distinct
seed-budgets disclosed per entry. The **pairwise selection audits use 8 seeds** `{50-57}`
("8 for the ≥7/8 bar", Exp 203): Exp 203 found `0.10 vs 0.15 invader_won 7/8` but the strict
CLAMPED_LR confound control came in `6/8` — *one world short* of the `7/8` bar, which alone
forced MIXED over POSITIVE (see Blind verification). Exp 204's same `0.10 vs 0.15` pairwise was
`2/8` (a fitness valley); Exp 206's was `3/8` (neutral). The **evolution arms use 5 fresh
seeds** — Exp 204 `{70-74}`, Exp 205 `{80-84}`, Exp 206 `{90-94}` — read as `k/5` bars (e.g.
Exp 204 evolution `0/5 functional` with `3/5 valid`; Exp 206 `0/5 functional, 5/5 valid`). The
**fresh-seeds-for-post-hoc rule** (LESSONS.md **L7**, Exp 70) is what makes these bars
trustworthy: re-running the *same* seeds is byte-identical and circular, so a pattern noticed in
old data is re-tested out-of-sample. Honest note: this lab deliberately uses **no formal
confidence intervals or p-values** — predeclared falsifiers plus fresh-seed confirmation do that
job (VALIDATION.md). LESSONS.md **L6** (Exp 78/79) is the small-`n` guard: "count thresholds
alone are weak on noisy endpoints — predeclare an effect size alongside the count, or use ≥ 8
seeds, or both" (the reason the *pairwise* audits jumped to 8 seeds while evolution stayed at 5).

**▸ In programmer terms.** `SE` is just `std/sqrt(n)`; the look-elsewhere effect is the same
math as "run enough A/B tests and one comes back significant by luck." The lab never computes a
p-value — it counts how many seeds cleared a *predeclared* bar, and re-runs any post-hoc idea on
seeds it has never touched.

```python
import statistics as st

def sem(xs):                      # standard error of the MEAN (not the spread)
    return st.stdev(xs) / len(xs) ** 0.5

compete = [0.0297, 0.0276, 0.0283, 0.0290, 0.0281]   # Exp 202 COMPETE
sem(compete)                      # ~0.00037  (vs std ~0.00082; SE = std/sqrt(5))

# look-elsewhere: family-wise false-positive rate over m predeclared levers
def family_wise(alpha, m):
    return 1 - (1 - alpha) ** m
family_wise(1/16, 7)              # ~0.36  → a spurious "win" is likely across many tests

# the lab's defense: predeclared bar, and FRESH seeds for any post-hoc pattern (L7)
PAIRWISE_SEEDS = range(50, 58)    # {50..57}, 8 seeds for the >=7/8 bar (Exp 203)
def passes(won_fraction, k=7, n=8):   # e.g. 7/8 bar, fixed at t0 — replaces a p-value
    return won_fraction >= k / n
assert PAIRWISE_SEEDS != range(50, 58) or True   # post-hoc retest must use NEVER-RUN seeds
```

---

## Cryptographic hashing as a byte-identical control

**Glossary.** A *cryptographic hash* `H` (here SHA-256) maps an input of any length to a
fixed-size 256-bit digest. Two properties make it a verification tool: it is *deterministic*
(same bytes in ⇒ same digest out), and it is *collision-resistant* (finding two distinct inputs
with the same digest is computationally infeasible). So equal digests imply, overwhelmingly,
byte-identical inputs — an accidental collision has probability about `2⁻²⁵⁶ ≈ 10⁻⁷⁷`. The repo
uses this as the *byte-identical control* (see Random seeds & reproducibility): hash the full
event trajectory of a run, and one 64-hex string then certifies the entire run is unchanged.

```
digest         d = H(bytes) ∈ {0,1}²⁵⁶            (SHA-256, fixed 256-bit output)
deterministic  bytes₁ = bytes₂  ⇒  H(bytes₁) = H(bytes₂)
collision      P[ H(a) = H(b)  ∧  a ≠ b ]  ≈  2⁻²⁵⁶     (negligible)
events_hash    e = H( canonical(events) ),   canonical = JSON(events, sort_keys=True)
control test   e_OFF = e_baseline   ⇒   the gated feature's OFF path changed NOTHING
```

`events` is the recorded per-step event log of a run; `canonical(·)` serializes it to a
deterministic byte string (sorted keys, no wall-clock) so that two runs with the same events
produce the same bytes — and therefore the same digest. `e_OFF` is the hash of a run with the
new feature gated OFF; `e_baseline` is the hash of the prior experiment. Matching digests prove
the refactor or new flag perturbed nothing (the strongest form of a matched control — see Null
hypotheses & matched controls, Confounds and how they are neutralized).

**Example.** Hash two short strings by hand to feel the determinism and avalanche. With any
SHA-256 implementation, `H("194") = 7559ca…` is reproduced *every* time you hash `"194"` — that
is the determinism guarantee. Change a single byte, `H("195")`, and the digest is *completely*
different (avalanche), so a one-step difference anywhere in a trajectory flips the 64-hex string
entirely. Now scale up: suppose run-A's event log serializes to 4 MB of canonical JSON and
hashes to `502e0539…`. Re-run the same seed with a new feature gated OFF; its 4 MB log hashes to
`502e0539…` again. You did not diff 4 MB — you compared two 64-character strings and concluded,
with confidence `1 − 2⁻²⁵⁶`, that all 4 MB are identical byte-for-byte.

**Data.** The byte-identical hash control runs through the whole Exp 194–207 foraging arc. The
engine computes it as `events_hash = SHA-256(canonical events)` (`ecology/engine.py`,
`hashlib.sha256`, `json.dumps(self.events, sort_keys=True, ensure_ascii=True)`). Exp 195's
senescence-OFF arm "reproduces Exp 194 170/628/458 with events_hash MATCH" — proving the L16
no-op guard left the prior model untouched. Exp 200's full 12000-step WIDE seed23 run hashes to
`502e0539…`, and Exp 201 reports "exp194-200 byte-identical (exp194 hash test + the full
12000-step exp200 WIDE seed23 hash 502e0539… reproduce exactly)"; Exp 202 likewise certifies
"exp194-201 byte-identical (exp194 hash + exp200 WIDE seed23 502e0539 reproduce)". The
*design-stage* Exp 207 used the same idea as anti-cheat guards: with the sensor cost OFF, intake
is byte-identical across the sensor trait `h` (`eaten 79368.597 = 79368.597` at confusion=0;
`8310.901 = 8310.901` at niche_weight=0), certifying that `h` only ever acts through the costed
channel — a NEGATIVE verdict made trustworthy precisely because the controls hash-match. (The
spine line uses the same machinery: Exp 45's saved creature has `state_hash=24197c338d576a8e…`,
and save→load round-trips reproduce it exactly.)

**▸ In programmer terms.** Serialize the run's events canonically (stable key order, no
timestamps), SHA-256 the bytes, and assert the digest equals the committed baseline. It is an
`O(1)`-to-compare equality check that stands in for diffing the entire trajectory.

```python
import hashlib, json

def events_hash(events) -> str:                       # ecology/engine.py
    canonical = json.dumps(events, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# gating control: the new feature's OFF path must be byte-identical to the baseline
off_run   = simulate(seed=23, feature_enabled=False)
assert events_hash(off_run) == "502e0539..."          # Exp 200 WIDE seed23 reproduces

# anti-cheat (Exp 207): with cost OFF, the trait must not touch the trajectory at all
assert events_hash(run(h=0.10, cost=False)) == events_hash(run(h=0.45, cost=False))
# one mismatched byte ⇒ a totally different digest (avalanche) ⇒ the OFF path is NOT inert
```

---
