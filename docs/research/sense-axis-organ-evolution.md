# When does a costed sense become an organ? — the sense-axis bridge

**Status:** CONVERGED (sense-evolution sub-arc, Exp 203–206 on the four walls 199–202; human steer
2026-06-13). Six structurally-distinct escapes exhausted — a costed sense is un-evolvable at this
substrate (`g(h_res) ≤ 0`); the FINAL SUMMARY below is filled. The formal direction decision (close the
card / Exp 207 / redirect) awaits an explicit human word (post-206 consult, `loop/IDEAS.md`).
**Scope:** a research note for the population-ecology direction (`loop/directions/population-ecology.md`).
The **FINDINGS** section is filled experiment-by-experiment as the sub-arc runs; the framework
below is fixed.

---

## 1. The question (reframed)

Exp 199–202 closed four walls. Under **avoidance** (199), **foraging** (200),
**increasing-returns tracking** (201), and **real interference competition** (202), a costed
thermosense organ never becomes functional — and under competition it is actively *suppressed*
below the founder. A *forced* strong sensor reproduces ~4× more than a no-sensor one (Exp 200),
so the installed benefit is real; evolution still does not climb.

The arc is therefore **not** "make thermosense win." It is the reusable meta-question:

> **What environmental, informational, and evolutionary conditions are required for a costed
> sense-like trait to become increasingly selected over generations — to become organ-like
> rather than merely useful when gifted?**

phrased so the answer transfers from thermosense to sight, hearing, localization, and
communication.

## 2. The load-bearing distinction: installed benefit vs local gradient

Write realized fitness as

    w_i = B(a_i, z, h_i, theta_i, rho) − C_h(h_i) − C_theta(theta_i)

with `h` = sensor precision/intensity, `theta` = the controller's ability to *use* the sensor,
`z` = hidden world state, `s ~ P(s|z,h)` the observation, `a = pi(s, theta)` the action, `rho`
the ecological state (density / depletion / residue / niche occupancy).

A population's mean sensor can only rise when the **local selection gradient** is positive near
the current value:

    g(h) = dE[w | h] / dh > 0,     i.e.   dB/dh > dC/dh + drift/noise/transmission-erosion.

The trap that produced all four walls:

    B(0.60) − B(0.00) ≫ 0      (a gifted strong sensor helps a lot)
    while   B'(0.08) − C'(0.08) ≤ 0   (a small heritable step does not pay).

A forced/behavioral probe measures the first; **evolvability is governed by the second**
(L22). Every experiment in this sub-arc instruments the **local slope**, not the endpoint.

## 3. The reusable sense-axis abstraction (`ecology/sense_axis.py`)

A `SenseAxis` treats a sense as a generic instance of `(h, C_h, z, P(s|z,h), theta, a=pi(s,theta),
rho, w)` and exposes the diagnostics that decide evolvability:

- **B(h)** — installed benefit (gross intake at a frozen sensor, cost OFF — the returns-probe pattern).
- **C(h)** — analytic upkeep `floor + inefficiency·h` (floored, never free).
- **the local slope** `dB/dh − dC/dh` at anchors near the resident.
- **the realized selection gradient** read from a live population with cost ON, via density-
  robust estimators that survive this engine's harsh founder cold-start mortality:
  - **N\*(h) / R\*(h)** — monomorphic carrying capacity + standing resource (Tilman R\* rule:
    the strategy that draws the limiting resource lowest competitively excludes the rest;
    `argmin_h R*(h)` is the competitively dominant sensor).
  - **pairwise selection coefficient s(h_inv vs h_res)** — equal-founder head-to-head
    competition; the cold-start differences out; `s = d ln(N_inv/N_res)/dt`. The direct gradient sign.
  - **common-garden r(h)** — multi-clamp early log-growth (descriptive landscape; noisy under
    fixation, so secondary).

Two gated, byte-identical engine features support it: `freeze_thermosense` (a clamp value breeds
true while upkeep stays charged — the unit of a realized-fitness-at-fixed-h measurement) and
`founder_mix` (one shared world seeded from an explicit polymorphic genotype list).

**Anti-cheat (binding).** Nothing writes fitness or food as `f(h)`. The clamp founders only set a
genotype trait; survival/reproduction read solely each creature's own state + local cell; cost is
the ordinary upkeep, charged by the unmodified engine. `assert_no_direct_h_reward` documents it;
the blinded verifier re-checks it.

**The meta-principle.** *A sense is evolvable when it exposes private actionable information whose
marginal value remains positive across small heritable improvements.*

## 4. The bridge to engineer (the eight conditions)

    hidden state z  →  sensory information s(z,h)  →  actionable affordance a  →
    positive LOCAL selection gradient g(h_res) > 0  →  heritable specialization  →  organ-like trait

A sense becomes organ-like when, jointly:
1. the world contains hidden state `z`;
2. the sense `h` increases usable information about `z`;
3. the controller `theta` can convert that information into better actions;
4. the benefit is **local and marginal**, not only visible at high `h`;
5. the benefit exceeds sensor + controller cost at the resident;
6. competitors / niches do not erase the advantage through herding;
7. lineage or niche structure preserves partial specialization long enough to compound;
8. knockouts show the trait is causally necessary.

Exp 203 measures condition (4)/(5) directly across the four existing walls. Exp 204 (residue
false-positives), 205 (barcode niches / sympatric divergence), and 206 (sensor–controller
co-adaptation) are constructive attempts to *create* a positive local gradient by adding,
respectively, a false-positive affordance, a private-niche affordance, and a controller coupling.

## 5. Future senses — the same fields (Phase 6)

| | hidden state `z` | sense trait `h` | false positives | affordance unlocked by precision |
|---|---|---|---|---|
| **thermosense (now)** | fresh/depleted food, drifting band, residue, niche temperature | thermal precision | residue read as fresh food | fresh-food discrimination, niche access, less search waste |
| **sight** | object identity, distance, occlusion, predator/prey/food class | spatial/angular resolution, FOV, channel discrimination | decoys, shadows, occluded objects | long-range planning, classification, obstacle avoidance, niche access |
| **hearing** | source location, frequency signature, social signal, movement | frequency/temporal resolution, localization precision | echo/noise read as signal | localization, coordination, threat detection, mate finding |
| **communication** | agent intention, local resource info, danger, role, identity | channel bandwidth, symbol discrimination, signal reliability, sender/receiver alignment | misdecoded messages | coordination, division of labour, teaching, recruitment, mate choice |

Each future sense is a different `SenseAxisSpec`, not a fork of the diagnostics. The same
clamp-grid gradient audit, the same B(h)/C(h) decomposition, the same N\*/R\*/pairwise estimators,
and the same eight conditions apply unchanged — that is the point of the abstraction.

---

## FINDINGS (filled as the sub-arc runs)

### Exp 203 — the selection-gradient audit (MIXED / NEW INSIGHT, blind-verified AGREE)

**The reframe paid off.** Rather than run a fifth evolution, the audit MEASURES the local gradient
`g(0.10)` directly across the four ecologies. The headline is a genuine surprise that REFUTES the
pre-registered NEGATIVE_GRADIENT prediction: **in the band-staleness FORAGE regime there is a
POSITIVE local gradient at the resident** — a step `0.10 → 0.15` beats the resident in **7/8** seeds
(time-averaged invader fraction 0.872), reaching up through 0.30 (6/8) and 0.45 (7/8), and it
**survives the learning-rate-frozen control (6/8, auc 0.749)** so it is genuine thermosense, not
memory substitution. This is the **first positive local gradient in the whole arc**, and it explains
exp201's transient climb (a real gradient was pushing up).

**But it is purely competitive, and weak.** Three readings together pin the mechanism:
- `N*(h)` monomorphic carrying capacity is **monotone-decreasing** in every ecology (FORAGE
  1518 → 279; `argmax N* = 0.0`) — the organ is **pure cost at the population level**.
- `B(h) = r_costOFF` density-independent intrinsic growth is **flat/negative** (gift
  `B(0.60)−B(0.00) = −0.0016`, `dB/dh|_{0.10} = −0.005`) — a **lone** strong forager grows no faster.
- Yet the **pairwise** (relative, contested-resource) signal is clearly positive.

So the foraging benefit of precision **has no density-independent or population-level component** —
it exists ONLY as a relative advantage when competing for the limiting drifting-band food. By the
predeclared three-way rule this is **MIXED**: not POSITIVE (the strict confound control CLAMPED_LR is
6/8, one seed short of ≥7/8), not NEGATIVE (the gift is absent AND the resident slope is positive).

**Why the four walls still stand, refined.** A positive local gradient is **necessary but not
sufficient**. This one is weak (`s ~ 0.01`), purely relational (negative-frequency-dependent in
spirit — the advantage erodes as the trait spreads), and competes against drift and the rising linear
cost — so full evolution (exp201) saw it as a transient climb that decayed, never reaching functional.

**Reusable meta-lesson (kin of L22).** A benefit can be **invisible at equilibrium AND
density-independently** yet still produce a positive RELATIVE gradient — only the head-to-head
competitive assay reveals it. Measuring the wrong quantity (equilibrium intake, carrying capacity, or
a forced/behavioral 4× gift) mis-reads a purely-competitive trait as having no gradient. This is why
the four prior evolutions could not see what the audit measures.

**Instrument honesty.** A first run used the exp200 free-band-read regime (band centre known for
free) where precision buys nothing — disclosed, and the reason FORAGE was switched to band-staleness.
The B(h) measure was corrected mid-iteration (equilibrium intake → density-independent intrinsic
growth) when equilibrium washout was diagnosed. Engine features gated + hash-guarded; runtime
pre-flight (L25) gated the run as bounded.

### Exp 204 — residue / false-positive bridge (MIXED / NEW INSIGHT, blind-verified AGREE)

The bridge CHANGES THE PAYOFF STRUCTURE: instead of precision grabbing marginally more food (a
saturating benefit — the four-wall trap), precision AVOIDS A COSTLY MISTAKE. Eaten food leaves a
decaying residue; the creature reads a noisy freshness percept `f̂ = f + N(0, residue_confusion·(1−h))`
and eating a residue-dominated cell costs `residue_loss` (a false positive). Avoiding a loss is steeper
than grabbing a little more, and competition raises residue (rivals deplete → traces accumulate).

**The first functional MONOMORPHIC optimum in the arc.** At the fairest-shot regime (residue_loss=1.5)
a PURE high-precision population is genuinely fitter — `N*(h)` RISES (h\*=0.60) vs the no-residue
control's pure-cost decline (h\*=0.06). So *"precision never helps even when gifted"* is now **FALSE**
for loss-avoidance — a real positive note.

**But it is un-earnable, three ways.** (1) The LOCAL gradient at the resident is ≤0 — a fitness VALLEY
(pairwise 0.10→0.15 wins only 2/8; only the big leap 0.10→0.45 pays, 5/8). (2) A NEW failure mode: the
false-positive cost that makes precision valuable **COLLAPSES 2/5 populations**, so the evolution arm is
under-licensed → NO_VERDICT/MIXED (F2). (3) Cost-dominated mediation: precise creatures DO make fewer
false positives (rate 0.88→0.66) but reproduce LESS (0.94→0.55). Evolution decays to 0.07 (0/5
functional). The L22 gap, sharpened: a genuinely functional optimum, still un-reachable by small steps.

### Exp 205 — survivable-loss sweep: the valley is the barrier, not collapse (MIXED / NEW INSIGHT, blind-verified AGREE)

Resolves Exp 204's NO_VERDICT and **REFUTES its own predeclared mechanism**. Sweeping
`residue_loss ∈ {0.5,0.8,1.0,1.2,1.5}`: at the SURVIVABLE losses (0.8/1.2/1.5, pops ≥4/5 valid) the
monomorphic optimum IS functional (h\*=0.60) **and** the population persists — yet evolution STILL keeps
the sensor primitive (0/5 functional; mean climbs only 0.04→0.09 with loss vs the no-residue control's
0.029; local pairwise ≤0). So the demographic collapse of Exp 204 was **INCIDENTAL, not the cause**: the
**FITNESS VALLEY — the local gradient g(0.10) ≤ 0 — is the SOLE binding barrier**, operative even when
the population survives and a precise population is fitter in bulk. The generalised L22 in its purest
isolated form: a functional, survivable, bulk-fitter optimum is still un-evolvable when small steps
don't pay.

### Exp 206 — rotating-class niche / sympatric divergence: the SIXTH wall (NEGATIVE / NEW INSIGHT, unanimously blind-verified 3/3)

_(Renumbering note: the survivable-loss sweep took the Exp 205 slot, so this niche bridge became Exp 206;
sensor–controller co-adaptation — §4's original "206" — is now Exp 207, untested.)_

The LAST structurally-distinct escape: a **private niche** a high-precision lineage can only access by
discriminating an overlapping class signature, so crowding makes precision pay MORE as rivals herd into
the common class (positive frequency-dependence). Designed by a **17-agent design + adversarial-confound-
audit workflow** (dossier: `experiments/outputs/exp206_design_audit.json`). The **load-bearing fix all
four confound-auditors converged on**: the niche class must be **NON-MEMORIZABLE** — carried on a
time-ROTATING per-cell field `j(pos,t)=floor(K·frac(class_phase[pos]+frac(t·niche_rotation)))` — because
a static spatial class is encoded for free by the learned map `m[cell]` and `freeze_learning_rate` cannot
kill it (the confound that sank every naive niche design). `h` keys ONLY the noisy read of a cell's
current class in routing; the eat step is h-BLIND (`kept = consume(deficit)/(1+niche_crowding·
occ_prev[j_true])`, the crowding divisor a creature-COUNT on the TRUE class); no access gate. Anti-cheat
verified by a guard test: at `niche_confusion=0` intake is byte-identical across `h`.

**Result: it fails like the rest — and it is the CLEANEST wall.** Across fresh seeds {90–94} precision
decays 0.10→**0.027** (0/5 functional), the monomorphic optimum is the cheapest sensor (h\*=0.0), and a
precise invader does not out-reproduce the resident (pairwise 0.10→0.15 won 3/8, auc 0.509; 0.10→0.30 won
1/8). Crucially the population stays **HEALTHY (min 586, 5/5 valid, corr(pop,h)=−0.15)** — so, unlike
Exp 204/205, the failure is **PURELY the local gradient ≤0, with demographic collapse fully excluded**.
The cost-off gift is real but tiny (+0.0014). Controls all primitive (STATIC_NICHE 0.027 — not memory;
CLAMPED_LR 0.030 — not memory-substitution; CONFUSION_0/BARCODE_SHUFFLED pairwise ≤4/8 — anti-cheat
clean). Unanimously blind-verified 3/3 (a standard recompute + an adversarial skeptic that found no
hidden positive + a validity auditor that confirmed the wall is valid, not a NO_VERDICT). Honest
caveats: the gift is tiny, and `I(h;niche)=0.0003 bits` is ~0 *circularly* (h stayed primitive → no
variation), so the niche's non-inertness rests on the control deltas (STATIC 2/8 vs base 3/8, CLAMPED_LR
5/8) + the SINGLE_NICHE collapse (no escape class → extinction), not on MI.

---

## Final summary (the six directive questions) — the sub-arc CONVERGED

Six structurally-distinct escapes — **avoidance (199), foraging (200), increasing-returns tracking
(201), interference competition (202), the selection-gradient audit + residue/false-positive bridge
(203–205), and a rotating private niche (206)** — converge on one answer: **a costed sense does not
become a functional organ at this toy substrate, because the local selection gradient at the resident
is ≤ 0.** (203 is the band-staleness FORAGE gradient *audit*; 204–205 the residue/false-positive bridge.)

1. **Did any environment create a positive local selection gradient?** Only a WEAK, PURELY RELATIONAL
   one. The band-staleness FORAGE regime (203) showed the first positive local gradient (0.10→0.15 wins
   7/8) and the residue regime a marginal push — but never strong, never density-independent, never
   enough to climb. The niche regime (206) was neutral-to-negative (3/8).
2. **Did full evolution follow that gradient?** NO. Where a weak relational gradient existed (201/203)
   evolution showed only a TRANSIENT climb that DECAYED; everywhere else `h` decayed to primitive
   (0.027–0.09 across the six escapes).
3. **Functional globally, by lineage, or not at all?** NOT AT ALL — 0/5 functional in every verdict arm;
   no stable high-h lineage (206 `max_lineage_h` = 0.027 ≪ 0.30).
4. **Robust across seeds?** YES — the failure is robust (8-seed audits + 5 fresh-seed evolutions per
   verdict, all converging on primitive), and blind-verified every time (203–206; 206 thrice).
5. **Information use, not hardcoding/confound?** The weak gradients that existed were GENUINE (survive
   the CLAMPED_LR / STATIC_NICHE / CONFUSION_0 / BARCODE_SHUFFLED controls; no direct-h-reward,
   guard-tested + blind-verified). The FAILURE is real, not a confound — it is `g(h_res) ≤ 0`.
6. **What condition appears necessary for future senses?** Condition **(4)/(5)** of §4: the LOCAL marginal
   benefit at the resident must exceed the marginal cost. The maximally-sharp generalised **L22**: a
   bulk / installed / functional-when-gifted / **demographically-affordable** optimum is **NOT sufficient**
   for evolvability — only the **sign of `g(h_res)`** decides. For a costed sense to become an organ,
   precision must pay at the MARGIN, locally; none of the six toy regimes achieved that. The likely
   missing ingredients, out of this toy's scope: a **non-saturating, non-relational, locally-steep**
   affordance, and/or a controller `θ` that converts a small precision gain into a step-change in action
   value (the untested **Exp 207** sensor–controller co-adaptation; the synthesis flags it as unlikely to
   change the gradient sign).

**Bottom line.** The sense-axis program turned "make thermosense win" into the reusable, falsifiable
meta-question of §1 and answered it cleanly at toy scale: across six structurally-distinct worlds, a
costed sense is **un-evolvable** because the local gradient at the resident never turns sufficiently
positive — even when precision is genuinely valuable in bulk and the population survives. This is the
strongest, best-earned closing statement of the population-ecology line. **STATUS: CONVERGED** — the
structurally-distinct escapes are exhausted; the formal direction decision (flip the card closed-negative
/ run Exp 207 / a direct-occupancy bullet-proofing re-run of 206 / redirect off the ecology line) awaits
an explicit human word (post-206 consult in `loop/IDEAS.md`).
