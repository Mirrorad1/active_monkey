# When does a costed sense become an organ? — the sense-axis bridge

**Status:** ACTIVE (sense-evolution sub-arc, Exp 203–206; human steer 2026-06-13).
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

### Exp 204 / 205 / 206 — the bridges
_(staged behind the post-203 consult.)_

## Final summary (the six directive questions)
_(answered after the sub-arc reaches its stop condition.)_
1. Did any environment create a positive local selection gradient?
2. Did full evolution follow that gradient?
3. Did the sensor become functional globally, by lineage, or not at all?
4. Was the result robust across seeds?
5. Was it caused by information use rather than hardcoding/confound?
6. What condition appears necessary for future senses?
