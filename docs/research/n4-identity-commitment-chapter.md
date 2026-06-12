# N4 Identity / Commitment Chapter

**Status:** CHAPTER CLOSED on the human's explicit word (2026-06-11): option (a) of the
Exp 183 consult — accept NEGATIVE-config as the rung-3 verdict, synthesize, close the
card. No Exp 184 was run. The seed-229 crack is logged as a future crack only (§7).
This document supersedes the same-day resumption-pattern draft
(`docs/research/n4-identity-chapter.md`, removed in this commit); every number below is
taken from the committed scripts, raw outputs, and rows/events/verdict JSON of
Exp 174–183, re-extracted from the artifacts for this synthesis.

**Honesty header.** "Identity" throughout means the creature's value vector and its
ordering — functional policy-continuity, no selfhood claims (loop/VALIDATION.md). Every
instrument and controller FORM (monitor, gates, triggers, thresholds, horizons) is
PROVIDED design; what the creature self-forms is the contents: its values, its drift
history, every mismatch. All experiments ran on forks of mirro; the spines
(mirro/vela/nira) were never touched.

---

## 1. Executive verdict

- **Rung 1: POSITIVE — displacement regime verified (Exp 176).** A constructible
  800-step captivity permanently re-makes the layerless, value-recency baseline's
  favorite: 8/8 forks, flips 24/24, recoveries 0/24 (and 96/96 flips across the four
  blocks of Exp 174–177).
- **Rung 2: POSITIVE — identity monitor real (Exp 180).** Read-only linear-drift
  self-prediction separates identity displacement from quiet life (median AUROC 0.894,
  8/8 ≥ 0.8) and from locally-matched value-neutral scramble (Δ 0.536–0.719 in 8/8
  pairs), argmax-independently, with a measured adaptation window (onset spike decays
  3.3–5.2× within a burst, 24/24).
- **Rung 3: NEGATIVE-config — regulated commitment not load-bearing (Exp 181–183).**
  Write-gain control defends nothing at any gain (181). The freeze surface is
  sufficient in principle — the oracle defends 8/8 (182, 183) — but the evidence-based
  concession surrenders mid-attack (183: F5, n4 2/8), while fixed horizons
  H1200/H1800/H2400/H3000 pass BOTH declared bars (183: F7).
- **Final chapter grade: the monitor is real; commitment control as a LAYER is
  unsupported at this richness.** Formally, F5/no-resistance is the binding verdict
  tier of the pre-registered ordered map; F7/commitment-is-config is the reportable
  second-order kill-test finding.

## 2. The question

The binding hypothesis (loop/directions/identity-n4.md):

    N4 is real iff self-predicted identity drift controls value-revision inertia
    such that transient pressure is resisted, sustained evidence is accepted,
    and no fixed constant matches both.

Pure rigidity is not N4 (it fails the revision arm); pure recency-following is not N4
(Exp 55's baseline: mirro's age-distant value vectors anti-correlate −0.71). The
anti-regress law governs: the layer is real only if it detects and corrects a
constructible degradation via a control surface the lower layers lack — and the
K-chapter's universal-constant law (Exp 173) is carried as an explicit kill-test arm:
if a fixed constant matches the regulator on both arms, the "layer" is config.

## 3. Core variables and equations

State and identity:

    v_t          = value vector (3 colors; equilibrium total mass 1/(1-lambda) ~ 3333)
    pi_t         = v_t / sum(v_t)            (identity as a point on the simplex)
    favorite_t   = argmax_i v_t              (expressed preference — an OUTCOME, not a gate)

The read-only identity monitor (rung 2; snapshot cadence 100 steps, drift window
1000 steps, horizon 100 steps):

    v_hat_k  = v_{k-1} + (v_{k-1} - v_{k-11}) / 10     (linear drift from own history)
    m_k      = || v_hat_k - v_k ||_2                    ("I am not who I predicted")
    m_bar    = median of trailing 30 snapshot mismatches

Value update (the Exp 26 rule; lambda = 0.9997, w_t = exp(−H(predictive)) ∈ (0,1] the
per-step predictability weight):

    v_{t+1} = lambda * v_t + w_t * e_{o_t}

Exp 181 write-gated update (g gates the increment only; decay untouched):

    g_k      = min(1, (m_bar / m_k)^2)
    v_{t+1}  = lambda * v_t + g_k * w_t * e_{o_t}

Exp 182/183 freeze update (state machine; trigger r_k = m_k / m_bar >= THETA = 3.5,
m_bar FROZEN at RESIST entry; Exp 183 trigger: 100-step horizon sliding at 25-step
stride, floor-guarded across history clears, 200-step refractory):

    NORMAL:  v_{t+1} = lambda * v_t + w_t * e_{o_t}
    RESIST:  v_{t+1} = v_t                             (no decay, no writes)

Evidence-concession failure (the surrender schedule, measured in Exp 183):

    tau_concede ~= tau_detect + E_STAR / mean_pressure
                ~= 75 + 600 / 0.956  ~=  703 steps  <  attack_length = 800
    (observed: freeze entry at onset+75 modal; E_blocked hits 600 at ~entry+628;
     exit at 6700–6725 vs burst end 6800 — a 75–100-step unprotected tail)

Fixed-H success condition (measured: H600 1/8; H900 6/8; H1200–H3000 7/8 on defense):

    H > attack_length - tau_detect          (= 725 steps here; H900 is marginal)
    and H <= revision_tolerance_window      (Phase-R latency ~ H + 150 must stay
                                             <= baseline + 3000; H3000 brushes it)

Anti-regress / kill-test condition (P7 as pre-registered: a constant arm passes by
meeting the absolute bars R >= 7/8 defense and Q >= 6/8 revision; the general form):

    N4 is load-bearing only if no constant H or c matches both:
    R(constant) >= R_N4  and  Q(constant) >= Q_N4

In Exp 183 this failed in the strongest way: R_N4 = 2/8 while R(H1200..H3000) = 7/8 —
the constants did not merely match the regulator; they dominated it.

## 4. The experiment arc (Exp 174–183)

- **Exp 174–175 (instrument; NO VERDICT ×2):** at λ=0.999 the baseline has no stable
  favorite even unpressured (the gate's precondition fails); at λ=0.9997 the residual
  instability is the spine's inherited near-tie (0↔2 gap 3.8%), still ~40% alive in the
  first stability window. Both blocks show the substance: captivity does not whipsaw
  the favorite, it REWRITES it (flips 24/24 in each; recoveries ~3/24 then 0/24 —
  slower forgetting makes the overwrite MORE durable).
- **Exp 176 (rung 1 PASSES):** a 6000-step settle washes out the inheritance; all
  preconditions hold and displacement is total (8/8 forks, 24/24 flips, 0/24
  recoveries). The displaced identity is WORLD-DETERMINED (the occupancy equilibrium,
  color 2 at 9/25 cells, in 7/8 forks) — named honestly. Recovery becomes rung 3's
  deliverable.
- **Exp 177–180 (instrument refinement, argmax → vector):** Exp 177's strict
  argmax-constancy precondition proved seed-block-fragile (8/8 then 5/8 under identical
  design — the third such block) while the monitor looked sensitive ungated (AUROC
  0.826–0.915, 8/8). On the human's word the gate was replaced, pre-registered, with
  the vector-grade PC2′ (quiet TV(π) ≤ 0.05): it passed first try and on every block
  since. The scramble control then took two more iterations: uniform-cell relocation
  failed its own validity check (Exp 178 — uniform ≠ the walk's stationary occupancy);
  a single global occupancy histogram failed it again (Exp 179 — quiet windows differ
  pairwise by median TV 0.057, and captivity itself shifts subsequent dwell). The fix:
  match each burst to its OWN immediately-preceding window.
- **Exp 180 (rung 2 PASSES):** with the per-burst-matched scramble certified at
  sampling-noise level (PC3b 24/24, mean TV 0.0198 vs theoretical 0.0204), every
  conjunct lands: sensitivity AUROC_A 0.859–0.911 (median 0.894); specificity
  Δ 0.536–0.719 with median AUROC_B 0.262 — an INVERSION (value-neutral chaos makes
  the signal quieter than quiet; 0.287/0.294/0.262 across three blocks; the control
  flips 0/24 identities); argmax-independence; adaptation D5 24/24 (ratios 3.26–5.19).
- **Exp 181 (rung 3 attempt 1 — the wrong-control-surface law):** write-gain control
  defends nothing. The adaptive gate engages (g = 0.22 at the first in-burst snapshot,
  seed 210 burst 0) but the monitor's absorption re-opens it mid-burst (g rising
  0.22→0.55 within 4 snapshots; leaked writing 256–498 per burst of ~655–686
  unresisted); every constant c ∈ {0.3..0.01} recovers 0/8 because low gain starves
  maintenance (equilibrium mass g/(1−λ): ~33 at c=0.01 vs ~1165 in the protected
  favorite — c0.02/c0.01 hold burst 0 in 8/8 forks, then lose bursts 1–2 in all 8).
  The recovery/revision frontier of write-gating is degenerate: P5 ≡ 0/8 while median
  revision latency runs 200→1694. P6 passed (Δ 25–465 vs tolerance 3000) — the
  controller costs little; it buys nothing. NEGATIVE by F5.
- **Exp 182 (rung 3 attempt 2 — freeze-surface sufficiency plus timing residue):**
  whole-dynamics freezing with a frozen monitor reference eliminates both Exp 181
  mechanisms (no decay erosion, no write leak, no absorption re-opening — m̄ frozen in
  RESIST, structurally asserted). The ORACLE (exact-burst freeze) defends 8/8 forks,
  23/24 bursts. The creature's own trigger defends 6/8 — one fork short of P5 — failing
  by exactly the two pre-registered instrument gaps: the 100-step snapshot-cadence lag
  dose (~85 units written before the freeze engages; election-deciding at the near-tied
  equilibrium: gap_end +10 to +68 in the borderline bursts) and the cleared-history
  dead zone (a stray transient release wipes the mismatch history; the 10-snapshot
  re-forming window swallows a burst arriving 300–400 steps later whole: gap_end ~ −622
  to −624, trigger silent). P6 8/8 (n4 concedes by evidence at E_blocked ≈ 645–676);
  the H-sweep differentiates for the first time (H600 4/8; H900–H3000 plateau at n4's
  exact 6/8; H3000's P6 brushed by 25 steps). MIXED, between bands.
- **Exp 183 (rung 3 attempt 3 — surrender-schedule law and commitment-as-config):**
  both gaps were closed and verified closed: the sliding 100-step-horizon / 25-step-
  stride trigger engages at 75 steps modal (range 50–125), and the floor guard +
  refractory eliminated the dead zone (the Exp 182 dead-zone seeds recovered; no alarm
  storms — 4–11 events/session, transients cheaply released). The faster instrument
  then exposed the design flaw: E_blocked accumulates at ~0.956/step inside the frozen
  window, reaches E*=600 at ~entry+628, and the concession exits at 6700–6725 against
  a burst ending at 6800 — n4 collapses to 2/8 (F5). Meanwhile H1200/H1800/H2400 pass
  defense 7/8 + revision 8/8, and H3000 passes 7/8 + 6/8 — every fixed horizon from
  1200 to 3000 meets BOTH pre-registered bars (F7). Defense failures among passing
  H-arms trace to seed 229 alone (repeated-burst-color cumulation; H900 also loses
  seed 227, the same [1,1,x] schedule class); H3000's second P6 miss is seed 232, a
  +10-step tolerance brush, a different mechanism. The oracle again defends 8/8.
  NEGATIVE on the ordered map: F5 binding, F7 reportable.

## 5. Laws / durable findings

**A. Vector-identity law.** Identity is a vector on the simplex, not an argmax label.
The argmax-constancy gate was seed-block-fragile and blocked three experiments for no
epistemic gain; the vector-grade gate (TV(π) ≤ 0.05) matched to what the monitor
actually reads passed on six consecutive blocks, and the monitor's sensitivity is
independent of label flicker (Exp 177–180).

**B. Like-with-like-in-time control law.** Scramble controls must be matched to the
immediately preceding local window, not a stale global histogram: quiet-window
occupancy is non-stationary at the 0.05 grain (median pairwise TV 0.057) and the
perturbation itself shifts subsequent dwell (Exp 178–180).

**C. Read-only identity-monitor law.** Linear drift mismatch over the value vector
detects identity displacement (median AUROC 0.894) and is specific against
value-neutral scramble — which not only fails to light the signal but suppresses it
below quiet baseline (inversion, three blocks running). Detection comes with measured
habituation: the onset spike decays 3–5× within an 800-step burst (Exp 180).

**D. Wrong-control-surface law.** Write-gain control does not defend identity: decay
keeps eroding the old favorite while the gate only slows the pen, and the monitor's
absorption re-opens the gate mid-attack; constant low gain starves the very mass it
protects (~33 vs ~1165). The write channel's recovery/revision frontier is degenerate
(Exp 181).

**E. Freeze-surface sufficiency.** Freezing the whole value dynamics (no decay, no
writes, monitor reference frozen) preserves both ordering and mass: perfectly timed,
it defends 8/8 forks in both Exp 182 and Exp 183. After Exp 181/182, what separates
defense from defeat is purely timing.

**F. Detection-floor law.** Ratio triggers on this body require roughly 75–100 steps
of onset dose: at short horizons there is NO signal/noise separation (quiet ratio
P99.5 2.56–3.19 vs burst-onset max 1.82–3.02 at the 25-step horizon — the burst signal
integrates ~linearly with horizon while slope noise shrinks slower). The Exp 183
calibration probe, committed pre-data.

**G. Surrender-schedule law.** Any finite evidence-integral concession threshold can
be outlasted by an attack and becomes an attacker-set release time: with E*=600 at
~0.956 units/step, concession lands ~703 steps into an 800-step attack — the faster
the trigger, the earlier the surrender completes inside the burst. A larger E* shifts,
not removes, the problem (Exp 183).

**H. Commitment-as-config law.** At this richness, fixed-horizon freeze gates pass the
resistance/revision tradeoff (H1200–H3000: defense 7/8, revision within tolerance);
regulated commitment is not necessary — and the regulated version measured strictly
worse (2/8). The Exp 173 universal-constant law, landing at the identity level on its
first kill-test firing.

## 6. Seed-229 note (future crack, NOT a next step)

Seed 229's regime — repeated-burst-color cumulation (the same color attacking in
consecutive bursts, [1,1,x] schedules) — defeats ALL current arms: n4, every H, and the
oracle-adjacent settings each lose it on defense (H900 additionally loses seed 227 to
the same class). This may motivate future adversarial schedules with variable lengths
and repeated colors, where no single H covers because the attacker controls the dose
an H must outlast. It does NOT rescue the current N4 claim: Exp 183 already found
fixed-H arms satisfying both declared bars, so the kill test stands as run. The
Exp 173 caution attaches: cracks of this form have so far dissolved into wider
constants or richer-body requirements when pursued. Status: DEFERRED — logged here,
not authorized as Exp 184.

## 7. Honest limitations

- The verdict is "at this richness / this body / this controller class," not a
  universal impossibility claim. "N4 is false in general" is NOT the claim; the claim
  is: N4 commitment control is unsupported at this body/richness; fixed-H config
  suffices here.
- The fixed-H config arms use the SAME mismatch trigger as n4 (the freeze machinery is
  shared; only the concession rule differs) — "config" means fixed-H-on-a-mismatch-
  trigger, not pure no-monitor clockwork. The constant rides on the rung-2 instrument.
- E_STAR, THETA, the pressure window/bar, the horizons, the refractory, and every
  verdict bar are PROVIDED design constants.
- Phase R is diet-driven captivity (relocation mechanics reused as "the world has
  changed"), not open-world evidence.
- The oracle is a diagnostic upper bound with perfect knowledge, excluded from all
  properties; it is not an achievable controller.
- P5's majority displacement-rejection form (bc-expression fraction < 0.5) was amended
  pre-data from the strict argmax-hold form, justified by the quantified oracle smoke
  table — but it remains a design choice, and Exp 182 flagged one-fork sensitivity
  near the 0.5 bar (the verifier confirmed no plausible bar shift changes any Exp 183
  count).
- The final binding tier is F5/no-resistance (the pre-registered verdict map is an
  ordered tree; F5 fires before the config branch); F7 is reportable as the
  second-order config finding, and is reported as exactly that.
- Toy scale: a 3-color value space means one overwrite IS a new identity; the settled
  self is world-determined (the occupancy equilibrium), not biography-determined.

## 8. Closing

The N4 monitor exists, but N4 commitment control was not shown to be a new layer. The
constructive result is that identity defense is possible on this body, but by a
fixed-horizon freeze mechanism. The central N4 layer hypothesis is unsupported at this
richness.

## 9. Reproduce

Scripts: `experiments/exp174_n4_gate.py` … `experiments/exp183_n4_freeze_gate2.py`
(plus `experiments/exp181_dynamics_audit.py`). Raw outputs and rows under
`experiments/outputs/` (exp174–exp183; verdict JSON for the gated experiments exp176,
exp178–exp183; the three NO-VERDICT instrument blocks 174/175/177 have rows +
diagnostics only; the Exp 183 calibration probe committed pre-data). Every gated
verdict was blind-verified (PROTOCOL 4.5); pre-registrations and pre-data amendments
are separate commits whose timestamps precede their data. The numbers in this synthesis
were re-extracted from the committed artifacts by four independent readers and
cross-checked against the EXPERIMENTS.md entries (three attribution refinements from
that pass are incorporated above: E*=600 with ~703-step concession arithmetic; H900's
two P5 misses are seeds 227 and 229; H3000's second P6 miss is seed 232, a tolerance
brush).
