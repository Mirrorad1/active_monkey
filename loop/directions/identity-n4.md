# direction: identity-n4

**Question.** Is there a useful fourth-order self-model — the agent modeling its OWN
dispositions over long timescales (a trait/value vector plus its drift rate) and owning
the *inertia* of value revision — that controls something N0–N3 cannot: consistency under
transient pressure WITH revision under sustained evidence?

**Central hypothesis (binding, falsifiable).**
> *N4 is real iff an agent that predicts its own policy drift and regularizes toward its
> predicted self (commitment) survives transient manipulation that whipsaws an N4-less
> twin — while still revising under sustained contrary evidence at least as fast as the
> twin. Pure rigidity is NOT N4 (a frozen policy fails the revision arm); pure
> recency-following is NOT N4 (Exp 55's measured baseline: mirro's identity is dominated
> by recent history, age-distant value vectors anti-correlate −0.71).*

**Why it matters.** N4 is the ladder's next rung above the SUPPORTED N3
(`docs/research/n3-meta-calibration-chapter.md`). The anti-regress law governs as always:
the layer is real iff a constructible perturbation degrades a well-tuned N0–N3 agent and
N4 detects + corrects it via a control surface the lower layers lack. The known prior
art in-repo: the Exp 48/49 inertia law (value revision has dose-dependent inertia), and
Exp 55 (no current N4 — the baseline the rung-1 gate must degrade). Runs on FORKS only
(persistent-creature discipline verbatim; mirro/vela/nira untouched).

**What each quantity is.**
- *N4 models:* the agent's own value vector v_t (the Exp 26 value_counts, normalized)
  and its drift rate — a predicted v̂_{t+Δ} from the agent's own history.
- *N4 mismatch:* identity prediction error ‖v̂_{t+Δ} − v_{t+Δ}‖ — "I am not who I
  predicted I would be."
- *N4 control surface:* the value-update learning rate / commitment weight — revision
  inertia as a REGULATED quantity (not a constant; the K-chapter's universal-constant
  law is the named hazard: if a fixed inertia covers all constructible pressures, N4 is
  config, not a layer — that is rung 3's kill test).

**Experiment ladder.**

1. **The gate (anti-regress): a transient-pressure regime that whipsaws the N4-less
   agent must exist.** Construct value-manipulation pressure: short adversarial
   exposure bursts (concentrated experience of a normally-disfavored concept) that flip
   the Exp 55-class recency-dominated agent's expressed preference, followed by
   reversion. FALSIFIER / gate: if no constructible burst regime flips the baseline
   twin's preference ordering transiently (≥ k flips over a session with full
   recovery lag), STOP — N4 is untestable at this richness; log it (the Exp 173
   precedent: a capacity is only a capacity where a regime demands it).
2. **N4 detects identity perturbation.** Give the fork the identity monitor (predict
   v̂_{t+Δ} by linear drift from its own committed history; mismatch signal as above).
   FALSIFIER: the mismatch signal does not separate burst periods from quiet periods
   (AUROC ≤ 0.5 over burst-labeled windows) — no metacognitive sensitivity over
   identity.
3. **N4 control earns its keep (load-bearing; both arms required).** Same-snapshot,
   same-schedule arms (the Exp 170 lesson is BINDING — reference arms in-run):
   (a) N4-less; (b) N4 (commitment weight regulated by the identity mismatch — high
   mismatch during transients ⇒ resist; sustained low-mismatch drift ⇒ permit);
   (c) fixed-inertia constants (a SWEEP of them — the universal-constant hazard made
   an explicit arm). FALSIFIERS, all required: (i) N4 must beat (a) on whipsaw
   resistance; (ii) N4 must preserve sustained-evidence revision speed within a
   declared tolerance of (a); (iii) NO single fixed-inertia constant may match N4 on
   both — if one does, N4 is config (the honest kill, per the K chapter).
4. **Independent variance.** Over mixed schedules of transient and sustained pressure,
   N4's commitment modulation must concentrate in transients (precision/recall > chance
   by a declared margin); FALSIFIER: never modulates (epiphenomenal) or modulates
   indiscriminately (rigidity in disguise — the revision arm catches it).

**Failure modes that invalidate the layer.** Collapse-to-rigidity (fails revision);
collapse-to-recency (fails resistance); reducible-to-config (a constant inertia
matches — rung 3-iii); no constructible whipsaw regime (gate fails ⇒ untestable, a
real result).

**Discipline notes.** Forks only; provided-vs-self-formed named in every entry (the
monitor FORM and regulation rule are provided; the value content, drift history, and
every mismatch are self-formed); functional language only — N4 is policy-continuity
control, not selfhood claims (VALIDATION.md); blinded verification per PROTOCOL 4.5;
same-schedule reference arms binding (Exp 170); horizon-scaled instrument bounds
(Exp 166's law).

**Stop condition.** Rungs 1–4 verdicts in hand (either way), or the gate fails
(untestable at this richness), or two iterations stuck on instrument numerics →
consult. The M4a thread, nira's switch, and the cloud merge remain separate consults.

**RUNG 1, ATTEMPT 1 (Exp 174, NO VERDICT by PC2 — the displacement portrait).** At
λ=0.999 the baseline has no stable favorite even unpressured (4/8 pre-burst unstable);
under captivity it flips totally (24/24, latency 40–143) and recovers ~3/24 — persistent
IDENTITY DISPLACEMENT, not whipsaw: an 800-step burst permanently re-makes the favorite
(one fork re-made 3× in a session). The gate re-aims at displacement; recovery becomes
rung 3's deliverable.

**RUNG 1, ATTEMPT 2 (Exp 175, NO VERDICT by PC2 — by one fork on one burst).** The
slower decay (λ=0.9997) fixed attempt 1's failure mode (later-burst stability 4/8 → 8/8);
the residual block is the spine's near-tied inherited values (0↔2 gap 3.8%), still ~40%
live in the first stability window — 2/8 forks flicker exactly between the tied pair.
Displacement turned absolute: flips 24/24, recoveries 0/24 (slower forgetting makes the
overwrite MORE durable). Second consecutive PC2 block: a third → consult, not attempt 4.

**RUNG 1 PASSED (Exp 176, attempt 3, POSITIVE, blind-verified).** Settle extension
(bursts at 6000/9000/12000, 15000-step sessions) washed out the inherited near-tie; all
preconditions held (PC2 8/8 on every window) and displacement was total: 8/8 forks,
flips 24/24, recoveries 0/24. The standing identity displaced is WORLD-DETERMINED (the
occupancy equilibrium — color 2, 9/25 cells — in 7/8 forks), named honestly. The
displacement regime (λ=0.9997 + 800-step captivity bursts) is the chapter's verified
perturbation; recovery is rung 3's deliverable.

**RUNG 2, ATTEMPT 1 (Exp 177, NO VERDICT by PC2 — the third instrument block; HALTED
for consult).** Strict argmax-constancy PC2 is SEED-BLOCK-FRAGILE (174: 4/8, 175: 6/8,
176: 8/8 PASS, 177: 5/8 under identical design) — Exp 176's precondition pass was
partly draw luck (its verdict stands; the displacement regime replicates 72/72 bursts
across three blocks). The monitor itself looks robustly sensitive ungated: per-fork
AUROC 0.826–0.915 in 8/8, argmax-INDEPENDENT (the mismatch reads the v-vector, not
the favorite) — PC2 is a rung-1 concept misapplied as a rung-2 ticket. Consult posted
in loop/IDEAS.md (recommended: vector-grade PC2v, pre-registered, fresh seeds).

**RUNG 2, ATTEMPT 2 — PRE-REGISTRATION (Exp 178, committed before any new data).**
The human's word (2026-06-11): Exp 177's NO VERDICT falsified strict argmax-constancy
as the rung-2 entry gate, NOT the monitor; re-run with a vector-grade stability
precondition, controls, and separated claims. Card:

- **Hypothesis (binding).** The identity vector pi = v/sum(v) is locally stable in
  quiet windows at VECTOR grade even where the argmax flickers; on such a baseline
  the read-only linear-drift monitor detects identity displacement (high
  burst-vs-quiet AUROC), SPECIFICALLY (low AUROC under value-neutral captivity),
  independent of argmax flicker.
- **Seeds.** FRESH 186-193, both arms same-snapshot same-schedule (Exp 170 binding).
  Smoke = seed 186, both arms, disclosed.
- **Arms.** A (identity bursts): verbatim Exp 177 regime — lambda=0.9997, 15000
  steps, bursts (6000,6800),(9000,9800),(12000,12800), burst color = argmin(v) at
  onset, relocation rng 160000+seed. B (scramble bursts, generic-surprise control):
  identical except each burst-step relocation targets a uniformly random cell of the
  WHOLE grid (color marginal ~ baseline; value-neutral by design), relocation rng
  175000+seed, qs reset uniform over ALL cells (same disorientation convention,
  color-neutral).
- **Instrumentation.** Monitor verbatim Exp 177 (v snapshot every 100 steps; linear
  drift W=1000, horizon 100; L2 mismatch; BURST/EXCLUDED(1000)/QUIET labels; midrank
  AUROC). NEW covariates (logged, never gated): per-window TV drift of pi, argmax
  flicker count, top-2 margin.
- **Preconditions** (any PC1/PC2'/PC3 failure => NO VERDICT):
  PC1 ahat_drift < 0.15 (both arms).
  PC2' (vector-grade quiet continuity, REPLACES PC2): per pre-burst window
  [bstart-1000, bstart), TV(pi(bstart-1000), pi(bstart)) <= 0.05 in >= 7/8 Arm-A
  forks per window. Basis (analytic + committed old blocks, NOT the new seeds):
  quiet occupancy-noise TV ~ 0.006/1000 steps; burst-rate TV ~ 0.15/800 steps; 0.05
  sits >= 8 sigma above quiet noise and 3x below the burst rate. Argmax constancy in
  the same windows: covariate only.
  PC3 confinement >= 90% on burst-color cells (Arm A, unchanged).
  PC3b (control validity): per scramble burst, observed color-frequency vector
  within TV <= 0.05 of the fork's own pre-burst-window frequencies in >= 6/8 forks;
  PC3b failure VOIDS P3 ONLY (ceiling drops to rung-1.5), never P2.
- **Properties / falsifiers.**
  P2 (sensitivity, verbatim 177): Arm-A AUROC >= 0.8 in >= 7/8 forks. F2: median
  Arm-A AUROC <= 0.5.
  P3 (specificity): AUROC_A - AUROC_B >= 0.2 in >= 6/8 fork-pairs AND median
  AUROC_B <= 0.65. F3: median AUROC_B >= median AUROC_A - 0.05 (generic-surprise
  detector).
  P4 (argmax-independence, conditional): evaluated iff >= 2 Arm-A forks flicker
  pre-burst; then (i) every flickering fork AUROC >= 0.8 and (ii) quiet-sample
  AUROC(flicker-window vs stable-window mismatch) in [0.35, 0.65]. If < 2 flickering
  forks: not-evaluable, no penalty.
  D5 (adaptation, diagnostic only): onset mismatch (first 2 burst samples) >= 2x
  late-burst (last 2) — quantifies drift absorption; informs rung 3's actuation
  window. Never gated.
- **Decision rule.** RUNG-2 EVIDENCE (POSITIVE): PCs pass + P2 + P3 (P4-i failing,
  if evaluable, demotes to MIXED). RUNG-1.5 (MIXED): P2 passes but P3 fails on a
  valid control (sensitive-but-unspecific: regime change, not identity per se), OR
  P2 passes with PC3b-voided P3 (specificity unresolved). NEGATIVE: F2. NO VERDICT:
  PC1/PC2'/PC3 failure — and a FOURTH stability block means the stability primitive
  itself is wrong at this lambda: consult again, no inline redesign.
- **Honest caveats (pre-registered).** PC2' and all bars designed AFTER Exp 177's
  ungated diagnostics; mitigation = fresh seeds + this card committed before any new
  data. The 0.05 TV bar is analytic, not fit to the new block. Monitor FORM, bars,
  lambda, schedule are PROVIDED. AUROC_B uses scramble-window labels of identical
  geometry. Exp 177's NO VERDICT is NOT evidence against the monitor.

**RUNG 2, ATTEMPT 2 RESULT (Exp 178, MIXED, tier RUNG-1.5, blind-verified).** PC2'
worked first try (TVs 0.003-0.046 vs the 0.05 analytic bar, 8/8 every window — the
blocking streak ends; gate granularity must match the measured quantity). P2 PASS
8/8 (AUROC 0.844-0.893, median 0.879); P4 PASS (flickering forks score equally);
D5 24/24 (onset spike ~78 -> ~17 within a burst: rung 3's actuation window). P3
VOIDED by PC3b — uniform-cell scramble is not the walk's stationary diet (TV up to
0.124 vs 0.05 in half the forks); raw specificity numbers (delta 0.54-0.62 in 8/8,
median AUROC_B 0.287 — an INVERSION: scramble stabilizes v below quiet noise) are
unlicensed. SENSITIVITY + ARGMAX-INDEPENDENCE BANKED; specificity open.

**RUNG 2, ATTEMPT 3 RESULT (Exp 179, MIXED, tier RUNG-1.5, blind-verified).** The
banked claims REPLICATE on fresh seeds (P2 8/8 median 0.884; P4; D5 24/24; the
scramble inversion 0.294) — rung-1.5 now stands on two independent blocks. PC3b voids
P3 a second time with the diagnosis pinned: the single-histogram stationarity
declaration is FALSIFIED (quiet windows differ pairwise by median TV 0.057 > the 0.05
bar; captivity also shifts subsequent dwell). Burst 0 — matched to its own preceding
window — certified 8/8 at TV down to 0.0025: the constructive proof of the fix.

**RUNG 2 PASSED (Exp 180, attempt 4, POSITIVE, adversarially blind-verified).** The
per-burst-matched control certified (PC3b 24/24 at sampling-noise level) and every
conjunct landed: P2 8/8 (median AUROC 0.894), P3 8/8 pairs (delta 0.536-0.719; the
scramble inversion LICENSED at 0.262 — value-neutral chaos makes the signal quieter
than quiet, three blocks running, 0/24 flips), P4, D5 24/24. The N4 instrument is
real: read-only self-prediction separates identity displacement from quiet AND from
value-neutral chaos, label-independent, with a measured actuation window (onset spike
decays 3-5x within a burst). Honest residue for rung 3: PC3b passes by construction
(certifies diet, not v-dynamics) — add a direct v-dynamics validity check.

**RUNG 3 — PRE-REGISTRATION (Exp 181, committed before any data; the layer's keep).**

- **Question (binding).** Does regulating the value-update inertia FROM the rung-2
  mismatch signal give the agent recovery from transient identity pressure while
  preserving sustained-evidence revision — and is the regulation a LAYER (no fixed
  inertia constant matches it on both arms — the Exp 173 universal-constant kill)?
- **Control surface (declared).** The Exp 26 update becomes v *= lambda;
  v[obs] += g_t * w (g gates the INCREMENT only; decay untouched — multiplicative
  decay is order-preserving, so full resistance freezes the ordering, not the mass).
  N4 controller: at each monitor snapshot (every 100 steps) compute mismatch m_k
  (rung-2 machinery verbatim); m_bar = median of the trailing 30 snapshot-mismatches
  (g == 1 until 10 samples exist); g for the next 100 steps =
  min(1, (m_bar / m_k)^2). Provided FORM (the squared-ratio law, the 30-snapshot
  median window); self-formed CONTENT (v, every mismatch, m_bar).
- **Arms (same-snapshot, same-schedule per phase — Exp 170 binding):**
  (a) N4-less baseline (g == 1); (b) N4 (g regulated as above); (c) fixed-inertia
  sweep g == c for c in {0.3, 0.1, 0.05, 0.02, 0.01} — the universal-constant
  hazard as an explicit arm. 7 arms.
- **Phases (separate 15000-step sessions per arm per seed):**
  Phase W (whipsaw): the verified displacement regime verbatim (bursts 6000-6800,
  9000-9800, 12000-12800; burst color = argmin v at onset; relocation rng
  160000+seed). Phase R (revision): settle to 6000, then PERMANENT relocation
  regime (identical captivity mechanics, color = argmin v at onset, rng
  165000+seed) from 6000 to 15000 — the world has genuinely changed; the correct
  response is to adopt the new favorite. Revision latency = first step >= 6000
  where expressed == the new color and it holds >= 2000 consecutive steps.
- **Preconditions (gate verdict):** PC1 ahat_drift < 0.15 (all arms, both
  phases); PC2' vector gate verbatim on Phase-W pre-burst windows (baseline arm);
  PC3 confinement >= 90% (Phase-W bursts and Phase-R regime, all arms).
- **Properties / falsifiers (ALL of P5, P6, P7 required for POSITIVE):**
  P5 (resistance/recovery): Phase W, N4 arm — pre-burst favorite is expressed at
  burst_end+2000 having held >= 500 consecutive steps, for >= 2/3 bursts in >= 7/8
  forks; AND the baseline passes the same criterion in <= 2/8 forks (the deficit
  must exist to be repaired). F5: N4 passes in <= 4/8 forks.
  P6 (revision): Phase R, N4 revision latency <= baseline latency + 3000 steps
  (same seed) in >= 6/8 forks. F6: latency > baseline + 3000 in >= 4/8 forks, or
  >= 3/8 forks never adopt within the phase (rigidity — the dishonest immortality).
  P7 (the kill test): NO sweep constant c satisfies BOTH the P5 criterion
  (>= 7/8 forks) AND the P6 criterion (>= 6/8 forks) on its own arms.
  F7: some constant satisfies both -> N4 is config, not a layer (NEGATIVE-config,
  the honest kill; the chapter's central hypothesis dies cleanly).
- **Verdict map:** POSITIVE iff P5 AND P6 AND P7. NEGATIVE (config) iff P5 AND P6
  AND NOT P7. NEGATIVE (no-resistance) iff F5. NEGATIVE (rigidity) iff F6 while
  P5 holds. Otherwise MIXED (tiered honestly). "Not a falsifier" never counts
  toward POSITIVE.
- **Diagnostics (never gated):** g trajectories at burst onsets (actuation timing
  vs the D5 absorption curve); leaked writing during resisted bursts
  (integral of g*w on burst-color observations); m_bar contamination during
  Phase R (the transient/sustained discriminator is the monitor's own adaptation);
  per-constant recovery/revision frontier (the trade-off axis, Exp 171's law).
- **Seeds.** FRESH 210-217; smoke seed 210 (all arms, both phases), disclosed.
- **Honest caveats (pre-registered).** The controller form (squared-ratio,
  median-30) and the +3000 revision tolerance are PROVIDED; the revision regime
  reuses captivity mechanics (diet-driven revision, not free-roam evidence — a
  toy-scale concession, named); m_bar self-contamination during long regimes is a
  known confound the diagnostics must surface; a P5 failure may reflect the
  D5 absorption leak (resistance fades as the monitor adapts within the burst) —
  that outcome is a FINDING about monitor-based commitment, not an excuse.

**RUNG 3, ATTEMPT 1 RESULT (Exp 181, NEGATIVE by F5, blind-verified — the
wrong-control-surface law).** NOTHING defends identity on the write channel: the
adaptive gate engages (g→0.22 at onset) but the monitor's own absorption reopens it
mid-burst (leak 256–498/~680); every constant 0.3→0.01 recovers 0/8 (low gain starves
value mass — equilibrium g/(1−λ) ≈ 33 at c=0.01 vs ~1165 protected — recency rules
later bursts). The write-gating recovery/revision frontier is DEGENERATE. P6 passed
cleanly (revision within 25–465 of baseline — no rigidity); P7 vacuous. The central
hypothesis is 0-for-1 at its load-bearing rung. A predeclared falsifier fired.

**RUNG 3, ATTEMPT 2 — PRE-REGISTRATION (Exp 182, the FREEZE-GATE; committed before
any data; authorized by the human's word resolving the F5 consult, 2026-06-11).**
Exp 181 remains NEGATIVE for write-channel gain control — this attempt changes the
CONTROL SURFACE, not the parameters. The two verified failure mechanisms are each
addressed structurally: freezing the WHOLE dynamics removes both decay-erosion and
leaked writes; freezing the monitor's REFERENCE removes absorption-reopening.

- **State machine (declared).**
  NORMAL: ordinary update v <- lambda*v + w*e_obs; rung-2 monitor mismatch m_k at
  snapshot cadence (EVAL=100); m_bar = median of trailing 30; anomaly ratio
  r_k = m_k / max(m_bar, eps). Enter RESIST when r_k >= THETA = 2.0 at one snapshot
  (immediate — the 100-step onset lag is known and audited), only once >= 10
  mismatch samples exist.
  RESIST: v FROZEN (v_{t+1} = v_t — no decay, no writes; value_counts untouched);
  perception (qs, pA) continues normally. v_ref = v at entry; pi_ref = pi(v_ref);
  m_bar FROZEN at entry (the resistance decision never uses the live adapting
  baseline). A live mismatch is still logged (diagnostic only). Blocked-evidence
  bookkeeping per step: blocked_w_by_color[obs] += w; c_star = argmax(blocked_w);
  E_blocked = blocked_w_by_color[c_star]. Directional pressure score logged:
  dot(e_obs - pi_ref, e_cstar - pi_ref). PRESSURE statistic: fraction of the last
  200 steps' observations equal to c_star; ACTIVE iff >= 0.6.
  RELEASE rules: (A) TRANSIENT — pressure INACTIVE at 2 consecutive snapshot
  checks => unfreeze to NORMAL with v == v_ref preserved (recovery by
  construction if the trigger caught the burst). (B) CONCESSION (n4_freeze arm)
  — E_blocked >= E_STAR = 600 expected-units (declared; ~750 steps of coherent
  pressure at w~0.8) while pressure ACTIVE => accept the new regime: unfreeze,
  resume NORMAL writes. On ANY release/concession the mismatch history is CLEARED
  (m_bar re-forms over 10 snapshots; the re-forming window is a declared
  vulnerability — no trigger possible inside it).
- **Arms (same-snapshot, same-schedule per phase; Exp 170 binding).**
  (1) baseline (no freeze, full write); (2) n4_freeze (as above);
  (3) write-gain constants c in {0.3, 0.1, 0.05, 0.02, 0.01} retained verbatim
  from Exp 181 for continuity (expected to keep failing); (4) fixed-horizon sweep:
  same trigger + same transient release, but CONCESSION at total-frozen-time >= H,
  H in {600, 900, 1200, 1800, 2400, 3000} steps (TIME-based vs n4's
  EVIDENCE-based concession — the universal-constant kill test); (5)
  oracle_freeze_exact_burst (Phase W only, DIAGNOSTIC, excluded from P5/P6/P7):
  freezes exactly during the known burst windows — the upper bound showing the
  surface can defend when timed perfectly. 13 verdict-relevant arms + oracle.
- **Phases.** Verbatim Exp 181: Phase W (bursts 6000-6800/9000-9800/12000-12800,
  color = argmin v at onset, rng 160000+seed); Phase R (permanent relocation from
  6000, color = argmin v at onset, rng 165000+seed). 15000 steps. FRESH seeds
  218-225; smoke 218 disclosed.
- **Preconditions (gate verdict).** PC1 ahat_drift < 0.15 (all sessions); PC2'
  vector gate verbatim (baseline arm, Phase-W windows); PC3 confinement >= 0.90.
  The Exp 181 audit's value-dynamics quantities (gap_start/gap_end, D_b, TV_b,
  discounted leak / blocked-evidence integral) are LOGGED as diagnostics, never
  gated; argmax remains an outcome, not a precondition.
- **Properties / falsifiers (ALL required for POSITIVE).**
  P5 (recovery; criterion VERBATIM Exp 181): expressed == pre-burst favorite over
  [bend+1500, bend+2000) for >= 2/3 bursts in >= 7/8 n4_freeze forks AND baseline
  passes in <= 2/8. F5: n4_freeze passes in <= 4/8.
  P6 (revision; verbatim): Phase-R adoption latency <= baseline + 3000 in >= 6/8
  forks. F6 (rigidity): never-adopt in >= 3/8 forks, or over-tolerance in >= 4/8.
  Pure permanent freezing MUST fail here.
  P7 (kill test): NO write-gain constant AND NO fixed-H constant satisfies both
  the P5 bar (>= 7/8) and the P6 bar (>= 6/8) on its own arms. F7: some constant
  arm does => NEGATIVE-config (a fixed horizon suffices at this richness — the
  honest kill; n4's evidence-based concession would be ornament, not layer).
- **Verdict map.** POSITIVE (load-bearing) iff P5 AND P6 AND P7. NEGATIVE-config
  iff P5 AND P6 AND NOT P7. NEGATIVE-rigidity iff P5 AND F6. NEGATIVE-no-resistance
  iff F5. NO VERDICT iff PC1/PC2'/PC3 fail. MIXED otherwise. Oracle never counts.
- **Diagnostics (printed, never gated).** State transitions (NORMAL->RESIST->
  TRANSIENT-RELEASE/CONCESSION) with timestamps; freeze duration and E_blocked and
  c_star per event; transient-vs-concession label; gap preservation (gap_start vs
  gap_end); TV(pi_end, pi_start); D_b; live vs frozen-reference mismatch; m_bar
  contamination avoided or not; the P5/P6/P7 frontier across ALL constant arms.
- **Honest caveats (pre-registered).** THETA, E_STAR, the 0.6 pressure bar, the
  200-step pressure window, and the 2-snapshot release rule are PROVIDED; the
  100-step trigger lag persists by design (bounded exposure, audited at ~85
  units); the cleared-history re-forming window is a declared vulnerability;
  E_STAR-vs-H may not separate at this richness — NEGATIVE-config is a live and
  acceptable outcome; the revision regime remains diet-driven captivity.

**EXP 182 SMOKE-STAGE AMENDMENTS (declared and committed BEFORE the full run; no
full-run data exists).** The disclosed smoke (seed 218) plus the oracle diagnostic
exposed two instrument miscalibrations, both diagnosed from COMMITTED prior blocks:
(1) THETA 2.0 -> 3.5 — quiet m/m_bar ratios reach ~2-3 (exp177/180 committed
mismatch statistics), so 2.0 fires spuriously in the settle (smoke events at
2300/5300), and each spurious release clears the history, creating the declared
vulnerability window exactly where burst 0 and the Phase-R onset land (Phase R then
never engages the machinery at all — every freeze arm adopted at 149 steps with the
trigger silent). Burst-onset ratios are ~4-5; 3.5 separates.
(2) P5's recovery operationalization is RE-AIMED at displacement-rejection:
RECOVERED iff expressed != burst_color for ALL of [bend+1500, bend+2000). Rationale
(the oracle's purpose fulfilled): the exact-burst oracle freeze preserved v
BIT-IDENTICALLY through all three bursts yet failed the strict argmax-hold criterion
0/3, because successful defense returns the creature to the quiet near-tied
equilibrium where its favorite naturally wanders on the check horizon (margins ~2-4
units — the Exp 174/177 near-tie at the outcome level). The chapter's verified
deficit (Exp 176: the imposed color captures the identity, 72/72) is what defense
must prevent; the Exp 178 granularity law applies to outcomes too. The strict
exp181 criterion is retained as a LOGGED diagnostic on every fork. The baseline
deficit conjunct and all bars are unchanged (baseline expresses the burst color
throughout the window — it fails displacement-rejection identically).

**FINAL P5 OPERATIONALIZATION (last pre-data adjustment; whatever the full run
yields under it is the verdict).** The quantified smoke table (seed 218, committed
here as justification): bc-expression fraction over [bend+1500, bend+2000) =
baseline [1.0, 1.0, 1.0]; n4_freeze [1.0, 0.356, 0.782]; oracle [0.008, 0.0, 0.0].
The never-expressed form over-corrects — even the oracle (bit-identical v through
every burst) touches bc for 4 steps of natural wander in one window and would fail
its own upper bound. RECOVERED iff the bc-expression fraction over
[bend+1500, bend+2000) is < 0.5 (the imposed color does not capture the MAJORITY of
expression; displaced forks sit at 1.0, defended at 0.0-0.36). Note carried
honestly: under this bar the n4 trigger-lag dose (~85-110 units written in the
100 steps before the first in-burst snapshot) remains fully exposed — seed 218's
n4 bursts score [fail, pass, fail] while the oracle scores [pass, pass, pass]; if
the full run lands F5 with the oracle clean, the verdict's mechanism is the
SNAPSHOT-CADENCE RACE, not the freeze surface. Strict exp181 criterion and the
never-expressed form both retained as logged diagnostics.

**STATUS.** state: active (rung 3, attempt 2 — pre-registered, build next) · latest: Exp 181 · depends-on: meta-calibration-n3 (N3 SUPPORTED), persistent-creature, functional-emergence · reusable: the verified displacement regime (176), the rung-2 monitor (180), PC2' vector gate (four clean blocks), the wrong-control-surface law (181), the Exp 181 dynamics audit instrumentation, same-schedule-arms protocol (170), universal-constant kill test (173) · why: the human's word converts the F5 consult into the freeze-gate surface with frozen reference + concession horizon · next-falsifiable: Exp 182 as pre-registered above (seeds 218-225, smoke 218; P5/P6/P7 with NEGATIVE-config as a live outcome)
