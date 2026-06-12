# direction: identity-n4-crack

**Question.** Does the seed-229-class regime — repeated-burst-color, variable-length attack
schedules — EMPTY the fixed-H interval that made Exp 183's rung-3 verdict NEGATIVE-config?
Formally (the human's mathematical target, 2026-06-11): a fixed horizon works iff some H
satisfies BOTH

    H >= max_b ( L_attack_b - d_trigger_b )        (defense: outlast every burst)
    H <= min_r ( T_revision_tolerance_r )          (revision: concede in time)

The crack exists only if the interval is empty —
`max_b(L_attack_b - d_trigger_b) > min_r(T_revision_tolerance_r)` — or if repeated-color
cumulation defeats all fixed-H arms across the declared defense/revision bars. Full supplied
program: `docs/research/n4-fixed-h-squeeze-plan.md` (read it before any rung).

**What this is NOT (binding fences, the human's words).** NOT a rerun of Exp 183; NOT a
reinterpretation of the closed N4 chapter — rung 1 POSITIVE (displacement regime), rung 2
POSITIVE (identity monitor), rung 3 NEGATIVE-config stand verbatim
(`docs/research/n4-identity-commitment-chapter.md`). Exp 183 outputs and verdicts are
READ-ONLY; the commitment-as-config result is not removed or weakened; no positive-N4 claim
is made without a later fresh-seed confirmation rung. The core discipline: the question is
not "can we make N4 pass?" — it is "does the seed-229/repeated-color regime remove the
fixed-H interval that made Exp 183 config?" If not, the chapter stays closed-negative.

**Why it matters.** Law H (commitment-as-config) rests on a covering constant EXISTING:
Exp 183 found the interval non-empty at one attack geometry (single 800-step bursts,
mostly distinct colors), where H1200–H3000 pass both bars. Seed 229's [1,1,x]
repeated-color schedule defeated every arm — n4, every H, and the passing arms each lose
exactly that seed. If attacker-controlled schedule geometry (length, repetition, spacing)
genuinely empties the interval while defense stays mechanically possible (oracle defends),
the universal-constant law (Exp 173, first kill at Exp 183) has its first real boundary —
and regulated commitment earns ONE honestly-gated re-test. If the crack dissolves into a
wider constant or a simple fixed add-on (the Exp 173 precedent, twice warned), the config
verdict is STRENGTHENED, with its domain mapped. Either way the chapter's §6 deferral is
resolved by measurement instead of caution.

**Validity requirements for a "real crack" (all four, predeclared).**
1. Fixed-H fails: no H in the swept range passes both declared bars in the cell.
2. Defense is mechanically possible there: the oracle (or a state-sensitive reference)
   defends — otherwise the cell is class E (impossible body), not a crack.
3. Not impossible-detection: cells where `L_attack - d_trigger <= 0` are detection-floor
   cells (Law F, ~75–100 steps irreducible for ratio triggers), not crack cells.
4. Not rescued by a modestly wider constant sweep or a simple fixed add-on (longer H,
   fixed cooldown, fixed refractory) — the Exp 173 dissolution test runs BEFORE any
   crack claim (class D check).

**Experiment ladder.**
1. **Seed-229 autopsy (diagnostic only; no verdict, no gate).**
   `experiments/exp183_seed229_autopsy.py` inspects/reconstructs seed 229 vs seeds
   226–233 from the COMMITTED Exp 183 artifacts (deterministic re-run only if the rows
   lack a needed series, with bit-match against committed numbers asserted). Logs: burst
   colors by index; repeats; v before/after each burst; pi_t = v_t/sum(v_t); directed
   drift D_b = pi_end[c] - pi_start[c] toward the repeated color; whole-vector
   TV_b = 0.5*sum_i |pi_end[i] - pi_start[i]|; cross-burst accumulation; mechanism class
   (trigger latency / release too early / repeated-color accumulation / insufficient
   refractory / revision-bar conflict / other). Outputs:
   `experiments/outputs/exp183_seed229_autopsy.{txt,json}`. MUST NOT modify any Exp 183
   artifact or verdict. FAILURE = no mechanism pinnable from committed artifacts and the
   reconstruction does not reproduce the committed trajectory → log and consult.
2. **Exploratory fixed-H squeeze map (Exp 184, labeled EXPLORATORY in docstring and
   entry; small seed count 3–4/cell, declared).** Grid: L_attack ∈ {400, 800, 1200,
   1600, 2400}; K_repeat ∈ {1, 2, 3, 4}; G ∈ {200, 600, 1200, 2400}; H ∈ {600, 900,
   1200, 1800, 2400, 3000, 4200, 6000}; revision-tolerance modes {normal (=Exp 183),
   tight, loose}. Arms: (1) baseline full-write; (2) fixed-H freeze arms; (3) the
   Exp 183 n4 evidence-concession arm as REFERENCE only; (4) optional oracle
   exact-burst freeze, diagnostic only. Factorization lever (design-time):
   revision_pass(H, mode) is attack-schedule-independent (Phase R has no bursts), so
   `both_pass(H, cell, mode) = defense_pass(H, cell) AND revision_pass(H, mode)` —
   Phase-R sessions run once per (H, mode, seed), not per cell. Per cell report
   `exists_H_both = any_H both_pass(H, cell)`; candidate crack cells are
   `exists_H_both == false` while the oracle succeeds (or defense is mechanically
   possible). FAILURE (of the exploration itself) = a degenerate map (all cells pass,
   or detection never fires off the Exp 183 geometry) — that is a result; log and
   classify, don't tune in place.
3. **Crack classification (per candidate cell).** A: length squeeze (transient needs H
   longer than revision permits). B: repeated-color cumulation (individually survivable
   bursts accumulate identity drift across releases). C: trigger floor (detection latency
   too large relative to attack length). D: config triviality (a larger H, refractory, or
   cooldown still solves it — the crack dissolves). E: impossible body (even oracle
   cannot defend — regime not useful for N4). Only A/B cells SURVIVING the D-check are
   crack candidates.
4. **Confirmation (only if a real squeeze survives rung 3).** Fully pre-registered fresh-
   seed experiment with the Exp 18x falsifier discipline; until it runs, the strongest
   permitted claim is "candidate squeeze regime found, unconfirmed."

**N5 fence (binding).** N5 (interoceptive/motivational economy: energy, drive budgets,
uncertainty tolerance, allostasis) stays CLOSED. Only if the crack map shows fixed-H
failing specifically when internal deadlines/costs are introduced does an N5 or N4×N5
interaction card become the recommendation — framed as "freeze duration must depend on
internal economy," never "more meta is automatically better."

**Outcome map (predeclared; one of these closes the exploration).**
1. Crack dissolves (some fixed H or simple fixed add-on covers every cell) → crack
   CLOSED, config still sufficient; Law H strengthened with mapped domain.
2. Real squeeze found (no fixed H covers defense+revision; oracle or a state-sensitive
   diagnostic can) → pre-register the confirmation experiment on fresh seeds (rung 4).
3. Requires internal economy (fixed-H fails only under internal deadline/cost regimes)
   → consult: an N5 or N4×N5 interaction card, not a pure N4 continuation.
4. Impossible body (no arm, including oracle, defends) → not a useful N4 test; enrich
   substrate or stop; consult.

**Stop condition.** The autopsy + squeeze map + classification are logged with one of
outcomes 1–4 (a dissolved crack is a COMPLETE result, positive for the config law) — or
two iterations stuck on instrument numerics → consult. Scope guards: forks only;
mirro/vela/nira untouched; FROZEN untouched; M4a increment 1c stays halted behind its own
fence; the N4 chapter doc is amended only by APPENDING a crack-resolution section, never
by editing its verdict.

**RUNG 1 RESULT (Exp 183 addendum, 2026-06-12 — the seed-229 autopsy; diagnostic,
bit-match licensed 17/17, blind-verified).** Mechanism PINNED; the pre-data hypothesis
PARTIALLY refuted, and the refinement reshapes Exp 184's predictions:

- **The dose component is CONFIRMED.** Every failing burst pays its erosion entirely in
  the ~75-step unfrozen detection head: head_frac = 1.000 with a frozen-plateau
  constancy assertion (v bit-constant across all in-freeze snapshots — also a freeze
  leak check, 0 leaks in 17 sessions); coverage 725/800 with zero early releases;
  refractory and revision-bar candidates REFUTED. Seed 229's per-burst head dose:
  72.6 / 72.4 / 71.5 units. Across seeds the head erosion runs ~72–97 (w-weighted; NOT
  uniform — bounds stated, no uniformity claim).
- **Repetition-as-discriminant is REFUTED (refuter d fired).** ALL 8 H1200 seeds have
  repeated burst colors; seeds 227/228/233 share seed 229's exact [1,1,0] schedule and
  PASS. The pooled repeat-vs-recovered cross-tab even inverts (failure 48% with repeats
  vs 96% without) — a selection artifact: successful defense keeps argmin(v) on the
  attacked color, so repeats are the DEFENDER's signature; displaced favorites rotate
  argmin. Exp 184's exogenous schedules de-confound this.
- **The operative discriminant is the margin ledger + post-release expression
  dynamics — NOT the gap at burst end.** Recovery (expr_frac < 0.5 over
  [bend+1500, bend+2000)) DECOUPLES from gap_end in both directions: 229 b1 FAILS at
  gap_end +5.2 and 227 b2 FAILS at +45.0, while 233 b1 PASSES at +1.0 and 228 b1
  PASSES at −33.5. Every observed burst with gap_end ≥ ~80 (≈ one head dose of slack)
  recovered; below that, post-release occupancy dynamics decide, with no deterministic
  gap threshold (blinded-verifier corrected framing — "near-tie lottery" was wrong;
  the causal variable is expr_frac). H is irrelevant to the head dose: H900–H3000
  share identical event tables on seed 229. Repetition is a real operating mechanism
  (b0's dose pre-erodes b1's same-color margin) but NON-DISCRIMINANT; the
  load-bearing mechanism is the trigger-latency dose against the margin ledger.
- **Found object:** seed 227 b0 was frozen-at-onset by a stray pre-burst alarm
  (unfrozen head = 0; gap untouched 29.0 → 29.0 → 29.0) — an accidental existence proof
  that freeze-at-onset pays zero dose (the oracle property, reached by config luck).
- **The card's first question answered:** seed 229 is NOT itself a crack instance of
  the interval-emptiness form — its failures defeat every H equally (the head dose is
  H-independent), so no H-interval arithmetic is implicated. Whether a ROBUST
  no-fixed-H regime is constructible from this mechanism is exactly Exp 184's question.
- **Exp 184 refinements (carry into its predeclaration):** (i) a SECOND crack route
  beyond interval-emptiness — H-INDEPENDENT dose-ledger exhaustion: cells where
  K·dose outpaces margin + regrowth(G) fail for EVERY H ≥ L−d_trigger while the oracle
  defends (class B carried by class C's floor); predicted geometry: K ≥ 2 same-color ×
  small G, any H. (ii) Log per-burst gap_start/gap_end AND expr_frac as diagnostics —
  cells whose post-dose margins fall below ~80 carry outcome variance that gap_end does
  not predict; seeds-per-cell must price it. (iii) The class-D dissolution check gains
  a named candidate: pre-armed / lower-THETA freezing (the seed-227 accident, promoted
  to a config arm).

**EXP 184 — PRE-REGISTRATION (the exploratory fixed-H squeeze map; plan Part 2;
committed BEFORE any data).**

- **Status: EXPLORATORY** (3 seeds/cell; predeclared SHAPE predictions per the ratified
  sweep rules; registered boundaries get fresh-seed confirmation at rung 4 ONLY for
  candidates surviving the Part-3 classification incl. the class-D dissolution check).
- **Runner.** `run_fork_schedule` adapted from `exp183_n4_freeze_gate2.run_fork` with
  the burst schedule parameterized (window list + exogenous color); the trigger/freeze
  machinery VERBATIM. **EQUIVALENCE GATE (licenses the grid):** configured to Exp 183's
  exact schedule (bursts 6000–6800/9000–9800/12000–12800, endogenous argmin-at-onset
  color mode, relocation rng 160000+seed), the new runner must bit-match the committed
  exp183 rows for (baseline × s229) and (H1200 × s229) — quantities + event fields,
  atol 1e-9. Spine safety per L14 (detached deepcopy); plateau dose reads per L13.
- **Grid (the plan's, verbatim).** L ∈ {400, 800, 1200, 1600, 2400}; K ∈ {1,2,3,4};
  G ∈ {200, 600, 1200, 2400}; H ∈ {600, 900, 1200, 1800, 2400, 3000, 4200, 6000};
  revision-tolerance modes normal/tight/loose = +3000/+1500/+6000 steps over the
  same-seed baseline latency — modes are EVALUATION BARS on the same Phase-R runs, not
  separate dynamics. Cells enumerated lexicographically by (L, K, G), cell_idx 0–79.
- **Sessions.** W: settle 6000 (attack color = argmin(v) at step 6000, FIXED for all K
  bursts — exogenous, the rung-1 de-confound) + K bursts of length L with gaps G +
  2500-step tail; length = 6000 + K·L + (K−1)·G + 2500 (max ≈ 25.3k). Captivity
  mechanics verbatim exp183; relocation rng default_rng(190000 + 1000·cell_idx + seed).
  R: verbatim exp183 Phase R (permanent relocation from 6000), arms baseline + H-sweep
  + n4-ref, latency criterion verbatim.
- **Arms.** baseline; the 8-H sweep; n4 evidence-concession E*=600 (REFERENCE only,
  never in exists_H_both); oracle exact-train freeze (DIAGNOSTIC, the
  defense-mechanically-possible witness). W-sessions 80 × 11 × 3 = 2640; R-sessions 30.
- **Seeds.** FRESH 240–242. Smoke: cell (L=800, K=3, G=2400), all arms, seed 240,
  disclosed.
- **Primary quantities.** defense(arm, cell, seed) = FINAL-window displacement-
  rejection: expr_frac(attack color) < 0.5 over [last_bend+1500, last_bend+2000) —
  per-burst windows collide with following bursts at G < 2000, so per-burst recovery is
  DIAGNOSTIC-only (gap ledger, plateau-read head dose, per-burst expr where the window
  fits in quiet). defense_pass(arm, cell) = ≥ 2/3 seeds. revision_pass(H, mode) =
  latency ≤ baseline + tolerance in ≥ 2/3 seeds. both_pass = defense ∧ revision.
  exists_H_both(cell, mode) = any H. **Candidate crack cells:** ¬exists_H_both while
  the oracle defends (≥ 2/3 seeds).
- **Shape predictions + falsifiers (the sweep binds the SHAPE).**
  P1 (dose route, rung 1): among H-arms with H ≥ L − 75, defense is H-INVARIANT (the
  head dose is paid regardless); failures concentrate at high K × small G. F1: defense
  improves monotonically with H within the H ≥ L stratum — the dose model is
  wrong/incomplete. P2 (interval route, arithmetic): (L=2400, tight) cells are
  interval-empty — defense needs H ≳ 2325, tight revision allows H ≲ 1650; predicted
  ¬exists_H_both there via route 1's arithmetic alone. F2: some H passes both. P3:
  the oracle defends every cell (≥ 2/3 seeds). F3: an oracle-failed cell (class-E
  territory, flagged, not tuned). F4 (map validity): a degenerate map (trigger silent
  off the Exp-183 geometry, or everything passes everywhere) is a logged result — no
  inline tuning.
- **PC gates (light, exploratory).** PC1 ahat_drift < 0.15 per session (gated); settle
  TV(pi) over [5000, 6000) logged and flagged > 0.05, NOT gated; argmin ≠ argmax at
  step 6000 asserted (the attack must target a non-favorite).
- **Outputs.** experiments/outputs/exp184_rows.json (JSONL: per arm × cell × seed
  defense quantities, final gaps, dose ledger, events, flags), exp184.txt
  (exists_H_both heat maps per mode, candidate-cell list, per-prediction verdicts),
  script experiments/exp184_n4_fixed_h_squeeze_map.py. The Part-3 classification runs
  NEXT iteration on these outputs; no crack claim without rung-4 confirmation.

**RUNG 2 RESULT (Exp 184, 2026-06-12 — the exploratory squeeze map; equivalence-gated
17→2640+30 sessions, blind-verified with both flagged gaps closed/logged).** The
commitment-as-config verdict does NOT extend across attack-schedule space: 36/80
cells (normal tolerance), 57/80 (tight), 32/80 (loose) have NO fixed H passing
defense+revision, while the oracle defends ALL 80 cells 3/3 (defense is mechanically
possible everywhere the constant fails). Both crack routes are REAL STRUCTURE: route 1
(tolerance-bound) confirmed exactly at (L=2400, tight) — the Phase-R law is clean,
latency = H + baseline, so the revision ceiling IS the tolerance, and the interval
empties when K·L + (K−1)·G − 75 > tol; route 2 (dose-bound) = the 32 loose-mode
persisters (e.g. L=400 K=4 G=2400 fails at EVERY H — long gaps release the freeze, K
full doses land). NEW LAW (P1's informative falsification, 8 cells): FREEZE-SPANNING —
against trains with short gaps a long H sleeps through the gaps and pays ONE detection
dose for several bursts; H controls the dose COUNT, interacting with G. Flagged:
(L=1200, K=1) candidates are seed-margin failures (election-variance class), and 3
seeds/cell makes the 2/3 bar single-flip — candidates are NOT cracks. Part-3
classification (A–E + the class-D dissolution arms: pre-armed / lower-THETA trigger)
and rung-4 confirmation are REQUIRED before any claim — both PAUSED on the human's
word (2026-06-12).

**EXP 185 — PRE-REGISTRATION (Part-3 classification + the class-D dissolution check;
committed BEFORE any data; authorized by the human's "continue" on the recommended
option, 2026-06-12).**

- **Question.** Classify every Exp 184 candidate cell into the plan's classes and
  decide dissolution: A (route-1 interval-empty — PROVABLE by the latency = H +
  baseline law: emptiness is exact arithmetic, no constant H can rescue), B
  (route-2 dose-bound, undissolved by every simple config), D (dissolved — a simple
  config covers the cell after all), V (seed-variance at the 3-seed bar — unresolved,
  needs rung-4 seeds), with E excluded (oracle 80/80) and pure C excluded (min L=400
  ≫ d=75; the floor contributes via dose only).
- **New simple-config arms (the class-D check), run on ALL 80 cells × seeds 240–242
  (SAME seeds/schedules as Exp 184 — the Exp 170 same-schedule lesson; this is
  diagnosis, not confirmation):** (i) H9000 (wider constant — predicted to fail
  revision in EVERY mode by the latency law: 9000 > 6000 = loose ceiling; included to
  test the law at scale and the defense side); (ii) THETA3.0 × {H1200, H3000}
  (pre-arm-prone trigger: quiet-tail false alarms are cheap and sometimes pre-arm
  bursts — the seed-227 b0 accident as config); (iii) CALM2600 × {H1200, H3000,
  H6000} (release-calm lengthened past max G — the freeze SPANS all gaps by
  construction, converting dose-bound cells into coverage-bound cells). 6 configs ×
  80 cells × 3 seeds = 1440 W-sessions + 18 Phase-R sessions (each new config needs
  its own revision latencies).
- **Predeclared shape predictions + falsifiers.**
  P1 (the unified coverage law under spanning): with CALM2600, defense_pass(cell, H)
  iff H ≥ K·L + (K−1)·G − 75 (one dose per train; the train is one coverage problem).
  F1: > 20% of CALM2600 cell×H outcomes land on the wrong side of that line — the
  spanning model is wrong.
  P2 (the latency law at scale): H9000 Phase-R latency = 9000 + baseline ± 50; it
  fails revision in all three modes. F2: deviation > 50 steps — the law breaks.
  P3 (no config regression): the new configs do not break formerly-covered cells.
  F3: > 5 cells flip from exists_H_both to no-H when the new arms are ADDED to the
  sweep (they can only add options, so F3 = bookkeeping error or interaction bug).
  F-cls: any candidate cell left without a class is a classification failure — log it.
- **Verdict semantics (predeclared).** A cell DISSOLVES (class D) if any simple-config
  arm passes BOTH bars in that cell × mode. Surviving A cells = route-1 crack
  candidates CONFIRMED-at-this-seed-grade (no constant in the family covers, provably;
  a discriminating controller could — n4-ref revises at ~725–885 while defending
  nothing: defense and revision separate exactly there). Surviving B cells = route-2
  crack candidates. V cells go to rung 4 unresolved. NO crack CLAIM at any grade
  without rung-4 fresh-seed confirmation — which needs its own word; after this
  classification the loop posts the consult and HOLDS.
- **Build.** Extend the exp184 runner with THETA and RELEASE_CALM as parameters
  (defaults unchanged); the equivalence gate re-runs through the new code path with an
  emitted evidence table (L15). Spine safety per L14. Outputs:
  experiments/outputs/exp185_rows.json, exp185.txt (per-candidate-cell class table +
  dissolution map + P1/P2/P3 verdicts), script
  experiments/exp185_n4_crack_classification.py.

**RUNG 3 RESULT (Exp 185, 2026-06-12 — classification + dissolution; blind-verified
CONFIRMED, every dissolution audited).** The crack mostly dissolves into config and
seed noise, but a HARD CORE survives. Normal mode: 36 → 8 D (CALM2600 dominant: the
freeze-spanning law as config dissolved 24 pairs across modes) + 22 V (single-seed
flips at the 3-seed bar) + 6 SURVIVING (A: (1600,2,200), (1600,3,200), (2400,3,200);
B: (1600,4,200), (2400,4,200), (2400,4,600)). Tight: 35 surviving (25 A + 10 B —
all (2400,·,·) and most (1600,·,·)). Loose: 3 surviving. H9000 dissolved ZERO (the
latency = H + baseline law holds: any H above tolerance fails revision — route-1
emptiness is arithmetic, not sweep luck). The surviving core's shape: defense
requires holding longer than the tolerance allows (A) or paying more detection dose
than the margin affords (B) — exactly the regime where a discriminating controller
(n4-ref revises at ~725–885 while the oracle defends everything) could earn its keep
over every constant. Survivors are same-seed diagnoses, NOT confirmed cracks.

**EXP 186 — PRE-REGISTRATION (rung 4: fresh-seed confirmation of the surviving core;
authorized by the human's "a", 2026-06-12; committed BEFORE any data).**

- **Cells (14).** The 6 surviving normal-mode core cells — A: (1600,2,200),
  (1600,3,200), (2400,3,200); B: (1600,4,200), (2400,4,200), (2400,4,600) — plus the
  V-resolution sample: the four (1200,1,G) anomaly cells and the first four OTHER
  normal-mode V cells by cell_idx (from the committed exp185 classification).
- **Arms (16).** baseline; H600–H6000 (8); n4_freeze (E*=600, reference); oracle
  (diagnostic witness); CALM2600×{H1200,H3000,H6000}; THETA3.0×{H1200,H3000}. H9000
  excluded analytically (fails revision in every mode by the latency law). W: 14 ×
  16 × 8 = 1792 sessions; R: 16 × 8 = 128 sessions; FRESH seeds 250–257.
- **Bars (CONFIRMATION-grade, primary = normal tolerance +3000; tight/loose logged
  as secondary diagnostics).** defense_pass(arm, cell) = ≥ 6/8 seeds (final-window
  displacement-rejection, verbatim exp184); revision_pass(arm) = ≥ 6/8 seeds at the
  arm's own fresh-seed Phase-R latencies. Per core cell: CRACK CONFIRMED iff (i) NO
  arm in the extended family passes both bars, (ii) oracle defends ≥ 7/8, (iii)
  baseline fails ≥ 7/8 (the deficit exists). CORE REPLICATES iff ≥ 4/6 core cells
  confirm; A/B labels re-derived per cell from which bar binds. V-sample verdicts:
  covered / crack / variance-dominated (best-arm defense in [3,5]/8).
- **Falsifiers.** F1: core fails to replicate (< 4/6) → the crack dissolves at seed
  grade; NEGATIVE — the chapter closes config-sufficient (refined family). F2: oracle
  fails ≥ 2/8 in a core cell → class-E contamination; that cell excluded + flagged.
  F3: baseline deficit absent (< 7/8 fail) in a core cell → precondition failure;
  cell excluded + flagged. PC1 ahat_drift < 0.15 gated. NO mid-run bar adjustments.
- **Stakes (predeclared).** Replicated core → the crack is REAL at this body (the
  two-inequality law named) and Exp 187 = ONE controller re-test in exactly the
  confirmed cells proceeds under the same word (kill-test discipline binding: the
  controller must beat EVERY constant in the extended family on both bars there).
  Dissolved core → config-sufficiency holds everywhere robust; closure consult.
- **Build.** Reuse the exp185 parameterized runner verbatim (THETA/CALM params);
  equivalence gate re-run through the exp186 code path with the emitted evidence
  table (L15); spine safety per L14. Outputs:
  experiments/outputs/exp186_rows.json, exp186.txt, script
  experiments/exp186_n4_core_confirmation.py.

**RUNG 4 RESULT (Exp 186, 2026-06-12 — fresh-seed confirmation; blind-verified
CONFIRMED, every conjunct recomputed from rows): 6/6 CORE CELLS CONFIRM — THE CRACK
IS REAL AT THIS BODY.** No arm in the extended family (8 H + n4 + CALM2600×3 +
THETA3.0×2) passes both bars in any core cell; oracle 8/8 and baseline-deficit 8/8
everywhere; F1/F2/F3 none fired, PC1 0/1792. Relabels at the 8-seed bar:
A = (1600,2,200) [CALM2600-H6000 defends 8/8, fails revision] and (2400,4,600)
[THETA3-H3000 defends 7/8 — the family's best showing anywhere — and STILL fails
revision: the two-inequality squeeze in one cell]; B = (1600,3,200) [exp185's A was
3-seed noise], (1600,4,200), (2400,3,200), (2400,4,200). The V-sample resolved the
(1200,1,·) anomaly as noise (all covered) and CONFIRMED three more cracks:
(400,4,600), (400,4,2400), (800,4,600) — the confirmed crack set is NINE cells.
Two core cells are tolerance-relative (dissolve under loose); four hold at all
tolerances. Exp 187 — ONE controller re-test in exactly the confirmed cells — is
LICENSED per the pre-registration, under the standing word.

**EXP 187 — PRE-REGISTRATION (rung 5: the licensed controller re-test in the confirmed
crack cells; under the standing word "a"; committed BEFORE any data).**

- **Design thesis (falsifiable).** In the confirmed cells the transient/sustained
  discriminator is pressure INTERMITTENCY: trains have gaps where the pressure
  statistic de-asserts; a changed world never lets up. The controller family
  INT = freeze machinery verbatim (THETA 3.5 sliding trigger) + gap-spanning release
  (calm 2600) + concession on CONTINUOUS pressure only — the concession accumulator
  RESETS whenever pressure de-asserts. Two concession forms, both constants on the
  new surface (the kill test runs ON the surface): INT-E600 (evidence integral
  E_blocked ≥ 600 of CONTINUOUS coherent pressure — Exp 183's surrender-schedule
  controller repaired at its root: an integral that resets on gaps cannot be
  outlasted by an intermittent attack, and a continuous attack IS functionally a
  changed world) and INT-C{1500,2000,2900} (continuous-pressure stopwatches). The
  N4 rung-2 monitor supplies the trigger; the concession discriminator is honest
  REGIME STRUCTURE (the Exp 167 move at the concession level), not meta-judgment —
  named as such.
- **Arms (10).** baseline; oracle (witness); reference constants from Exp 186:
  H3000, CALM2600-H6000, THETA3.0-H3000; the surrender-schedule contrast: n4_freeze
  E*=600 WITHOUT gap reset (verbatim — same integral, no reset: the minimal pair);
  INT-E600; INT-C1500; INT-C2000; INT-C2900.
- **Cells (9, the confirmed set).** (1600,2,200), (1600,3,200), (2400,3,200),
  (1600,4,200), (2400,4,200), (2400,4,600), (400,4,600), (400,4,2400), (800,4,600).
- **Seeds.** FRESH 260–267 (8). W: 9 × 10 × 8 = 720 sessions; R: 9 non-oracle arms
  × 8 = 72 sessions. Bars VERBATIM Exp 186 (defense ≥ 6/8; revision ≥ 6/8 vs
  same-seed baseline + 3000 normal, tight/loose secondary; oracle ≥ 7/8; deficit
  ≥ 7/8). PC1 gated. No mid-run adjustments.
- **Predictions + falsifiers.**
  P1: INT-E600 passes BOTH bars in ≥ 7/9 cells (defense: every gap resets the
  integral, one head dose per train; revision: concedes at ~703 continuous steps —
  also passes TIGHT, which no frozen-time arm can). F1: INT-E600 fails defense in
  ≥ 3 cells (spurious resets or dose exhaustion) — intermittency insufficient, the
  crack stands as an open wall.
  P2: the C-sweep is tolerance-trivial on the new surface (any C ≤ tolerance passes
  both bars; C2900 fails tight while E600 passes) — constants-on-the-right-surface
  suffice: the UNIVERSAL-CONSTANT LAW REFINED (sufficient-surface at the concession
  level), not broken. F2: no single C covers all 9 cells while cell-dependent C's
  exist — a genuine regulation requirement (the chapter-reversing outcome; report
  with both hands).
  P3: the minimal pair separates — non-reset E*=600 fails defense in the train
  cells (concedes mid-train) while INT-E600 defends: the gap-reset is the entire
  difference. F3: the pair does NOT separate — the mechanism story is wrong.
- **Verdict semantics.** INT family passing everywhere → the crack is CLOSED
  CONSTRUCTIVELY: identity defense + revision coexist in the crack cells via the
  continuity surface; the chapter's negative is refined (what was missing was the
  SURFACE, not judgment); rung-2 instruments necessary, regulation still
  unnecessary. INT failing → the crack stands; consult. Either way the chapter
  synthesis follows as the next consolidation step, then consult.
- **Build.** Extend the exp185/186 runner with the INT concession modes
  (parameterized: gap-reset flag, continuity threshold form C-or-E); equivalence
  gate with evidence table (L15); spine safety (L14); outputs exp187_rows.json /
  exp187.txt; script experiments/exp187_n4_intermittency_controller.py.

**RUNG 5 RESULT (Exp 187, 2026-06-12 — blind-verified CONFIRMED): MIXED — THE CRACK
IS CLOSED CONSTRUCTIVELY at normal tolerance.** INT-C2900 (a stopwatch on the
CONTINUOUS-pressure clock, reset by every gap) passes both bars in 9/9 cells
(defense 7/8 each; revision ~2703 ≤ +3000). The C-sweep traces the interval law on
the new surface: working C ∈ (max continuous stretch ≈ L+window, tolerance] —
C1500 closes 4/9, C2000 6/9 (fails exactly the L=2400 cells), C2900 9/9, zero
exceptions. P1/P3 NEGATIVE honestly: INT-E600 0/9 — with a named instrument-fidelity
gap (the accumulator counted QUIET blocked evidence; the pre-registered
pressure-gated E-form went UNTESTED — residual); the minimal pair separated in the
wrong direction ((400,4,·): non-reset 5/8 vs reset 0/8 as implemented). THE
SUFFICIENT-SURFACE LAW completes at the concession level: at every chapter level
(write-gate → freeze → concession) what was missing was the measured quantity,
never judgment — the universal-constant law refined, not broken. The boundary
persists: under TIGHT tolerance no surface tried covers (interval empty whenever
tolerance < single-burst stretch) — the residual crack exactly characterized.
Residuals needing words: the corrected pressure-gated E-form; the tight-mode core;
post-release retention (spanning arms still frozen at the scoring window;
calm 2600 > the 2500-step tail).

**EXP 188 — PRE-REGISTRATION (the human's Exp-187 question, renumbered per the
append-only log; explicit directive 2026-06-12; committed BEFORE any data).**

- **Question (the human's, verbatim in substance).** Can ONE online regulated
  controller pass both defense and revision in the nine confirmed crack cells,
  using no oracle timing and no cell-specific tuning? Graded against the binding
  constant comparator INT-C2900 (same seeds): parity = the universal-constant
  refrain (with the single-stretch ambiguity bound named); a beat (tight-mode pass
  or materially faster revision) = the first genuine win for regulation.
- **The controller (single, locked here).** REG-TB: rung-2 monitor trigger (THETA
  3.5 verbatim); gap-spanning release (calm 2600); concession when the current
  CONTINUOUS-ACTIVE-pressure stretch exceeds max(κ · S_max, T₀) where S_max = the
  longest COMPLETED continuous stretch observed so far in this freeze (online tempo
  learning), κ = 1.5, T₀ = 2800 — all provided, GLOBAL, no per-cell anything.
  Pressure-gating FIXED vs Exp 187's flaw: the stretch counter increments ONLY on
  pressure-ACTIVE snapshots; quiet contributes nothing (auditable per session:
  rows carry the accumulator trace summary).
- **Arms (5).** baseline; oracle (witness); REG-TB; INT-C2900 (the comparator);
  INT-E600-FIXED (pressure-gated evidence integral, reset on de-assert, E*=600 —
  retires the Exp 187 instrument-fidelity residual; PREDICTION: it fails every
  cell with L ≥ 800 — a single long burst accumulates 600 coherent units in ~628
  steps and concedes MID-BURST: the surrender-schedule law on the E-surface).
- **Cells (9, confirmed).** Verbatim Exp 186/187. **Seeds FRESH 270–277.**
  W: 9 × 5 × 8 = 360; R: 4 non-oracle arms × 8 = 32. Bars VERBATIM Exp 186
  (defense ≥ 6/8; revision ≥ 6/8 at normal +3000; tight/loose secondary; oracle
  ≥ 7/8; deficit ≥ 7/8; PC1 gated; no mid-run adjustments).
- **Predictions + falsifiers.**
  P1 (the human's bar): REG-TB passes BOTH bars in ≥ 8/9 cells (predicted
  mechanism: every train's burst stretch tops out at ~2525 (L ≤ 2400 + the
  pressure-window tail) which stays BELOW T₀ = 2800, so trains never trigger
  concession and the freeze spans them; Phase R's unbroken stretch crosses T₀ →
  concession at ~2875 ≤ baseline + 3000). F1: ≤ 6/9 — regulation fails where the
  constant succeeds (a hard negative for the chapter's reopened hypothesis).
  P2 (parity-vs-beat): REG-TB ≈ INT-C2900 at normal tolerance AND REG-TB does NOT
  pass tight (its T₀ budget exceeds +1500 by construction) — parity, the
  single-stretch ambiguity bound: any online controller without attack-length
  priors must hold ambiguous first stretches up to its revision budget, so at
  fixed-L geometry it cannot beat the tolerance-bounded constant. F2 (either
  direction is reportable): REG-TB passes cells C2900 fails, or fails cells it
  passes — regulation and the constant separate.
  P3 (the E-form retirement): INT-E600-FIXED fails all L ≥ 800 cells by mid-burst
  concession while showing ZERO quiet accumulation (the fidelity fix verified in
  the same run). F3: it defends long-L cells — the surrender-schedule law is wrong
  on the properly-gated E-surface.
- **Verdict semantics.** P1+P2-parity → the chapter closes with: regulation CAN
  do it but constants on the right surface do it too — the sufficient-surface law
  is the chapter's final word at this richness; tight-mode remains the open core.
  P1+F2-beat → regulation earns its first keep; consult before any further claim.
  F1 → regulation loses to its own budget; the constant stands alone; consult.
- **Build.** exp187 runner + the REG-TB/E600-FIXED modes (pressure-gated
  accumulators); equivalence gate with evidence table (L15); spine safety (L14);
  outputs exp188_rows.json / exp188.txt; script
  experiments/exp188_n4_regulated_controller.py.

**EXP 188 RESULT (2026-06-12 — blind-verified, P1/P2/P3 all independently confirmed
from rows): POSITIVE — THE HUMAN'S QUESTION IS ANSWERED: YES, one online regulated
controller (REG-TB, no oracle, no cell-specific tuning) passes both bars in 9/9
crack cells — at EXACT defense parity with INT-C2900** (identical 6/8 per cell, the
same two seeds failing everywhere; the seeds-272/275 failures are margin/election
class, arm-independent). The single-stretch ambiguity bound STANDS: at fixed-L
geometry regulation ties the right clock; REG-TB's tempo-learning component never
actually bound behavior (stretches stayed below both thresholds) — judgment was
along for the ride. Beneath the bars: REG-TB revision 8/8 vs C2900 7/8 (seed 277
missed by 32 steps) — a construction artifact (T₀ 2800 < C 2900), not earned
regulation. E-surface retired: INT-E600-FIXED fails 0/9 with mid-burst surrender in
every L ≥ 800 cell (the surrender-schedule law, properly gated; 0 quiet leaks in
392 sessions) — with one failed sub-prediction flagged (concession at ~372 active
steps, not ~628; accumulation-rate accounting open). THE CHAPTER'S THREE OPEN
EDGES: (i) variable-L attack geometry (where any single budget fails and tempo
learning could finally bind — the Exp 173 multi-tempo move at the identity level);
(ii) the tight-tolerance core (no surface ever tried covers); (iii) post-release
retention (spanning arms still frozen at the scoring window).

**CHAPTER CLOSED (2026-06-12), then RE-OPENED at rung 6 on the human's word "a"
(variable-L geometry).** Synthesis at `docs/research/n4-crack-chapter.md`
(fact-checked by three parallel blinded readers, ~46 claims, 1 correction folded).

**EXP 189 — PRE-REGISTRATION (rung 6: the variable-L separation test; the human's
word "a", 2026-06-12; committed BEFORE any data).**

- **Question.** Does online tempo learning finally BIND — and beat every constant —
  where burst lengths vary such that no revision-safe constant covers? The
  controller is REG-TB VERBATIM from Exp 188 (κ = 1.5, T₀ = 2800; pressure-gated;
  NO retuning — the same controller that tied constants at fixed-L).
- **THE κ-REACH LAW (predeclared shape).** REG-TB defends a train iff (i) its FIRST
  burst stretch ≤ T₀ = 2800 and (ii) every subsequent burst stretch ≤ κ × the
  longest COMPLETED stretch so far (stretch ≈ L + ~125 pressure-window tail).
  Tempo learning extends the hold only for foreshadowed lengths; first encounters
  stay budget-bounded (the ambiguity bound's per-encounter form).
- **Cells (6; settle 6000, tail 2500, exogenous attack color verbatim).**
  E1 escalating: L-seq (1200, 2400, 3200), G=600 — REG-TB PASS predicted (each
  step ≤ ×1.5; final stretch 3325 < κ·2525 = 3787).
  E2 doubling-jump: (800, 1600, 3200), G=600 — REG-TB FAIL predicted (3325 >
  max(κ·1725 = 2587, 2800)).
  E3 escalating, wider gaps: (1200, 2400, 3200), G=1200 — REG-TB PASS predicted.
  E4 doubling train: (600, 1200, 2400, 4800), G=600 — REG-TB FAIL predicted
  (4925 > κ·2525).
  D1 descending: (3200, 2400, 1200), G=600 — REG-TB FAIL predicted
  (first burst 3325 > T₀ = 2800; first-encounter ambiguity).
  C1 fixed control: (2400, 2400, 2400), G=600 — REG-TB and C2900 BOTH pass
  predicted (the Exp 188 parity regime; sanity anchor).
- **Arms (6).** baseline; oracle (exact-train freeze, witness); C2900 (the
  revision-safe constant — predicted to concede mid-3200/4800 bursts and fail
  defense in E1–E4, pass C1); C3500 (the deliberately super-tolerance constant —
  predicted to DEFEND E1/E3 (3325 < 3500) but FAIL revision (latency ~3575 >
  baseline + 3000): the constant's dilemma made visible); CALM2600-H3000 (the best
  frozen-time arm — predicted to concede mid-train everywhere the train exceeds
  3000 frozen steps); REG-TB.
- **Seeds.** FRESH 280–287. W: 6 × 6 × 8 = 288; R: 5 non-oracle arms × 8 = 40.
  Bars VERBATIM Exp 186/188 (defense ≥ 6/8; revision ≥ 6/8 at normal +3000;
  tight/loose secondary; oracle ≥ 7/8; deficit ≥ 7/8; PC1 gated; no mid-run
  adjustments). Variable-L uses the runner's existing window-list parameter — the
  freeze machinery is untouched; equivalence gate re-run with evidence table (L15).
- **Predictions + falsifiers.**
  P1 (THE SEPARATION): in ≥ 2 cells (predicted: E1, E3) REG-TB passes BOTH bars
  while NO constant arm passes both. F1: zero separation cells — the ambiguity
  bound extends to variable-L; regulation loses everywhere it could have won (a
  clean negative for the chapter's last open hope for judgment).
  P2 (revision invariance): REG-TB Phase-R latency ≈ T₀ + trigger ≈ 2875 ≤
  baseline + 3000 (no completed stretches in a changed world → threshold stays
  T₀). F2: REG-TB fails revision — tempo learning contaminates concession.
  P3 (the κ-reach law): REG-TB's six cell-level defense signs land exactly as
  predicted (E1✓ E3✓ C1✓ E2✗ E4✗ D1✗). F3: ≥ 2 cells off — the law is wrong.
- **Verdict semantics.** P1+P3 → REGULATION EARNS ITS KEEP for the first time in
  the program: a controller that no constant matches, exactly where the chapter's
  laws predicted, with its failure boundary (κ-reach, first-encounter) honestly
  mapped — the N4-chapter question gets its constructive postscript. F1 → the
  ambiguity bound is geometry-general at this richness; the chapter's negative is
  total; close. Either way: synthesis appendix + consult, then HOLD.
- **Build.** exp188 runner verbatim (variable windows via the existing parameter);
  outputs exp189_rows.json / exp189.txt; script
  experiments/exp189_n4_variable_length.py.

**RUNG 6 RESULT (Exp 189, 2026-06-12 — blind-verified CONFIRMED): NEGATIVE by F1+F2
with THE KAPPA-REACH LAW CONFIRMED 6/6.** REG-TB (verbatim, untouched) DEFENDS the
escalating trains no constant survives — E1/E3 8/8 exactly as predicted, riding
each step within kappa-reach while C2900/CALM-H3000 concede mid-burst and C3500
defends-but-cannot-revise (the dilemma arm, 0/8 revision) — but loses the
separation on its OWN revision: 5/8 (seeds 281/283/286 over +3000 by 7/39/67
steps) via THE FLICKER TAX: onset pressure-flicker resets the continuity clock and
delays the T0 crossing (1-5 pre-stretches per seed); the tax is a property of the
reset-based surface itself (C2900 also fell to 5/8 on this block vs 7/8 on the
previous). The fight is exactly localized: a flicker-robust concession form
(total-active-time-since-entry, or de-assert hysteresis) is the named crack —
keep the reset's defense benefit without paying the revision tax, no oracle
knowledge. HALTED on the fired falsifiers; consult posted.

**EXP 190 — THE FLICKER-ROBUST CONCESSION ATTEMPT (the human's word "(a)" on the
Exp 189 consult, 2026-06-12): NEGATIVE AT DESIGN TIME, blind-verified.** Both named
forms refuted BEFORE any fresh-seed run — the design-time diagnostic (predeclared
admissibility + buildability rules fixed in the build spec before any trace data;
bit-match-gated against the committed exp189 rows, 137 evidence lines, 0
mismatches) measured what Exp 189 only named:

- **Total-active-time:** refuted by arithmetic — defending E1 needs the budget to
  cover ~7,175 active steps; revision allows ~2,900. Empty by >4,000. Not run.
- **De-assert hysteresis:** refuted by measurement — changed-world onset de-assert
  runs span 1-104 snaps (25-2,600 steps, calm-capped) while attack-train gap runs
  are 21-47 snaps (525-1,175 steps): FULL OVERLAP, no admissible h (0 of [2,20] at
  the predeclared margins). The counterfactual sweep through the REAL runner on the
  committed seeds (the controller's best case) closes it completely: revision needs
  h >= 36 for even 6/8 (per-seed rescue minima 36/58/76) while E1 defense dies at
  h >= 21 (mid-train concessions), C1 at 21-24, E3 at 46 — 0/19 h values satisfy
  the buildability bar; the revision floor exceeds the defense ceiling by 400 steps.
- **THE TIMESCALE-OVERLAP LAW (new):** the flicker tax is structural to the reset
  surface — the reset that buys train defense and the reset that taxes revision are
  the same event class at overlapping timescales; bridging flickers bridges gaps.
  Bounded to the two named channels; the measured pressure-density inversion (the
  trains are DENSER in pressure than the changed world's onset, ~39% active vs
  near-continuous) is noted as an observation against monotone occupancy
  functionals generally.
- META: the first build's trace recorder silently no-opped (unmatched str.replace
  patches) and produced the right headline for the wrong reason — caught against
  the stretch_log invariant, repaired with `_must_replace` (L16 distilled: patch
  builders are instruments; gates must exercise both branches of a new parameter).

Option (a) is CLOSED refuted-as-buildable per the consult's own terms. The
kappa-reach law (defense) and the flicker tax + timescale overlap (revision) are
the rung-6 legacy: regulation's defense half is proven and its revision half is
unrepairable on the de-assert channel at this body.

**STATUS.** state: halted (rung-6 NEGATIVE + the option-(a) attempt CLOSED refuted-as-buildable at design time, Exp 190; CONSULT in loop/IDEAS.md: (a) RECOMMENDED fold the rung-6 postscript + three laws into docs/research/n4-crack-chapter.md and close this card / (b) a form outside the named channels needs its own design-time diagnostic / (c) redirect-stop; the loop HOLDS) · previously closed-positive — 9 confirmed cells; closed constructively on the continuity clock; the regulated-controller question answered at exact parity; chapter synthesis committed) · latest: Exp 188 + the chapter synthesis · depends-on: identity-n4 (closed-negative), Exp 183–188 artifacts · reusable: the equivalence-gated schedule runner, continuity-surface concession modes (C / gated-E / REG-TB), the interval laws on both surfaces, the single-stretch ambiguity bound, the margin-ledger instruments, the sufficient-surface law (three levels) · why: the question the human reopened is fully graded — crack real, closed by the right clock, regulation ties it; three exactly-named open edges await words · next-falsifiable: ON A WORD — variable-L geometry (where tempo learning must bind), the tight-tolerance core, or post-release retention · latest: Exp 187 (MIXED — constructive closure at normal tolerance + the continuous-stretch interval law) · depends-on: identity-n4 (closed-negative), Exp 183–186 artifacts · reusable: the INT runner (continuity-surface concession, parameterized), the continuous-stretch interval law (C ∈ (L+window, tolerance]), the sufficient-surface law at the concession level, everything in rungs 1–4 · why: the crack chapter has its constructive answer at normal tolerance and an exactly-characterized residual boundary at tight · next-falsifiable: ON A WORD — the pressure-gated E-form (untested as pre-registered), or the tight-mode core (no surface covers), or post-release retention · latest: Exp 186 (BREAKTHROUGH — crack confirmed, 9 cells) · depends-on: identity-n4 (closed-negative), Exp 183–185 artifacts · reusable: everything in the rung-1–4 toolkit + the confirmed crack-cell set · why: defense and revision provably separate in the confirmed cells while the oracle defends — the first regime where a discriminating controller could beat every constant · next-falsifiable: Exp 187 — does ANY realizable controller pass both bars in the confirmed cells (and does it beat the constants honestly, same-schedule, fresh seeds)? · latest: Exp 185 (classification, blind-verified CONFIRMED) · depends-on: identity-n4 (closed-negative), Exp 183/184 artifacts · reusable: parameterized THETA/CALM runner (exp185), CALM2600 gap-spanning config (dissolved 24 pairs), the unified coverage law H ≥ K·L+(K−1)·G−75, the latency = H + baseline revision law, the A/B/D/V classification instruments · why: the surviving core (6 normal / 35 tight / 3 loose cells) is where no constant in the extended family covers while the oracle defends — the candidate regime where regulated commitment could finally beat config · next-falsifiable: rung 4 (ON A WORD) — fresh seeds × more seeds/cell on the surviving core: do the A/B cells replicate, and does a discriminating controller beat every constant there? · latest: Exp 184 (exploratory squeeze map) + Exp 183 addendum (seed-229 autopsy) · depends-on: identity-n4 (closed-negative; read-only inputs), Exp 183 artifacts · reusable: the equivalence-gated generalized schedule runner (exp184), Exp 183 freeze machinery, rung-2 monitor (180), displacement regime (176), detection-floor law (F), the margin-ledger/election instruments, the freeze-spanning/dose-count law, the latency = H + baseline revision law · why: the map shows config-sufficiency is a property of benign attack geometry — candidate crack cells exist on both routes with the oracle clean; classification + confirmation pending · next-falsifiable: ON RESUMPTION — Part-3 classification of the candidate sets (is any candidate a REAL crack after the class-D dissolution check?), then rung-4 fresh-seed confirmation
