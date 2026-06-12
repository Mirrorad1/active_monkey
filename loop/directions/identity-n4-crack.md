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

**STATUS.** state: active (rung 3 = Part-3 classification, authorized by the human's "continue" 2026-06-12; after it the loop posts the crack-or-dissolved consult and HOLDS — rung-4 confirmation needs its own word) · latest: Exp 184 (exploratory squeeze map) + Exp 183 addendum (seed-229 autopsy) · depends-on: identity-n4 (closed-negative; read-only inputs), Exp 183 artifacts · reusable: the equivalence-gated generalized schedule runner (exp184), Exp 183 freeze machinery, rung-2 monitor (180), displacement regime (176), detection-floor law (F), the margin-ledger/election instruments, the freeze-spanning/dose-count law, the latency = H + baseline revision law · why: the map shows config-sufficiency is a property of benign attack geometry — candidate crack cells exist on both routes with the oracle clean; classification + confirmation pending · next-falsifiable: ON RESUMPTION — Part-3 classification of the candidate sets (is any candidate a REAL crack after the class-D dissolution check?), then rung-4 fresh-seed confirmation
