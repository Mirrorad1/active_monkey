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

**STATUS.** state: active (predesign committed; autopsy next) · latest: — (opens at the seed-229 autopsy, then Exp 184 exploratory) · depends-on: identity-n4 (closed-negative; read-only inputs), Exp 183 artifacts · reusable: Exp 183 freeze machinery + sliding trigger, rung-2 monitor (180), displacement regime (176), detection-floor law (F), surrender-schedule law (G), universal-constant kill test (173) · why: the human's explicit word (2026-06-11) reopens the deferred seed-229 crack as a NEW investigation; the closed chapter verdict is untouched · next-falsifiable: seed-229 autopsy (mechanism pinnable from committed artifacts?), then Exp 184 (does any schedule cell empty the fixed-H interval?)
