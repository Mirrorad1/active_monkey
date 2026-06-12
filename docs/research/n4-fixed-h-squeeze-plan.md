# N4 fixed-H squeeze — predesign plan (the supplied crack program)

**Provenance.** Human directive, 2026-06-11, given in-session as the explicit word
reopening the identity-N4 line (loop/IDEAS.md entry of the same date). This document
captures the supplied program faithfully; the direction card
(`loop/directions/identity-n4-crack.md`) is the executable summary. Where the two
differ, this plan is the source of truth for WHAT was asked; the card + PROTOCOL govern
HOW it runs.

**Status of the closed chapter (unchanged by this plan).** The N4 identity/commitment
chapter (`docs/research/n4-identity-commitment-chapter.md`) remains CLOSED with: rung 1
POSITIVE (displacement regime verified, Exp 176); rung 2 POSITIVE (identity monitor real,
Exp 180); rung 3 NEGATIVE-config at the prior richness (Exp 183: fixed-H freeze matched
defense and revision; regulated commitment did not earn its keep). This is NOT a rerun of
Exp 183, NOT an attempt to reinterpret the chapter, and Exp 183's outputs and verdicts are
read-only.

---

## 1. The new question

Does the seed-229 repeated-burst-color failure point to a regime where NO single fixed
horizon H can defend transient attacks while still revising under sustained evidence?

**Mathematical target.** A fixed H works if there exists an H satisfying both:

    H >= max_b ( L_attack_b - d_trigger_b )

and

    H <= min_r ( T_revision_tolerance_r )

The crack exists only if the interval is empty:

    max_b ( L_attack_b - d_trigger_b )  >  min_r ( T_revision_tolerance_r )

or if repeated-color cumulation defeats all fixed-H arms across the declared
defense/revision bars.

**Background numbers (from the closed chapter, for orientation only).** At Exp 183's
geometry (L=800, d_trigger ≈ 75): the lower edge is ~725 (H900 marginal at 6/8; H1200+
pass 7/8) and the upper edge is set by Phase-R latency ≈ H + 150 ≤ baseline + 3000
(H3000 brushes it). Seed 229's [1,1,x] repeated-color schedule defeats ALL current arms —
including every passing H and the n4 reference — and H900 additionally loses seed 227 to
the same class. The chapter's §6 deferred exactly this crack, with the Exp 173 caution:
cracks of this form have so far dissolved into wider constants or richer-body
requirements.

## 2. Read-first list (before any code)

- `loop/directions/identity-n4.md` (the closed card: regimes, bars, pre-registrations)
- `docs/research/n4-identity-commitment-chapter.md` (verdict, laws A–H, §6 crack note)
- EXPERIMENTS.md entries for Exp 181, Exp 182, Exp 183
- `experiments/outputs/exp183_rows.json`
- `experiments/outputs/exp183_events.json` if present (else the events tables inside
  `exp183.txt` / rows)
- `experiments/outputs/exp183_verdict.json`
- `docs/specs/n-order-self-modeling.md`

## 3. Part 1 — seed-229 autopsy (diagnostic only)

Script: `experiments/exp183_seed229_autopsy.py`. Inspect or reconstruct seed 229 and
compare it to seeds 226–233. Log:

1. Burst colors by burst index.
2. Whether burst colors repeat.
3. Value vector before/after each burst.
4. Normalized identity vector: `pi_t = v_t / sum(v_t)`.
5. Directed drift toward the repeated color: `D_b = pi_end[c] - pi_start[c]`.
6. Whole-vector movement: `TV_b = 0.5 * sum_i |pi_end[i] - pi_start[i]|`.
7. Whether repeated-color pressure accumulates across bursts.
8. Whether failure occurs because of: trigger latency; release too early;
   repeated-color accumulation; insufficient refractory; revision-bar conflict; or
   some other mechanism.

Outputs: `experiments/outputs/exp183_seed229_autopsy.txt` and `.json`.
This is diagnostic only. It must not modify Exp 183's verdict.

## 4. Part 2 — exploratory fixed-H squeeze map (Exp 184)

Script: `experiments/exp184_n4_fixed_h_squeeze_map.py`, labeled EXPLORATORY in the
docstring and in EXPERIMENTS.md. Small seed count (3–4 seeds per cell), clearly labeled
exploratory.

**Grid.**

- attack length `L_attack` ∈ {400, 800, 1200, 1600, 2400}
- repeated same-color burst count `K_repeat` ∈ {1, 2, 3, 4}
- inter-burst gap `G` ∈ {200, 600, 1200, 2400}
- fixed horizons `H` ∈ {600, 900, 1200, 1800, 2400, 3000, 4200, 6000}
- revision tolerance modes: `normal` (same as Exp 183), `tight` (stricter revision
  deadline), `loose` (looser revision deadline)

**Arms.**

1. baseline full-write
2. fixed-H freeze arms
3. prior n4 evidence-concession arm, as a REFERENCE only
4. optional oracle exact-burst freeze, diagnostic only

**Primary exploratory question.** For each regime cell, does there exist any fixed H
satisfying both defense against transient/repeated attacks AND revision under sustained
evidence? Compute `defense_pass(H, cell)`, `revision_pass(H, cell)`,
`both_pass(H, cell)`; report `exists_H_both = any_H both_pass(H, cell)`. Candidate crack
cells: `exists_H_both == false` while oracle succeeds or defense is at least mechanically
possible.

**Factorization note (loop, design-time; doesn't change what's measured).** Revision
passing depends on (H, mode), not on the attack-schedule cell (Phase R contains no
bursts), so Phase-R sessions run once per (H, mode, seed) and
`both_pass(H, cell, mode) = defense_pass(H, cell) AND revision_pass(H, mode)` — this
keeps the full grid tractable without subsampling the defense map.

## 5. Part 3 — crack classification

For each no-fixed-H candidate cell, classify the mechanism:

- **A. length squeeze:** transient attack requires H longer than revision permits.
- **B. repeated-color cumulation:** each burst individually survivable, but repeated
  same-color bursts accumulate identity drift across releases.
- **C. trigger floor:** detection latency too large relative to attack length.
- **D. controller/config triviality:** a simple larger H, refractory, or cooldown still
  solves it.
- **E. impossible body:** even oracle cannot defend — the regime is not useful for N4.

**A valid N4 crack requires ALL of:** fixed-H fails; oracle or some state-sensitive
reference indicates defense is possible; failure is not just impossible detection;
failure is not rescued by a slightly wider constant sweep.

## 6. Part 4 — do not build N5 yet

N5 is separate. It models interoceptive/motivational economy: energy, drive budgets,
uncertainty tolerance, allostasis. Do not open N5 unless the N4 crack map shows that
fixed-H only fails when internal deadlines/costs are introduced. If N5 becomes relevant
later, frame it as: fixed-H commitment fails because freeze duration must depend on
internal economy — not because "more meta" is automatically better.

## 7. Part 5 — update logs

After the autopsy and exploratory map:

- Append EXPERIMENTS.md with the diagnostic/autopsy and the exploratory map.
- Update `loop/directions/identity-n4-crack.md` with: whether seed 229 was a real crack;
  whether any no-fixed-H regime was found; whether the crack dissolved into a wider
  constant; whether a confirmation experiment is warranted.

Do not claim success unless a fresh-seed confirmation is later run.

## 8. Possible next outcomes

1. **Crack dissolves:** some fixed H or simple fixed cooldown covers the regime.
   Verdict: N4 crack closed / config still sufficient.
2. **Real squeeze found:** no fixed H covers defense+revision, but oracle or a
   state-sensitive diagnostic can. Verdict: pre-register a confirmation experiment on
   fresh seeds.
3. **Requires internal economy:** fixed-H only fails when freeze has
   drive/energy/urgency costs. Verdict: open an N5 or N4+N5 interaction card, not a pure
   N4 continuation.
4. **Impossible body:** no arm, including oracle, can defend. Verdict: not a useful N4
   test; enrich substrate or stop.

## 9. Core discipline

The question is not "can we make N4 pass?" The question is:

    Does the seed-229/repeated-color regime remove the fixed-H interval
    that made Exp 183 config?

If not, the N4 chapter stays closed-negative. If yes, then and only then open a
confirmation experiment.
