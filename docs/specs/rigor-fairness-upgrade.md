# Spec: Rigor & Fairness Upgrade (the lab hardening pass)

**Status:** ACTIVE. **Origin:** external critique session, 2026-06-10 (full critique in the
session transcript; condensed rationale below). **Owner:** the human; execution by
dispatched subagents (Sonnet for code, Haiku only where this spec provides exact text).

## 1. Purpose

The lab's *methodology* (predeclared falsifiers, blinded verifier, append-only log,
committed raw outputs, honored halts) is stronger than its *presentation* and parts of
its *engineering*. This spec closes the gap so the project is harder to dismiss:

1. **Rigor:** the most novel code (alarm / probation / growth machinery) is untested,
   script-embedded, and copy-pasted across experiments; designed constants are scattered;
   deps are under-pinned; there is no machine-readable per-experiment verdict.
2. **Fairness:** the public layer ("a little mind coming online") overclaims relative to
   a 4×4 toy world; the hedging exists (README, open_problem tab 4) but is back-loaded;
   there is no related-work section, no scale statement up front, and the blinded-verifier
   protocol — the lab's best credibility asset — is invisible publicly.
3. **Structure:** directions have cards but no status registry; mature arcs have no
   flagship summary layer over the 4,600-line EXPERIMENTS.md.

## 2. Non-goals (binding)

- **Do NOT collapse the lab into the flagship arc.** `active_monkey` stays a multi-act,
  multi-direction lab. The "worldview too small" arc is one direction among many.
- **Do NOT redesign experiments or claims.** Experiment design, predeclaration, running,
  and EXPERIMENTS.md entries are Loop B work (`loop/PROTOCOL.md`), never subagent work.
- **Do NOT soften or strengthen any scientific claim** beyond the exact wording given in
  this spec. Wording tasks are mechanical: apply the provided text.

## 3. Ground rules for EVERY task below

These bind every subagent. Include them verbatim in each dispatch prompt.

- **Never edit FROZEN paths** (see `FROZEN` manifest at repo root). New modules are
  ADDITIVE. `eval/` is never touched.
- **Never edit `EXPERIMENTS.md`** — it is append-only and owned by Loop B. Subagents do
  not append to it either.
- **Never touch creature spines** (`creature/state/mirro|vela|nira/`).
- **Suite stays green:** run `uv run --python .venv pytest -q` before and after; same or
  larger pass count. Always run Python via `uv run --python .venv ...` from repo root.
- **Shared checkout:** a live cron loop commits to this same working tree. Before any
  git mutation: `git status` + `git log --oneline -3`; if a merge/rebase is in progress
  or the tree is mid-change, stop and report instead of proceeding.
- **One task = one commit**, message `rigor: T<N> <short title>`, ending with the
  standard Claude co-author line. Commit experiment-grade outputs with their scripts.
- **No FROZEN diff check:** `git diff --name-only HEAD~1 | grep -f FROZEN` must be empty
  after your commit (adapt to the manifest's format).

## 4. Sequencing and gates

```
GATE-1 (open consult): answer Exp 153's halt — "154" runs the confirmation increment
        (fresh seeds, alarm-answered criterion, no structure proxy). LOOP B WORK.
GATE-2: Exp 154 verdict. If POSITIVE → growth wall re-bounded to the unnormalized
        convention; wall docs amended (Loop B); benchmark workstream F unblocks with a
        working hero mechanism. If NEGATIVE → workstream F still proceeds, but the
        flagship arc is the honest-negative story; T14's skeleton already supports both.
Workstreams B, C, D, E are NOT gated — dispatch any time, any order within their
listed dependencies. Workstream F is gated on GATE-2.
```

## 5. Task list

Format: **T# — title** · tier · depends-on · files. Each task is self-contained for a
fresh subagent with no conversation context; pointers below are sufficient.

---

### Workstream B — extract and test the earned machinery

**T1 — Extract `active_loop/growth.py`** · Sonnet · no deps ·
new file `active_loop/growth.py`.
Lift (do not rewrite) from `experiments/exp145_m3c_live_probation.py` and
`experiments/exp153_*.py`: per-color + global ceiling detection (rolling windows,
mean/slope conjunction), `LiveProbation` (snapshot → provisional install → resolve
keep/revert with exact snapshot restore), round-robin alarm scheduling,
`burnin_em_color`, and the mixture predictive utilities in BOTH conventions —
unnormalized-footprint (the historical default) and normalized-density (Exp 153 Arm B)
— selected by an explicit `convention=` argument, never silently.
Acceptance: module imports clean; suite green; NO existing experiment script modified
(historical scripts stay byte-stable — they are the audit trail); docstring of each
function names the experiment that validated it (145, 152, 153).

**T2 — Unit + regression tests for growth machinery** · Sonnet · T1 ·
new `tests/test_growth.py`.
Unit: detector fires on a synthetic flat plateau >0.7 nats; stays silent on a descending
trace and during a slow-learning transient; probation keeps at margin −0.11 nats and
reverts at −0.09 (the 0.1-nat boundary); revert restores the snapshot bit-exactly
(hash the component state); round-robin never starves a persistently-alarmed color;
burn-in EM recovers two synthetic modes. Regression: load 1–2 committed
`experiments/outputs/exp145_*.json` rows and assert the extracted code reproduces the
recorded keep/revert decisions for a sampled subset.
Acceptance: all new tests pass in <5s; suite green.

**T3 — Procedural worlds module** · Sonnet · no deps ·
new `active_loop/worlds.py`, new `tests/test_worlds.py`.
Generators for the four benchmark families, constants matching their source experiments:
`learnable()` (Exp 132 standard arm), `noisy(p_true)` (Exp 132 noise arm — include an
`analytic_floor()` helper returning the irreducible-surprise value), `aliased(n_colors,
n_cells_per_color, layout_seed)` (Exp 143's three layouts must be reproducible from
seeds 7/11/13), `nonstationary(base, remap_at_step, remap_seed)` (abrupt color↔cell
remap). Tests: determinism per seed; aliased layouts for seeds 7/11/13 match the layouts
printed in the committed exp143/exp144 outputs; analytic floor for p=0.7 ≈ 0.82 nats.
Acceptance: suite green; module used by nothing yet (additive).

**T4 — Constants registry with provenance** · Sonnet · T1 ·
edit `active_loop/growth.py` (+ comment-only pointers elsewhere if helpful).
Every designed constant (0.7 nats ceiling mean, 5e-4 slope, 200-step window, 50-obs
per-color window, 100-obs pre-spawn window, 400-step probation, 0.1-nat keep margin,
burn-in iters/floors) becomes a named class attribute with a one-line provenance
comment: which experiment validated it and at what noise level/scale. Add a short
"power note" docstring on the probation window: with ~uniform color visitation,
400 steps ≈ 100 observations of the probated color vs the 50-obs alarm window.
Acceptance: no behavior change (tests from T2 still pass unchanged).

### Workstream C — reproducibility hardening

**T5 — Canonical environment + repro section** · Haiku · no deps ·
edit `README.md`; verify `uv.lock` is committed and current (`uv lock --check` or
equivalent; if stale, regenerate and commit separately).
Append a `## Reproduce` section to README (exact text):

> All results were produced inside the committed lockfile environment.
> ```bash
> uv sync                                  # creates .venv from uv.lock
> uv run --python .venv pytest -q          # fast suite (~2s)
> uv run --python .venv active-monkey-converse-demo    # capstone demo
> uv run --python .venv python experiments/exp145_m3c_live_probation.py  # any experiment re-runs from its script
> ```
> Every experiment script is committed together with its raw outputs under
> `experiments/outputs/`; seeds are fixed in-script; headline numbers can be
> recomputed from the committed rows (see `tools/`).

Acceptance: README renders; lockfile committed; nothing else changed.

**T6 — Machine-readable verdicts going forward** · Sonnet · no deps ·
new `active_loop/verdict.py`, new `tests/test_verdict.py`.
A small helper future experiment scripts call at the end:
`write_verdict(path, experiment, arms={name: {"pass": bool, "reason": str}},
verdict="POSITIVE|NEGATIVE|MIXED", halted=bool, notes=str)` → writes
`experiments/outputs/exp<N>_verdict.json`. Schema documented in the module docstring.
Do NOT retrofit past experiments (their outputs are frozen history).
Acceptance: helper + schema test green; nothing imports it yet.

**T7 — Headline audit tool for the M-series** · Sonnet · no deps ·
new `tools/audit_headlines.py`.
Recompute, from committed `experiments/outputs/exp1[4-5]*_*.json` rows, the headline
numbers quoted in the Exp 141–153 entries that are derivable from rows (e.g., per-layout
keep counts and probation deltas in 145, acceptance rates in 152, drop/quiet conjuncts
in 153, localization medians throughout) and print a MATCH/MISMATCH table against
expected values hardcoded in the tool *with a citation comment per number* (entry line
ranges in EXPERIMENTS.md). Read EXPERIMENTS.md if needed but never write it. Follow the
Exp 140 / exp102 audit-script precedent (`experiments/exp140_chapter_audit.py`).
Acceptance: tool runs clean end-to-end; mismatches (if any) are REPORTED in the commit
message and to the human, never silently "fixed."

### Workstream D — documentation fairness

**T8 — README "what this is / is not" + scale** · Haiku · no deps · edit `README.md`.
Insert near the top (exact text, adjust heading levels to match the file):

> ## What this is
> A toy active-inference lab: persistent agents at deliberately small scale
> (4×4 gridworlds, ≤16 observation classes, lives of a few thousand steps) that learn,
> act, form preferences, forget, and revise their world models through lived
> experience. Built on the active-inference framework (Friston et al.) and pymdp
> patterns — the framework is not ours; the testbed, benchmark discipline, and
> negative-results catalog are.
>
> ## What this is not
> - Not a consciousness claim. "Wants," "feels," "opinion" are functional labels for
>   measured quantities (valence = −free energy), documented as such.
> - Not AGI or a scaling claim. Nothing here is claimed to transfer beyond toy scale.
> - Not "model expansion works." As of Exp 153, five growth designs failed honestly;
>   the current best hypothesis (the emission-convention finding) is one predeclared
>   re-test from confirmation. See EXPERIMENTS.md Exp 143–153.
> - Not self-graded only: every experiment verdict since Exp 152 is checked by a
>   blinded verifier (loop/PROTOCOL.md step 4.5), and earlier chapters were
>   retro-audited (Exp 140, Exp 102).

Acceptance: README renders; no other claims touched.

**T9 — Public site language pass** · Haiku (exact replacements) · no deps ·
edit `index.html` (and `am-shared.jsx`/`experiments-data.js` ONLY if the same strings
live there; search first). NOTE: the site auto-deploys via git hooks — one commit, no
force-push.
Apply these exact replacements (verify each old string exists; if drifted, find the
nearest match and report what you changed):
1. meta description "one little mind coming online" → "a tiny active-inference agent
   learning its world from experience — a toy-scale open research lab".
2. "slowly builds a little mind that perceives, learns facts, wants things, and acts"
   → "slowly assembles a toy agent that localizes, learns its map, forms preferences,
   and acts — in a 4×4 world it has to figure out from experience".
3. "a mind is just a thing that tries to not be surprised" → "the bet borrowed from
   theoretical neuroscience: adaptive behavior falls out of minimizing surprise".
4. heading "A mind built from math" → "An agent built from math".
5. "a little mind made of math, figuring out its world from nothing" → "a small agent
   made of math, learning its world from experience plus one innate anchor".
6. heading "A little mind, coming online" → "A small agent, learning its world".
7. "Why a mind won't grow a concept on its own" → "Why unsupervised concepts didn't
   form in the regime we tested".
Then add to the hero section, styled like adjacent body text: one sentence —
"Everything here is toy scale (4×4 worlds, 16 colors), every experiment predeclares
its falsifiers before running, negative results are logged as negatives, and since
Exp 152 every verdict is checked by a blinded verifier." — followed by a visible link
to the GitHub repo next to the existing CTA links.
Acceptance: all seven replacements applied (or drift reported); page renders
(open the file, sanity-check structure); single commit.

**T10 — Related-work page** · Sonnet · no deps · new `docs/RELATED_WORK.md`,
plus one link line added to README.
Write the prior-art map as prose+table. Required entries with the honest overlap
statement for each (one short paragraph per row): active inference / EFE (Friston et
al.; total framework overlap — contribution is the lab, not the theory); Bayesian model
reduction (Friston & Penny; implemented in `structure_bmr`); structure learning in
active inference (Smith, Schwartenbeck, Parr, Friston ~2020; this lab's measured result
that replay-scored acceptance disagrees with live outcomes 57–74% of the time is a
pointed datum for that literature); prequential evaluation (Dawid — live probation IS
prequential acceptance applied to structure growth); perceptual aliasing & state
splitting (McCallum's Utile Suffix Memory / U-Tree — the closest prior art to the
aliased-world growth problem); HDP-HMM / infinite HMM (Beal; Fox sticky HDP-HMM — the
nonparametric answer to unknown state counts; note the background-floor design's kinship
to DP residual mass); split-merge mixtures (Ueda SMEM; Jain & Neal — the greedy-growth
valley found in Exp 144–146 is the classical motivation for these moves, rediscovered
under online honest acceptance); concept drift detection (DDM/ADWIN — cousins of the
ceiling detector, baselines for nonstationary worlds); forgetting factors in adaptive
filtering (Exp 137's decay-counts law); world models / MBRL (conceptual kinship only —
this lab is deliberately at auditable toy scale). Close with one paragraph: "no single
mechanism here is novel; the contribution is the integrated, falsifier-disciplined,
audited toy lab and its negative-results catalog."
Acceptance: page exists, README links it under a "Related work" bullet.

**T11 — Caveats ride along on the public experiment cards** · Sonnet · no deps ·
inspect how `journey.html` / `experiments-data.js` cards are generated
(see `active_loop/site_data.py` and its tests); add the entry's "Honest caveat" text
(or its first sentence, if length-constrained) to each experiment card's data so the
public timeline never shows a BREAKTHROUGH/POSITIVE card without its caveat.
Acceptance: regenerated site data includes caveat fields; site_data tests extended and
green; one spot-check documented in the commit message (e.g., Exp 20's card now carries
its "cells are directly observed" caveat).

### Workstream E — lab structure

**T12 — STATUS blocks on direction cards** · Haiku · no deps ·
edit every file in `loop/directions/` (and `_TEMPLATE.md`).
Append to each card a fenced block:

```
**STATUS.** state: <exploratory|active|halted|closed-positive|closed-negative|
flagship-candidate|published> · latest: Exp <N> · depends-on: <cards/modules> ·
reusable: <modules> · why: <one line> · next-falsifiable: <one line>
```

Fill from the cards' own content plus these anchors: continuous-creature →
halted-awaiting-consult, latest Exp 153 (growth thread) / ladder complete Exp 151;
continuous-substrate → closed-positive, latest Exp 138–140; transfer / sequence-substrate
/ relational-thoughts / social-emergence → exploratory unless the card says otherwise;
red-team → active (standing). Where the card content is insufficient to fill a field,
write `TBD-human` rather than inventing.
Acceptance: every card has the block; no other card text altered.

**T13 — Generated DIRECTIONS.md index** · Sonnet · T12 ·
new `tools/gen_directions_index.py`, new `DIRECTIONS.md`, new test.
Parse the STATUS blocks into a one-table index at repo root (direction · state · latest
Exp · next falsifiable step), with a header line "GENERATED — edit the cards, then
re-run tools/gen_directions_index.py". Test: regenerating produces identical bytes
(staleness guard, same pattern as the site-data tests).
Acceptance: index committed; staleness test green.

**T14 — Flagship page skeleton** · Sonnet · no deps ·
new `docs/flagship/worldview-too-small.md`.
Title: "Agents That Know When Their Worldview Is Too Small". Sections, in order:
30-second summary · 5-minute summary · problem statement · ordinary uncertainty vs
irreducible noise vs structural inadequacy vs drift (define all four) · toy worlds
(scale stated in the first line) · agent architecture · benchmark environments (A
learnable / B noise / C aliased / D nonstationary) · baseline suite · main result table
(PLACEHOLDER) · ablations (PLACEHOLDER) · live-vs-replay acceptance (write this section
NOW from Exp 144–146 numbers: the 57–74% disagreement, probation honesty 70–86%) ·
useful failures (write NOW: the growth valley Exp 144–146; the dilution autopsy Exp
152–153) · limitations · related work (link T10) · reproduction · what this does NOT
prove · next directions. Mark the page DRAFT — placeholders are filled only after
GATE-2 and the benchmark, by Loop B. The page must read correctly under BOTH GATE-2
outcomes (the failure sections are load-bearing either way).
Acceptance: skeleton committed with the two "write NOW" sections fully drafted from
the cited EXPERIMENTS.md entries (quote numbers, cite entry headers).

### Workstream F — benchmark harness (GATED on GATE-2: do not dispatch before Exp 154 is logged)

**T15 — Bench runner** · Sonnet · T1, T3, T6 ·
new `experiments/bench_worldview/run_bench.py`.
CLI: `--world {A,B,C,D} --mechanism {none,grow,decay,both,random_accept,replay_accept,
bigger_fixed,oracle} --seeds 8 --layouts 3 --convention {unnormalized,normalized}`.
Emits JSON rows (existing output conventions), a `verdict.json` via T6, and
`summary.md`. Mechanisms compose from `active_loop/growth.py` + `worlds.py`; no logic
re-implemented in-script. The predeclared bars per arm are NOT set by the subagent —
the runner takes them from a `bars.json` the research loop writes first.
**T16 — Baseline mechanisms** · Sonnet · T15 · same dir.
no-expansion; random-accept (accept spawns at the observed base acceptance rate, no
probation); replay-only acceptance (the Exp 144 rule, kept as the dishonest control);
bigger-fixed (true K per color installed at birth, both conventions); oracle (true K,
true assignments, fitted); decay-only (Exp 137's count-decay law).
**T17 — Bench plots + tables** · Sonnet · T15 ·
surprise traces with alarm/probation bands; the 4×4 world-type × response confusion
matrix; nats-regret vs oracle table; structure-economy (components used / needed).
Acceptance for F: full matrix runs end-to-end on 2 seeds in CI-ish time; outputs
committed for the smoke run; bars.json absent → runner refuses to grade (exit with
"awaiting predeclaration").

## 6. Definition of done

- Workstreams B–E: all tasks merged, suite green, FROZEN untouched, site renders, and
  `tools/audit_headlines.py` reports all-MATCH (or its mismatches are surfaced to the
  human and resolved in Loop B).
- Workstream F: harness smoke-tested and waiting on Loop B predeclaration.
- The growth/benchmark *science* (Exp 154 and the benchmark runs themselves) remains
  Loop B work under loop/PROTOCOL.md and loop/VALIDATION.md, as always.
