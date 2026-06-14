# Lab Infrastructure Roadmap — design

- **Date:** 2026-06-13
- **Status:** approved-shape, spec under review
- **Author:** Claude (Loop B, design session)
- **Topic:** make active-loop cheap to drive and clean to grow — an invocation layer over the
  existing `loop/` prompt-OS, plus a staged reorganization toward a DeepMind-grade repo.

---

## 1. Context & problem

`active-loop` is a mature active-inference research repo (Exp 1–207, two loops, a modular prompt-OS
in `loop/`, a live showcase site, several code substrates). It is driven almost entirely by a human
**pasting prompts into Claude sessions**. The friction and the debt, ranked by the human:

1. **Start-up courier hop.** `loop/compose.py` already takes `--direction/--persona/--idea`, prints
   ~9.5 KB, and the human copies that and pastes it after the built-in `/loop`. The composer is
   already CLI-shaped; the missing piece is a thin in-session command. There is **no
   `.claude/commands/` directory yet** — greenfield.
2. **Root clutter / "doesn't look like a real repo."** ~670 KB of website files (`experiments-data.js`
   512 KB, `*.jsx/.css/.js`, `*.jpeg`) sit at root alongside code, governance `.md`s, and data.
3. **Finding docs & directions.** Referencing a chapter/experiment/direction means typing paths or
   grepping a 716 KB `EXPERIMENTS.md`; no smart `@`-refs; root `DIRECTIONS.md` (generated) vs
   `loop/directions/` (canonical) ownership is invisible.
4. **Steering / CONSULT replies.** Mid-flight steering = hand-edit `loop/IDEAS.md`; answering a
   consult = type a word into IDEAS.md *and* re-paste.

### What is already good (build ON, do NOT rebuild)
`loop/compose.py` (parameterized CLI), the `loop/` prompt-OS (PROTOCOL/VALIDATION/METHODOLOGY/
LESSONS/META/EFFICIENCY), `check_iteration.py` + the blinded verifier, the `IDEAS.md` inbox model,
the auto-publish pipeline (`tools/autosync.sh` Stop hook + native git push → GitHub Pages),
`active_loop/site_data.py` + `tools/gen_directions_index.py` generators, the `FROZEN` manifest, and
`RESUME.md` as the single excellent entry point.

---

## 2. Goals & non-goals

**Goals**
- G1. Collapse the start-up courier hop into one in-session command (`/lab <direction>`) that
  **composes, shows the prompt, and iterates only on explicit human confirmation**.
- G2. Make mid-flight steering and consult-replies a single command, **without eroding the
  human-vs-automated consent boundary**.
- G3. Make the repo root read like a clean OSS project: code / site / docs / governance separated.
- G4. Make docs, directions, and experiments **referenceable by short key** (`@n4-identity`,
  `@exp190`) and queryable by an index.
- G5. Remove the scaling cliffs in the 700 KB+ monoliths without breaking the append-only culture.

**Non-goals**
- N1. Not rebuilding `compose.py`, the prompt-OS, or the honesty gates — we wrap them.
- N2. Not changing the *science* (PROTOCOL/VALIDATION semantics stay identical).
- N3. Not migrating GitHub Pages to a CI/Action deploy in this roadmap (noted as a future upgrade).
- N4. Not auto-resuming or auto-steering from automation — automation stays guardrail-only.

---

## 3. Invariants the design must not break

These are load-bearing properties verified during mapping. Every stream is checked against them.

- **INV-1 (consent boundary).** `VALIDATION.md §5`: a cron/automation fire is **NOT** a human
  resumption. Any steer/resume command must keep human-initiated and automation-initiated actions
  distinguishable. A human typing `/steer` *is* a human action and is fine; automation must never
  call it.
- **INV-2 (append-only honesty).** `EXPERIMENTS.md` and `IDEAS.md` are append-only / never-rewrite.
  Data-scaling adds *index/offset layers and generated views*, never rewrites history in place.
- **INV-3 (FROZEN).** `eval/` and the Loop-A trust boundary in `FROZEN` are never edited.
- **INV-4 (Pages-from-root).** Pages serves the repo root of `Mirrorad1/active_monkey` (no CNAME, no
  Action). Branch-based Pages can only serve `/` or `/docs`. So site relocation keeps the HTML entry
  points + `.nojekyll` at root and moves *assets* under `site/`.
- **INV-5 (shared-checkout concurrency).** A live cron loop commits to this same working tree;
  `tools/autosync.sh` sweeps a fixed `managed_paths` set on every Stop. Any path move must update
  `managed_paths` in the same change, or artifacts get stranded/auto-committed wrong.
- **INV-6 (atomic experiment commit).** PROTOCOL step 6: script + output + entry + site data in one
  commit per experiment. The invocation layer must not interpose extra commits mid-iteration.
- **INV-7 (division of labor).** Design/validation stays with the main (Fable/Opus) model; coding is
  dispatched to Sonnet subagents (the standing "main ideates, Sonnet codes" working preference).

---

## 4. Architecture decision: hybrid

Ship the invocation layer as a **thin command wrapping the existing `compose.py`** (immediate win),
and introduce **one small generated index** — the *backbone* — that smart-referencing, doc-finding,
and later data-scaling progressively adopt. We do not build a maximal registry up front, and we do
not leave referencing grep-backed forever.

```
                 ┌─────────────────────────────────────────────┐
   human  ──/lab──▶  .claude/commands/lab.md                    │
          ──/steer─▶  .claude/commands/steer.md                 │  Stream A
                 │     │ shell out                              │
                 │     ▼                                        │
                 │   loop/compose.py  (+positional direction)   │
                 │   loop/IDEAS.md    (human-marked append)     │
                 └───────────────┬─────────────────────────────┘
                                 │ resolves names against
                                 ▼
                 ┌─────────────────────────────────────────────┐
                 │   loop/registry.json  (generated backbone)   │  Stream C
                 │   directions ▸ personas ▸ chapters ▸ exps    │
                 └───────────────┬─────────────────────────────┘
                                 │ also powers
                 ┌───────────────┴───────────┐
                 ▼                           ▼
         docs/INDEX.md  (@-refs)      experiments_parser.py     Stream C/D
                                      (one parser, 4 consumers)
```

The backbone (`loop/registry.json`) is generated by extending the existing
`tools/gen_directions_index.py` — not a new subsystem.

---

## 5. Stream A — Invocation layer (BUILD-OUT, ships first)

### A.1 `/lab` — compose + confirm + iterate

**New file:** `.claude/commands/lab.md`

```markdown
---
description: Compose and launch a research-loop iteration (compose → confirm → iterate)
argument-hint: <direction> [--persona NAME] [--idea "..."]
allowed-tools: Bash(uv run --python .venv python loop/compose.py:*)
---
Assembled /loop prompt for this run:

!`uv run --python .venv python loop/compose.py $ARGUMENTS`

That block is the composed prompt (premise + direction + persona + discipline). Do this:
1. Summarize in ONE line what this run will do: direction, persona, and any one-off idea.
2. If the direction did not resolve (compose.py printed an error), show
   `uv run --python .venv python loop/compose.py --list` and ask which direction I meant.
3. STOP and wait for my explicit "go" (or "edit ..."). Do not begin iterating yet — this
   confirm step is the human-consent gate (VALIDATION.md §5 / INV-1).
4. On "go": read RESUME.md + the EXPERIMENTS.md tail + loop/IDEAS.md, then iterate under
   loop/PROTOCOL.md, self-pacing across wakes via the standard /loop dynamic mechanism.
```

**Supporting change (tiny, backward-compatible) — `loop/compose.py`:** accept the direction as an
optional **positional** arg so `/lab ecology --persona skeptic` maps cleanly, while `--direction`
stays as an alias. Add a **name resolver**: exact card stem first; else fuzzy/alias match against the
backbone (`ecology` → `population-ecology`); else error listing options. Until Stream C lands, the
resolver falls back to exact-stem + a hardcoded alias map.

**Behavior contract**
- `/lab` with no args → composes the default direction (current steer), still confirm-gated.
- `/lab population-ecology --idea "try a foraging gradient"` → composes with the idea injected.
- The command never commits, never edits IDEAS.md, never starts iterating before "go".

### A.2 `/steer` — human-marked inbox writes

**New file:** `.claude/commands/steer.md`. Two modes:
- `/steer "<free-text idea or redirection>"` → append a bullet to the `## Inbox` section of
  `loop/IDEAS.md`, tagged `[from human, <UTC date>]` (mirrors the existing hand-written convention),
  then confirm what was written.
- `/steer --resume <word>` → append the human's consult reply as a clearly human-attributed line and
  surface it so the next iteration consumes it.

**INV-1 handling:** the bullet is stamped `[from human, …]` exactly as hand authored. The command is
only ever invoked interactively by the human; `tools/autosync.sh` and cron never call it. The
distinction the consent model relies on is preserved because *who pressed enter* is unchanged — we
only remove the open-file/edit/save mechanics, not the human's authorship.

`allowed-tools` is scoped to a single append helper (a 15-line `tools/steer_append.py` that appends
under `## Inbox` and refuses to rewrite existing lines — enforces INV-2) so the command cannot edit
arbitrary files.

### A.3 Stream E hygiene, folded in
- Make `tools/autosync.sh` `managed_paths` **config-driven**: read from `loop/managed-paths.json`
  (a list) instead of the hardcoded array (`autosync.sh:25-37`). This is a prerequisite for Stream B
  (so moving site files can't silently break the sweep) and removes a class of INV-5 footguns.
- Document the two new commands in `loop/README.md` and `RESUME.md §6` (replace the "paste the
  output" instructions with "`/lab <direction>`").

### A.4 Acceptance (Stream A)
- `/lab ecology` composes, shows a one-line summary, and waits; typing "go" begins a PROTOCOL
  iteration; no commit happens before "go".
- `/steer "..."` adds a `[from human, …]` bullet to `loop/IDEAS.md` `## Inbox` and nothing else.
- `autosync.sh` reads `loop/managed-paths.json`; existing managed set reproduced exactly; suite green.
- `RESUME.md §6` and `loop/README.md` document the command-first flow; the paste fallback is retained
  as "if a command is unavailable."

---

## 6. Stream B — Root cleanup → `site/` (design level; ships second)

**Move into `site/`:** `experiments-data.js`, `lab-status.js`, `am-live.js`, `am-shared.jsx`,
`tweaks-panel.jsx`, `am.css`, and the `*.jpeg` showcase images (→ `site/img/`).
**Keep at root (INV-4):** `index.html`, `journey.html`, `open_problem.html`, `.nojekyll`,
`favicon.ico`.

**Exact path edits required (enumerated so nothing breaks silently):**
- `index.html`, `journey.html`, `open_problem.html`: script/style `src`/`href` → `site/…`
  (the `?v=5` cache-bust strings move with them).
- `active_loop/site_data.py`: output paths for `experiments-data.js` + `lab-status.js` → `site/`.
- `meta_monkey/collect_iteration.py`: any `experiments-data.js` path → `site/`.
- `am-live.js`: relative fetch/asset paths.
- `tools/autosync.sh` `managed_paths` (now `loop/managed-paths.json` from A.3): update entries.
- Tests: `tests/test_lab_status.py`, `tests/test_site_data.py`, `tests/test_meta_monkey.py` expected
  paths → `site/`.

**Cruft pass:** track-or-drop the root `*.jpeg` screenshots (currently `.gitignore`'d by `*.jpeg` yet
referenced by `RESUME.md` → broken on clone — either commit them under `site/img/` with a `.jpeg`
un-ignore, or replace doc refs); delete the 90-byte `REPORT.md` stub; stop tracking `.venv` /
`.pytest_cache` (add to `.gitignore` if not already; `git rm --cached`).

**Risk:** MEDIUM-HIGH — mechanical but a single missed reference silently breaks the live site.
**Mitigation:** a deploy smoke-test (serve root locally, load `index.html`, assert the three scripts
+ data file 200 and the lab-status renders) added as the last step; run before push. Future upgrade
(N3): GitHub Action that publishes `site/` to `gh-pages`, removing the keep-HTML-at-root constraint.

---

## 7. Stream C — Doc & direction taxonomy + backbone (design level; ships third)

Unlocks the smart-referencing half of A and the human's #3 pain.

**Backbone index — `loop/registry.json`** (generated by extending `tools/gen_directions_index.py`):
for each **direction** (name, aliases, status line, ladder, file), **persona**, **research chapter**
(`docs/research/*.md` → key, title, arc, experiments covered), and **experiment** (expNN → slug,
verdict, branch, output path, EXPERIMENTS.md offset). Regenerated by the existing autosync self-heal
step (`autosync.sh:19`).

**`@`-referencing resolver:** a short helper (`tools/ref.py`, or inline in commands) that maps
`@n4-identity` → `docs/research/n4-identity-commitment-chapter.md`, `@exp190` →
`experiments/exp190_*.py` + its EXPERIMENTS.md offset, `@ecology` → the direction card. Used by
`/lab`'s resolver (A.1) and citeable in cards/prompts/IDEAS.

**`docs/INDEX.md`** (generated, human-facing view of the backbone): chapters ↔ arcs ↔ directions ↔
experiments. Replaces grep-the-monolith for navigation.

**Governance consolidation (de-dup, INV-2-safe):**
- Single-source the "two loops" explainer in `RESUME.md §4`; make `CLAUDE.md` / `AGENTS.md` thin
  generated pointers (one short paragraph + "see RESUME.md") rather than near-duplicates.
- Note `MISSION.md` + `policy.md` + `FROZEN` as the Loop-A governance triad in one place; keep
  `FROZEN` authoritative and untouched (INV-3).
- Clarify `DIRECTIONS.md` (root, generated) ownership: add a "generated — edit `loop/directions/`"
  banner so edits don't get lost on regen.

**Risk:** LOW-MEDIUM — additive indexes are safe; governance consolidation needs care not to drop a
load-bearing rule. **Mitigation:** consolidation is pointer-only (no rule deletion); a diff review of
every removed sentence.

---

## 8. Stream D — Data scaling (design level; ships last; backward-compatible)

Highest risk because `EXPERIMENTS.md` is the append-only source of truth touched by ≥4 consumers.

- **Keep `EXPERIMENTS.md` one append-only file** (INV-2) but add a **generated index/offset table**
  (`experiments-index.json`: expNN → byte offset, verdict, arc) so consumers can seek instead of
  parsing 716 KB. (Chosen over physically splitting the file — preserves append-only simplicity.)
- **Extract one shared `experiments_parser.py`** and route `check_iteration.py`,
  `active_loop/site_data.py`, `am-live.js` (via the generated JSON), and `checkpoint.py` through it —
  turning 4 fragile re-implemented regex parsers into 1. This also closes the latent "add a field →
  silently break a consumer" cliff.
- **Lazy-load / paginate `experiments-data.js`:** ship only the latest N inline; fetch older on
  demand. Add ETag/`If-None-Match` to `am-live.js`'s GitHub fetch (currently a hardcoded 7 s pull of
  the raw file). Automate the `?v=` cache-bust (derive from a content hash) so the 3 HTML files stop
  drifting.
- **Archive policy** for `experiments/outputs/` and a note resolving the hidden
  `experiments/recovered/` second directory so the true count is legible.

**Risk:** HIGH. **Mitigation:** every change behind a regenerate-and-diff check; the shared parser
must reproduce each current consumer's output byte-for-byte on the existing corpus before cutover.

---

## 9. Sequencing & dependencies

```
A (invocation + E hygiene)  ──▶  ships the #1 win immediately; no deps
        │  (A.3 config-driven autosync paths)
        ▼
B (root → site/)            ──▶  depends on A.3 so the move can't break the sweep
        │
        ▼
C (backbone + @-refs + governance)  ──▶  completes A's smart-referencing; needs B's
        │                                 settled paths for the registry's file links
        ▼
D (data scaling)            ──▶  last; benefits from C's index/parser groundwork
```

B and D both touch `active_loop/site_data.py` + `tools/autosync.sh`; sequencing them apart (B before
D) avoids path-edit collisions. Each stream is independently shippable and independently revertible.

---

## 10. Risk summary

| Stream | Risk | Primary mitigation |
|---|---|---|
| A | MEDIUM | Confirm-gate preserves INV-1; commands write nothing before "go"; steer is append-only |
| B | MED-HIGH | Enumerated path edits + deploy smoke-test before push; keep HTML at root (INV-4) |
| C | LOW-MED | Additive indexes; governance edits are pointer-only with per-sentence diff review |
| D | HIGH | Index/offset layer not a rewrite (INV-2); shared parser must byte-match all consumers pre-cutover |

---

## 11. Build notes
- Per INV-7, each stream's **coding is dispatched to Sonnet subagents** with tight specs (exact files,
  exact behavior, exact verification command); design/validation/verdicts stay with the main model.
- Each stream lands as its own small PR/branch with the suite green; no FROZEN edits (INV-3).
- This is infra (Loop-A/B machinery), **not** an experiment — no `EXPERIMENTS.md` entry; `META.md`
  applies (institutionalize a guard with each fix, e.g. the deploy smoke-test, the byte-match gate).

## 12. Open decisions deferred (not blocking)
- D: automate `?v=` cache-bust via content hash vs keep manual — decide at D time.
- N3 future: GitHub Action `gh-pages` deploy (removes INV-4's keep-HTML-at-root constraint).
- C: `tools/ref.py` standalone vs inline-in-command resolver — decide at C time.

---

## 13. Acceptance criteria (whole roadmap)
- A: `/lab <direction>` and `/steer` exist, confirm-gated, documented in `RESUME.md`; suite green.
- B: root contains no JS/JSX/CSS/data/image clutter (only HTML entry points + config); live site loads
  identically (smoke-test passes); clone yields no broken doc image refs.
- C: `loop/registry.json` + `docs/INDEX.md` generate from source; `@key` refs resolve; governance docs
  de-duplicated to single sources.
- D: a shared `experiments_parser.py` backs all consumers (byte-match verified); site lazy-loads;
  `EXPERIMENTS.md` remains one append-only file with a generated index.
