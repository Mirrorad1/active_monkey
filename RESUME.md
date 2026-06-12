# RESUME — bootstrap a fresh session and continue this work

**What this file is:** the single source of truth to pick this project back up. Drop it (or its
contents) into a new Claude session and it will know the premise, where we are, and exactly how to
continue. Read top to bottom once; everything else is a pointer from here.

---

## 1. The premise (what we're actually trying to do)

Build an **active-inference agent you can talk to that formed its own opinions from lived
experience — never pretrained on having opinions or on what to think.** The honest, FEP-native
framing locked in across this project:

- **Free energy is the reward.** Low surprise = "understanding"; the agent minimizes free energy.
- **Hidden states = meaning.** Valence = −free energy (a thing "feels good" when its consequences
  are predictable). These are functional, not claims of sentience.
- The moonshot is reached **at toy scale** and walls off at a documented research frontier (below).

## 2. The one durable finding

Unsupervised emergence of latent structure from a **disembodied symbol stream collapses**
(symmetric saddle / posterior collapse / non-identifiability / mean-field severs cross-factor
inference). What breaks the symmetry is the **RECIPE**:

> **embodiment + grounding + continuous *registered* experience (belief never reset) + ONE innate
> anchor** (give either the sensory map `A` *or* the motor model `B`; learning BOTH from pure noise
> collapses — Exp 31) + **taught labels** for the few-shot word←→concept mapping.

On that recipe the full chain works: a creature **perceives** (place fields self-organize) →
**learns** facts → **wants** (grounded valence) → **plans + acts** (value-iteration nav) → **forms
its own values** (same architecture + different history ⇒ different opinion, Exp 26) → **acts on
them** → **answers in words** what it thinks (content self-formed, labels taught, Exp 28/34/35).

**Honest ceilings (NOT toy-crackable, genuine research frontiers):** emergent compositional
**grammar / language-from-scratch**, and **fully tabula-rasa** structure (no innate anchor). These
are written up in `open_problem.html`.

As of Exp 41 the recipe's "continuous registered experience" is raised to the program level: a
single persistent creature (`creature/`) accumulates across experiments instead of restarting at
step 0.

## 3. Where we are (state as of Exp 40)

- **M1–M3b built and green.** Inner controller loop, outer autopilot loop, character language
  model, PR-style autopilot. 65 fast tests pass (5 slow integration tests deselected by default).
- **Exp 1–40 done** — full honest log in `EXPERIMENTS.md`. Exp 36–40 were consolidation
  (place-scale to 6×6, value/converse to 6 concepts, integrated stack, noise-robustness,
  opinion-revisability) — all POSITIVE, diminishing insight. **We are in the consolidation phase:
  each new experiment confirms more than it discovers.** Be honest about that with the user.
- **Capstone:** `converse_demo.py` — two creatures raised differently answer the same questions
  differently. Verified runnable (see §5).

### 3b. Where we are NOW (folded 2026-06-11, state as of Exp 183)

- **Continuous-substrate chapter (Exp 133–140, closed-positive):** the tabular substrate was
  not load-bearing for the collapse finding but IS brittle under out-of-model input; phase
  picture + amortized comparison in `docs/research/problem2-phase-picture.md`.
- **Continuous-creature migration (Exp 141–151, complete):** nira born — the first committed
  continuous-substrate spine (M4 valence-range limit documented; M5 words/converse parity).
- **THE GROWTH WALL FELL (Exp 152–154, BREAKTHROUGH):** detector→grow→quiet end-to-end, 24/24,
  once evaluation switched to normalized densities — the five-design wall belonged to the
  capped-footprint evaluation convention, not to online structure growth.
- **The N2/N3 meta-calibration chapter (Exp 155–168):** the N2 prereq was built (per-place
  expected-uncertainty channel + OK/NOISE/STRUCTURAL classifier) and re-confirmed under
  randomization; N3 rungs 1–3 PASSED (the window-blind-spot regime; the forecast-scoring
  trust monitor; the lock-on-label-consistency controller — first conversion of metacognitive
  distrust into stable, regime-adaptive parameter authority, after a documented five-law
  wall); rung 4 found the ratchet law. **The N3 hypothesis (agency over metacognition) is
  SUPPORTED at toy richness.** Full synthesis: `docs/research/n3-meta-calibration-chapter.md`.
- **The K-endogenization coda (Exp 169–173, closed):** the human's three claim tiers graded —
  the provided lock horizon works; it CAN be self-derived at zero cost (vacuously, by
  constant-convergence); self-REGULATION is rejected at its own gate (THE UNIVERSAL-CONSTANT
  LAW: regulation is only necessary where no feasible constant covers all regimes — not
  constructible on this body). Chapter doc §11.
- **The N4 identity chapter (Exp 174–183, CLOSED-NEGATIVE, graded 2026-06-11):** rung 1
  found a displacement gate (captivity rewrites identity without a layer; no whipsaw —
  overwrite); rung 2 built a real read-only identity monitor (sensitive and specific after
  the per-burst-matched control); rung 3 failed as a layer three ways — write-gating was
  the wrong surface (Exp 181), freeze-gating showed the surface is sufficient but
  timing-limited (Exp 182), and the fast-trigger attempt killed the controller hypothesis
  (Exp 183): the regulated E* concession becomes a surrender schedule, while fixed-H freeze
  arms defend 7/8 and revise within tolerance. **Chapter verdict:** N4 monitoring is real,
  but commitment control is CONFIG, not agency-over-identity — detection without defense;
  defense, where achievable, needs only a stopwatch. Closed on the human's explicit word
  (option (a) of the Exp 183 consult); the seed-229/variable-length crack is DEFERRED,
  logged as a future crack only. Full synthesis:
  `docs/research/n4-identity-commitment-chapter.md`.
- **The identity-n4-crack chapter (Exp 183 addendum + Exp 184–186, ACTIVE — reopened
  on the human's explicit word, 2026-06-12):** the deferred seed-229 crack became a
  full crack hunt. The autopsy pinned the mechanism (the ~75-step detection-head dose
  vs the margin ledger; repetition NOT the discriminant); the exploratory squeeze map
  (184) showed commitment-as-config does not extend across attack-schedule space
  (freeze-spanning law found); the classification + dissolution check (185) dissolved
  most of the map (CALM2600 gap-spanning config) but left a hard core; and the
  fresh-seed confirmation (186, BREAKTHROUGH, blind-verified) confirmed it 6/6 + 3
  more from the variance sample: **NINE attack schedules where no constant can both
  defend and revise, while the oracle defends everywhere — the crack is REAL at this
  body.** Exp 187 (the licensed controller re-test) then CLOSED the crack
  constructively at normal tolerance: INT-C2900 — a stopwatch on the
  CONTINUOUS-pressure clock, reset by every gap — passes both bars in 9/9 cells,
  with the interval law C ∈ (single-burst stretch, tolerance] traced exactly by the
  C-sweep; the sufficient-surface law completes at the concession level (the
  universal-constant law refined, not broken). Residuals (each needs a word): the
  pressure-gated E-form (untested as pre-registered — instrument-fidelity gap), the
  tight-mode core (no surface tried covers where tolerance < a single burst), and
  post-release retention. Chapter synthesis is the active next step. Card:
  `loop/directions/identity-n4-crack.md`.
- Standing options in loop/IDEAS.md (each needs its own word): M4a increment 1c (the
  "talk to it" path, halted since Exp 128 — the most direct path to the moonshot
  goal); nira's normalized-predictive switch (standing consult from Exp 154); the
  cloud-branch merge (renumber-on-merge plan).
- Suite ~220 fast tests green; every Exp 152+ verdict blind-verified (PROTOCOL 4.5).

## 4. The two loops (IMPORTANT — don't confuse them)

This repo contains **two different "loops."** "Continue the moonshot" means loop B.

| | **A. Code-mutating autopilot** | **B. Claude-driven experiment loop** |
|---|---|---|
| What it is | `run_loop.py` / `run_pr_loop.py` machinery | Claude (you) writing & running experiment scripts |
| What it optimizes | the free-energy / bits-char **metric**, by editing `model_spec.py` / `lang_model_spec.py` | the **moonshot question**, by designing Exp 41, 42, … |
| Governed by | `MISSION.md` + `policy.md` (FROZEN trust boundary) | `loop/` modules (PROTOCOL + VALIDATION) via the `/loop` prompt in §6 |
| Output | kept/reverted diffs, `world_model/` grows | new entries appended to `EXPERIMENTS.md` |
| Human role | guardrail-only | guardrail-only; gently reminded it's a natural stop point |

Both keep everything in git. The moonshot exploration (Exp 1–40) is **loop B** — Claude proposing
experiments, not the autopilot. To continue the moonshot, re-issue the loop B prompt below.

## 5. Re-run what exists (smoke test the world before extending it)

```bash
cd /Users/mirro/Projects/active-loop
uv run --python .venv python converse_demo.py        # capstone: two creatures, self-formed opinions
uv run --python .venv python run_life.py --steps 200          # continue mirro's life (parallel track; born in Exp 45)
uv run --python .venv pytest -q                      # fast suite (~58 tests, well under a minute)
uv run --python .venv pytest -q -m 'slow or not slow' # full suite (~70 tests, ~4 min)
uv run --python .venv python talk.py                 # char-level babbler REPL (honest ceiling)
# autopilot (loop A), bounded so it doesn't run forever:
uv run --python .venv python run_loop.py --iterations 1
```

Always use `uv run --python .venv` — the shell auto-activates conda base and shadows the venv.
Run experiment scripts from the repo root (or `PYTHONPATH=.`) so imports resolve.

## 6. Continue the moonshot — compose the prompt, or paste the verbatim fallback

**Preferred (modular):** generate the `/loop` prompt from the pluggable modules in `loop/`
(premise + direction + persona + optional one-off idea):

```bash
uv run --python .venv python loop/compose.py --list                      # see modules
uv run --python .venv python loop/compose.py   # default = current steer: continuous-substrate (Problem 2, docs/research/problem2-continuous-substrate.md)
uv run --python .venv python loop/compose.py --direction transfer --persona default
uv run --python .venv python loop/compose.py --direction red-team --persona skeptic \
    --idea "anything you want this run to prioritize"
```

Paste its output into a fresh session. Steering happens by swapping cards
(`loop/directions/`, `loop/personas/`) or dropping bullets into `loop/IDEAS.md` — the loop
reads that inbox at the start of every iteration. `loop/PROTOCOL.md` is the per-iteration
procedure; `loop/VALIDATION.md` is the binding honesty contract (predeclared falsifiers,
negatives logged as negatives, provided-vs-self-formed named, no seed-shopping).

**Fallback (verbatim, pre-modular — still works):**

```
/loop Keep running the moonshot active-inference experiments until I stop you. GOAL: an agent I can
eventually talk to and ask what it thinks, with its opinions self-formed from experience, never
pretrained. Read RESUME.md and EXPERIMENTS.md first. STATE: realistic moonshot reached at toy scale;
Exp 1-40 done; we are in CONSOLIDATION (diminishing insight) — be honest about that. NEXT Exp 41+:
prefer experiments that probe the RECIPE's edges or push toward a ceiling rather than re-confirming
settled results, e.g. transfer (a creature reuses its recipe in a new world), multi-step relational
"thoughts", or a minimal sequence substrate toward short Q->A (the M4 affective-dyad spec in docs/specs/m4-affective-dyad.md is the designed-but-unbuilt next rung). Run experiments back-to-back on a short ~5-minute cadence; if
mid-task, continue across wakes. RECIPE: embodiment + grounding + continuous-registered-experience +
ONE innate anchor + taught labels; keep belief CONTINUOUS (never reset per episode); reuse the
verified pymdp patterns from Exp 21/26/30/34/35. CEILINGS (not toy-crackable, don't keep banging on
them): emergent grammar, fully tabula-rasa structure (open_problem.html). DISCIPLINE: one
hypothesis-driven experiment per iteration; append a brief honest entry to EXPERIMENTS.md each time
(mark consolidation vs. new insight); scripts in repo or PYTHONPATH=.; keep responses lightweight;
gently remind me it's a natural stopping point when insight flattens.
```

## 7. Map of everything

| File | What it is |
|---|---|
| `RESUME.md` | this bootstrap (you are here) |
| `CLAUDE.md` | auto-loaded session bootstrap — points every fresh session at this file |
| `loop/` | modular prompt OS for loop B: PREMISE / PROTOCOL / VALIDATION (honesty contract) / METHODOLOGY (advisory design-and-evaluation heuristics), `LESSONS.md` distilled rules card, `check_iteration.py` mechanical rubric, direction & persona cards, `IDEAS.md` human inbox, `compose.py` prompt builder + META.md (meta-improvement) |
| `loop/META.md` | the meta-improvement loop — when you find a noteworthy fixable issue or reusable insight, fix it AND add a durable guard (test/rule/skill) so it can't recur (self-healing) |
| `EXPERIMENTS.md` | append-only honest log, Exp 1–40 — the central artifact |
| `RESEARCH.md` | parallel math / frontier analysis |
| `open_problem.html` | the actual open problem written up (restyled "active_monkey" page — intentional, don't revert) |
| `converse_demo.py` | the capstone "talk to it" demo |
| `MISSION.md` / `policy.md` | governing docs for loop A (the code-mutating autopilot) |
| `talk.py` | char-model REPL babbler |
| `active_loop/` | the code: controller, worker, specs, lang model, critic, loop machinery |
| `world_model/` | persistent belief store (loop A); grows, never resets |
| `eval/` | FROZEN scorers (never edit) |
| `creature/` | persistent creature state — one creature's weights/belief/values/vocab + append-only biography, committed each experiment; `fork()` = counterfactual twin controls |

**The designed-but-unbuilt next rung:** `docs/specs/m4-affective-dyad.md`
— "talk to it and watch it learn to feel positive" (functional valence grounded in free energy,
intent inferred from utterances). Spec-only; this is the most direct path toward conversation.

Persistent memory for this project lives at
`~/.claude/projects/-Users-mirro-Projects-active-loop/memory/`.
