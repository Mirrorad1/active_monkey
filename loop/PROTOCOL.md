# PROTOCOL — one iteration of the self-guided research loop

Each iteration is ONE hypothesis-driven experiment, run start-to-finish. Never batch
several half-experiments. Open `loop/LESSONS.md` (the distilled rules card) at the
start of every iteration — it is the one-page digest of every incident-derived rule;
the full stories stay in EXPERIMENTS.md. Steps:

**Session bootstrap (once per `/loop` session, before the first iteration):**
run `uv run --python .venv python -m meta_monkey.preflight`. Treat its checklist as
ADVISORY ONLY: passive process memory, not a controller, not a verdict-grader, and
not permission to weaken `loop/VALIDATION.md`.

0. **Inbox.** Read `loop/IDEAS.md`. If a human dropped an idea/redirection, it outranks
   your own queue: address it (run it, or log in IDEAS.md why it's deferred — never
   silently ignore it). Mark items you've consumed.

1. **Choose.** Pick the next experiment from the active direction card
   (`loop/directions/`). Check `EXPERIMENTS.md` first — don't repeat settled results.
   Prefer experiments that can FAIL: an experiment whose every outcome confirms the
   premise is consolidation at best.

2. **Predeclare.** Before writing code, write down (in the script's docstring):
   - Hypothesis: one sentence.
   - Prediction: what specific result you expect if the hypothesis is TRUE.
   - Falsifier: what result would count as the hypothesis being FALSE.
   No result may be interpreted against a prediction made after seeing it.
   When the run is stochastic or involves the persistent creature, predictions and
   falsifiers must be property-level with explicit thresholds (see VALIDATION.md).
   Design-time pass (advisory, `loop/METHODOLOGY.md` Part 1): mechanism vs outcome,
   specific null, confound list, strongest-version comparison, one independent variable
   (or a declared sweep with a shape-level falsifier), and name the bridge domain.

3. **Build small.** Smallest script that tests the hypothesis. Reuse verified patterns
   (Exp 21/26/30/34/35). Repo root or `PYTHONPATH=.`; run via `uv run --python .venv`.
   Fixed seed by default; if you try multiple seeds, report ALL of them.
   Write the script at `experiments/expNN_<slug>.py` inside the repo — never `/tmp` or
   any path outside the repo — so it is committed with the entry (see step 6).
   **Runtime / complexity pre-flight (binding, L25 — human request 2026-06-13):** BEFORE
   launching any full batch, do an algorithmic-complexity / runtime check so a bug cannot
   quietly burn hours of compute. Verify the inner loop is bounded (`O(alive)`/`O(cells)`,
   not `O(total-ever-born)` or `O(alive²)`); verify population is bounded (no no-scarcity
   regime growing toward the runaway cap — the Exp 202 ABUNDANT lesson); and project the
   wall-clock. For ecology runs use `ecology/runtime_budget.preflight(...)` (a logistic-aware
   probe that flags EXPLOSION / SUPERLINEAR / OVER_BUDGET) and call it with
   `require_safe=True` at the top of the experiment's `main()` so the batch refuses to launch
   on a flagged config. For non-ecology runs, do the equivalent back-of-envelope (jobs ×
   horizon × work/step) and a short smoke-timed extrapolation before the full run.
   **Parallelism & memory (binding, L31 — human request 2026-06-15):** experiment wall-clock is
   `c · (pop·horizon) · (arms·seeds)`, and the per-run sim is GIL-bound, so a multi-run batch
   (sweeps, multi-seed gates, multi-arm preflights) MUST run its INDEPENDENT runs in PARALLEL
   across cores, not in a serial `for seed in seeds:` loop. Independence holds because each run's
   `events_hash` depends only on its own seed (so parallelism changes NO result — guard with a
   serial-vs-parallel byte-match where feasible). Use `ecology/batch.py` (`run_batch` /
   `default_workers()`) for batch runs; for the Evolvability Preflight, the generic gates
   `run_local_pairwise_gradient` / `run_invasion_from_rarity` take a `max_workers` param — pass the
   memory-sized `recommended_workers` (below) to parallelize their per-seed runs (omit ⇒ serial;
   parallel is bit-identical, guarded by `tests/test_gates_parallel.py`). CONFIG-DRIVEN preflights
   (`run_preflight(cfg)` / `python -m ecology.evolvability --config`) need NO wiring — the runner
   auto-sizes via `runtime_budget.recommended_workers_for` and records it in `repro["workers"]`; set
   `PreflightConfig.max_workers` only to lower the ceiling (None ⇒ auto, 1 ⇒ serial). **BUT cap workers so memory cannot swap-thrash**
   (the L26 hours-scale failure mode): `max_workers × peak_RSS_per_run ≤ ~60% physical RAM`. Do NOT
   hand-set `max_workers` to the core count — pass the runs through
   `ecology.runtime_budget.preflight(..., max_workers=<cores>)`, which returns a `recommended_workers`
   already sized to the memory budget, and use THAT as the pool size. Over-subscribing workers (more
   than RAM fits) turns minutes into hours via swap. Also right-size the cheap knobs to the science
   first: carrying-capacity/pop (cap50 ≈ 5× cheaper than cap250 — only go big when a drift control
   actually shows drift, L29), seed count (8 standard; more only for a contested signal), arm set,
   and horizon (trim once the metric has plateaued, L23).
   **Division of labor:** follow `loop/ROUTING.md`. Claude ideates, designs, and
   validates; the CODING is dispatched to a Codex plugin worker on the `codex
   high-fast` profile (explicit fallback: `gpt-5.4-mini` with high reasoning)
   with a tight spec — files, expected behavior, exact verification command. The
   main model reviews the returned code against the spec and VALIDATION.md before
   running the experiment.
   Review must include the verdict logic conjunct-by-conjunct: the printed verdict
   line is the coder's claim, not the experiment's result, and "not a falsifier"
   never counts toward POSITIVE (L1, L2 in `loop/LESSONS.md`). Step 4.5's blinded
   verifier is the independent backstop, not a substitute for this review.

4. **Run & validate.** Apply `loop/VALIDATION.md` (binding) to the raw output before
   interpreting anything.

4.5. **Blinded verify (binding, added 2026-06-10 — independent verdict before logging).**
   Before writing the entry, dispatch a separate Codex plugin VERIFIER worker
   (`codex high-fast`, or `gpt-5.4-mini` with high reasoning) that is blinded to
   your interpretation. It receives ONLY:
   (a) the script's predeclared docstring (hypothesis / predictions / falsifiers), and
   (b) the committed raw output (`experiments/outputs/expNN.txt`) — never the main
   session's reading, never the draft entry. Instruct it to IGNORE any verdict/summary
   claims printed inside the output and recompute from the raw numbers. It returns a
   three-way verdict (POSITIVE / NEGATIVE / MIXED) derived by applying the predeclared
   rule conjunct-by-conjunct, with the mapping written out.
   - **Agree** → proceed to step 5; the entry records `- Verifier: agree` (a required
     entry line from Exp 152 on; `loop/check_iteration.py` enforces it).
   - **Disagree** → do NOT log yet; investigate. If the disagreement survives
     investigation (a genuine ambiguity in the predeclaration), take the STRICTER
     verdict and record both readings: `- Verifier: disagreed — <one line on what and
     how resolved>`.
   Rationale: the same mind that designed an experiment grades it leniently;
   independent verification beats self-critique on exactly the failure class this
   loop has logged repeatedly (L1/L2/L9 in `loop/LESSONS.md`).

5. **Log.** Append one honest entry to `EXPERIMENTS.md` in the established format
   (Plain / Setup / Result / Implication / Honest caveat / Verdict / Verifier / Next),
   explicitly tagged:
   POSITIVE / NEGATIVE / MIXED, and CONSOLIDATION / NEW INSIGHT. Negative results get
   the same care as positive ones.
   Evaluation pass before grading (advisory, `loop/METHODOLOGY.md` Parts 3–4): confound
   audit against the predeclared list, boundary conditions (where the effect STOPS),
   effect size not just significance, surprise audit; the Implication line names the
   generalizability tier claimed (analytic / functional-form / parameter-level / none).

   **Plain (mandatory, first line of every entry):** one or two jargon-free sentences —
   what we're *really* testing and what the result means, said simply, for a reader who
   knows none of the machinery. The in-depth Setup stays exactly as before; the Plain line
   sits above it as the simple, broad-base reference. This is the same text that becomes
   the entry's `plain` field in `experiments-data.js` (rendered above the technical setup
   in the journey). No double-quotes inside it (the site stores it as a JS double-quoted
   string).

   **Re-run re-quote rule:** if a script is re-run after ANY patch, the entry quotes
   the FINAL committed output — re-check every non-deterministic number (L3 in
   `loop/LESSONS.md`; `loop/check_iteration.py` warns on entry numbers absent from
   the committed output).

   **Self-grade (mandatory for every POSITIVE entry):** declare BREAKTHROUGH or
   POSITIVE-SINGLE.

   - **BREAKTHROUGH** = first demonstration of a capability the system did not have
     before, OR a result that materially redirected the research program. Use sparingly.
     Warning: confirmations, extensions, scale tests, and "the thing we built works"
     default to POSITIVE-SINGLE. Over-grading destroys the signal — if everything is a
     breakthrough, nothing is.
   - **POSITIVE-SINGLE** = a positive result that advances the work, but is not the
     first of its kind and is broadly consistent with predictions from prior experiments.

   **When BREAKTHROUGH:** the iteration is NOT complete until a "Story so far" synthesis
   is written. It must be 4–8 sentences covering: (a) what was being worked toward at
   this point in the arc, (b) what just emerged and precisely why it is new (what the
   system could not do before), and (c) the honest caveat — what is still provided,
   templated, or approximate. Write it for a reader who has read nothing else. This
   synthesis is placed BOTH in the EXPERIMENTS.md entry AND in the new curated entry's
   `story` field in `experiments-data.js`.

6. **Commit.** One commit per experiment containing the script
   (`experiments/expNN_<slug>.py`), its raw output (`experiments/outputs/expNN.txt`),
   AND the EXPERIMENTS.md entry: `expNN: <one-line honest summary>`. All land in the
   same atomic commit, written and committed within ONE turn (atomicity norm, binding
   — L4 in `loop/LESSONS.md`). A log entry without committed script and output is an
   unverified claim.
   **Mechanical rubric (binding, added 2026-06-10):** before committing, run
   `uv run --python .venv python loop/check_iteration.py` — it checks the entry's
   required lines/tags, the committed artifacts, and the docstring predeclaration.
   Hard failures block the commit; warnings (entry numbers not found in the raw
   output) must each be confirmed derived-not-stale.
   - **Persistent-creature experiments (Exp 58+) advance ONE continuous spine** (`mirro`
     by default). They do NOT re-birth or branch the spine: wrap the episode in
     `active_loop.checkpoint.mirro_episode("Exp NN")`, which loads the committed snapshot,
     records a checkpoint BEFORE (age + state_hash), and on clean exit saves the spine
     and records a checkpoint AFTER. The entry quotes that before/after block
     (`ep.report()`). Commit the updated `creature/state/mirro/` snapshot (arrays.npz +
     manifest.json + biography) in the same atomic commit — it is the resume-from point.
     **Forks stay the scientific control, but only as SIDE-controls:** branch them with
     `ep.fork_control(name)` from the spine's pre-experiment state and save them under a
     non-spine path; a fork never replaces the spine. A branch raised long enough is a peer
     species — promote it to its own committed line (`twin.save(creature/state/<name>/)`);
     mirro is then the **root ancestor** of a clade (the long arc:
     `loop/directions/social-emergence.md`). The continuity guard
     (`tests/test_creature_continuity.py`) fails CI on a SILENT reset of an established line —
     so the life is never-reset by default. The escape hatch (anti-lock-in): recover from a
     bad epoch by `git checkout` of a prior committed snapshot, and an intentional restart is
     allowed only when logged as an explicit `rebirth` biography event.

   **Site update (mandatory):** the same commit also updates `experiments-data.js` with
   the new curated entry — kind graded honestly (breakthrough / positive / wall /
   partial), a `plain` field (the layman's "in plain terms" from step 5, required on
   every entry), and a `trace` block with:
   - `script`: `"experiments/expNN_<slug>.py"` — the script committed above
   - `output`: `"experiments/outputs/expNN.txt"` — the raw output committed above
   - `rerun` / `verified`: omit these fields for now; add them only after an actual
     independent re-run confirms the output matches.

   `experiments/recovered/` is the **sealed historical archive for Exp 1–40 only**.
   Nothing new goes there. Future experiments (Exp 41+) use the paths above.

   The `story` field is added if BREAKTHROUGH. The fast test suite
   (`uv run --python .venv pytest -q`) enforces that the curated count equals the log
   count, that every entry carries a non-empty `plain` field, and that
   `trace.script` / `trace.output` exist on disk; a skipped site update
   will fail CI and must be fixed before the commit lands.

6.5. **Record passive process memory.** Immediately AFTER the experiment commit lands,
   write the passive Meta Monkey episode record:
   `uv run --python .venv python -m meta_monkey.collect_iteration --latest --write`
   This is loop-scaffolding evidence, not the scientific verdict: it records the
   completed iteration's process state under `meta/episodes/expNN.json` so later
   sessions can consult the process record without scraping prose. It must not edit
   `EXPERIMENTS.md`, choose the next experiment, or overrule `loop/VALIDATION.md`.

7. **Reflect.** If the last ~3 entries are all consolidation, say so to the human and
   suggest either a direction switch (other cards), a harder edge, or stopping — a
   natural stop point is a finding, not a failure.
   **Meta-check:** if this iteration surfaced a noteworthy NON-research issue (a stale doc, an unreproducible claim, a harness/site bug, an honesty slip) or a reusable insight, run `loop/META.md` before continuing — fix it AND add a durable guard (test / rule / skill) so it can't recur. Patching the instance without the guard is incomplete.
