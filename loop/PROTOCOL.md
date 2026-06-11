# PROTOCOL — one iteration of the self-guided research loop

Each iteration is ONE hypothesis-driven experiment, run start-to-finish. Never batch
several half-experiments. Steps:

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

3. **Build small.** Smallest script that tests the hypothesis. Reuse verified patterns
   (Exp 21/26/30/34/35). Repo root or `PYTHONPATH=.`; run via `uv run --python .venv`.
   Fixed seed by default; if you try multiple seeds, report ALL of them.
   Write the script at `experiments/expNN_<slug>.py` inside the repo — never `/tmp` or
   any path outside the repo — so it is committed with the entry (see step 6).
   **Division of labor:** the main model ideates, designs, and validates; the CODING is
   dispatched to a Sonnet subagent (`Agent` tool, `model: "sonnet"`) with a tight spec —
   files, expected behavior, exact verification command. The main model reviews the
   returned code against the spec and VALIDATION.md before running the experiment.
   **Review must include the verdict logic:** check every compound predeclaration
   (AND/OR conjuncts, conditional branches) is implemented exactly — Exp 72's script
   printed a passing verdict after silently dropping a conjunct; the per-seed raw data
   plus validation against the docstring caught it. The printed verdict line is the
   coder's claim, not the experiment's result; the entry's verdict comes from applying
   the predeclared rule to the committed raw output.
   **Three-way verdict rule (added after Exp 136):** when a predeclaration says
   "TRUE iff all", the POSITIVE branch must require EVERY conjunct — a condition
   labeled "not a falsifier" still blocks POSITIVE and routes to MIXED. "Not a
   falsifier" never means "counts toward POSITIVE". Coder subagents soften this
   distinction reliably (Exp 133, Exp 136); check the three-way branch explicitly.

4. **Run & validate.** Apply `loop/VALIDATION.md` (binding) to the raw output before
   interpreting anything.

5. **Log.** Append one honest entry to `EXPERIMENTS.md` in the established format
   (Plain / Setup / Result / Implication / Honest caveat / Next), explicitly tagged:
   POSITIVE / NEGATIVE / MIXED, and CONSOLIDATION / NEW INSIGHT. Negative results get
   the same care as positive ones.

   **Plain (mandatory, first line of every entry):** one or two jargon-free sentences —
   what we're *really* testing and what the result means, said simply, for a reader who
   knows none of the machinery. The in-depth Setup stays exactly as before; the Plain line
   sits above it as the simple, broad-base reference. This is the same text that becomes
   the entry's `plain` field in `experiments-data.js` (rendered above the technical setup
   in the journey). No double-quotes inside it (the site stores it as a JS double-quoted
   string).

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

6. **Commit.** One commit per experiment containing the script.
   **Atomicity norm (resolves the Exp 41/42 autosync split):** write script + raw output +
   EXPERIMENTS.md entry + experiments-data.js AND commit them within ONE turn — the Stop-
   hook autosync fires between turns and will sweep a half-finished working set into an
   `auto-sync:` commit. Exp 64-81 all committed atomically this way; treat it as binding.
   One commit per experiment containing the script
   (`experiments/expNN_<slug>.py`), its raw output (`experiments/outputs/expNN.txt`),
   AND the EXPERIMENTS.md entry: `expNN: <one-line honest summary>`. All three land in
   the same atomic commit. A log entry without committed script and output is an
   unverified claim.
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

7. **Reflect.** If the last ~3 entries are all consolidation, say so to the human and
   suggest either a direction switch (other cards), a harder edge, or stopping — a
   natural stop point is a finding, not a failure.
   **Meta-check:** if this iteration surfaced a noteworthy NON-research issue (a stale doc, an unreproducible claim, a harness/site bug, an honesty slip) or a reusable insight, run `loop/META.md` before continuing — fix it AND add a durable guard (test / rule / skill) so it can't recur. Patching the instance without the guard is incomplete.
