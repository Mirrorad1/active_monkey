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

3. **Build small.** Smallest script that tests the hypothesis. Reuse verified patterns
   (Exp 21/26/30/34/35). Repo root or `PYTHONPATH=.`; run via `uv run --python .venv`.
   Fixed seed by default; if you try multiple seeds, report ALL of them.
   Write the script at `experiments/expNN_<slug>.py` inside the repo — never `/tmp` or
   any path outside the repo — so it is committed with the entry (see step 6).
   **Division of labor:** the main model ideates, designs, and validates; the CODING is
   dispatched to a Sonnet subagent (`Agent` tool, `model: "sonnet"`) with a tight spec —
   files, expected behavior, exact verification command. The main model reviews the
   returned code against the spec and VALIDATION.md before running the experiment.

4. **Run & validate.** Apply `loop/VALIDATION.md` (binding) to the raw output before
   interpreting anything.

5. **Log.** Append one honest entry to `EXPERIMENTS.md` in the established format
   (Setup / Result / Implication / Honest caveat / Next), explicitly tagged:
   POSITIVE / NEGATIVE / MIXED, and CONSOLIDATION / NEW INSIGHT. Negative results get
   the same care as positive ones.

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
   AND the EXPERIMENTS.md entry: `expNN: <one-line honest summary>`. All three land in
   the same atomic commit. A log entry without committed script and output is an
   unverified claim.

   **Site update (mandatory):** the same commit also updates `experiments-data.js` with
   the new curated entry — kind graded honestly (breakthrough / positive / wall /
   partial), and a `trace` block with:
   - `script`: `"experiments/expNN_<slug>.py"` — the script committed above
   - `output`: `"experiments/outputs/expNN.txt"` — the raw output committed above
   - `rerun` / `verified`: omit these fields for now; add them only after an actual
     independent re-run confirms the output matches.

   `experiments/recovered/` is the **sealed historical archive for Exp 1–40 only**.
   Nothing new goes there. Future experiments (Exp 41+) use the paths above.

   The `story` field is added if BREAKTHROUGH. The fast test suite
   (`uv run --python .venv pytest -q`) enforces that the curated count equals the log
   count and that `trace.script` / `trace.output` exist on disk; a skipped site update
   will fail CI and must be fixed before the commit lands.

7. **Reflect.** If the last ~3 entries are all consolidation, say so to the human and
   suggest either a direction switch (other cards), a harder edge, or stopping — a
   natural stop point is a finding, not a failure.
   **Meta-check:** if this iteration surfaced a noteworthy NON-research issue (a stale doc, an unreproducible claim, a harness/site bug, an honesty slip) or a reusable insight, run `loop/META.md` before continuing — fix it AND add a durable guard (test / rule / skill) so it can't recur. Patching the instance without the guard is incomplete.
