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

4. **Run & validate.** Apply `loop/VALIDATION.md` (binding) to the raw output before
   interpreting anything.

5. **Log.** Append one honest entry to `EXPERIMENTS.md` in the established format
   (Setup / Result / Implication / Honest caveat / Next), explicitly tagged:
   POSITIVE / NEGATIVE / MIXED, and CONSOLIDATION / NEW INSIGHT. Negative results get
   the same care as positive ones.

6. **Commit.** One commit per experiment: `expNN: <one-line honest summary>`.

7. **Reflect.** If the last ~3 entries are all consolidation, say so to the human and
   suggest either a direction switch (other cards), a harder edge, or stopping — a
   natural stop point is a finding, not a failure.
