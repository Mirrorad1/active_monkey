# creature/state — persistent creature snapshots

## What lives here

One directory per creature, e.g. `creature/state/creature-A/`:

| File | Contents |
|------|----------|
| `arrays.npz` | All learned arrays: `pA` (Dirichlet counts = the weights), `qs` (place belief), `value_counts` (grounded valence), `vocab__<word>` (taught word associations). |
| `manifest.json` | Scalars: name, lineage, age_steps, true_pos, world config (rows, cols, cmap, n_colors), rng_counter, birth seed, state_hash, saved_at. |
| `BIOGRAPHY.jsonl` | Append-only event log — one JSON line per `live()`, `teach_word()`, `save()`, `fork()` call. Never truncated. |

## Why state is committed to git

The RECIPE requires **continuous registered experience across sessions** — the
creature's belief (`qs`) and learned sensory map (`pA`) must persist.  Committing
`arrays.npz` + `manifest.json` IS the "weights recording".  Any empirical claim
("after 900 steps the creature prefers red") is resumable from that snapshot.

Snapshot workflow for an experiment:
1. `creature.save("creature/state/<name>")` + `git commit` before the experiment.
2. Run the experiment (multiple `live()` calls, possibly `teach_word()`).
3. `creature.save(...)` + `git commit` after.  The diff shows exactly what changed.

## Fork = counterfactual twin

`fork(new_name)` deep-copies the committed snapshot.  The twin starts with the
same history but can be placed in a different world or run with a different seed.
Any divergence in values, map accuracy, or language is causally attributable to
post-fork experience.  This is the project's main experimental control.

## Analogy to `experiments/recovered/`

`experiments/recovered/` holds the scripts of past experiments as the historical
record.  `creature/state/` holds the living record — the actual belief and value
state of each creature that has been run.  `BIOGRAPHY.jsonl` is the append-only
narrative, just as `EXPERIMENTS.md` is the append-only scientific log.
