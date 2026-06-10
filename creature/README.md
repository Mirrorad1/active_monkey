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

## The spine: one continuous life, checkpointed before & after (Exp 58+)

`mirro` is the **spine** — the single continuous creature that every persistent
experiment from Exp 58 on advances. An experiment never re-births or rewinds the spine;
it loads the committed snapshot, lives, and saves back, so the world-model accumulates as
one never-reset life. Use the helper so this is uniform and guaranteed:

```python
from active_loop.checkpoint import mirro_episode

with mirro_episode("Exp 58") as ep:
    ep.creature.live(500)            # advance the one continuous life
    twin = ep.fork_control("exp58-twinA")   # SIDE-control only — never the spine
    twin.live(500)
# on exit: spine saved (AFTER checkpoint); ep.report() prints before/after age + hash
```

`mirro_episode` records a checkpoint BEFORE (age + `state_hash`, a `checkpoint_before`
biography event) and, on clean exit, an AFTER checkpoint. If the body raises, the spine
is left untouched — a failed run can't corrupt the life. Paste `ep.report()` into the
EXPERIMENTS.md entry. **Forks remain the scientific control, but only as side-controls**
branched via `ep.fork_control(...)` and saved under a non-spine path. The continuity guard
`tests/test_creature_continuity.py` fails CI if an established spine's age ever resets.

Snapshot workflow for an experiment:
1. `with mirro_episode("Exp NN") as ep:` — loads the spine + checkpoints BEFORE.
2. Run the experiment on `ep.creature` (`live()` calls, possibly `teach_word()`);
   branch any controls with `ep.fork_control(...)`.
3. On block exit the spine is saved (checkpoint AFTER); `git commit` the snapshot in the
   experiment's atomic commit. The diff shows exactly what changed.

## The clade: mirro is the root ancestor, branches can become peer spines

The persistent creature is a **family tree**, not a single sacred object:

- **mirro is the trunk / root ancestor** — the one continuous line that keeps accumulating.
- **A "different history" is a branch.** Fork mirro at a checkpoint, raise it in a different
  world, and it is a named line of its own. A branch lived long enough is *functionally a
  different species* (same architecture = genome; different lived history = development).
- **Promote a branch to a committed peer spine** by saving it under its own directory:
  `twin.save(Path("creature/state") / twin.name)`. It then accumulates as a peer trunk and
  obeys the same continuity discipline.
- **The common ancestor is frozen and revisitable.** Each committed snapshot is permanent in
  git; `git checkout <commit> -- creature/state/mirro/` re-derives the exact ancestor brain to
  re-fork from. Because every fork stamps `lineage: ["mirro@AGE#HASH"]`, two descendants can
  *prove* their shared ancestor by hash — divergence between them is causally clean.
- **Recovery, not in-place wipe.** If an epoch corrupts a line, restore the prior committed
  snapshot (`git checkout`). The continuity guard (`tests/test_creature_continuity.py`) forbids
  a *silent* reset of an established line; a deliberate restart is allowed only when logged as
  an explicit `rebirth` biography event — the sanctioned escape hatch, so you are never locked
  in.
- A from-scratch, no-inheritance **newborn** is a *separate root* (not a mirro descendant),
  legitimate only when an experiment needs a zero-history baseline (cf. Exp 57).

The long arc (see `loop/directions/social-emergence.md`): freeze a common ancestor → let
descendants speciate → reunite them through an M4-style channel → look for social emergence.

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
