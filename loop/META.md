# META — the meta-improvement loop (binding)

PROTOCOL.md governs the EXPERIMENTS (the subject being studied). META.md governs the
LOOP ITSELF (the scaffolding). "Self-healing" means the loop gets harder to break each
time it breaks: when you find something noteworthy to fix, you fix the *class*, not just
the instance.

## When this fires
Any time — mid-iteration or any session — you notice a noteworthy NON-RESEARCH issue or
a reusable insight. Examples (all real, from this project's own history):
- a claim in the record that can't be reproduced or traced to a committed artifact
- a doc, pointer, or count that is stale or contradicts the code
- a harness / git / site / tooling bug (an unguarded hook, fabricated data, snapshot drift)
- an honesty violation in the log (over-grading, provided-as-self-formed, unlogged failure)
- a mistake you or a past session made more than once
- a non-obvious technique worth keeping (a recovery method, a debugging trick)

"Noteworthy" = it would bite again if left uninstitutionalized. Trivial one-offs don't qualify.

## The five moves: notice -> institutionalize
1. **Notice & name.** One line: the problem, and *why it recurs* if unfixed.
2. **Log.** Open a GitHub issue — honest title, root cause, prevention. Single-purpose.
3. **Fix.** Reviewed PR. Never push main. Same dispatch/review discipline as everything here.
4. **Institutionalize (the point — REQUIRED).** Add the durable guard so it cannot recur.
   Pick the strongest available, preferring a MECHANICAL guard over a prose reminder:
   - mechanical invariant (data/site/count drift, a checkable property) -> add a FAST TEST
     that fails on recurrence (e.g. tests/test_site_data.py)
   - process rule -> amend the relevant `loop/` module (PROTOCOL / VALIDATION / this file)
     AND add/update the one-line distilled rule in `loop/LESSONS.md` in the same commit
     (the module holds the binding text; LESSONS is the consult-at-start digest)
   - reusable technique (non-obvious debugging / recovery / workaround) -> extract a skill
     via the `claudeception` skill (`/claudeception`)
   - environment / config gotcha -> record in project memory AND a CLAUDE.md standing rule
5. **Record.** One line in the PR/commit naming the guard: `prevented by: <test | rule | skill>`.

## Honesty (binding)
A meta-fix that adds no guard is INCOMPLETE — you fixed the instance, not the class. If no
durable guard is possible, say so explicitly and why. Never silently "fix and move on": an
unlogged "I'll prevent it later" is how the record rots. Run META inline when an issue
surfaces; do not defer it to a tidy-up pass that never comes.

## Relation to the rest of the loop
- The research loop's Reflect step (PROTOCOL.md) routes here when a non-research issue surfaces.
- META shares every standing discipline: honest, reviewed, single-purpose, everything in git.
- The global `claudeception` skill is the extraction engine for move 4's "reusable technique"
  branch; META is what decides *when* a lesson is worth extracting for THIS project.
