# Bundle: exp31

**Direction:** embodiment-valence-recipe
**Status:** completed
**Verdict:** NEGATIVE
**Confidence:** medium
**Backfill level:** repro_bundle

## Question

learn A AND B from scratch: COLLAPSES

## Evidence included

- Source scripts: 1
- Metrics refs: 0
- Raw data refs: 0
- Scorer refs: 1
- State refs: 0

## Evidence NOT included at this level

- trajectory_bundle (not exported at this level)
- checkpoint_bundle (not exported at this level)
- mechanism_bundle (not exported at this level)

## Reproduction

Raw data is referenced in place, not copied.
Re-runs are new reproduction runs, not replays of the original data.

    uv run --python .venv python experiments/recovered/exp31_learn_a_and_b_fails.py

Repo commit: 023463bcc15778283c2d5df68ec93f93dae5d998


## Anchors

Load-bearing evidence for: recipe-symmetry-breaking-v0 / disembodied-stream-collapse-v0 (the anchor law: A+B from noise collapses)
