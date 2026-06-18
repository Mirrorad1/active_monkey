# Bundle: exp220

**Direction:** affective-dyad
**Status:** completed
**Verdict:** POSITIVE
**Confidence:** high
**Backfill level:** repro_bundle

## Question

M4a precision schedule: gradually annealing decisiveness (γ 1→8) gives the FIRST reliable GENUINE discrimination at realistic capacity K=4 (sched_full 13/16) — it fixes the learn-but-don't-exploit decoupling and beats fixed precision; but at the LONG 300-turn session only

## Evidence included

- Source scripts: 1
- Metrics refs: 4
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

    uv run --python .venv python experiments/exp220_m4a_schedule.py

Repo commit: 023463bcc15778283c2d5df68ec93f93dae5d998


## Anchors

Load-bearing evidence for: functional-valence-dyad-v0 (precision schedule: first reliable genuine discrimination)
