# Bundle: exp209

**Direction:** hidden-state-memory
**Status:** completed
**Verdict:** NEGATIVE
**Confidence:** medium
**Backfill level:** repro_bundle

## Question

Phase 3 rung 1b: the CONTINUOUS belief_persistence trait (ρ 0.5→0.55, a true small-ε step) — is the wall a granularity artifact?

## Evidence included

- Source scripts: 1
- Metrics refs: 0
- Raw data refs: 0
- Scorer refs: 0
- State refs: 0

## Evidence NOT included at this level

- trajectory_bundle (not exported at this level)
- checkpoint_bundle (not exported at this level)
- mechanism_bundle (not exported at this level)

## Reproduction

Raw data is referenced in place, not copied.
Re-runs are new reproduction runs, not replays of the original data.

    uv run --python .venv python experiments/exp209_belief_persistence_preflight.py

Repo commit: 023463bcc15778283c2d5df68ec93f93dae5d998


## Anchors

Load-bearing evidence for: hidden-state-memory-boundary-v0 (continuous belief-persistence; wall not a granularity artifact)
