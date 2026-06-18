# Bundle: exp211

**Direction:** uncertainty-gated-active-sensing
**Status:** completed
**Verdict:** NEGATIVE
**Confidence:** medium
**Backfill level:** repro_bundle

## Question

Phase 4 / Rung 4: does UNCERTAINTY-GATED active sensing (probe only when unsure) escape Exp 210's wall?

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

    uv run --python .venv python experiments/exp211_uncertainty_gated_active_sensing.py

Repo commit: 023463bcc15778283c2d5df68ec93f93dae5d998


## Anchors

Load-bearing evidence for: active-sensing-benefit-wall-v0 (uncertainty-gated probing; benefit ceiling ~0)
