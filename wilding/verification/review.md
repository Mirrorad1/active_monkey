# Engineering verification, 2026-10-28

This is verification of a game implementation, not an experiment entry or a
scientific novelty claim.

- C# project builds with zero warnings/errors (`build.txt`).
- Thirteen core behavioral checks pass (`core-checks.txt`).
- Godot headless and graphical smoke modes exit 0 and print `WILDING_SMOKE_OK`.
  They exercise the clock, terrain picking, zoom, save/load, independent snapshot
  continuation, newborn chart selection and current food-overlay sampling.
- `smoke.png` is an actual native engine capture; it is the previously rendered
  frame, so UI/clock values may precede the final smoke assertions by one frame.
- Reviewer read-only inspection identified stale food colors, spatial-bucket
  boundary misses, and insufficient save validation. All three are fixed.
  Spatial and invalid-save checks were observed failing before the fixes, then
  passing. Repeated forest painting also has a failing-then-passing regression.
- The Play clock regression was observed failing before its accumulator clamp
  was fixed. Its smoke assertion remains active.
- Food availability now uses a sampled 64×64 texture updated every 0.25 seconds;
  updates do not rebuild terrain geometry.
- Both tool archives matched the official published hashes before extraction.

## Performance scope

The retained Release benchmark times one step from exactly 1,000 or 3,000
creatures on an Apple M3 Pro. Each of 200 trials restores the same complete
snapshot before timing; restoration is excluded. Post-step population is also
reported. The result is not a guarantee of rendered frame rate or sustained
ecological coexistence. Initial development long runs showed population loss
and, at higher initial density, extinction; this is why the final benchmark
does not average through a collapsing population.

## Existing repository baseline

The full existing `uv run --python .venv pytest -q` suite completed with three
failures outside `wilding/`:

- `tests/test_embodied.py::test_mujoco_offscreen_render_nonblack`
- `tests/test_embodied.py::test_render_writes_two_videos`
- `tests/test_loop_routing.py::test_resume_mentions_recent_logged_experiment`

The first two failed with `invalid CoreGraphics connection` inside the sandbox;
both passed when rerun with desktop access (`existing-render-tests.txt`). The
remaining check finds the highest space-separated `Exp N` reference in RESUME
to be 280 while the journal reaches 283. No research log, RESUME, or existing
Python source was edited by this implementation. The repository baseline is
therefore not reported as fully green.

## Boundaries

Only Apple Silicon macOS was exercised. The engine model is simplified and its
policies, costs, mutation rule and founder lineages are provided. There is no
Python parity claim, autonomous learning claim or guaranteed adaptive outcome.
Forks can be saved and resumed; side-by-side comparative replay is future work.
