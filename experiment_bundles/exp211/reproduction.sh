#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/exp211_uncertainty_gated_active_sensing.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/exp211_uncertainty_gated_active_sensing.py
