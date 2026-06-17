#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/exp210_active_sensing_preflight.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/exp210_active_sensing_preflight.py
