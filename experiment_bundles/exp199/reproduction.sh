#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/exp199_n5_valley_sweep.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/exp199_n5_valley_sweep.py
