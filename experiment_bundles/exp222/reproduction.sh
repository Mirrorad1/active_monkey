#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/exp222_m4a_converse_milestone.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/exp222_m4a_converse_milestone.py
