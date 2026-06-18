#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/exp209_belief_persistence_preflight.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/exp209_belief_persistence_preflight.py
