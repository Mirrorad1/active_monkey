#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/exp154_growth_confirmation.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/exp154_growth_confirmation.py
