#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/exp168_n3_rung4_mixed.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/exp168_n3_rung4_mixed.py
