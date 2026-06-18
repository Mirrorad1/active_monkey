#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/exp205_n5_survivable_loss.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/exp205_n5_survivable_loss.py
