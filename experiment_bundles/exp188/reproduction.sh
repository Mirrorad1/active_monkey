#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/exp188_n4_regulated_controller.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/exp188_n4_regulated_controller.py
