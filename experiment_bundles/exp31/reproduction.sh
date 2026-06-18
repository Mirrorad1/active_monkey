#!/usr/bin/env bash
# Reproduction script — re-run is a NEW reproduction, not a replay.
# Original script: experiments/recovered/exp31_learn_a_and_b_fails.py
cd "$(git rev-parse --show-toplevel)"
uv run --python .venv python experiments/recovered/exp31_learn_a_and_b_fails.py
