"""Guard: loop/managed-paths.txt holds exactly the expected managed set.

Prevents a silent drop (e.g. EXPERIMENTS.md falling out of the autosync sweep).
Run:  uv run --python .venv pytest tests/test_managed_paths.py -q
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).parent.parent


def _configured() -> set[str]:
    text = (ROOT / "loop" / "managed-paths.txt").read_text(encoding="utf-8")
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def test_managed_set_matches_expected():
    expected = {
        "EXPERIMENTS.md", "experiments", "site/data/experiments-data.js", "site/data/lab-status.js",
        "DIRECTIONS.md", "loop/IDEAS.md", "loop/directions", "loop/managed-paths.txt",
        "creature/state", "world_model", "reports", "REPORT.md",
    }
    assert _configured() == expected
