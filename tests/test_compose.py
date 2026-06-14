"""Tests for loop/compose.py — the direction resolver and CLI wiring.

loop/ is not an importable package, so load compose.py by file path.
Run:  uv run --python .venv pytest tests/test_compose.py -q
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _load_compose():
    spec = importlib.util.spec_from_file_location(
        "loop_compose", ROOT / "loop" / "compose.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CANDIDATES = [
    "population-ecology",
    "continuous-substrate",
    "sequence-substrate",
    "identity-n4",
    "identity-n4-crack",
    "graded-uncertainty",
    "transfer",
]


def test_exact_stem_resolves_to_itself():
    c = _load_compose()
    assert c.resolve_direction("transfer", candidates=CANDIDATES) == "transfer"


def test_alias_resolves():
    c = _load_compose()
    assert c.resolve_direction("ecology", candidates=CANDIDATES) == "population-ecology"


def test_unique_substring_resolves():
    c = _load_compose()
    assert c.resolve_direction("graded", candidates=CANDIDATES) == "graded-uncertainty"


def test_ambiguous_substring_exits():
    c = _load_compose()
    with pytest.raises(SystemExit):
        c.resolve_direction("identity", candidates=CANDIDATES)


def test_unknown_exits():
    c = _load_compose()
    with pytest.raises(SystemExit):
        c.resolve_direction("zzznope", candidates=CANDIDATES)


def test_cli_positional_direction_composes():
    # stdlib-only module → plain interpreter, no venv needed
    r = subprocess.run(
        [sys.executable, "loop/compose.py", "transfer"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "=== DIRECTION (what to work on) ===" in r.stdout


def test_cli_unknown_direction_errors():
    r = subprocess.run(
        [sys.executable, "loop/compose.py", "zzznope"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "no direction" in (r.stderr + r.stdout).lower()
