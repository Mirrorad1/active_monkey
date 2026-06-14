"""Tests for tools/ref.py — the @-reference resolver.

tools/ is not an importable package, so load ref.py by file path. Tests use the
real filesystem (stable, closed cards/experiments) like test_directions_index.py.
Run:  uv run --python .venv pytest tests/test_ref.py -q
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("ref", ROOT / "tools" / "ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_direction():
    r = _load()
    assert r.resolve("@transfer") == [pathlib.Path("loop/directions/transfer.md")]


def test_alias_direction():
    r = _load()
    assert r.resolve("ecology") == [pathlib.Path("loop/directions/population-ecology.md")]


def test_research_chapter_unique_substring():
    r = _load()
    assert r.resolve("@n4-identity") == [
        pathlib.Path("docs/research/n4-identity-commitment-chapter.md")
    ]


def test_experiment_glob():
    r = _load()
    got = r.resolve("@exp201")
    assert got, "no exp201 scripts found"
    assert all(
        str(p).startswith("experiments/exp201_") and str(p).endswith(".py")
        for p in got
    )


def test_unknown_exits():
    r = _load()
    with pytest.raises(SystemExit):
        r.resolve("@zzznope")


def test_ambiguous_exits():
    r = _load()
    with pytest.raises(SystemExit):
        r.resolve("@problem2")  # two docs/research/problem2-*.md


def test_list_index_mentions_kinds():
    r = _load()
    out = r.list_index()
    assert "@transfer" in out
    assert "experiment" in out
