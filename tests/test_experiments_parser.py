"""Tests for active_loop/experiments_parser.py — the canonical EXPERIMENTS.md parser.

These are RELATIONAL invariants: the parser must agree with the regexes the existing
consumers use (site_data.py, check_iteration.py, the site tests) on the SAME text.
They read the live EXPERIMENTS.md but assert internal agreement, so they are robust
to the loop appending experiments (both sides see the new entry). A format change that
splits the consumers fails here LOUDLY instead of silently breaking one of them.

Run:  uv run --python .venv pytest tests/test_experiments_parser.py -q
"""
from __future__ import annotations

import pathlib
import re

import pytest

from active_loop.experiments_parser import Experiment, by_number, parse

ROOT = pathlib.Path(__file__).parent.parent
TEXT = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")


def test_numbers_match_site_data_canonical_regex():
    canonical = sorted(int(x) for x in re.findall(r"^## Exp (\d+) ", TEXT, re.MULTILINE))
    assert sorted(e.n for e in parse(TEXT)) == canonical


def test_numbers_match_check_iteration_regex():
    ci = sorted(int(x) for x in re.findall(r"^## Exp (\d+)\b", TEXT, re.MULTILINE))
    assert sorted(e.n for e in parse(TEXT)) == ci


def test_bodies_partition_the_log():
    exps = parse(TEXT)
    assert exps, "no experiments parsed"
    for e in exps:
        assert e.body.startswith(e.header)
    for a, b in zip(exps, exps[1:]):
        assert TEXT[a.start:b.start] == a.body


def test_every_experiment_has_a_title():
    assert all(e.title for e in parse(TEXT))


def test_known_title_exp1():
    by = by_number(TEXT)
    assert by[1].title.startswith("does the character HMM learn")


def test_by_number_raises_on_duplicate():
    dup = "## Exp 5 — a\nbody a\n## Exp 5 — b\nbody b\n"
    with pytest.raises(ValueError):
        by_number(dup)


def test_load_reads_repo_experiments_md():
    from active_loop.experiments_parser import load
    assert len(load()) == len(parse(TEXT))
