"""Tests for tools/steer_append.py — append-only human-marked inbox writes.

Run:  uv run --python .venv pytest tests/test_steer_append.py -q
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "steer_append", ROOT / "tools" / "steer_append.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IDEAS = (
    "# IDEAS — human inbox\n\nintro line\n\n## Inbox\n\n"
    "- [from human, 2026-06-01] an older idea\n\n"
    "## Consumed\n\n- something consumed\n"
)


def _write(tmp_path) -> pathlib.Path:
    p = tmp_path / "IDEAS.md"
    p.write_text(IDEAS, encoding="utf-8")
    return p


def test_append_idea_preserves_original_bytes(tmp_path):
    m = _load()
    p = _write(tmp_path)
    m.append_idea("try a foraging gradient", date="2026-06-13", ideas_path=p)
    out = p.read_text(encoding="utf-8")
    # every original line still present, in order (append-only)
    for line in IDEAS.splitlines():
        assert line in out
    assert "- [from human, 2026-06-13] try a foraging gradient" in out


def test_append_idea_lands_inside_inbox_not_consumed(tmp_path):
    m = _load()
    p = _write(tmp_path)
    m.append_idea("new steer", date="2026-06-13", ideas_path=p)
    out = p.read_text(encoding="utf-8")
    inbox_i = out.index("## Inbox")
    consumed_i = out.index("## Consumed")
    new_i = out.index("new steer")
    assert inbox_i < new_i < consumed_i


def test_append_resume_marks_human_reply(tmp_path):
    m = _load()
    p = _write(tmp_path)
    m.append_resume("a", date="2026-06-13", ideas_path=p)
    out = p.read_text(encoding="utf-8")
    assert "[from human, 2026-06-13]" in out
    assert "consult reply: a" in out


def test_missing_inbox_section_raises(tmp_path):
    m = _load()
    p = tmp_path / "IDEAS.md"
    p.write_text("# IDEAS\n\nno inbox header here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        m.append_idea("x", date="2026-06-13", ideas_path=p)
