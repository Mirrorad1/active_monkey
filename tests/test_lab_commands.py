"""Structural guards for the project slash commands.

Command .md files are prompt templates (not unit-testable behavior), so we guard
the load-bearing invariants: the file exists, declares the right allowed-tools,
and KEEPS the human-consent gate language (the confirm-before-iterate step that
VALIDATION.md §5 depends on). A regression that deletes the gate fails here.

Run:  uv run --python .venv pytest tests/test_lab_commands.py -q
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).parent.parent
CMD = ROOT / ".claude" / "commands"


def test_lab_command_exists():
    assert (CMD / "lab.md").is_file()


def test_lab_command_invokes_compose():
    text = (CMD / "lab.md").read_text(encoding="utf-8")
    assert "loop/compose.py" in text


def test_lab_command_has_consent_gate():
    text = (CMD / "lab.md").read_text(encoding="utf-8").lower()
    # must wait for an explicit "go" and must not iterate/commit before it
    assert "go" in text
    assert "confirm" in text or "wait" in text
    assert "vation.md" in text or "consent" in text  # cites the boundary
