#!/usr/bin/env python3
"""Append a human-marked bullet to loop/IDEAS.md under '## Inbox'.

Append-only: inserts a new bullet inside the Inbox section and never modifies or
deletes existing lines. Used by the /steer slash command so the human does not
hand-edit the inbox. A human running this IS a human action — it preserves the
VALIDATION.md §5 consent boundary (cron/automation never calls it).

Usage:
  python tools/steer_append.py "free-text idea or redirection"
  python tools/steer_append.py --resume a
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IDEAS_PATH = ROOT / "loop" / "IDEAS.md"


def _insert_into_inbox(content: str, bullet: str) -> str:
    lines = content.splitlines(keepends=True)
    try:
        inbox_i = next(i for i, l in enumerate(lines) if l.strip() == "## Inbox")
    except StopIteration:
        raise ValueError("no '## Inbox' section in IDEAS.md")
    end_i = len(lines)
    for j in range(inbox_i + 1, len(lines)):
        if lines[j].startswith("## "):
            end_i = j
            break
    if not bullet.endswith("\n"):
        bullet += "\n"
    # Ensure the preceding line terminates so the bullet starts on its own line
    # (handles '## Inbox' as the final line with no trailing newline).
    if end_i > 0 and not lines[end_i - 1].endswith("\n"):
        lines[end_i - 1] = lines[end_i - 1] + "\n"
    return "".join(lines[:end_i] + [bullet] + lines[end_i:])


def append_idea(text: str, *, date: str, ideas_path: pathlib.Path = IDEAS_PATH) -> str:
    bullet = f"- [from human, {date}] {text.strip()}"
    ideas_path.write_text(
        _insert_into_inbox(ideas_path.read_text(encoding="utf-8"), bullet),
        encoding="utf-8",
    )
    return bullet


def append_resume(word: str, *, date: str, ideas_path: pathlib.Path = IDEAS_PATH) -> str:
    return append_idea(f"consult reply: {word.strip()}", date=date, ideas_path=ideas_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="?", default=None, help="idea / redirection text")
    ap.add_argument("--resume", default=None, help="answer an open consult with WORD")
    args = ap.parse_args()
    date = datetime.date.today().isoformat()

    if args.resume is not None:
        print("appended:", append_resume(args.resume, date=date))
    elif args.text:
        print("appended:", append_idea(args.text, date=date))
    else:
        sys.exit('error: provide idea text or --resume WORD')


if __name__ == "__main__":
    main()
