"""Print a non-binding process checklist from passive Meta Monkey memory."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

from meta_monkey.report import load_episodes


def _read_optional(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _recent_repeated_risks(root: pathlib.Path, *, limit: int = 8) -> list[str]:
    episodes = sorted(load_episodes(root), key=lambda episode: episode.exp, reverse=True)[:limit]
    counts: Counter[str] = Counter(
        risk for episode in episodes for risk in episode.process.likely_risks
    )
    return [
        f"- {risk}: {count}"
        for risk, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ]


def build_checklist(root: pathlib.Path) -> str:
    lessons_text = _read_optional(root / "loop" / "LESSONS.md")
    repeated_risks = _recent_repeated_risks(root)
    if not repeated_risks:
        repeated_risks = ["- None in recent meta episodes."]

    lesson_sample = "none"
    for line in lessons_text.splitlines():
        stripped = line.strip()
        if stripped:
            lesson_sample = stripped
            break

    lines = [
        "Meta Monkey preflight",
        "Advisory only: this is passive meta memory, not a controller.",
        "",
        "Standing checks:",
        "- Follow loop/PROTOCOL.md.",
        "- Follow loop/ROUTING.md.",
        "- Run loop/check_iteration.py before commit.",
        "- Use blinded verifier when required.",
        "- Do not treat printed script verdict as the scientific result.",
        "- Quote final committed output after reruns.",
        "- Prefer mechanical guards over prose reminders for recurring process failures.",
        "",
        "Recent repeated risks:",
    ]
    lines.extend(repeated_risks)
    lines.extend(
        [
            "",
            "Boundaries:",
            "- This checklist does not choose the next experiment.",
            "- This checklist does not grade science.",
            "- This checklist does not mutate files.",
            "",
            "References:",
            "- loop/LESSONS.md",
            "- loop/PROTOCOL.md",
            "- loop/ROUTING.md",
            "",
            "Lesson sample:",
            lesson_sample,
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Print passive Meta Monkey preflight guidance.")


def main(argv: list[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    try:
        print(build_checklist(pathlib.Path.cwd()), end="")
        return 0
    except Exception as exc:
        print(f"preflight error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
