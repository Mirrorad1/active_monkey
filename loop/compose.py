#!/usr/bin/env python3
"""Compose a ready-to-paste /loop prompt for the self-guided research loop (loop B).

Concatenates: header + premise + direction card + persona card + protocol/validation
pointers + optional one-off human idea. No magic — the cards ARE the modules.

Usage:
  python loop/compose.py                                  # defaults
  python loop/compose.py --direction transfer --persona skeptic
  python loop/compose.py --idea "what if the anchor can be weaker?"
  python loop/compose.py --list
"""

import argparse
import sys
from pathlib import Path

LOOP_DIR = Path(__file__).resolve().parent


def available(kind: str) -> list[str]:
    return sorted(
        p.stem for p in (LOOP_DIR / kind).glob("*.md") if not p.stem.startswith("_")
    )


def load_card(kind: str, name: str) -> str:
    path = LOOP_DIR / kind / f"{name}.md"
    if not path.exists():
        options = ", ".join(available(kind))
        sys.exit(f"error: no {kind[:-1]} card '{name}' (have: {options})")
    return path.read_text().strip()


def compose(direction: str, persona: str, idea: str | None) -> str:
    premise = (LOOP_DIR / "PREMISE.md").read_text().strip()
    parts = [
        "/loop Keep running the moonshot active-inference experiments until I stop you.",
        "BOOTSTRAP: read RESUME.md and EXPERIMENTS.md before anything else; this prompt's"
        " modules live in loop/ and are the source of truth for this run.",
        "",
        "=== PREMISE (world model) ===",
        premise,
        "",
        "=== DIRECTION (what to work on) ===",
        load_card("directions", direction),
        "",
        "=== PERSONA (how to work) ===",
        load_card("personas", persona),
    ]
    if idea:
        parts += [
            "",
            "=== HUMAN IDEA (outranks the direction's queue this run) ===",
            idea.strip(),
        ]
    parts += [
        "",
        "=== DISCIPLINE ===",
        "Follow loop/PROTOCOL.md every iteration (inbox -> choose -> predeclare ->",
        "build -> validate -> BLINDED-VERIFY -> log -> commit -> reflect).",
        "Consult loop/LESSONS.md (the distilled rules card) at the start of every",
        "iteration. loop/VALIDATION.md is BINDING:",
        "predeclared falsifiers, negative results logged as negatives, provided-vs-",
        "self-formed named, consolidation tagged honestly, no seed-shopping, mandatory",
        "honest-caveat line. Verdicts are checked by a blinded verifier subagent",
        "(PROTOCOL step 4.5) and the mechanical rubric (loop/check_iteration.py) runs",
        "before every experiment commit. loop/METHODOLOGY.md is the ADVISORY",
        "heuristics layer (design-time questions before predeclaring; evaluation +",
        "generalizability checks before logging/grading; VALIDATION stays binding",
        "where they differ). Check loop/IDEAS.md at every wake; human entries outrank",
        "your queue. One experiment per iteration on a short ~5-minute cadence; if",
        "mid-task, continue across wakes. Keep responses lightweight. When insight",
        "flattens for ~3 iterations, say so and gently suggest a direction switch or a",
        "natural stop.",
        "Follow loop/ROUTING.md for model budget: the highest-thinking Claude loop owns",
        "research design/verdicts; Codex high-fast handles bounded coding and blinded verifier",
        "work (explicit fallback: gpt-5.4-mini with high reasoning); Codex explorer handles",
        "low-risk read-only or clerking tasks.",
        "loop/META.md is also BINDING: when you find a noteworthy non-research issue or a"
        " reusable insight, institutionalize a durable guard (test / loop-module rule / skill"
        " via /claudeception) — don't just patch the instance.",
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--direction",
        default="continuous-substrate",  # current human steer (see loop/IDEAS.md 2026-06-10)
        help="direction card name",
    )
    ap.add_argument("--persona", default="default", help="persona card name")
    ap.add_argument("--idea", default=None, help="one-off human idea to inject")
    ap.add_argument("--list", action="store_true", help="list available modules")
    args = ap.parse_args()

    if args.list:
        print("directions:", ", ".join(available("directions")))
        print("personas:  ", ", ".join(available("personas")))
        return

    print(compose(args.direction, args.persona, args.idea))


if __name__ == "__main__":
    main()
