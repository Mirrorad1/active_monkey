#!/usr/bin/env python3
"""@-reference resolver for active-loop — turn a short key into a canonical path.

Resolves references used in prompts, cards, and IDEAS to repo paths so you can
write @n4-crack-chapter or @exp201 instead of a full path. Computed on demand
from the filesystem (no committed index → never stale).

Usage:
  python tools/ref.py @exp201            # -> experiments/exp201_<slug>.py
  python tools/ref.py @n4-identity       # -> docs/research/n4-identity-commitment-chapter.md
  python tools/ref.py ecology            # -> loop/directions/population-ecology.md (alias)
  python tools/ref.py --list             # print the whole index
  python tools/ref.py --list research    # print one kind
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# kinds in priority order (earlier wins on an exact-stem collision)
KINDS = {
    "direction": ROOT / "loop" / "directions",
    "persona": ROOT / "loop" / "personas",
    "research": ROOT / "docs" / "research",
    "spec": ROOT / "docs" / "specs",
}
ALIASES = {"ecology": "population-ecology", "pop": "population-ecology"}


def build_index(kinds: dict[str, pathlib.Path] = KINDS) -> dict[str, dict[str, pathlib.Path]]:
    idx: dict[str, dict[str, pathlib.Path]] = {}
    for kind, d in kinds.items():
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            if p.stem.startswith("_"):
                continue
            idx.setdefault(kind, {})[p.stem] = p
    return idx


def _experiment(ref: str) -> list[pathlib.Path] | None:
    m = re.fullmatch(r"exp0*(\d+)", ref, re.IGNORECASE)
    if not m:
        return None
    n = int(m.group(1))
    matches = sorted((ROOT / "experiments").glob(f"exp{n}_*.py"))
    if matches:
        return matches[:1]
    sys.exit(f"error: no experiment script experiments/exp{n}_*.py")


def resolve(ref: str, index: dict | None = None) -> list[pathlib.Path]:
    """Resolve @ref (or ref) to repo-relative Paths. sys.exit on miss/ambiguity."""
    ref = ref.lstrip("@").strip()
    exp = _experiment(ref)
    if exp is not None:
        return [p.relative_to(ROOT) for p in exp]
    if index is None:
        index = build_index()
    flat: dict[str, pathlib.Path] = {}  # stem -> path; first kind wins
    for _kind, d in index.items():
        for stem, path in d.items():
            flat.setdefault(stem, path)
    if ref in ALIASES and ALIASES[ref] in flat:
        ref = ALIASES[ref]
    if ref in flat:
        return [flat[ref].relative_to(ROOT)]
    subs = sorted((s, p) for s, p in flat.items() if ref.lower() in s.lower())
    if len(subs) == 1:
        return [subs[0][1].relative_to(ROOT)]
    if not subs:
        sys.exit(f"error: no match for @{ref}")
    sys.exit(f"error: @{ref} is ambiguous — matches {', '.join(s for s, _ in subs)}")


def list_index(kind: str | None = None, index: dict | None = None) -> str:
    if index is None:
        index = build_index()
    lines: list[str] = []
    for k, d in index.items():
        if kind and k != kind:
            continue
        lines.append(f"== {k} ==")
        for stem in sorted(d):
            lines.append(f"  @{stem}  ->  {d[stem].relative_to(ROOT)}")
    if not kind or kind == "experiment":
        n = len(list((ROOT / "experiments").glob("exp*_*.py")))
        lines.append(f"== experiment ==\n  @exp<N>  ->  experiments/exp<N>_*.py  ({n} scripts)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", nargs="?", help="@key to resolve (e.g. @exp201, @n4-crack-chapter)")
    ap.add_argument(
        "--list", nargs="?", const="", default=None, metavar="KIND",
        help="list the index (optionally one kind: direction/persona/research/spec)",
    )
    args = ap.parse_args()
    if args.list is not None:
        print(list_index(args.list or None))
        return
    if not args.ref:
        sys.exit("usage: ref.py @key   |   ref.py --list [kind]")
    for p in resolve(args.ref):
        print(p)


if __name__ == "__main__":
    main()
