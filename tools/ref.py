#!/usr/bin/env python3
"""@-reference resolver for active-loop — turn a short key into a canonical path.

Resolves references used in prompts, cards, and IDEAS to repo paths so you can
write @n4-crack-chapter or @exp201 instead of a full path. Computed on demand
from the filesystem (no committed index → never stale).

Usage:
  python tools/ref.py @exp201            # -> experiments/exp201_<slug>.py
  python tools/ref.py @n4-identity       # -> docs/research/n4-identity-commitment-chapter.md
  python tools/ref.py ecology            # -> loop/directions/population-ecology.md (alias)
  python tools/ref.py @direction:foo    # qualify a stem shared across kinds
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
        return matches
    sys.exit(f"error: no experiment script experiments/exp{n}_*.py")


def resolve(ref: str, index: dict | None = None) -> list[pathlib.Path]:
    """Resolve @ref (or ref) to repo-relative Paths. sys.exit on miss/ambiguity.

    Forms: @exp<N> (all matching scripts); a bare key matched by exact stem /
    alias / unique substring; or a kind-qualified `kind:stem` (e.g.
    `direction:transfer`) to disambiguate a stem shared across kinds. Never
    silently picks among competing matches.
    """
    ref = ref.lstrip("@").strip()
    exp = _experiment(ref)
    if exp is not None:
        return [p.relative_to(ROOT) for p in exp]
    if index is None:
        index = build_index()
    # kind-qualified form: "direction:transfer"
    if ":" in ref:
        kind, _, stem = ref.partition(":")
        d = index.get(kind, {})
        if stem in d:
            return [d[stem].relative_to(ROOT)]
        sys.exit(f"error: no {kind} '{stem}' (kinds: {', '.join(index)})")
    by_stem: dict[str, list[tuple[str, pathlib.Path]]] = {}
    for kind, d in index.items():
        for stem, path in d.items():
            by_stem.setdefault(stem, []).append((kind, path))
    if ref in ALIASES and ALIASES[ref] in by_stem:
        ref = ALIASES[ref]
    if ref in by_stem:
        stems = [ref]                       # exact stem
    else:
        stems = sorted(s for s in by_stem if ref.lower() in s.lower())  # substring
    hits = [(kind, stem, path) for stem in stems for (kind, path) in by_stem[stem]]
    if len(hits) == 1:
        return [hits[0][2].relative_to(ROOT)]
    if not hits:
        sys.exit(f"error: no match for @{ref}")
    sys.exit(
        f"error: @{ref} is ambiguous — matches "
        + ", ".join(f"{k}:{s}" for k, s, _ in hits)
        + " (qualify with kind:stem)"
    )


def list_index(kind: str | None = None, index: dict | None = None) -> str:
    if kind and kind not in set(KINDS) | {"experiment"}:
        sys.exit(
            f"error: unknown kind '{kind}' "
            f"(have: {', '.join(list(KINDS) + ['experiment'])})"
        )
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
