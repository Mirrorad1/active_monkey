#!/usr/bin/env python3
"""Build GitHub-Pages deploy entrypoints from site/pages sources.

GitHub Pages serves the repository root for this project, so the root HTML
entrypoints are committed deploy artifacts. The edited source pages live under
site/pages; this script copies them to the root and gives tests a byte-level
staleness guard.
"""

from __future__ import annotations

import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGES = (
    "index.html",
    "journey.html",
    "open_problem.html",
    "sense-evolution.html",
    "math.html",
)


def deploy_outputs(root: pathlib.Path | str = ROOT) -> dict[str, str]:
    """Return {root_relative_output: text} for generated deploy pages."""
    repo = pathlib.Path(root)
    source_dir = repo / "site" / "pages"
    return {
        page: (source_dir / page).read_text(encoding="utf-8")
        for page in PAGES
    }


def build(root: pathlib.Path | str = ROOT) -> list[pathlib.Path]:
    """Write root deploy pages from site/pages and return written paths."""
    repo = pathlib.Path(root)
    written: list[pathlib.Path] = []
    for rel, text in deploy_outputs(repo).items():
        out = repo / rel
        out.write_text(text, encoding="utf-8")
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if root deploy pages differ from site/pages sources",
    )
    args = parser.parse_args(argv)

    outputs = deploy_outputs(ROOT)
    stale = [
        rel
        for rel, expected in outputs.items()
        if not (ROOT / rel).exists()
        or (ROOT / rel).read_text(encoding="utf-8") != expected
    ]
    if args.check:
        if stale:
            for rel in stale:
                print(f"stale: {rel}")
            return 1
        return 0

    build(ROOT)
    for rel in outputs:
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
