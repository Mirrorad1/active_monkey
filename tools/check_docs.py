#!/usr/bin/env python3
"""Lightweight consistency checks for public-facing Markdown docs.

The checks are intentionally small:
- local Markdown links from README/docs point to committed files;
- experiment IDs cited in docs/CLAIMS.md exist in EXPERIMENTS.md.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections.abc import Iterable


ROOT = pathlib.Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
EXP_RE = re.compile(r"\bExp(?:eriments?)?\s+(\d+)(?:\s*[-–]\s*(\d+))?", re.IGNORECASE)
HEADING_RE = re.compile(r"^## Exp (\d+)\b", re.MULTILINE)


def markdown_files(root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    readme = root / "README.md"
    if readme.exists():
        files.append(readme)
    docs = root / "docs"
    if docs.exists():
        files.extend(sorted(docs.rglob("*.md")))
    return files


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _is_external_or_anchor(target: str) -> bool:
    lower = target.lower()
    return (
        lower.startswith(("http://", "https://", "mailto:"))
        or lower.startswith("#")
        or not target
    )


def _local_target(path: pathlib.Path, target: str) -> pathlib.Path | None:
    clean = target.split("#", 1)[0].strip()
    if _is_external_or_anchor(clean):
        return None
    if clean.startswith("<") and clean.endswith(">"):
        clean = clean[1:-1]
    base = ROOT if clean.startswith("/") else path.parent
    return (base / clean.lstrip("/")).resolve()


def check_markdown_links(root: pathlib.Path, files: Iterable[pathlib.Path]) -> list[str]:
    errors: list[str] = []
    root = root.resolve()
    for path in files:
        text = _strip_code_fences(path.read_text(encoding="utf-8"))
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            local = _local_target(path, target)
            if local is None or local.exists():
                continue
            display = target.split("#", 1)[0].strip()
            errors.append(
                f"{path.relative_to(root)} links to missing local file {display}"
            )
    return errors


def experiment_ids(text: str) -> set[int]:
    return {int(n) for n in HEADING_RE.findall(text)}


def cited_experiment_ids(text: str) -> set[int]:
    ids: set[int] = set()
    for start_text, end_text in EXP_RE.findall(text):
        start = int(start_text)
        end = int(end_text) if end_text else start
        if end < start:
            start, end = end, start
        ids.update(range(start, end + 1))
    return ids


def check_claim_experiments(root: pathlib.Path) -> list[str]:
    claims = root / "docs" / "CLAIMS.md"
    experiments = root / "EXPERIMENTS.md"
    if not claims.exists():
        return ["docs/CLAIMS.md is missing"]
    if not experiments.exists():
        return ["EXPERIMENTS.md is missing"]

    known = experiment_ids(experiments.read_text(encoding="utf-8"))
    cited = cited_experiment_ids(claims.read_text(encoding="utf-8"))
    missing = sorted(cited - known)
    if not missing:
        return []
    missing_text = ", ".join(str(n) for n in missing)
    return [f"docs/CLAIMS.md cites missing experiment IDs: {missing_text}"]


def run(root: pathlib.Path = ROOT) -> list[str]:
    root = root.resolve()
    return [
        *check_markdown_links(root, markdown_files(root)),
        *check_claim_experiments(root),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=ROOT,
        help="repository root to check",
    )
    args = parser.parse_args(argv)
    errors = run(args.root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("docs consistency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
