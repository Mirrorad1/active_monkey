"""Single canonical parser for EXPERIMENTS.md — the append-only research log.

Multiple consumers (loop/check_iteration.py, active_loop/site_data.py, am-live.js,
and the site tests) each re-implement '## Exp N' header parsing; a format change can
silently break some while others keep passing. This module is the one parser they
should share. tests/test_experiments_parser.py pins it to the regexes those consumers
use, so a divergence fails LOUDLY.

An Experiment is the header line '## Exp N — title' plus its body, up to the next
header — matching check_iteration.py's entry-split semantics.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPERIMENTS_MD = ROOT / "EXPERIMENTS.md"

# Entry header. The \b after the number matches check_iteration.py:39; the title
# (after the em-dash) matches site_data.py:277.
_HEADER_RE = re.compile(r"^## Exp (\d+)\b[^\n]*$", re.MULTILINE)
_TITLE_RE = re.compile(r"^## Exp \d+\s*[—-]\s*(.+)$")


@dataclass(frozen=True)
class Experiment:
    n: int            # the experiment number
    title: str        # header text after the em-dash ("" if a header has none)
    header: str       # the full header line
    body: str         # the header line through the text just before the next header
    start: int        # character offset of the header in the source text


def parse(text: str) -> list[Experiment]:
    """Parse EXPERIMENTS.md text into ordered Experiment records (one per ## Exp N)."""
    matches = list(_HEADER_RE.finditer(text))
    out: list[Experiment] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = m.group(0)
        tm = _TITLE_RE.match(header)
        out.append(
            Experiment(
                n=int(m.group(1)),
                title=tm.group(1).strip() if tm else "",
                header=header,
                body=text[m.start():end],
                start=m.start(),
            )
        )
    return out


def load(path: pathlib.Path = EXPERIMENTS_MD) -> list[Experiment]:
    """Parse the repo's EXPERIMENTS.md."""
    return parse(path.read_text(encoding="utf-8"))


def by_number(text: str) -> dict[int, Experiment]:
    """Map number → Experiment. Raises ValueError on a duplicate (never silently overwrites)."""
    out: dict[int, Experiment] = {}
    for e in parse(text):
        if e.n in out:
            raise ValueError(f"duplicate experiment number {e.n} in EXPERIMENTS.md")
        out[e.n] = e
    return out
