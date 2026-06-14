# Stream D (experiments parser foundation) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Close the most insidious data-scaling cliff — the FOUR re-implemented `## Exp N` parsers (`loop/check_iteration.py`, `active_loop/site_data.py`, `am-live.js`, the site tests) where a format change silently breaks some while others pass. Ship one canonical `active_loop/experiments_parser.py` plus a **format-drift guard** that pins it to every consumer's regex, so any divergence fails LOUDLY.

**Architecture:** Purely additive — a new module + its tests. The tests are **relational invariants** (the parser's experiment-number set equals what each consumer regex finds on the *same* `EXPERIMENTS.md` text), NOT committed snapshots — so they are robust to the live loop appending experiments and never red on loop activity. **Deferred (loop-critical / collides with the live loop, flagged):** rerouting the 4 consumers through the parser + deleting the duplicates (changes loop-critical pre-commit/site code; needs byte-match cutover); a generated `experiments-index.json` (per-experiment churn → autosync-staleness coupling); lazy-loading/paginating `experiments-data.js` + ETag on `am-live.js` (touches the loop-regenerated site files deferred in Stream B).

**Tech stack:** Python 3.11+ stdlib (`re`, `dataclasses`), pytest 8.

**Branch:** `infra/experiments-parser` (worktree `/Users/mirro/Projects/active-loop-data`).

**Verified facts (origin/main @ 2c1b807):** 207 `## Exp N` headers, format `## Exp N — <title>` (em-dash U+2014). Consumer regexes: `site_data.py:60` `^## Exp (\d+) `, `site_data.py:277` `^## Exp (\d+) — (.+)$`, `check_iteration.py:39` `^## Exp (\d+)\b` (finditer entry-split), tests `^## Exp (\d+) `.

---

## Task 1: `active_loop/experiments_parser.py` + tests (TDD)

**Files:**
- Create: `active_loop/experiments_parser.py`
- Test: `tests/test_experiments_parser.py`

- [ ] **Step 1: Write the failing test** `tests/test_experiments_parser.py`:

```python
"""Tests for active_loop/experiments_parser.py — the canonical EXPERIMENTS.md parser.

These are RELATIONAL invariants: the parser must agree with the regexes the existing
consumers use (site_data.py, check_iteration.py, the site tests) on the SAME text.
They read the live EXPERIMENTS.md but assert internal agreement, so they are robust
to the loop appending experiments (both sides see the new entry). A format change that
splits the consumers fails here LOUDLY instead of silently breaking one of them.

Run:  uv run --python .venv pytest tests/test_experiments_parser.py -q
"""
from __future__ import annotations

import pathlib
import re

import pytest

from active_loop.experiments_parser import Experiment, by_number, parse

ROOT = pathlib.Path(__file__).parent.parent
TEXT = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")


def test_numbers_match_site_data_canonical_regex():
    canonical = sorted(int(x) for x in re.findall(r"^## Exp (\d+) ", TEXT, re.MULTILINE))
    assert sorted(e.n for e in parse(TEXT)) == canonical


def test_numbers_match_check_iteration_regex():
    ci = sorted(int(x) for x in re.findall(r"^## Exp (\d+)\b", TEXT, re.MULTILINE))
    assert sorted(e.n for e in parse(TEXT)) == ci


def test_bodies_partition_the_log():
    exps = parse(TEXT)
    assert exps, "no experiments parsed"
    for e in exps:
        assert e.body.startswith(e.header)
    # each entry's body is exactly the text from its header to the next header
    for a, b in zip(exps, exps[1:]):
        assert TEXT[a.start:b.start] == a.body


def test_every_experiment_has_a_title():
    # the '## Exp N — title' em-dash format is universal; a future entry that
    # breaks it should fail here (loud), not silently lose its title.
    assert all(e.title for e in parse(TEXT))


def test_known_title_exp1():
    by = by_number(TEXT)
    assert by[1].title.startswith("does the character HMM learn")


def test_by_number_raises_on_duplicate():
    dup = "## Exp 5 — a\nbody a\n## Exp 5 — b\nbody b\n"
    with pytest.raises(ValueError):
        by_number(dup)


def test_load_reads_repo_experiments_md():
    from active_loop.experiments_parser import load
    assert len(load()) == len(parse(TEXT))
```

- [ ] **Step 2: Run — expect FAIL** (module missing)

Run: `cd /Users/mirro/Projects/active-loop-data && uv run --python .venv pytest tests/test_experiments_parser.py -q`

- [ ] **Step 3: Implement `active_loop/experiments_parser.py`** with EXACTLY:

```python
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
```

- [ ] **Step 4: Run — expect PASS** (7 passed)

Run: `cd /Users/mirro/Projects/active-loop-data && uv run --python .venv pytest tests/test_experiments_parser.py -q`

- [ ] **Step 5: Sanity-check**

```bash
cd /Users/mirro/Projects/active-loop-data
uv run --python .venv python -c "from active_loop.experiments_parser import load, by_number; xs=load(); print(len(xs), 'experiments;', xs[0].n, xs[0].title[:40], '...', xs[-1].n)"
```
Expected: prints the count (207+), `1 does the character HMM learn ...`, and the latest number.

- [ ] **Step 6: Commit**

```bash
cd /Users/mirro/Projects/active-loop-data
git add active_loop/experiments_parser.py tests/test_experiments_parser.py
git commit -m "feat(parser): canonical EXPERIMENTS.md parser + format-drift guard (consumers unchanged)"
```

---

## Task 2: Full-suite regression

- [ ] **Step 1:** `cd /Users/mirro/Projects/active-loop-data && uv run --python .venv pytest -p no:warnings -q` → exit 0, zero failures. (The new module is additive; no consumer was rerouted, so existing tests are unaffected.)

---

## Self-review (author — completed)
- Delivers spec §8's parser-unification cliff as a SAFE foundation + a loud format-drift guard (serves the no-silent-failure value). Consumer cutover + index + site lazy-load explicitly deferred (loop-critical) and named.
- Additive only: `active_loop/experiments_parser.py` + `tests/test_experiments_parser.py`. No edits to `check_iteration.py`, `site_data.py`, `am-live.js`, `EXPERIMENTS.md`, `autosync.sh`, or `managed-paths.txt` → zero conflict with the live loop.
- Tests are relational invariants (robust to loop appends), not committed snapshots; `by_number` raises (no silent overwrite); no placeholders.
