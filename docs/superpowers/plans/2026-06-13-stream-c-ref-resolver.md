# Stream C (ref resolver) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Deliver smart `@`-referencing (the human's ranked #3 pain): a single on-demand resolver that turns a short key (`@exp201`, `@n4-identity`, `@ecology`) into the canonical repo path, plus a `--list` discovery view.

**Architecture:** One additive script, `tools/ref.py`, computed **on demand from the filesystem** — no committed registry/index, so it can never go stale and has zero coupling to the live loop's per-iteration commits. (A committed `loop/registry.json` / `docs/INDEX.md` and governance-doc de-dup from spec §7 are deliberately **deferred**: they'd need autosync wiring or `RESUME.md` edits that collide with the loop, for lower value.)

**Tech stack:** Python 3.11+ stdlib, pytest 8 (`uv run --python .venv pytest`).

**Branch:** `infra/doc-backbone` (worktree `/Users/mirro/Projects/active-loop-docs`).

**Verified facts:** directions live in `loop/directions/*.md` (skip `_TEMPLATE`), personas in `loop/personas/*.md`, research chapters in `docs/research/*.md`, specs in `docs/specs/*.md`; experiment scripts are `experiments/exp<N>_<slug>.py` (176 of them); `compose.py` aliases `ecology`/`pop` → `population-ecology`.

---

## Task 1: `tools/ref.py` + tests (TDD)

**Files:**
- Create: `tools/ref.py`
- Test: `tests/test_ref.py`

- [ ] **Step 1: Write the failing test** `tests/test_ref.py`:

```python
"""Tests for tools/ref.py — the @-reference resolver.

tools/ is not an importable package, so load ref.py by file path. Tests use the
real filesystem (stable, closed cards/experiments) like test_directions_index.py.
Run:  uv run --python .venv pytest tests/test_ref.py -q
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("ref", ROOT / "tools" / "ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_direction():
    r = _load()
    assert r.resolve("@transfer") == [pathlib.Path("loop/directions/transfer.md")]


def test_alias_direction():
    r = _load()
    assert r.resolve("ecology") == [pathlib.Path("loop/directions/population-ecology.md")]


def test_research_chapter_unique_substring():
    r = _load()
    assert r.resolve("@n4-identity") == [
        pathlib.Path("docs/research/n4-identity-commitment-chapter.md")
    ]


def test_experiment_glob():
    r = _load()
    got = r.resolve("@exp201")
    assert len(got) == 1
    assert str(got[0]).startswith("experiments/exp201_")
    assert str(got[0]).endswith(".py")


def test_unknown_exits():
    r = _load()
    with pytest.raises(SystemExit):
        r.resolve("@zzznope")


def test_ambiguous_exits():
    r = _load()
    with pytest.raises(SystemExit):
        r.resolve("@problem2")  # two docs/research/problem2-*.md


def test_list_index_mentions_kinds():
    r = _load()
    out = r.list_index()
    assert "@transfer" in out
    assert "experiment" in out
```

- [ ] **Step 2: Run — expect FAIL** (`tools/ref.py` does not exist)

Run: `cd /Users/mirro/Projects/active-loop-docs && uv run --python .venv pytest tests/test_ref.py -q`

- [ ] **Step 3: Implement `tools/ref.py`** with EXACTLY this content:

```python
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
        return matches
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
```

- [ ] **Step 4: Run — expect PASS** (7 passed)

Run: `cd /Users/mirro/Projects/active-loop-docs && uv run --python .venv pytest tests/test_ref.py -q`

- [ ] **Step 5: Sanity-check the CLI**

```bash
cd /Users/mirro/Projects/active-loop-docs
uv run --python .venv python tools/ref.py @exp201
uv run --python .venv python tools/ref.py @n4-crack-chapter
uv run --python .venv python tools/ref.py --list research
```
Expected: prints `experiments/exp201_*.py`; `docs/research/n4-crack-chapter.md`; the research list.

- [ ] **Step 6: Commit**

```bash
cd /Users/mirro/Projects/active-loop-docs
git add tools/ref.py tests/test_ref.py
git commit -m "feat(ref): @-reference resolver for directions/personas/chapters/specs/experiments"
```

---

## Task 2: Document `@`-refs

**Files:** Modify `loop/README.md`

- [ ] **Step 1: Append a reference note to `loop/README.md`**

READ `loop/README.md`. After the `### In-session commands` block (added in Stream A) and before `## Extending`, append:

```markdown
### Referencing (tools/ref.py)

Resolve a short key to a canonical path instead of typing/grepping it:

```bash
uv run --python .venv python tools/ref.py @exp201          # experiments/exp201_<slug>.py
uv run --python .venv python tools/ref.py @n4-crack-chapter  # docs/research/...
uv run --python .venv python tools/ref.py --list            # discover everything
```

`@<direction>` / `@<persona>` / `@<chapter>` / `@<spec>` resolve by exact stem,
alias (e.g. `ecology` → `population-ecology`), or unique substring; `@exp<N>`
globs the experiment script. Computed on demand — never stale.
```

(If `loop/README.md` has no `### In-session commands` block, append the note at the end of the `## Use` section instead.)

- [ ] **Step 2: Verify the suite is unaffected + commit**

Run: `cd /Users/mirro/Projects/active-loop-docs && uv run --python .venv pytest tests/test_ref.py -q` (still 7 passed)
```bash
cd /Users/mirro/Projects/active-loop-docs
git add loop/README.md
git commit -m "docs: document tools/ref.py @-referencing"
```

---

## Task 3: Full-suite regression

- [ ] **Step 1:** `cd /Users/mirro/Projects/active-loop-docs && uv run --python .venv pytest -p no:warnings -q` → exit 0, zero failures.

---

## Self-review (author — completed)
- Delivers spec §7's smart-referencing core (the ranked #3 pain). Committed registry/INDEX + governance de-dup explicitly deferred (loop-coupling / RESUME-collision risk) and flagged.
- Additive only: one new script + test + a doc note. No edits to loop-critical files (`autosync.sh`, `managed-paths.txt`, `site_data.py`, `RESUME.md`, `EXPERIMENTS.md`), so zero conflict with the live loop.
- No placeholders; exact code + commands throughout. Tests cover exact/alias/substring/experiment/unknown/ambiguous + list.
