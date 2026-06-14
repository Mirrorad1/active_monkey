# Stream A — Invocation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the compose→copy→paste courier hop with two in-session slash commands — `/lab <direction>` (compose → confirm → iterate) and `/steer` (human-marked inbox writes) — wrapping the existing `loop/compose.py`, plus config-driven `autosync.sh` paths.

**Architecture:** Thin command layer over existing machinery. `.claude/commands/lab.md` and `.claude/commands/steer.md` are prompt templates that shell out to `loop/compose.py` and a new `tools/steer_append.py`. `compose.py` gains a positional direction arg + a name resolver (exact → alias → unique-substring). `tools/autosync.sh`'s hardcoded `managed_paths` becomes a newline-delimited `loop/managed-paths.txt` so Stream B can relocate site files without breaking the Stop-hook sweep. No prompt-OS rewrite; the confirm-gate preserves the `VALIDATION.md §5` consent boundary.

**Tech Stack:** Python 3.11+ (stdlib only for the touched code), pytest 8 (`uv run --python .venv pytest`), bash (`autosync.sh`), Claude Code project slash commands (`.claude/commands/*.md`).

**Refinement vs spec §A.3:** managed-paths is a newline-delimited `.txt`, not `.json` — the sole consumer is the bash Stop hook; a text file needs no JSON parser and works in the existing sandbox test without `uv`/`.venv`. (The spec's `loop/registry.json` backbone in Stream C is separate.)

**Branch:** `infra/lab-invocation` (draft PR #39). All tasks commit here.

---

## File structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `loop/compose.py` | Modify (`:21-113`) | Add positional direction arg + `resolve_direction()` (exact → alias → unique-substring → error) |
| `tests/test_compose.py` | Create | Unit-test the resolver (injected candidates) + subprocess wiring |
| `.claude/commands/lab.md` | Create | `/lab` prompt template: compose → summary → **wait for "go"** → iterate |
| `.claude/commands/steer.md` | Create | `/steer` prompt template: append idea / `--resume <word>` to IDEAS.md |
| `tests/test_lab_commands.py` | Create | Structural guard: command files exist + contain the consent-gate language |
| `tools/steer_append.py` | Create | Append-only human-marked bullet under `## Inbox` of `loop/IDEAS.md` |
| `tests/test_steer_append.py` | Create | Unit-test append-only behavior (original bytes preserved, bullet added under Inbox) |
| `loop/managed-paths.txt` | Create | Newline-delimited managed-paths list (the current hardcoded set) |
| `tools/autosync.sh` | Modify (`:25-37`) | Read `managed_paths` from `loop/managed-paths.txt` (graceful skip if absent) |
| `tests/test_autosync_branch_guard.py` | Modify (`_make_sandbox`) | Copy `loop/managed-paths.txt` into the sandbox so the real script finds it |
| `tests/test_managed_paths.py` | Create | Assert the configured managed set equals the expected list (no silent drops) |
| `RESUME.md` | Modify (`§6`) | Command-first flow (`/lab <direction>`); paste fallback retained |
| `loop/README.md` | Modify (`## Use`) | Document `/lab` and `/steer` |

---

## Task 1: `compose.py` — positional direction + resolver

**Files:**
- Modify: `loop/compose.py:21-113`
- Test: `tests/test_compose.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compose.py`:

```python
"""Tests for loop/compose.py — the direction resolver and CLI wiring.

loop/ is not an importable package, so load compose.py by file path.
Run:  uv run --python .venv pytest tests/test_compose.py -q
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _load_compose():
    spec = importlib.util.spec_from_file_location(
        "loop_compose", ROOT / "loop" / "compose.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CANDIDATES = [
    "population-ecology",
    "continuous-substrate",
    "sequence-substrate",
    "identity-n4",
    "identity-n4-crack",
    "graded-uncertainty",
    "transfer",
]


def test_exact_stem_resolves_to_itself():
    c = _load_compose()
    assert c.resolve_direction("transfer", candidates=CANDIDATES) == "transfer"


def test_alias_resolves():
    c = _load_compose()
    assert c.resolve_direction("ecology", candidates=CANDIDATES) == "population-ecology"


def test_unique_substring_resolves():
    c = _load_compose()
    assert c.resolve_direction("graded", candidates=CANDIDATES) == "graded-uncertainty"


def test_ambiguous_substring_exits():
    c = _load_compose()
    with pytest.raises(SystemExit):
        c.resolve_direction("identity", candidates=CANDIDATES)


def test_unknown_exits():
    c = _load_compose()
    with pytest.raises(SystemExit):
        c.resolve_direction("zzznope", candidates=CANDIDATES)


def test_cli_positional_direction_composes():
    # stdlib-only module → plain interpreter, no venv needed
    r = subprocess.run(
        [sys.executable, "loop/compose.py", "transfer"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "=== DIRECTION (what to work on) ===" in r.stdout


def test_cli_unknown_direction_errors():
    r = subprocess.run(
        [sys.executable, "loop/compose.py", "zzznope"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "no direction" in (r.stderr + r.stdout).lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --python .venv pytest tests/test_compose.py -q`
Expected: FAIL — `AttributeError: module 'loop_compose' has no attribute 'resolve_direction'`.

- [ ] **Step 3: Implement the resolver + positional arg in `loop/compose.py`**

Add a module-level constant + function near the top (after the `LOOP_DIR` definition, around `compose.py:18`):

```python
DEFAULT_DIRECTION = "continuous-substrate"  # current human steer (see loop/IDEAS.md)

# Short ergonomic aliases checked before substring matching. Substring already
# covers most cases (e.g. "graded" → "graded-uncertainty"); aliases disambiguate
# the common shortcuts the human types.
ALIASES = {
    "ecology": "population-ecology",
    "pop": "population-ecology",
}


def resolve_direction(name: str, candidates: list[str] | None = None) -> str:
    """Resolve a user-typed direction token to a real card stem.

    Order: exact stem → alias → unique case-insensitive substring → sys.exit.
    """
    if candidates is None:
        candidates = available("directions")
    if name in candidates:
        return name
    if name in ALIASES and ALIASES[name] in candidates:
        return ALIASES[name]
    matches = [c for c in candidates if name.lower() in c.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit(
            f"error: no direction matches '{name}' "
            f"(have: {', '.join(candidates)})"
        )
    sys.exit(
        f"error: '{name}' is ambiguous — matches {', '.join(matches)}; be more specific"
    )
```

Then change `main()` (`compose.py:92-109`) to accept a positional direction and resolve it. Replace the `--direction` argument block and the `compose(...)` call:

```python
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "direction_pos", nargs="?", default=None,
        help="direction card name or short alias (positional; e.g. 'ecology')",
    )
    ap.add_argument(
        "--direction", default=None,
        help="direction card name (alternative to the positional form)",
    )
    ap.add_argument("--persona", default="default", help="persona card name")
    ap.add_argument("--idea", default=None, help="one-off human idea to inject")
    ap.add_argument("--list", action="store_true", help="list available modules")
    args = ap.parse_args()

    if args.list:
        print("directions:", ", ".join(available("directions")))
        print("personas:  ", ", ".join(available("personas")))
        return

    requested = args.direction_pos or args.direction or DEFAULT_DIRECTION
    direction = resolve_direction(requested)
    print(compose(direction, args.persona, args.idea))
```

`compose()` and `load_card()` are unchanged — `direction` is now always a real stem, so `load_card("directions", direction)` still works.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --python .venv pytest tests/test_compose.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Sanity-check the CLI by hand**

Run: `uv run --python .venv python loop/compose.py ecology | head -1`
Expected: `/loop Keep running the moonshot active-inference experiments until I stop you.`
Run: `uv run --python .venv python loop/compose.py --list`
Expected: prints `directions: ...` including `population-ecology`.

- [ ] **Step 6: Commit**

```bash
git add loop/compose.py tests/test_compose.py
git commit -m "feat(compose): positional direction + name resolver (exact/alias/substring)"
```

---

## Task 2: `/lab` command

**Files:**
- Create: `.claude/commands/lab.md`
- Test: `tests/test_lab_commands.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lab_commands.py`:

```python
"""Structural guards for the project slash commands.

Command .md files are prompt templates (not unit-testable behavior), so we guard
the load-bearing invariants: the file exists, declares the right allowed-tools,
and KEEPS the human-consent gate language (the confirm-before-iterate step that
VALIDATION.md §5 depends on). A regression that deletes the gate fails here.

Run:  uv run --python .venv pytest tests/test_lab_commands.py -q
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).parent.parent
CMD = ROOT / ".claude" / "commands"


def test_lab_command_exists():
    assert (CMD / "lab.md").is_file()


def test_lab_command_invokes_compose():
    text = (CMD / "lab.md").read_text(encoding="utf-8")
    assert "loop/compose.py" in text


def test_lab_command_has_consent_gate():
    text = (CMD / "lab.md").read_text(encoding="utf-8").lower()
    # must wait for an explicit "go" and must not iterate/commit before it
    assert "go" in text
    assert "confirm" in text or "wait" in text
    assert "vation.md" in text or "consent" in text  # cites the boundary
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --python .venv pytest tests/test_lab_commands.py -q`
Expected: FAIL — `assert (CMD / "lab.md").is_file()` (file missing).

- [ ] **Step 3: Create `.claude/commands/lab.md`**

```markdown
---
description: Compose and launch a research-loop iteration (compose → confirm → iterate)
argument-hint: <direction> [--persona NAME] [--idea "..."]
allowed-tools: Bash(uv run --python .venv python loop/compose.py:*)
---
The human invoked `/lab` with arguments: `$ARGUMENTS`

Do this, in order:

1. Run `uv run --python .venv python loop/compose.py $ARGUMENTS` via the Bash tool
   (the first positional token is the direction; `--persona` / `--idea` pass through).
   Capture the composed `/loop` prompt it prints.
2. If it exited non-zero (unknown or ambiguous direction), run
   `uv run --python .venv python loop/compose.py --list`, show the options, and ask
   me which direction I meant. Do not guess.
3. Show a ONE-line summary of what this run will do: direction, persona, and any
   `--idea`.
4. STOP. Ask me to confirm with "go" (or "edit ...") before you begin iterating.
   Do NOT start iterating and do NOT commit anything yet — this confirm step is the
   human-consent gate that VALIDATION.md §5 depends on (a cron fire is not a human
   resumption; neither is composing a prompt).
5. On my "go": read RESUME.md, the tail of EXPERIMENTS.md, and loop/IDEAS.md, then
   iterate under loop/PROTOCOL.md, self-pacing across wakes via the standard /loop
   dynamic mechanism. Honor loop/VALIDATION.md (binding) throughout.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --python .venv pytest tests/test_lab_commands.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add .claude/commands/lab.md tests/test_lab_commands.py
git commit -m "feat(commands): /lab — compose, confirm-gate, then iterate"
```

---

## Task 3: `tools/steer_append.py`

**Files:**
- Create: `tools/steer_append.py`
- Test: `tests/test_steer_append.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_steer_append.py`:

```python
"""Tests for tools/steer_append.py — append-only human-marked inbox writes.

Run:  uv run --python .venv pytest tests/test_steer_append.py -q
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "steer_append", ROOT / "tools" / "steer_append.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IDEAS = (
    "# IDEAS — human inbox\n\nintro line\n\n## Inbox\n\n"
    "- [from human, 2026-06-01] an older idea\n\n"
    "## Consumed\n\n- something consumed\n"
)


def _write(tmp_path) -> pathlib.Path:
    p = tmp_path / "IDEAS.md"
    p.write_text(IDEAS, encoding="utf-8")
    return p


def test_append_idea_preserves_original_bytes(tmp_path):
    m = _load()
    p = _write(tmp_path)
    m.append_idea("try a foraging gradient", date="2026-06-13", ideas_path=p)
    out = p.read_text(encoding="utf-8")
    # every original line still present, in order (append-only)
    for line in IDEAS.splitlines():
        assert line in out
    assert "- [from human, 2026-06-13] try a foraging gradient" in out


def test_append_idea_lands_inside_inbox_not_consumed(tmp_path):
    m = _load()
    p = _write(tmp_path)
    m.append_idea("new steer", date="2026-06-13", ideas_path=p)
    out = p.read_text(encoding="utf-8")
    inbox_i = out.index("## Inbox")
    consumed_i = out.index("## Consumed")
    new_i = out.index("new steer")
    assert inbox_i < new_i < consumed_i


def test_append_resume_marks_human_reply(tmp_path):
    m = _load()
    p = _write(tmp_path)
    m.append_resume("a", date="2026-06-13", ideas_path=p)
    out = p.read_text(encoding="utf-8")
    assert "[from human, 2026-06-13]" in out
    assert "consult reply: a" in out


def test_missing_inbox_section_raises(tmp_path):
    m = _load()
    p = tmp_path / "IDEAS.md"
    p.write_text("# IDEAS\n\nno inbox header here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        m.append_idea("x", date="2026-06-13", ideas_path=p)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --python .venv pytest tests/test_steer_append.py -q`
Expected: FAIL — file `tools/steer_append.py` does not exist (import error).

- [ ] **Step 3: Implement `tools/steer_append.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --python .venv pytest tests/test_steer_append.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/steer_append.py tests/test_steer_append.py
git commit -m "feat(steer): append-only human-marked inbox writer"
```

---

## Task 4: `/steer` command

**Files:**
- Create: `.claude/commands/steer.md`
- Modify: `tests/test_lab_commands.py` (add steer guards)

- [ ] **Step 1: Add failing guards to `tests/test_lab_commands.py`**

Append to `tests/test_lab_commands.py`:

```python
def test_steer_command_exists():
    assert (CMD / "steer.md").is_file()


def test_steer_command_uses_append_script():
    text = (CMD / "steer.md").read_text(encoding="utf-8")
    assert "tools/steer_append.py" in text


def test_steer_command_marks_human_authorship():
    text = (CMD / "steer.md").read_text(encoding="utf-8").lower()
    assert "human" in text  # the bullet must be attributed to the human
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --python .venv pytest tests/test_lab_commands.py -q`
Expected: FAIL — `assert (CMD / "steer.md").is_file()`.

- [ ] **Step 3: Create `.claude/commands/steer.md`**

```markdown
---
description: Drop a human steer/idea into the research loop's inbox (loop/IDEAS.md)
argument-hint: "<idea text>"  |  --resume <word>
allowed-tools: Bash(uv run --python .venv python tools/steer_append.py:*)
---
The human invoked `/steer` with arguments: `$ARGUMENTS`

This records a HUMAN steer into loop/IDEAS.md so the loop consumes it at the next
iteration (PROTOCOL.md step 0). Because I am running it at the human's request, it
counts as a human action — it does NOT erode the VALIDATION.md §5 boundary (a cron
fire is not a human resumption; this is).

Do this:
1. If the arguments start with `--resume`, run
   `uv run --python .venv python tools/steer_append.py --resume <word>`.
   Otherwise run `uv run --python .venv python tools/steer_append.py "<all the text>"`,
   passing everything after `/steer` as the idea text (quote it so spaces survive).
2. Show me the exact bullet that was appended and confirm it landed under `## Inbox`.
3. Do not start or resume any iteration yourself — `/steer` only records the steer.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --python .venv pytest tests/test_lab_commands.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add .claude/commands/steer.md tests/test_lab_commands.py
git commit -m "feat(commands): /steer — human-marked inbox writes"
```

---

## Task 5: Config-driven `autosync.sh` managed paths

**Files:**
- Create: `loop/managed-paths.txt`
- Modify: `tools/autosync.sh:25-37`
- Modify: `tests/test_autosync_branch_guard.py` (`_make_sandbox` copies the file)
- Test: `tests/test_managed_paths.py`

- [ ] **Step 1: Create `loop/managed-paths.txt`** (the current hardcoded set, verbatim)

```text
# Paths the autosync Stop hook (tools/autosync.sh) may sweep into an auto-sync
# commit. One path per line; blank lines and #-comments ignored. The loop owns
# experiment/site artifacts; arbitrary docs/code stay unstaged for intentional
# commits. Stream B will repoint experiments-data.js / lab-status.js to site/.
EXPERIMENTS.md
experiments
experiments-data.js
lab-status.js
DIRECTIONS.md
loop/IDEAS.md
loop/directions
loop/managed-paths.txt
creature/state
world_model
reports
REPORT.md
```

- [ ] **Step 2: Write the failing guard test** `tests/test_managed_paths.py`

```python
"""Guard: loop/managed-paths.txt holds exactly the expected managed set.

Prevents a silent drop (e.g. EXPERIMENTS.md falling out of the autosync sweep).
Run:  uv run --python .venv pytest tests/test_managed_paths.py -q
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).parent.parent


def _configured() -> set[str]:
    text = (ROOT / "loop" / "managed-paths.txt").read_text(encoding="utf-8")
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def test_managed_set_matches_expected():
    expected = {
        "EXPERIMENTS.md", "experiments", "experiments-data.js", "lab-status.js",
        "DIRECTIONS.md", "loop/IDEAS.md", "loop/directions", "loop/managed-paths.txt",
        "creature/state", "world_model", "reports", "REPORT.md",
    }
    assert _configured() == expected
```

- [ ] **Step 3: Run it to verify it passes already** (the file from Step 1 satisfies it)

Run: `uv run --python .venv pytest tests/test_managed_paths.py -q`
Expected: PASS (1 passed). (This guard locks the file; it has no pre-impl failing phase.)

- [ ] **Step 4: Make `tests/test_autosync_branch_guard.py` fail by tightening the sandbox**

In `tests/test_autosync_branch_guard.py`, edit `_make_sandbox` to also install the
managed-paths file BEFORE the seed commit (so the real script can read it). Add,
right after the `shutil.copy(SCRIPT, tools / "autosync.sh")` line:

```python
    loop_dir = work / "loop"
    loop_dir.mkdir()
    shutil.copy(ROOT / "loop" / "managed-paths.txt", loop_dir / "managed-paths.txt")
```

Run now (script still hardcoded): `uv run --python .venv pytest tests/test_autosync_branch_guard.py -q`
Expected: PASS still (hardcoded list happens to match) — this step only prepares the
sandbox. The real failing assertion comes from Step 5's transition: after we delete
the hardcoded array the sandbox MUST have the file or `EXPERIMENTS.md` won't be swept.

- [ ] **Step 5: Replace the hardcoded array in `tools/autosync.sh`**

Replace lines `25-37` (the `managed_paths=( ... )` literal array) with a reader:

```bash
  # Managed paths are configured in loop/managed-paths.txt (one per line,
  # # comments allowed). Keeps autosync from sweeping scratch files or
  # parallel-agent edits; lets Stream B repoint site files without editing this
  # script. If the file is absent, sweep nothing (safe: leaves the tree for an
  # intentional commit rather than guessing).
  managed_paths=()
  mp_file="loop/managed-paths.txt"
  if [ -f "$mp_file" ]; then
    while IFS= read -r p; do
      case "$p" in ''|\#*) continue ;; esac
      managed_paths+=("$p")
    done < "$mp_file"
  fi
```

The existing loop just below (`for path in "${managed_paths[@]}"; do ...`) is unchanged.

- [ ] **Step 6: Run the autosync tests to verify they pass**

Run: `uv run --python .venv pytest tests/test_autosync_branch_guard.py tests/test_managed_paths.py -q`
Expected: PASS (4 passed) — the three sandbox guards (now reading the copied file) +
the managed-set guard.

- [ ] **Step 7: Commit**

```bash
git add loop/managed-paths.txt tools/autosync.sh tests/test_autosync_branch_guard.py tests/test_managed_paths.py
git commit -m "refactor(autosync): config-driven managed paths via loop/managed-paths.txt"
```

---

## Task 6: Document the command-first flow

**Files:**
- Modify: `RESUME.md` (§6)
- Modify: `loop/README.md` (`## Use`)
- Test: `tests/test_lab_commands.py` (add a docs guard)

- [ ] **Step 1: Add a failing docs guard to `tests/test_lab_commands.py`**

```python
def test_resume_documents_lab_command():
    text = (ROOT / "RESUME.md").read_text(encoding="utf-8")
    assert "/lab" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --python .venv pytest tests/test_lab_commands.py::test_resume_documents_lab_command -q`
Expected: FAIL — `/lab` not yet in `RESUME.md`.

- [ ] **Step 3: Update `RESUME.md` §6**

At the top of the "Preferred (modular)" block in §6, before the `compose.py` examples,
insert:

```markdown
**Fastest (in-session command):** type `/lab <direction>` in a Claude session — it
runs `loop/compose.py`, shows the assembled prompt + a one-line summary, and begins
iterating only after you reply "go" (the human-consent gate). Steer mid-flight with
`/steer "<idea>"`; answer an open consult with `/steer --resume <word>`. The terminal
`compose.py` recipe below still works as a fallback.
```

- [ ] **Step 4: Update `loop/README.md`**

At the end of the `## Use` section (after the `--list` example block), add:

```markdown
### In-session commands (preferred over copy-paste)

- `/lab <direction> [--persona NAME] [--idea "..."]` — compose, show a one-line
  summary, and iterate only after you confirm with "go". Wraps `compose.py`
  (which now also accepts the direction positionally, e.g. `compose.py ecology`).
- `/steer "<idea>"` / `/steer --resume <word>` — append a human-marked bullet to
  `IDEAS.md` without opening the file.

The commands live in `.claude/commands/`. They never start iterating or commit
before your explicit "go" — that confirm step is the `VALIDATION.md §5` consent gate.
```

- [ ] **Step 5: Run the docs guard to verify it passes**

Run: `uv run --python .venv pytest tests/test_lab_commands.py -q`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
git add RESUME.md loop/README.md tests/test_lab_commands.py
git commit -m "docs: document /lab + /steer command-first flow"
```

---

## Task 7: Full-suite regression + PR refresh

**Files:** none (verification only)

- [ ] **Step 1: Run the full fast suite**

Run: `uv run --python .venv pytest -q`
Expected: PASS — the pre-existing fast suite (~250 tests) plus the new
`test_compose.py`, `test_lab_commands.py`, `test_steer_append.py`,
`test_managed_paths.py`, and the modified `test_autosync_branch_guard.py`. Zero
failures, zero errors. (Slow tests stay deselected by default per `pyproject.toml`.)

- [ ] **Step 2: Manual end-to-end smoke (optional, no commit)**

In a scratch session: `/lab ecology` → confirm it shows a one-line summary and waits
for "go" without iterating; `/steer "smoke-test idea"` → confirm a `[from human, …]`
bullet appears under `## Inbox` of `loop/IDEAS.md`; then `git checkout -- loop/IDEAS.md`
to discard the smoke bullet.

- [ ] **Step 3: Mark PR #39 ready (or leave draft for review)**

```bash
git push origin infra/lab-invocation
# optionally: gh pr ready 39
```

Leave the final ready/merge decision to the human.

---

## Self-review notes (author checklist — completed)

- **Spec coverage:** §A.1 `/lab` + positional + resolver → Tasks 1–2; §A.2 `/steer` +
  append helper → Tasks 3–4; §A.3 config-driven autosync + doc updates → Tasks 5–6;
  §A.4 acceptance → Task 7. All Stream-A items mapped.
- **Placeholder scan:** every code/step shows complete code or an exact command +
  expected output. No TBD/TODO.
- **Type/name consistency:** `resolve_direction(name, candidates=None)`,
  `append_idea(text, *, date, ideas_path)`, `append_resume(word, *, date, ideas_path)`,
  `_insert_into_inbox`, `loop/managed-paths.txt`, `.claude/commands/{lab,steer}.md`
  are referenced identically across tasks.
- **Consent invariant (INV-1):** the gate language is created in Task 2 and guarded by
  a test that fails if removed — so the safety property can't silently regress.
