# loop/ — the modular self-guided research loop (loop B)

This directory is the **prompt operating system** for the Claude-driven experiment loop.
It exists so the research can (1) resume across sessions, (2) be steered modularly —
swap in a direction, a persona, or a human idea without rewriting the whole prompt —
and (3) stay honest by construction.

## Parts

| File | Role |
|---|---|
| `PREMISE.md` | the world model: what we're building, the RECIPE, the ceilings. Stable. |
| `PROTOCOL.md` | the per-iteration procedure (hypothesis → predict → run → validate → log). |
| `VALIDATION.md` | the brutal-honesty rules. Binding. Every experiment entry must satisfy it. |
| `IDEAS.md` | human inbox. The loop reads it every iteration; humans drop ideas/redirections here. |
| `directions/` | pluggable research directions (what to work on). One card per direction. |
| `personas/` | pluggable working styles (how to approach it). One card per persona. |
| `compose.py` | assembles a ready-to-paste `/loop` prompt from the parts. |

## Use

```bash
# default direction + persona:
uv run --python .venv python loop/compose.py

# pick modules, inject a one-off idea:
uv run --python .venv python loop/compose.py \
  --direction transfer --persona skeptic --idea "what if the anchor can be weaker?"

# list available modules:
uv run --python .venv python loop/compose.py --list
```

Paste the output into a fresh Claude session (it is a `/loop` command). The session will
read `RESUME.md` + `EXPERIMENTS.md`, then iterate under the chosen direction/persona,
honoring `VALIDATION.md` and checking `IDEAS.md` each wake.

## Extending

Add a new direction or persona by copying `_TEMPLATE.md` in the respective folder.
Keep cards short (< 40 lines). The card is the module; `compose.py` does no magic —
it just concatenates, so anything you can say in a card steers the loop.
