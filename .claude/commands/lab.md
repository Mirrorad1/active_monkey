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
2. If it exited non-zero (no direction given, or an unknown/ambiguous one), run
   `uv run --python .venv python loop/compose.py --list`, show the options, and ask
   me which direction I meant. If none of the listed cards applies, offer to
   create a new direction by copying `loop/directions/_TEMPLATE.md` to
   `loop/directions/<slug>.md`, filling in the question, ladder, stop condition,
   and STATUS line, then composing with that slug. Do not guess.
3. Show a ONE-line summary of what this run will do: direction, persona, and any
   `--idea`.
4. STOP. Ask me to confirm with "go" (or "edit ...") before you begin iterating.
   Do NOT start iterating and do NOT commit anything yet — this confirm step is the
   human-consent gate that VALIDATION.md §5 depends on (a cron fire is not a human
   resumption; neither is composing a prompt).
5. On my "go": read RESUME.md, the tail of EXPERIMENTS.md, and loop/IDEAS.md, then
   iterate under loop/PROTOCOL.md, self-pacing across wakes via the standard /loop
   dynamic mechanism. Honor loop/VALIDATION.md (binding) throughout.
