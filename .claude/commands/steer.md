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
