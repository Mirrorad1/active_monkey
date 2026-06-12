#!/usr/bin/env bash
# Auto-commit loop-owned pending changes and push to GitHub (which redeploys Pages).
#
# Used by the Claude Code "Stop" hook so that whenever a session/loop turn ends
# with uncommitted loop artifacts, they get committed and pushed. Safe to run anytime:
# it is a no-op when the tree is clean and already up to date, and it refuses
# to touch the repo while a merge/rebase is in progress.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 0

# Never interfere with an in-progress merge / rebase / cherry-pick.
[ -e .git/MERGE_HEAD ] && exit 0
[ -d .git/rebase-merge ] && exit 0
[ -d .git/rebase-apply ] && exit 0

if [ -n "$(git status --porcelain)" ]; then
  # Self-heal generated indexes before staging, so a direction-card edit or
  # experiment entry can never strand the suite red on a staleness guard.
  uv run --python .venv python tools/gen_directions_index.py >/dev/null 2>&1 || true
  uv run --python .venv python -m active_loop.site_data --lab-status >/dev/null 2>&1 || true

  # Keep autosync from sweeping scratch files or parallel-agent edits into a
  # drive-by commit. The main loop owns experiment/site artifacts; PR work and
  # arbitrary docs/code changes should be committed intentionally.
  managed_paths=(
    "EXPERIMENTS.md"
    "experiments"
    "experiments-data.js"
    "lab-status.js"
    "DIRECTIONS.md"
    "loop/IDEAS.md"
    "loop/directions"
    "creature/state"
    "world_model"
    "reports"
    "REPORT.md"
  )
  for path in "${managed_paths[@]}"; do
    [ -e "$path" ] && git add -A -- "$path"
  done

  if ! git diff --cached --quiet; then
    git commit -q -m "auto-sync: experiments + site ($(date -u +%FT%TZ))" || true
  fi
fi

# Push policy (fixed 2026-06-10): only main may push to main. On any other
# branch, push the branch under its own name — an unconditional HEAD:main here
# once swept feature-branch work straight onto origin/main, silently bypassing
# the PR review path that META.md mandates for non-experiment changes.
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" = "main" ]; then
  git push -q origin "HEAD:main" 2>/dev/null || true
elif [ "$branch" != "HEAD" ]; then
  git push -q origin "HEAD:refs/heads/$branch" 2>/dev/null || true
fi
