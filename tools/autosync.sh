#!/usr/bin/env bash
# Auto-commit any pending changes and push to GitHub (which redeploys Pages).
#
# Used by the Claude Code "Stop" hook so that whenever a session/loop turn ends
# with uncommitted work, it gets committed and pushed. Safe to run anytime:
# it is a no-op when the tree is clean and already up to date, and it refuses
# to touch the repo while a merge/rebase is in progress.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 0

# Never interfere with an in-progress merge / rebase / cherry-pick.
[ -e .git/MERGE_HEAD ] && exit 0
[ -d .git/rebase-merge ] && exit 0
[ -d .git/rebase-apply ] && exit 0

if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -q -m "auto-sync: experiments + site ($(date -u +%FT%TZ))" || true
fi

# Push whatever is ahead of origin/main; harmless no-op when already in sync.
git push -q origin "HEAD:main" 2>/dev/null || true
