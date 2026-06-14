# Stream B (stable assets) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Declutter the repo root by moving the four low-churn web code-assets (`am.css`, `am-live.js`, `am-shared.jsx`, `tweaks-panel.jsx`) into `site/` and repointing the HTML, without breaking the live GitHub-Pages deploy.

**Architecture:** GitHub Pages serves the repo ROOT (`Mirrorad1/active_monkey`, no CNAME/Action), so the HTML entry pages stay at root and only the assets move to `site/`. The loop-generated `experiments-data.js` and `lab-status.js` (regenerated every iteration via `site_data.py`, listed in `loop/managed-paths.txt`) **stay at root** this round — deferred to a tight follow-up to avoid colliding with the live loop's per-iteration writes. Therefore `site_data.py`, `meta_monkey/collect_iteration.py`, `loop/managed-paths.txt`, and their tests are NOT touched.

**Cache-bust:** keep `?v=5` unchanged — the path change (root → `site/`) is itself a new URL, so it busts the cache; and `test_asset_cache_versions_are_consistent` requires all `?v=` numbers to match (`experiments-data.js?v=5` stays at root), so bumping would break it. Only paths change.

**Tech stack:** static HTML/CSS/JS, pytest 8 (`uv run --python .venv pytest`), `git mv`.

**Branch:** `infra/site-relocation` (worktree `/Users/mirro/Projects/active-loop-site`).

---

## Exact reference map (verified against `origin/main` @ 5e9d982)

Move (root → `site/`): `am.css`, `am-live.js`, `am-shared.jsx`, `tweaks-panel.jsx`.
Keep at root: `index.html`, `journey.html`, `open_problem.html`, `sense-evolution.html`, `reports/index.html`, `.nojekyll`, `favicon.ico`, `assets/` (tidy dir, referenced as `assets/monkey-*.png`), and the generated `experiments-data.js` / `lab-status.js` (deferred).

HTML references to update (path only, keep `?v=5`):
- `index.html:20` `am.css?v=5` → `site/am.css?v=5`
- `open_problem.html:14` `am.css?v=5` → `site/am.css?v=5`
- `sense-evolution.html:19` `am.css?v=5` → `site/am.css?v=5`
- `journey.html:14` `am.css?v=5` → `site/am.css?v=5`
- `journey.html:182` `am-live.js?v=5` → `site/am-live.js?v=5`
- `journey.html:183` `tweaks-panel.jsx?v=5` → `site/tweaks-panel.jsx?v=5`
- `journey.html:184` `am-shared.jsx?v=5` → `site/am-shared.jsx?v=5`

`journey.html:181` `experiments-data.js?v=5` — **unchanged** (stays at root). `reports/index.html` references none of the four. `am-live.js` fetches only the absolute `RAW_URL` GitHub URL and reads the `window.AM_EXPERIMENTS` global (set by the root `experiments-data.js` `<script>` that loads before it in `journey.html`) — moving `am-live.js` does not affect either.

---

## Task 1: Move the four assets to `site/` and repoint HTML (TDD)

**Files:**
- Create: `tests/test_site_assets.py`
- Move: `am.css`, `am-live.js`, `am-shared.jsx`, `tweaks-panel.jsx` → `site/`
- Modify: `index.html`, `journey.html`, `open_problem.html`, `sense-evolution.html`

- [ ] **Step 1: Write the failing guard test** `tests/test_site_assets.py`

```python
"""Guard: web code-assets live under site/ and HTML references them there.

Stream B (stable assets) moved am.css / am-live.js / am-shared.jsx /
tweaks-panel.jsx into site/. A regression that re-introduces a bare root
reference, or a missing site/ file, would break the live GitHub-Pages deploy.
(experiments-data.js / lab-status.js intentionally remain at root this round —
the loop regenerates them every iteration.)

Run:  uv run --python .venv pytest tests/test_site_assets.py -q
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
MOVED = ["am.css", "am-live.js", "am-shared.jsx", "tweaks-panel.jsx"]
HTML = ["index.html", "journey.html", "open_problem.html", "sense-evolution.html"]


def test_moved_assets_live_under_site_only():
    for f in MOVED:
        assert (ROOT / "site" / f).is_file(), f"site/{f} is missing"
        assert not (ROOT / f).exists(), f"stale root copy of {f} still present"


def test_html_references_moved_assets_under_site():
    for page in HTML:
        text = (ROOT / page).read_text(encoding="utf-8")
        for f in MOVED:
            for m in re.finditer(rf'(?:src|href)="([^"]*{re.escape(f)}[^"?]*)', text):
                ref = m.group(1)
                assert ref.startswith("site/"), (
                    f"{page} references {f} at a non-site path: {ref!r}"
                )
```

- [ ] **Step 2: Run it — expect failure**

Run: `cd /Users/mirro/Projects/active-loop-site && uv run --python .venv pytest tests/test_site_assets.py -q`
Expected: FAIL — assets still at root / HTML still bare-references them.

- [ ] **Step 3: Move the four files with `git mv` (preserves history)**

```bash
cd /Users/mirro/Projects/active-loop-site
mkdir -p site
git mv am.css site/am.css
git mv am-live.js site/am-live.js
git mv am-shared.jsx site/am-shared.jsx
git mv tweaks-panel.jsx site/tweaks-panel.jsx
```

- [ ] **Step 4: Repoint the HTML references (path only, keep `?v=5`)**

Make exactly these replacements (each is a unique string in its file):
- `index.html`: `href="am.css?v=5"` → `href="site/am.css?v=5"`
- `open_problem.html`: `href="am.css?v=5"` → `href="site/am.css?v=5"`
- `sense-evolution.html`: `href="am.css?v=5"` → `href="site/am.css?v=5"`
- `journey.html`: `href="am.css?v=5"` → `href="site/am.css?v=5"`
- `journey.html`: `src="am-live.js?v=5"` → `src="site/am-live.js?v=5"`
- `journey.html`: `src="tweaks-panel.jsx?v=5"` → `src="site/tweaks-panel.jsx?v=5"`
- `journey.html`: `src="am-shared.jsx?v=5"` → `src="site/am-shared.jsx?v=5"`

Do NOT change `journey.html`'s `experiments-data.js?v=5` line. Do NOT change `assets/...` icon/logo refs.

- [ ] **Step 5: Run the guard test + the cache-consistency test**

Run: `cd /Users/mirro/Projects/active-loop-site && uv run --python .venv pytest tests/test_site_assets.py tests/test_site_data.py::test_asset_cache_versions_are_consistent -q`
Expected: PASS (3 passed) — assets under `site/`, HTML repointed, all `?v=` still `5`.

- [ ] **Step 6: Full fast suite (no regressions)**

Run: `cd /Users/mirro/Projects/active-loop-site && uv run --python .venv pytest -p no:warnings -q`
Expected: exit 0, zero failures.

- [ ] **Step 7: Deploy smoke-test (serve root like Pages, verify URLs resolve)**

```bash
cd /Users/mirro/Projects/active-loop-site
python3 -m http.server 8911 >/tmp/pages_smoke.log 2>&1 &
SRV=$!; sleep 1
for url in index.html journey.html open_problem.html sense-evolution.html \
           site/am.css site/am-live.js site/am-shared.jsx site/tweaks-panel.jsx \
           experiments-data.js; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8911/$url")
  echo "$code  $url"
done
# old root paths should now 404:
for url in am.css am-live.js; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8911/$url")
  echo "$code  (expect 404) $url"
done
kill $SRV
```
Expected: `200` for all four HTML pages, all four `site/...` assets, and root `experiments-data.js`; `404` for the old root `am.css` / `am-live.js`.

- [ ] **Step 8: Commit**

```bash
cd /Users/mirro/Projects/active-loop-site
git add -A
git commit -m "refactor(site): move stable web assets (css/js/jsx) into site/, repoint HTML"
```

---

## Self-review checklist (author — completed)
- Spec coverage: stable-asset subset of spec §B (web → site/, HTML at root for Pages-from-root). Generated files + governance/cruft explicitly deferred and named.
- No placeholders: every step has exact commands/edits + expected output.
- Cache test interaction handled (keep `?v=5`, change paths only).
- Durable guard: `test_site_assets.py` fails if a bare root reference or missing `site/` file regresses the deploy.
- Loop-collision avoided: untouched `site_data.py` / `managed-paths.txt` / generated files → no fight with the live loop's per-iteration writes.
