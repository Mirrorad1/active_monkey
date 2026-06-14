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
HTML = ["index.html", "journey.html", "open_problem.html", "sense-evolution.html", "math.html"]


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
