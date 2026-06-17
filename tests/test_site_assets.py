"""Guard: public web source lives under site/.

The root HTML files are GitHub-Pages deploy artifacts. Authoring sources live in
site/pages, generated browser data lives in site/data, visual assets live in
site/assets, and shared client code is grouped by purpose. A regression that
reintroduces root-owned JS/assets or stale generated pages makes the repo look
like an ad-hoc static dump again.

Run:  uv run --python .venv pytest tests/test_site_assets.py -q
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
PAGES = ["index.html", "journey.html", "open_problem.html", "sense-evolution.html", "math.html"]
DATA = ["experiments-data.js", "lab-status.js", "math-data.js"]
SHARED = [
    "styles/am.css",
    "runtime/am-live.js",
    "components/am-shared.jsx",
    "components/tweaks-panel.jsx",
]
ASSETS = ["monkey.png", "monkey-32.png", "monkey-180.png", "monkey-512.png", "monkey-raw.png"]


def _read_source_page(page: str) -> str:
    return (ROOT / "site" / "pages" / page).read_text(encoding="utf-8")


def test_site_source_tree_owns_web_sources():
    for page in PAGES:
        assert (ROOT / "site" / "pages" / page).is_file(), f"site/pages/{page} is missing"
    for data in DATA:
        assert (ROOT / "site" / "data" / data).is_file(), f"site/data/{data} is missing"
    for shared in SHARED:
        assert (ROOT / "site" / shared).is_file(), f"site/{shared} is missing"
    for asset in ASSETS:
        assert (ROOT / "site" / "assets" / asset).is_file(), f"site/assets/{asset} is missing"
    assert (ROOT / "tools" / "site" / "make_logo.py").is_file(), "tools/site/make_logo.py is missing"


def test_root_has_only_deploy_pages_not_web_sources():
    for data in DATA:
        assert not (ROOT / data).exists(), f"stale root data file still present: {data}"
    assert not (ROOT / "assets").exists(), "stale root assets/ directory still present"
    for old in ("am.css", "am-live.js", "am-shared.jsx", "tweaks-panel.jsx"):
        assert not (ROOT / "site" / old).exists(), f"uncategorized site/{old} still present"
    assert not list((ROOT / "site" / "assets").glob("*.py")), "site/assets should contain static assets only"


def test_root_deploy_pages_match_site_sources():
    from tools.site.build_static import deploy_outputs  # type: ignore

    outputs = deploy_outputs(ROOT)
    assert set(outputs) == set(PAGES)
    for page in PAGES:
        committed = (ROOT / page).read_text(encoding="utf-8")
        assert outputs[page] == committed, (
            f"{page} is stale; regenerate from site/pages/{page} with "
            "`uv run --python .venv python tools/site/build_static.py`"
        )


def test_source_html_references_categorized_site_paths():
    expected_refs = [
        "site/styles/am.css",
        "site/assets/monkey-",
    ]
    forbidden_refs = [
        'href="assets/',
        'src="assets/',
        "active_monkey/assets/",
        "site/am.css",
        "site/am-live.js",
        "site/am-shared.jsx",
        "site/tweaks-panel.jsx",
        'src="experiments-data.js',
        'src="lab-status.js',
        'src="math-data.js',
    ]
    for page in PAGES:
        text = _read_source_page(page)
        for ref in expected_refs:
            assert ref in text, f"site/pages/{page} does not reference {ref}"
        for ref in forbidden_refs:
            assert ref not in text, f"site/pages/{page} still references {ref}"


def test_shared_asset_references_stay_under_site():
    for page in PAGES:
        text = _read_source_page(page)
        for f in ("am.css", "am-live.js", "am-shared.jsx", "tweaks-panel.jsx"):
            for m in re.finditer(rf'(?:src|href)="([^"]*{re.escape(f)}[^"?]*)', text):
                ref = m.group(1)
                assert ref.startswith(("site/styles/", "site/runtime/", "site/components/")), (
                    f"site/pages/{page} references {f} at an uncategorized path: {ref!r}"
                )
