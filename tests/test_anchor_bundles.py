"""Guard: the committed milestone anchor bundles stay schema-valid and ref-complete.

Each anchor experiment materialises a citable ExperimentBundle for a seeded card
(see tools/export_anchor_bundles.py). These guards fail loudly if a bundle's manifest
rots or starts referencing a file it does not have.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from active_loop.coalescence import validate as V

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _anchor_ids():
    path = ROOT / "tools" / "export_anchor_bundles.py"
    spec = importlib.util.spec_from_file_location("export_anchor_bundles", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [f"exp{n}" for n, _ in mod.ANCHORS]


ANCHOR_IDS = _anchor_ids()


def test_anchors_nonempty():
    assert len(ANCHOR_IDS) >= 8


@pytest.mark.parametrize("eid", ANCHOR_IDS)
def test_anchor_bundle_validates(eid):
    bdir = ROOT / "experiment_bundles" / eid
    if not bdir.exists():
        pytest.skip(f"{eid} bundle not committed")
    report = V.validate_bundle(str(bdir))
    assert report["ok"] is True


def test_validate_all_clean_with_anchor_bundles():
    report = V.validate_all(str(ROOT))
    assert report["failed"] == [], report["failed"]
