"""Guards for the committed coalescence library (cards, maps, notes, sample bundles).

These lock the seeded artifacts so they cannot silently rot: every committed card must stay
schema-valid, every sample bundle's referenced files must keep existing, and the seed
generator must stay runnable. Run from the repo root.
"""
from __future__ import annotations

import pathlib

import pytest

from active_loop.coalescence import validate as V
from active_loop.coalescence.schema import load

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_validate_all_clean_on_committed_repo():
    """Every committed card/bundle/manifest validates; nothing fails."""
    report = V.validate_all(str(ROOT))
    assert report["failed"] == [], report["failed"]
    # The seeded library + the existing checkpoint artifact all pass.
    assert len(report["passed"]) >= 11, report["passed"]


@pytest.mark.parametrize("bundle", ["exp222", "exp199", "exp210"])
def test_sample_bundle_refs_exist(bundle):
    """A committed sample bundle must not claim a file it does not have."""
    bdir = ROOT / "experiment_bundles" / bundle
    if not bdir.exists():
        pytest.skip(f"{bundle} bundle not committed")
    report = V.validate_bundle(str(bdir))
    assert report["ok"] is True


def test_seed_mechanism_cards_have_honest_status():
    """functional-valence-dyad-v0 is validated; communication-scaffold-v0 is a scaffold."""
    fv = load(ROOT / "mechanisms/functional-valence-dyad-v0/mechanism_card.json")
    assert fv["status"] == "validated"
    assert fv["source_experiments"]  # non-empty evidence
    comm = load(ROOT / "mechanisms/communication-scaffold-v0/mechanism_card.json")
    assert comm["status"] == "scaffold"
    assert comm["source_experiments"] == []  # no run yet — honestly empty


def test_seed_generator_importable():
    import importlib.util

    path = ROOT / "tools" / "seed_coalescence_cards.py"
    spec = importlib.util.spec_from_file_location("seed_coalescence_cards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert len(mod.SEEDS) >= 10
