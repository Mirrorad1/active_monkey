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
    # The seeded library (6 mechanisms + 5 boundary notes + 3 geometry maps + adapter +
    # scorer) + the existing checkpoint artifact all pass.
    assert len(report["passed"]) >= 21, report["passed"]


@pytest.mark.parametrize("bundle", ["exp222", "exp199", "exp210"])
def test_sample_bundle_refs_exist(bundle):
    """A committed sample bundle must not claim a file it does not have."""
    bdir = ROOT / "experiment_bundles" / bundle
    if not bdir.exists():
        pytest.skip(f"{bundle} bundle not committed")
    report = V.validate_bundle(str(bdir))
    assert report["ok"] is True


def test_seed_mechanism_cards_have_honest_status():
    """Each mechanism card's status matches its evidence (no overclaiming)."""
    expected = {
        "functional-valence-dyad-v0": "validated",
        "recipe-symmetry-breaking-v0": "validated",
        "meta-calibration-n3-v0": "validated",
        "online-structure-growth-v0": "validated",
        "identity-n4-monitor-v0": "constrained",
        "communication-scaffold-v0": "scaffold",
    }
    for mid, status in expected.items():
        card = load(ROOT / "mechanisms" / mid / "mechanism_card.json")
        assert card["status"] == status, (mid, card["status"])
        if status == "scaffold":
            assert card["source_experiments"] == []  # no run yet — honestly empty
        else:
            assert card["source_experiments"]  # validated/constrained need evidence


def test_recipe_and_collapse_are_two_halves():
    """The flagship finding: the collapse boundary and the recipe that breaks its symmetry."""
    collapse = load(ROOT / "boundary_notes/disembodied-stream-collapse-v0.json")
    assert collapse["artifact_type"] == "boundary_note"
    assert 31 in collapse["source_experiments"]
    commitment = load(ROOT / "boundary_notes/identity-n4-commitment-v0.json")
    assert commitment["artifact_type"] == "boundary_note"


def test_seed_generator_importable():
    import importlib.util

    path = ROOT / "tools" / "seed_coalescence_cards.py"
    spec = importlib.util.spec_from_file_location("seed_coalescence_cards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert len(mod.SEEDS) >= 10


def test_seed_generator_is_idempotent_against_committed_library():
    """Lock the committed library to its generator source so a re-run of
    tools/seed_coalescence_cards.py can never SILENTLY REVERT a committed card.

    Exactly one artifact is a KNOWN, DOCUMENTED drift:
    local-gradient-wall-locomotion-v0 was hand-EXPANDED post-seed (Exp 235-237
    -> 235-246) and its generator source was never updated, so the generator
    would shrink it back. We tolerate that one LOUDLY (assert it stayed a
    schema-valid SUPERSET — expansion, not corruption) and lock every other
    card, including newly added ones, to byte-exact agreement. (Caught while
    closing the identity-ecological direction: adding new SEEDS and running the
    generator reverted the hand-expanded note.)
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "seed_coalescence_cards", ROOT / "tools" / "seed_coalescence_cards.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    KNOWN_DRIFT = {"boundary_notes/local-gradient-wall-locomotion-v0.json"}
    drifted = []
    for rel, card in gen.SEEDS:
        committed = load(ROOT / rel)
        if rel in KNOWN_DRIFT:
            gen_exps = set(card.to_dict().get("source_experiments", []))
            committed_exps = set(committed.get("source_experiments", []))
            assert gen_exps.issubset(committed_exps), (
                f"{rel}: committed file is no longer a superset of the generator "
                "source — the documented hand-expansion may have been corrupted"
            )
            continue
        if card.to_dict() != committed:
            drifted.append(rel)

    assert not drifted, (
        "committed coalescence artifacts diverge from "
        "tools/seed_coalescence_cards.py (re-running the generator would silently "
        "revert these):\n" + "\n".join(drifted)
    )
