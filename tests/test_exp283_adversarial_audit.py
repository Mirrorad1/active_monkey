"""Guards for the additive audit; original executed tests remain unchanged."""
from dataclasses import replace
from pathlib import Path
import pytest

from experiments.exp283_adversarial_audit import (
    original, fingerprint, assert_equivalent, reserve_outputs, window_statistics,
)


def test_guard_catches_focal_change_invisible_to_event_hash():
    eco = original.Ecology(original.make_config("NEUTRAL", 1), seed=9499)
    before = fingerprint(eco)
    c = eco._creatures[0]
    c.genotype = replace(c.genotype, thermosense_intensity=.123456)
    after = fingerprint(eco)
    assert before["events"] == after["events"]
    with pytest.raises(ValueError, match="genotypes"):
        assert_equivalent(before, after)


def test_guard_catches_rng_advance():
    eco = original.Ecology(original.make_config("NEUTRAL", 1), seed=9499)
    before = fingerprint(eco)
    eco.rng.random()
    with pytest.raises(ValueError, match="RNG"):
        assert_equivalent(before, fingerprint(eco))


def test_window_maps_birth_time_to_after_census_and_retains_dead_births():
    births = [[1, 0, 0, 0., .9], [2, 0, 1, 0., .2], [3, 0, 2, 0., .4], [4, 0, 2, 0., .6]]
    rows = [{"t": 1, "births": 1, "mean_after": .9},
            {"t": 2, "births": 1, "mean_after": .1},
            {"t": 3, "births": 2, "mean_after": .2}]
    got = window_statistics(births, rows, horizon=3, width=2)
    assert got["newborn_count"] == 3
    assert got["newborn_mean"] == pytest.approx(.4)
    assert got["time_mean_living"] == pytest.approx(.15)
    assert got["birth_weighted_living"] == pytest.approx(1/6)
    with pytest.raises(ValueError, match="mismatch"):
        window_statistics(births[:-1], rows, horizon=3, width=2)


@pytest.mark.parametrize("kind", ["json", "txt", "directory", "dangling_symlink"])
def test_output_guard_preserves_existing_evidence(tmp_path, kind):
    prefix = tmp_path / "audit"
    if kind == "directory":
        target = tmp_path / "audit_births"
        target.mkdir()
    else:
        target = prefix.with_suffix(".txt" if kind == "txt" else ".json")
        if kind == "dangling_symlink":
            target.symlink_to(tmp_path / "missing")
        else:
            target.write_text("keep")
    with pytest.raises(FileExistsError):
        reserve_outputs(prefix)
    assert target.exists() or target.is_symlink()
