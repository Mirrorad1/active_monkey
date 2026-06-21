"""tests/test_acoustic.py — the gated acoustic substrate (direction: acoustic-ecology).

Guards:
  - OFF is byte-identical to the patch-mosaic ring golden (the substrate is unchanged
    unless enable_acoustic_field is set).
  - FIELD ON but acoustic_response OFF is STILL byte-identical (the field is purely
    observational; only acting on it changes behavior) — separates "does sound carry
    info" from "does acting on it help".
  - Acoustic response ON (gifted hearing) DOES change the trajectory (the channel is
    causally live).
  - Determinism holds with the field on.
  - The physics behaves: attenuation falls with distance; high frequencies absorb
    faster than low; propagation delay is respected.
  - No semantic leakage: agents are driven only by per-band intensity helpers; the
    public API exposes no source-position / identity / event-type accessor.
"""
import math

import numpy as np
import pytest

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
from ecology import acoustic as ac


# The ring golden from tests/test_patchmosaic.py (T10) — acoustics OFF must match it.
_RING_GOLDEN_HASH = "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"


# ---------------------------------------------------------------------------
# A1: OFF is byte-identical to the ring golden.
# ---------------------------------------------------------------------------
def test_acoustic_off_byte_identical():
    cfg_default = PatchMosaicConfig(horizon=200)
    cfg_off = PatchMosaicConfig(horizon=200, enable_acoustic_field=False)
    h_default = PatchMosaicSim(cfg_default, seed=1).run()["events_hash"]
    h_off = PatchMosaicSim(cfg_off, seed=1).run()["events_hash"]
    assert h_default == _RING_GOLDEN_HASH, (
        f"default hash changed from ring golden: {h_default!r}")
    assert h_off == _RING_GOLDEN_HASH, (
        f"enable_acoustic_field=False changed the rng stream: {h_off!r}")


# ---------------------------------------------------------------------------
# A2: FIELD ON, RESPONSE OFF is still byte-identical (passive field).
# A migration-active regime, so the difference (if any) WOULD show up.
# ---------------------------------------------------------------------------
def test_acoustic_field_passive_byte_identical():
    common = dict(horizon=300, migration_rate_prey=0.1, migration_rate_pred=0.1,
                  n_prey0_per_patch=30, n_pred0_per_patch=8)
    base = PatchMosaicSim(PatchMosaicConfig(**common), seed=3).run()["events_hash"]
    field = PatchMosaicSim(
        PatchMosaicConfig(enable_acoustic_field=True, acoustic_response=False, **common),
        seed=3).run()["events_hash"]
    assert base == field, (
        "field present but acoustic_response=False must NOT change the trajectory "
        "(the acoustic field is observational only)")


# ---------------------------------------------------------------------------
# A3: RESPONSE ON (gifted prey hearing) DOES change the trajectory.
# ---------------------------------------------------------------------------
def test_acoustic_response_changes_behavior():
    common = dict(horizon=300, migration_rate_prey=0.2, migration_rate_pred=0.1,
                  n_prey0_per_patch=30, n_pred0_per_patch=8)
    passive = PatchMosaicSim(
        PatchMosaicConfig(enable_acoustic_field=True, acoustic_response=False, **common),
        seed=5).run()["events_hash"]
    gifted = PatchMosaicSim(
        PatchMosaicConfig(enable_acoustic_field=True, acoustic_response=True,
                          gifted_hearing="prey", **common),
        seed=5).run()["events_hash"]
    assert passive != gifted, (
        "gifted hearing (acoustic_response=True) must change the trajectory — "
        "the channel is causally live")


# ---------------------------------------------------------------------------
# A4: Determinism with the field on.
# ---------------------------------------------------------------------------
def test_acoustic_determinism():
    cfg = PatchMosaicConfig(horizon=200, enable_acoustic_field=True,
                            acoustic_response=True, gifted_hearing="prey",
                            migration_rate_prey=0.15, hearing_precision=0.4)
    h1 = PatchMosaicSim(cfg, seed=2).run()["events_hash"]
    h2 = PatchMosaicSim(cfg, seed=2).run()["events_hash"]
    assert h1 == h2, "field-on run must be deterministic under a fixed seed"


# ---------------------------------------------------------------------------
# A5: result carries the acoustic analysis block.
# ---------------------------------------------------------------------------
def test_acoustic_result_block():
    cfg = PatchMosaicConfig(horizon=200, enable_acoustic_field=True,
                            migration_rate_prey=0.1, migration_rate_pred=0.1)
    res = PatchMosaicSim(cfg, seed=4).run()
    a = res["acoustic"]
    for k in ("mi_banded_bits", "mi_scalar_bits", "false_positive_rate",
              "false_negative_rate", "prey_capture_hazard", "attenuation_curve"):
        assert k in a, f"missing acoustic metric {k!r}"
    assert a["mi_banded_bits"] >= 0.0 and a["mi_scalar_bits"] >= 0.0


# ---------------------------------------------------------------------------
# A6: physics — attenuation FALLS with distance, and is NOT perfect at long range.
# ---------------------------------------------------------------------------
def test_attenuation_falls_with_distance():
    neighbors = [[(i - 1) % 8, (i + 1) % 8] for i in range(8)]
    for i in range(8):
        neighbors[i].sort()
    field = ac.AcousticField(ac.AcousticConfig(), neighbors)
    curve = dict(field.attenuation_curve())
    dists = sorted(curve)
    # strictly decreasing gain with distance
    for a, b in zip(dists, dists[1:]):
        assert curve[b] < curve[a], f"gain did not fall from d={a} to d={b}"
    # not perfect at long range: far gain is a small fraction of near gain
    assert curve[max(dists)] < 0.25 * curve[min(dists)], "detection too generous at range"


# ---------------------------------------------------------------------------
# A7: physics — HIGH frequencies absorb faster than LOW over distance.
# ---------------------------------------------------------------------------
def test_high_freq_absorbs_faster():
    neighbors = [[(i - 1) % 8, (i + 1) % 8] for i in range(8)]
    for i in range(8):
        neighbors[i].sort()
    acfg = ac.AcousticConfig()
    field = ac.AcousticField(acfg, neighbors)
    # gain[listener, source, band]; compare band ratios near vs far.
    near, far = 1, 4
    # find a source at hop distance `far` and one at `near` from patch 0
    src_near = next(j for j in range(8) if round(field.hops[0, j]) == near)
    src_far = next(j for j in range(8) if round(field.hops[0, j]) == far)
    g_near = field.gain[0, src_near]
    g_far = field.gain[0, src_far]
    # ratio far/near should be SMALLER for high band than low band (high decays faster)
    ratio_low = g_far[0] / g_near[0]
    ratio_high = g_far[2] / g_near[2]
    assert ratio_high < ratio_low, (
        "high band should retain LESS of its near intensity at range than low band")


# ---------------------------------------------------------------------------
# A8: propagation DELAY — a sound emitted now is not audible at a far patch
# instantly; it arrives after round(distance / sound_speed) steps.
# ---------------------------------------------------------------------------
def test_propagation_delay():
    neighbors = [[(i - 1) % 8, (i + 1) % 8] for i in range(8)]
    for i in range(8):
        neighbors[i].sort()
    acfg = ac.AcousticConfig(sound_speed=1.0)  # 1 hop/step => delay == hop distance
    field = ac.AcousticField(acfg, neighbors)
    emit = np.zeros((8, 3))
    emit[4] = [1.0, 1.0, 1.0]  # patch 4 (hop distance 4 from patch 0) emits at t=0
    field.push_emissions(0, emit)
    # at t=0..3 patch 0 hears nothing (sound still in flight)
    for t in range(0, 4):
        assert field.received(t)[0].sum() == 0.0, f"sound arrived early at t={t}"
    # at t=4 (== distance/speed) patch 0 hears it
    assert field.received(4)[0].sum() > 0.0, "sound never arrived at the expected delay"


# ---------------------------------------------------------------------------
# A9: no semantic leakage — public API exposes no source-position / identity /
# event-type accessor; agents are driven only by per-band intensity.
# ---------------------------------------------------------------------------
def test_no_semantic_leakage():
    public = [n for n in dir(PatchMosaicSim) if not n.startswith("_")]
    for n in public:
        ln = n.lower()
        assert not any(b in ln for b in ("source_pos", "predator_pos", "event_type",
                                         "identity", "label")), (
            f"public API leaks a semantic acoustic accessor: {n}")


# ---------------------------------------------------------------------------
# A10: mutual_information is 0 for independent streams and >0 for a coupled one.
# ---------------------------------------------------------------------------
def test_mutual_information_sanity():
    rng = np.random.default_rng(0)
    n = 4000
    indep_obs = list(rng.integers(0, 2, n))
    indep_hid = list(rng.integers(0, 3, n))
    assert ac.mutual_information_bits(indep_obs, indep_hid) < 0.02
    hid = list(rng.integers(0, 2, n))
    # obs copies hidden 90% of the time => clearly >0 bits
    obs = [h if rng.random() < 0.9 else 1 - h for h in hid]
    assert ac.mutual_information_bits(obs, hid) > 0.3
