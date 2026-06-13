"""Exp 202 shuffled-order + strip auditor — regression + liveness guards.

The escape rests on a gated SHUFFLED processing order (id-order neutraliser) + a gated strip auditor,
consume() unchanged. These guards pin: the OFF path is byte-identical to exp194-201 (no shuffle rng,
no strip telemetry), and the ON path is genuinely live (shuffle changes the order/hash; strip logged).
"""
from __future__ import annotations

import dataclasses as D

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS, FOUNDER

_BASE = dict(
    enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
    thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05, thermosense_noise_base=0.5,
    thermal_avoidance_weight=4.0, food_optimal_base=0.5, food_optimal_amplitude=0.3,
    food_optimal_period=1500.0, food_concentration=14.0, food_band_width=0.08,
    enable_food_coupling=True, thermosense_forage_mode=True,
)


def _cfg(horizon=400, shuffle=False, track=False):
    f = D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2, temperature_tolerance=0.10)
    return D.replace(SCENARIOS["balanced"], horizon=horizon, max_population=6000, founder=f,
                     regen_rate=0.08, shuffle_creature_order=shuffle, track_band_strip=track, **_BASE)


def test_off_path_byte_identical():
    """shuffle + track both OFF must equal a plain run with neither flag (no rng/telemetry drift)."""
    base = Ecology(_cfg(), 38); base.run()
    off = Ecology(_cfg(shuffle=False, track=False), 38); off.run()
    assert off.events_hash() == base.events_hash()
    assert off.strip_log == []


def test_track_band_strip_not_in_hash():
    """Turning ON the strip auditor (telemetry only) must NOT change events_hash."""
    no_track = Ecology(_cfg(shuffle=False, track=False), 38); no_track.run()
    track = Ecology(_cfg(shuffle=False, track=True), 38); track.run()
    assert track.events_hash() == no_track.events_hash()      # telemetry is not in the hash
    assert len(track.strip_log) > 0                           # but it IS populated (live)


def test_shuffle_changes_order_and_hash():
    off = Ecology(_cfg(shuffle=False), 38); off.run()
    on = Ecology(_cfg(shuffle=True), 38); on.run()
    assert on.events_hash() != off.events_hash()              # shuffled order genuinely differs


def test_shuffle_deterministic_same_seed():
    a = Ecology(_cfg(shuffle=True, track=True), 38); a.run()
    b = Ecology(_cfg(shuffle=True, track=True), 38); b.run()
    assert a.events_hash() == b.events_hash()


def test_strip_auditor_reports_depletion():
    """In the depleting-band regime the auditor must record genuine strip (>0) — the go/no-go."""
    eco = Ecology(_cfg(horizon=800, shuffle=True, track=True), 38); eco.run()
    late = [s["strip"] for s in eco.strip_log if s["t"] >= 600]
    assert late and (sum(x > 0 for x in late) / len(late)) > 0.5
