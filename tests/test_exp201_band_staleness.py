"""Exp 201 band-staleness — regression + liveness guards for the ecology engine change.

The Exp 201 mechanism replaces exp200's FREE read of world.current_food_optimal with a
per-creature EMA tracker (band_estimate) whose responsiveness + reading-noise are keyed to
thermosense_intensity.  The new forage sub-branch sits INSIDE the exp200 forage path, so these
guards pin:
  - the FORAGE path stays byte-identical when enable_band_staleness is OFF (a fast sentinel on
    the exact code path the new branch sits in — the full 12000-step exp200 WIDE seed23 hash
    502e053989c703616931e4edf7d4ffbc171fcf1a1adc597e33bd9f0b0ec01124 was verified by hand;
    this is the fast unit-test version), AND
  - the new branch is LIVE and NON-TRIVIAL when ON (L16: a gated feature must be exercised —
    precision must actually buy tighter tracking), AND
  - freeze_learning_rate pins learning_rate (confound-killer arm), default OFF is a no-op, AND
  - enable_band_staleness without a drifting food band fails loudly (no silent None-deref).
"""
from __future__ import annotations

import dataclasses as D

import numpy as np
import pytest

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS, FOUNDER


# Fast sentinel on the FORAGE path with band-staleness OFF (== the exp200 code path).
# If a future edit perturbs the exp200 forage branch, this hash changes.
FORAGE_OFF_H400_SEED23 = "7ce19bb4375740b02d3824b8a51db5826a3539bbd165e1516a6a904caacd367e"

_BASE = dict(
    enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
    tolerance_cost_scale=0.0, comfort_amplitude=0.0, thermosense_upkeep_floor=0.0,
    thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
    food_optimal_base=0.5, food_concentration=8.0,
)


def _founder(intensity=0.10):
    return D.replace(FOUNDER, thermosense_intensity=intensity, thermosense_inefficiency=0.2,
                     temperature_tolerance=0.10)


def _forage_cfg(horizon=400, band_staleness=False, period=1500.0, band=0.15,
                intensity=0.10, freeze_lr=False, responsiveness=1.0):
    return D.replace(
        SCENARIOS["balanced"], horizon=horizon, max_population=5000, founder=_founder(intensity),
        enable_food_coupling=True, thermosense_forage_mode=True,
        food_band_width=band, food_optimal_amplitude=0.3, food_optimal_period=period,
        enable_band_staleness=band_staleness, band_responsiveness=responsiveness,
        freeze_learning_rate=freeze_lr, **_BASE,
    )


class TestForagePathRegression:
    def test_forage_off_path_byte_identical(self):
        """band_staleness OFF must reproduce the pinned exp200 forage-path hash."""
        eco = Ecology(_forage_cfg(band_staleness=False), seed=23)
        eco.run()
        assert eco.events_hash() == FORAGE_OFF_H400_SEED23, (
            "exp200 forage path drifted — the band-staleness branch must not affect the OFF path"
        )

    def test_off_path_deterministic(self):
        a = Ecology(_forage_cfg(band_staleness=False), seed=23); a.run()
        b = Ecology(_forage_cfg(band_staleness=False), seed=23); b.run()
        assert a.events_hash() == b.events_hash()


class TestBandStalenessLive:
    def test_branch_fires_and_sets_estimate(self):
        """L16: with band_staleness ON the tracker is actually used (band_estimate set)."""
        eco = Ecology(_forage_cfg(horizon=200, band_staleness=True, period=80.0, band=0.12), seed=7)
        eco.run()
        estimates = [c.policy.band_estimate for c in eco._creatures if c.policy.band_estimate is not None]
        assert len(estimates) > 0, "band-staleness branch never ran — feature is silently dead"

    def test_precision_buys_tighter_tracking(self):
        """A precise tracker stays closer to the drifting band center than a crude one.

        This is the mechanism's load-bearing claim; if it fails the experiment is meaningless.
        """
        def mean_tracker_error(intensity):
            eco = Ecology(_forage_cfg(horizon=300, band_staleness=True, period=80.0,
                                      band=0.12, intensity=intensity), seed=7)
            errs = []
            while eco.t < eco.cfg.horizon and not eco.exploded:
                eco.step()
                for c in eco._alive():
                    be = c.policy.band_estimate
                    if be is not None:
                        errs.append(abs(be - eco.world.current_food_optimal))
                if not eco._alive():
                    break
            return float(np.mean(errs)) if errs else float("nan")

        crude = mean_tracker_error(0.10)
        precise = mean_tracker_error(0.80)
        assert precise < crude, f"precision did not improve tracking (precise={precise}, crude={crude})"

    def test_band_staleness_does_not_touch_off_hash(self):
        """ON vs OFF must differ (the branch genuinely changes behaviour)."""
        on = Ecology(_forage_cfg(horizon=400, band_staleness=True), seed=23); on.run()
        off = Ecology(_forage_cfg(horizon=400, band_staleness=False), seed=23); off.run()
        assert on.events_hash() != off.events_hash()


class TestFreezeLearningRate:
    def test_freeze_pins_learning_rate(self):
        """With freeze_learning_rate, every creature keeps the founder learning_rate."""
        cfg = _forage_cfg(horizon=600, band_staleness=True, period=80.0, band=0.12, freeze_lr=True)
        eco = Ecology(cfg, seed=5)
        eco.run()
        founder_lr = cfg.founder.learning_rate
        lrs = {round(c.genotype.learning_rate, 9) for c in eco._creatures}
        assert lrs == {round(founder_lr, 9)}, f"learning_rate drifted despite freeze: {sorted(lrs)[:5]}"

    def test_unfrozen_learning_rate_drifts(self):
        cfg = _forage_cfg(horizon=600, band_staleness=True, period=80.0, band=0.12, freeze_lr=False)
        eco = Ecology(cfg, seed=5)
        eco.run()
        lrs = {round(c.genotype.learning_rate, 6) for c in eco._creatures}
        assert len(lrs) > 1, "learning_rate did not drift without freeze (mutation may be broken)"


class TestBandStalenessGuards:
    def test_requires_food_and_temperature(self):
        """enable_band_staleness without a drifting food band must fail loudly."""
        cfg = D.replace(
            SCENARIOS["balanced"], horizon=50, founder=_founder(),
            enable_band_staleness=True, enable_food_coupling=False, enable_temperature=False,
        )
        with pytest.raises(AssertionError):
            Ecology(cfg, seed=1)
