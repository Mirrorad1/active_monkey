"""tests/test_ecology_senescence.py — Fast deterministic tests for the senescence
model added in Exp 195.

L16 guard: test_senescence_off_reproduces_exp194 is the key regression test — it
ensures the OFF path is byte-identical to the committed Exp 194 run.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP194_SUMMARY = Path(__file__).parent.parent / (
    "experiments/outputs/exp194_n5_homeostatic_population/balanced_seed0/summary.json"
)
EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"

# Canonical treatment parameters (same as the experiment script)
SENES_PARAMS = dict(
    enable_senescence=True,
    senescence_onset0=155.0,
    senescence_onset_frailty=0.65,
    senescence_rate_frailty=2.0,
    senescence_base=0.002,
    senescence_self_maintenance=1.5,
    senescence_exp=1.5,
)


# ---------------------------------------------------------------------------
# L16 guard — the key regression test
# ---------------------------------------------------------------------------
class TestSenescenceOffReproducesExp194:
    """CRITICAL: ensure enable_senescence=False is byte-identical to Exp 194."""

    def test_senescence_off_reproduces_exp194(self):
        """OFF path must reproduce Exp 194: fp=170, births=628, deaths=458, hash match."""
        # Load committed hash from the actual Exp 194 output
        with open(EXP194_SUMMARY) as f:
            exp194 = json.load(f)
        committed_hash = exp194["events_hash"]

        # Run with senescence OFF (default)
        eco = Ecology(SCENARIOS["balanced"], seed=0)
        summary = eco.run()

        assert summary["final_pop"] == 170, (
            f"OFF path: expected final_pop=170, got {summary['final_pop']}"
        )
        assert summary["births"] == 628, (
            f"OFF path: expected births=628, got {summary['births']}"
        )
        assert summary["deaths"] == 458, (
            f"OFF path: expected deaths=458, got {summary['deaths']}"
        )
        assert summary["events_hash"] == committed_hash, (
            f"OFF path hash mismatch: got {summary['events_hash']!r}, "
            f"expected {committed_hash!r}. The senescence-OFF branch leaked!"
        )
        # Also check the hardcoded constant matches
        assert committed_hash == EXP194_HASH, (
            "The hardcoded EXP194_HASH constant is wrong — update it."
        )


# ---------------------------------------------------------------------------
# Core mechanism tests
# ---------------------------------------------------------------------------
class TestSenescenceFires:
    """Senescence deaths must appear in treatment arm."""

    def test_senescence_fires(self):
        """In an abundant setting with long horizon, senescence deaths > 0."""
        cfg = replace(SCENARIOS["balanced"], **SENES_PARAMS)
        eco = Ecology(cfg, seed=0)
        summary = eco.run()
        senes = summary["cause_of_death_tally"].get("senescence", 0)
        assert senes > 0, (
            f"Expected senescence deaths > 0 in treatment/balanced/seed0, got {senes}"
        )

    def test_no_senescence_death_in_control(self):
        """Control arm (OFF) must have 0 senescence deaths."""
        for scenario in ["balanced", "scarce", "overabundant"]:
            eco = Ecology(SCENARIOS[scenario], seed=0)
            summary = eco.run()
            senes = summary["cause_of_death_tally"].get("senescence", 0)
            assert senes == 0, (
                f"Control/OFF {scenario}/seed0 has {senes} senescence deaths — leak!"
            )


class TestTwoCausesCoexist:
    """Both starvation and senescence deaths must appear in >= 1 scenario."""

    def test_two_causes_coexist(self):
        """Treatment balanced has both starvation and senescence deaths."""
        any_coexist = False
        for seed in [0, 1, 2]:
            cfg = replace(SCENARIOS["balanced"], **SENES_PARAMS)
            eco = Ecology(cfg, seed=seed)
            summary = eco.run()
            cod = summary["cause_of_death_tally"]
            if cod.get("starvation", 0) > 0 and cod.get("senescence", 0) > 0:
                any_coexist = True
                break
        assert any_coexist, (
            "No seed of treatment/balanced had both starvation and senescence deaths"
        )


class TestSenescenceDeterministic:
    """Senescence must add no RNG draws — same seed gives identical hash."""

    def test_senescence_deterministic(self):
        """Two runs with senescence ON and same seed must produce identical event hash."""
        cfg = replace(SCENARIOS["balanced"], **SENES_PARAMS)
        eco1 = Ecology(cfg, seed=0)
        s1 = eco1.run()
        eco2 = Ecology(cfg, seed=0)
        s2 = eco2.run()
        assert s1["events_hash"] == s2["events_hash"], (
            "Senescence added non-determinism: hash1 != hash2. "
            "Check for rng draws in the senescence block."
        )

    def test_senescence_deterministic_scarce(self):
        """Scarce scenario also deterministic with senescence ON."""
        cfg = replace(SCENARIOS["scarce"], **SENES_PARAMS)
        eco1 = Ecology(cfg, seed=1)
        s1 = eco1.run()
        eco2 = Ecology(cfg, seed=1)
        s2 = eco2.run()
        assert s1["events_hash"] == s2["events_hash"]


# ---------------------------------------------------------------------------
# Cause-of-death event structure
# ---------------------------------------------------------------------------
class TestDeathCauseRecorded:
    """Death events must carry cause, age, and complexity."""

    def test_death_cause_recorded(self):
        """All death events must have cause in {starvation, senescence}, age, complexity."""
        cfg = replace(SCENARIOS["balanced"], **SENES_PARAMS)
        eco = Ecology(cfg, seed=0)
        eco.run()
        death_events = [e for e in eco.events if e["event_type"] == "death"]
        assert len(death_events) > 0, "No death events found"
        valid_causes = {"starvation", "senescence"}
        for evt in death_events:
            details = evt.get("details", {})
            cause = details.get("cause")
            assert cause in valid_causes, (
                f"Death event has cause={cause!r}, expected one of {valid_causes}"
            )
            assert "age" in details, f"Death event missing 'age' field: {details}"
            assert details["age"] >= 0, f"Death event age={details['age']} is negative"
            assert "complexity" in details, (
                f"Death event (treatment) missing 'complexity': {details}"
            )

    def test_control_death_has_no_complexity(self):
        """Control arm death events must NOT have 'complexity' key (byte-identical to Exp 194)."""
        eco = Ecology(SCENARIOS["balanced"], seed=0)
        eco.run()
        death_events = [e for e in eco.events if e["event_type"] == "death"]
        for evt in death_events:
            details = evt.get("details", {})
            assert "complexity" not in details, (
                f"Control death event has unexpected 'complexity' key: {details}"
            )


# ---------------------------------------------------------------------------
# Complexity-lifespan relationship (P3 unit check)
# ---------------------------------------------------------------------------
class TestComplexityShortensLifespan:
    """Higher complexity -> shorter senescence lifespan (P3 core claim)."""

    def test_complexity_shortens_lifespan_population(self):
        """Pool senescence deaths from balanced/seed0; Spearman rho < 0."""
        cfg = replace(SCENARIOS["balanced"], **SENES_PARAMS)
        eco = Ecology(cfg, seed=0)
        eco.run()
        senes_deaths = [
            (e["details"]["complexity"], e["details"]["age"])
            for e in eco.events
            if e["event_type"] == "death"
            and e.get("details", {}).get("cause") == "senescence"
        ]
        assert len(senes_deaths) >= 10, (
            f"Too few senescence deaths for rho test: {len(senes_deaths)}"
        )
        cs = np.array([x[0] for x in senes_deaths])
        as_ = np.array([x[1] for x in senes_deaths])
        # Spearman rho
        rc = np.argsort(np.argsort(cs)).astype(float)
        ra = np.argsort(np.argsort(as_)).astype(float)
        rho = float(np.corrcoef(rc, ra)[0, 1])
        assert rho < 0, (
            f"Expected negative Spearman rho(complexity, age); got {rho:.4f}"
        )

    def test_complexity_shortens_lifespan_full_p3(self):
        """Full P3 check over all 3 seeds: rho <= -0.15 AND gap >= 15."""
        failures = []
        for seed in [0, 1, 2]:
            all_evs = []
            for scenario in ["balanced", "scarce", "overabundant"]:
                cfg = replace(SCENARIOS[scenario], **SENES_PARAMS)
                eco = Ecology(cfg, seed=seed)
                eco.run()
                for e in eco.events:
                    if (e["event_type"] == "death"
                            and e.get("details", {}).get("cause") == "senescence"):
                        all_evs.append((e["details"]["complexity"], e["details"]["age"]))

            if len(all_evs) < 6:
                failures.append(f"seed{seed}: insufficient events ({len(all_evs)})")
                continue

            cs = np.array([x[0] for x in all_evs])
            as_ = np.array([x[1] for x in all_evs])
            rc = np.argsort(np.argsort(cs)).astype(float)
            ra = np.argsort(np.argsort(as_)).astype(float)
            rho = float(np.corrcoef(rc, ra)[0, 1])
            t33 = float(np.percentile(cs, 33.33))
            t67 = float(np.percentile(cs, 66.67))
            bot_ages = as_[cs <= t33]
            top_ages = as_[cs >= t67]
            gap = float(np.mean(bot_ages) - np.mean(top_ages))

            if not (rho <= -0.15 and gap >= 15.0):
                failures.append(
                    f"seed{seed}: rho={rho:.4f}, gap={gap:.1f} (need rho<=-0.15 AND gap>=15)"
                )

        assert not failures, "P3 failed for:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Well-fed outlives starving (self-maintenance is operative)
# ---------------------------------------------------------------------------
class TestWellFedOutlivesStarving:
    """A well-fed creature must outlive a starving creature of equal age+complexity.

    This directly verifies that senescence_self_maintenance > 0 is operative:
    damage accumulates more slowly (or not at all) when energy is high.
    We drive the senescence model analytically using the same formula as engine.py,
    so this test is a fast, deterministic proof of the mechanism without needing
    a full ecology run.
    """

    def test_well_fed_outlives_starving_analytic(self):
        """Analytically prove well-fed creature accumulates less damage.

        Drive two 'virtual' creatures through the damage accumulation formula:
          deg = base * (1 + rate_f * c) * dt^exp
          maintenance = self_maintenance * (energy / energy_capacity)
          damage += max(0, deg - maintenance)

        One creature is well-fed (energy = capacity -> energy_frac = 1.0).
        The other is starving (energy = 0 -> energy_frac = 0.0).
        Same genotype (founder complexity ~0.415), same onset.

        Expectation: starving creature hits damage_death threshold strictly earlier.
        """
        from ecology.engine import EcologyConfig
        from ecology.genotype import complexity as genotype_complexity, founder

        # Use canonical treatment constants
        base = SENES_PARAMS["senescence_base"]
        rate_f = SENES_PARAMS["senescence_rate_frailty"]
        exp = SENES_PARAMS["senescence_exp"]
        onset0 = SENES_PARAMS["senescence_onset0"]
        frailty = SENES_PARAMS["senescence_onset_frailty"]
        maintenance_factor = SENES_PARAMS["senescence_self_maintenance"]
        damage_death = 1.0  # EcologyConfig default

        g = founder()
        c = genotype_complexity(g)
        onset = onset0 * (1.0 - frailty * c)

        # Run damage accumulation for well-fed (ef=1.0) and starving (ef=0.0)
        damage_wf = 0.0
        damage_st = 0.0
        death_step_wf = None
        death_step_st = None

        for dt in range(1, 2000):
            deg = base * (1.0 + rate_f * c) * (dt ** exp)

            # Well-fed
            if death_step_wf is None:
                m_wf = maintenance_factor * 1.0
                damage_wf += max(0.0, deg - m_wf)
                if damage_wf >= damage_death:
                    death_step_wf = dt

            # Starving
            if death_step_st is None:
                m_st = maintenance_factor * 0.0
                damage_st += max(0.0, deg - m_st)
                if damage_st >= damage_death:
                    death_step_st = dt

            if death_step_wf is not None and death_step_st is not None:
                break

        assert death_step_wf is not None, (
            "Well-fed creature never died in 2000 steps — raise base or lower damage_death"
        )
        assert death_step_st is not None, (
            "Starving creature never died in 2000 steps — raise base or lower damage_death"
        )

        death_age_wf = onset + death_step_wf
        death_age_st = onset + death_step_st

        assert death_age_wf > death_age_st, (
            f"Well-fed creature (death_age={death_age_wf:.0f}) should outlive "
            f"starving creature (death_age={death_age_st:.0f}) of equal age+complexity. "
            f"complexity={c:.3f}, onset={onset:.1f}, "
            f"post-onset steps: well-fed={death_step_wf}, starving={death_step_st}. "
            "Check that senescence_self_maintenance > 0."
        )
