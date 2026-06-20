"""Continuous-substrate VIABILITY guard (Exp 248 self-heal).

The continuous depletion-intake substrate is only viable under the Exp-240/242 CALIBRATED
founder (low metabolic/movement cost + low repro threshold) WITH a resource regen rate set.
The raw ``founder()`` (the DISCRETE 'balanced' calibration: energy_capacity=20, bmc=0.5,
aging=0.02) starves to extinction within a few dozen steps — and NO other test catches this,
because every other continuous test hashes only 50-step runs for DETERMINISM, never long-horizon
PERSISTENCE. This bug has now bitten twice (Exp 243 cert_run founder calibration; Exp 248
geometry/`_pred_cfg` probes). These two tests make the calibration requirement explicit and
regression-proof:

  * test_calibrated_continuous_founder_persists  — the viable config must still sustain a
    population to a long horizon (guards the engine against a viability regression).
  * test_raw_founder_is_non_viable_on_continuous — documents+pins the trap: founder() alone
    collapses, so any experiment on this substrate MUST recalibrate + verify persistence first.
"""
import dataclasses as D

from ecology.engine import Ecology
from ecology.genotype import founder
from ecology.scenarios import SCENARIOS

# Exp-240/242 calibrated continuous founder (see experiments/exp242_regulated_ess.py).
_CAL = dict(
    baseline_metabolic_cost=0.05,
    movement_cost=0.03,
    energy_capacity=10.0,
    reproduction_energy_threshold=4.2,
    reproduction_energy_transfer_fraction=0.35,
    reproduction_cost_fraction=0.06,
    aging_cost=0.003,
)


def _cont_base(**over):
    base = dict(
        enable_continuous_locomotion=True,
        continuous_layout="bump",
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.0,
        enable_continuous_depletion_intake=True,
        continuous_logistic_regen=False,
        continuous_regen_rate=1.0,
        continuous_capacity=2.0,
        min_survival_energy=0.5,
        mutation_rate=0.0,
        # max_population MATTERS: the viable continuous population blooms to ~500 before
        # settling; the SCENARIOS['balanced'] default of 200 strangles that bloom into
        # collapse (extinct by t~18). The Exp-242 viable config uses 4000.
        max_population=4000,
        horizon=150,
    )
    base.update(over)
    return D.replace(SCENARIOS["balanced"], **base)


def _alive_at_horizon(cfg, seed=0):
    eco = Ecology(cfg, seed=seed)
    while eco.has_alive() and not eco.exploded and eco.t < cfg.horizon:
        eco.step()
    return eco.t, eco.alive_count()


def test_calibrated_continuous_founder_persists():
    """The Exp-240 calibrated founder + regen must sustain a population to the horizon.

    Guards against an engine regression that silently breaks continuous viability (which the
    determinism goldens would NOT catch — they would just change hash and get repinned)."""
    g = D.replace(founder(), locomotor_speed=1.0, **_CAL)
    cfg = _cont_base(founder=g, initial_population=21)
    t_end, n_alive = _alive_at_horizon(cfg)
    assert t_end == cfg.horizon, f"calibrated founder went extinct early at t={t_end}"
    assert n_alive > 0, "calibrated continuous founder failed to persist to the horizon"


def test_raw_founder_is_non_viable_on_continuous():
    """The raw founder() (energy_capacity=20, bmc=0.5, no regen tuning) is NON-VIABLE here.

    Pins the Exp-243/Exp-248 trap: any experiment on the continuous depletion substrate MUST
    use the calibrated founder and verify persistence — never the raw founder()."""
    g = D.replace(founder(), locomotor_speed=1.0)  # raw discrete 'balanced' calibration
    cfg = _cont_base(founder=g, continuous_regen_rate=0.05, initial_population=21)
    t_end, n_alive = _alive_at_horizon(cfg)
    assert t_end < cfg.horizon and n_alive == 0, (
        f"raw founder() unexpectedly survived on the continuous substrate "
        f"(t_end={t_end}, n_alive={n_alive}) — the viability trap may have changed; "
        f"re-examine the continuous calibration assumptions."
    )
