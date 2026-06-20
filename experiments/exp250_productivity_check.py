"""Exp 250: Quick productivity check — ON vs OFF prey count at t=40."""
import dataclasses as D
from ecology.engine import Ecology
from ecology.genotype import founder
from ecology.scenarios import SCENARIOS

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
        max_population=4000,
        horizon=150,
    )
    base.update(over)
    return D.replace(SCENARIOS["balanced"], **base)


def _viable_founder():
    return D.replace(founder(), locomotor_speed=1.0, **_CAL)


def count_at(cfg, t_target, seed=0):
    eco = Ecology(cfg, seed=seed)
    while eco.has_alive() and not eco.exploded and eco.t < t_target:
        eco.step()
    return eco.alive_count()


if __name__ == "__main__":
    g = _viable_founder()
    T_HORIZON = 40
    K = 120.0
    INIT_POP = 5

    cfg_off = _cont_base(
        founder=g, initial_population=INIT_POP, horizon=T_HORIZON,
        enable_decoupled_prey_birth=False, prey_birth_rate=0.3, prey_carrying_capacity=K,
    )
    cfg_on = _cont_base(
        founder=g, initial_population=INIT_POP, horizon=T_HORIZON,
        enable_decoupled_prey_birth=True, prey_birth_rate=0.3, prey_carrying_capacity=K,
    )

    n_off = count_at(cfg_off, T_HORIZON)
    n_on = count_at(cfg_on, T_HORIZON)
    print(f"OFF prey count at t={T_HORIZON}: {n_off}")
    print(f"ON  prey count at t={T_HORIZON}: {n_on}")
    print(f"Delta: ON - OFF = {n_on - n_off}")
    print(f"ON > OFF: {n_on > n_off}")
