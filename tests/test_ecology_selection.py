"""tests/test_ecology_selection.py — Fast deterministic tests for the long-horizon
senescence complexity-selection tested in Exp 196.

All tests use SHORT horizons to stay fast.  The full 5000-step predeclared
analysis is done in experiments/exp196_n5_senescence_selection.py.
"""
from __future__ import annotations

from dataclasses import replace
from statistics import mean

import pytest

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS
from ecology.genotype import complexity

# ---------------------------------------------------------------------------
# Canonical senescence params (same as exp195/exp196 treatment arm)
# ---------------------------------------------------------------------------
SENES_PARAMS = dict(
    enable_senescence=True,
    senescence_onset0=155.0,
    senescence_onset_frailty=0.65,
    senescence_rate_frailty=2.0,
    senescence_base=0.002,
    senescence_self_maintenance=1.5,
    senescence_exp=1.5,
)

SHORT_HORIZON = 1500
MEDIUM_HORIZON = 2000


def _mean_complexity(eco: Ecology) -> float | None:
    alive = eco._alive()
    if not alive:
        return None
    return mean(complexity(c.genotype) for c in alive)


def _run_to_horizon(cfg, seed: int):
    eco = Ecology(cfg, seed=seed)
    while eco.t < cfg.horizon and not eco.exploded:
        eco.step()
    return eco


# ---------------------------------------------------------------------------
# test_long_horizon_determinism
# ---------------------------------------------------------------------------
class TestLongHorizonDeterminism:
    """Same seed, same config -> identical events_hash (short horizon for speed)."""

    def test_long_horizon_determinism(self):
        """balanced + senescence ON, horizon ~1500, max_population 5000 — two runs identical."""
        cfg = replace(
            SCENARIOS["balanced"],
            horizon=SHORT_HORIZON,
            max_population=5000,
            **SENES_PARAMS,
        )
        eco1 = _run_to_horizon(cfg, seed=3)
        eco2 = _run_to_horizon(cfg, seed=3)
        assert eco1.events_hash() == eco2.events_hash(), (
            "Long-horizon determinism failed: two runs with same seed produced "
            f"different events_hash.\n"
            f"hash1={eco1.events_hash()!r}\nhash2={eco2.events_hash()!r}"
        )

    def test_long_horizon_determinism_control(self):
        """Control arm (senescence OFF) also deterministic at longer horizon."""
        cfg = replace(
            SCENARIOS["balanced"],
            horizon=SHORT_HORIZON,
            max_population=5000,
        )
        eco1 = _run_to_horizon(cfg, seed=4)
        eco2 = _run_to_horizon(cfg, seed=4)
        assert eco1.events_hash() == eco2.events_hash(), (
            "Control arm determinism failed at long horizon."
        )


# ---------------------------------------------------------------------------
# test_trajectory_sampling
# ---------------------------------------------------------------------------
class TestTrajectorySampling:
    """Stepping manually and sampling complexity returns a valid trajectory."""

    def test_trajectory_sampling(self):
        """Manual stepping with complexity sampling produces non-empty trajectory
        with monotonically increasing t-coverage and pop > 0 at samples."""
        cfg = replace(
            SCENARIOS["balanced"],
            horizon=1000,
            max_population=5000,
            **SENES_PARAMS,
        )
        eco = Ecology(cfg, seed=5)
        sample_steps = {200, 400, 600, 800, 1000}
        trajectory = []

        while eco.t < cfg.horizon and not eco.exploded:
            eco.step()
            if eco.t in sample_steps:
                alive = eco._alive()
                mc = mean(complexity(c.genotype) for c in alive) if alive else None
                trajectory.append({"t": eco.t, "pop": len(alive), "mean_complexity": mc})

        assert len(trajectory) > 0, "Trajectory is empty — no sample points captured."

        # t-values must be strictly increasing
        ts = [e["t"] for e in trajectory]
        assert ts == sorted(ts), f"Trajectory t-values not increasing: {ts}"
        assert ts == sorted(set(ts)), "Duplicate t-values in trajectory."

        # For a run that persists, pop > 0 at all sampled steps
        final_pop = len(eco._alive())
        if final_pop > 0 and not eco.exploded:
            for entry in trajectory:
                assert entry["pop"] > 0, (
                    f"Pop=0 at t={entry['t']} in a persisting run — sampling error."
                )

        # mean_complexity in [0, 1] wherever population is non-zero
        for entry in trajectory:
            if entry["mean_complexity"] is not None:
                assert 0.0 <= entry["mean_complexity"] <= 1.0, (
                    f"mean_complexity={entry['mean_complexity']} out of [0,1] at t={entry['t']}"
                )


# ---------------------------------------------------------------------------
# test_senescence_lowers_complexity_vs_control_shorthorizon
# ---------------------------------------------------------------------------
class TestSenescenceLowersComplexity:
    """Direction check at shorter horizon: treatment final complexity < control final
    for at least one seed.  A fast sanity check, not the full predeclared bar.
    """

    def test_senescence_lowers_complexity_vs_control_shorthorizon(self):
        """At horizon=2000, treatment final mean complexity < control for >=1 seed."""
        found_lower = False
        results = []
        for seed in [3, 4, 5]:
            ctrl_cfg = replace(
                SCENARIOS["balanced"],
                horizon=MEDIUM_HORIZON,
                max_population=5000,
            )
            trt_cfg = replace(
                SCENARIOS["balanced"],
                horizon=MEDIUM_HORIZON,
                max_population=5000,
                **SENES_PARAMS,
            )
            eco_ctrl = _run_to_horizon(ctrl_cfg, seed=seed)
            eco_trt = _run_to_horizon(trt_cfg, seed=seed)

            mc_ctrl = _mean_complexity(eco_ctrl)
            mc_trt = _mean_complexity(eco_trt)

            results.append((seed, mc_ctrl, mc_trt))
            if mc_ctrl is not None and mc_trt is not None and mc_trt < mc_ctrl:
                found_lower = True

        details = "; ".join(
            f"seed{s}: ctrl={f'{c:.4f}' if c is not None else 'extinct'}, "
            f"trt={f'{t:.4f}' if t is not None else 'extinct'}"
            for s, c, t in results
        )
        assert found_lower, (
            "Expected treatment final mean complexity < control for at least one seed "
            f"at horizon={MEDIUM_HORIZON}. Results: {details}"
        )


# ---------------------------------------------------------------------------
# test_control_complexity_flatter_than_treatment
# ---------------------------------------------------------------------------
class TestControlComplexityFlatter:
    """Control complexity should be flatter than treatment over the run.

    |control@end - control@start| < |treatment@end - treatment@start| for a
    representative seed — treatment diverges downward; control stays near its
    initial value.
    """

    def test_control_complexity_flatter_than_treatment(self):
        """Over a medium horizon, control complexity change < treatment complexity change
        for seed 3 (representative)."""
        seed = 3
        sample_early = 400   # proxy for 'start' (after initial transient)
        horizon = MEDIUM_HORIZON

        def _run_and_sample(cfg, sample_t: int):
            eco = Ecology(cfg, seed=seed)
            mc_early = None
            while eco.t < cfg.horizon and not eco.exploded:
                eco.step()
                if eco.t == sample_t:
                    alive = eco._alive()
                    mc_early = mean(complexity(c.genotype) for c in alive) if alive else None
            mc_final = _mean_complexity(eco)
            return mc_early, mc_final, eco

        ctrl_cfg = replace(SCENARIOS["balanced"], horizon=horizon, max_population=5000)
        trt_cfg = replace(
            SCENARIOS["balanced"], horizon=horizon, max_population=5000, **SENES_PARAMS
        )

        mc_ctrl_early, mc_ctrl_final, eco_ctrl = _run_and_sample(ctrl_cfg, sample_early)
        mc_trt_early, mc_trt_final, eco_trt = _run_and_sample(trt_cfg, sample_early)

        # Both must have persisted for this test to be meaningful
        assert mc_ctrl_final is not None, "Control went extinct — cannot compare flatness."
        assert mc_trt_final is not None, "Treatment went extinct — cannot compare flatness."
        assert mc_ctrl_early is not None, "Control: no early sample — adjust sample_early."
        assert mc_trt_early is not None, "Treatment: no early sample — adjust sample_early."

        ctrl_change = abs(mc_ctrl_final - mc_ctrl_early)
        trt_change = abs(mc_trt_final - mc_trt_early)

        assert ctrl_change < trt_change, (
            f"Expected |control change| < |treatment change|, but got "
            f"ctrl_change={ctrl_change:.4f}, trt_change={trt_change:.4f} "
            f"(ctrl: {mc_ctrl_early:.4f}->{mc_ctrl_final:.4f}, "
            f"trt: {mc_trt_early:.4f}->{mc_trt_final:.4f})"
        )
