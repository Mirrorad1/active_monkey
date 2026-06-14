"""Tests for ecology.evolvability.gates — engine-coupled gate runners.

All tests use TINY run parameters to stay fast (<~30 s total).
Seeds: [38, 39] (two seeds for speed; enough to exercise validity logic).

The key anti-cheat test is test_null_guard_byte_identical: when the trait is
disconnected (cost OFF + food coupling OFF), events_hash must be identical
regardless of what h value is clamped.
"""
from __future__ import annotations

import dataclasses as D
import math

import pytest

from ecology.scenarios import SCENARIOS, FOUNDER
from ecology.evolvability.gates import (
    GATE_REGISTRY,
    GateOutcome,
    ControllerSpec,
    build_base_cfg,
    run_local_pairwise_gradient,
    run_monomorphic_sweep,
    run_gifted_benefit,
    run_null_guards,
    run_invasion_from_rarity,
    run_density_independent_growth,
    run_cost_sensitivity,
    run_controller_cross_partial,
)
from ecology.evolvability.trait_axis import THERMOSENSE_AXIS
from ecology.evolvability.verdicts import (
    GradientVerdict, BenefitVerdict, InvasionVerdict, GuardStatus
)


# ---------------------------------------------------------------------------
# Tiny config factory (mirrors test_exp203_sense_axis.py pattern)
# ---------------------------------------------------------------------------

_BASE = dict(
    enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
    thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05, thermosense_noise_base=0.5,
    thermal_avoidance_weight=4.0, food_optimal_base=0.5, food_optimal_amplitude=0.3,
    food_optimal_period=1500.0, food_concentration=14.0, food_band_width=0.08,
    enable_food_coupling=True, thermosense_forage_mode=True,
)


def tiny_cfg(horizon: int = 300, regen_rate: float = 0.20, **kw):
    f = D.replace(
        FOUNDER,
        thermosense_intensity=0.10,
        thermosense_inefficiency=0.2,
        temperature_tolerance=0.10,
    )
    fields = dict(
        horizon=horizon,
        max_population=4000,
        founder=f,
        regen_rate=regen_rate,
        shuffle_creature_order=True,
    )
    fields.update(_BASE)
    fields.update(kw)  # explicit kwargs override the defaults (no duplicate-keyword error)
    return D.replace(SCENARIOS["balanced"], **fields)


SEEDS = [38, 39]
AXIS = THERMOSENSE_AXIS


# ---------------------------------------------------------------------------
# A) build_base_cfg smoke test
# ---------------------------------------------------------------------------

def test_build_base_cfg_smoke():
    cfg = build_base_cfg("balanced", 200, {})
    assert cfg.horizon == 200


# ---------------------------------------------------------------------------
# B) run_local_pairwise_gradient
# ---------------------------------------------------------------------------

def test_local_pairwise_gradient_returns_gate_outcome():
    cfg = tiny_cfg(horizon=200)
    out = run_local_pairwise_gradient(
        cfg, AXIS, SEEDS,
        win_threshold=2, lose_threshold=0, min_valid=1,
        count_each=5, window=(50, 180), min_pop=2,
    )
    assert isinstance(out, GateOutcome)
    assert out.name == "local_pairwise_gradient"
    # Verdict must be a valid GradientVerdict value
    assert out.verdict in {v.value for v in GradientVerdict}


def test_local_pairwise_gradient_raw_rows_keys():
    cfg = tiny_cfg(horizon=200)
    out = run_local_pairwise_gradient(
        cfg, AXIS, SEEDS,
        win_threshold=2, lose_threshold=0, min_valid=1,
        count_each=5, window=(50, 180), min_pop=2,
    )
    assert len(out.raw_rows) == len(SEEDS)
    required_keys = {"gate", "seed", "h_res", "h_mut", "s", "inv_frac_auc",
                     "inv_frac_final", "inv_won", "final_pop", "extinct", "valid"}
    for row in out.raw_rows:
        assert required_keys <= set(row.keys()), f"Missing keys: {required_keys - set(row.keys())}"
        assert row["gate"] == "local_pairwise_gradient"


def test_local_pairwise_gradient_n_valid_lte_seeds():
    cfg = tiny_cfg(horizon=200)
    out = run_local_pairwise_gradient(
        cfg, AXIS, SEEDS,
        win_threshold=2, lose_threshold=0, min_valid=1,
        count_each=5, window=(50, 180), min_pop=2,
    )
    assert out.aggregate["n_valid"] <= len(SEEDS)


def test_local_pairwise_gradient_determinism():
    """Calling twice with the same seeds produces identical aggregate wins/n_valid."""
    cfg = tiny_cfg(horizon=200)
    kw = dict(win_threshold=2, lose_threshold=0, min_valid=1,
               count_each=5, window=(50, 180), min_pop=2)
    out1 = run_local_pairwise_gradient(cfg, AXIS, SEEDS, **kw)
    out2 = run_local_pairwise_gradient(cfg, AXIS, SEEDS, **kw)
    assert out1.aggregate["wins"] == out2.aggregate["wins"]
    assert out1.aggregate["n_valid"] == out2.aggregate["n_valid"]


# ---------------------------------------------------------------------------
# C) run_monomorphic_sweep
# ---------------------------------------------------------------------------

def test_monomorphic_sweep_curve_keys():
    cfg = tiny_cfg(horizon=200)
    grid = [0.0, 0.10, 0.60]
    out = run_monomorphic_sweep(cfg, AXIS, SEEDS, grid, min_pop=2)
    assert isinstance(out, GateOutcome)
    assert out.name == "monomorphic_sweep"
    # curve_Nstar must have all grid values as keys
    curve = out.aggregate["curve_Nstar"]
    for h in grid:
        assert h in curve, f"h={h} missing from curve_Nstar"


def test_monomorphic_sweep_optimum_in_grid():
    cfg = tiny_cfg(horizon=200)
    grid = [0.0, 0.10, 0.60]
    out = run_monomorphic_sweep(cfg, AXIS, SEEDS, grid, min_pop=2)
    opt_h = out.aggregate["optimum_h"]
    # optimum_h must be one of the grid values or nan
    if not math.isnan(opt_h):
        assert any(abs(opt_h - g) < 1e-9 for g in grid), \
            f"optimum_h={opt_h} not in grid {grid}"


def test_monomorphic_sweep_above_resident_and_survivable():
    cfg = tiny_cfg(horizon=200)
    grid = [0.0, 0.10, 0.60]
    out = run_monomorphic_sweep(cfg, AXIS, SEEDS, grid, min_pop=2)
    assert isinstance(out.aggregate["above_resident"], bool)
    assert isinstance(out.aggregate["survivable"], bool)


def test_monomorphic_sweep_no_verdict():
    cfg = tiny_cfg(horizon=200)
    out = run_monomorphic_sweep(cfg, AXIS, SEEDS, [0.0, 0.10, 0.60], min_pop=2)
    assert out.verdict == "", "monomorphic_sweep should have no pass/fail verdict"


# ---------------------------------------------------------------------------
# D) run_gifted_benefit
# ---------------------------------------------------------------------------

def test_gifted_benefit_returns_delta_and_verdict():
    cfg = tiny_cfg(horizon=200)
    out = run_gifted_benefit(cfg, AXIS, SEEDS, eps=1e-6)
    assert isinstance(out, GateOutcome)
    assert out.name == "gifted_benefit"
    # Verdict must be a valid BenefitVerdict value
    assert out.verdict in {v.value for v in BenefitVerdict}
    assert len(out.raw_rows) == len(SEEDS)
    for row in out.raw_rows:
        assert "delta" in row
        assert row["gate"] == "gifted_benefit"


def test_gifted_benefit_interpretation_no_evolvability_claim():
    cfg = tiny_cfg(horizon=200)
    out = run_gifted_benefit(cfg, AXIS, SEEDS)
    # The interpretation MUST state it does NOT prove evolvability
    interp_lower = out.interpretation.lower()
    assert "not" in interp_lower and ("evolvable" in interp_lower or "evolvability" in interp_lower), \
        "interpretation must state gifted benefit does NOT prove evolvability"


# ---------------------------------------------------------------------------
# E) run_null_guards — the KEY anti-cheat test
# ---------------------------------------------------------------------------

def test_null_guard_byte_identical_passes():
    """ANTI-CHEAT: when trait is disconnected (cost OFF + coupling OFF), events_hash
    must be identical for h=0.0 and h=0.60.  This is the core null check."""
    cfg = tiny_cfg(horizon=150)
    out = run_null_guards(cfg, AXIS, SEEDS, min_pop=2)
    assert isinstance(out, GateOutcome)
    assert out.name == "null_guards"

    guards_by_name = {g["name"]: g for g in out.aggregate["guards"]}
    g1 = guards_by_name["cost_off_disconnected_byte_identical"]
    assert g1["status"] == GuardStatus.PASS.value, \
        f"cost_off_disconnected_byte_identical FAILED: {g1['reason']}"


def test_null_guard_no_direct_h_reward_passes():
    """no_direct_h_reward must PASS for the thermosense axis (assert_no_direct_h_reward)."""
    cfg = tiny_cfg(horizon=150)
    out = run_null_guards(cfg, AXIS, SEEDS, min_pop=2)
    guards_by_name = {g["name"]: g for g in out.aggregate["guards"]}
    g2 = guards_by_name["no_direct_h_reward"]
    assert g2["status"] == GuardStatus.PASS.value, \
        f"no_direct_h_reward FAILED: {g2['reason']}"


def test_null_guard_trait_disabled_null_passes():
    """trait_disabled_null: per-capita intake must be identical when trait is disconnected."""
    cfg = tiny_cfg(horizon=150)
    out = run_null_guards(cfg, AXIS, SEEDS, min_pop=2)
    guards_by_name = {g["name"]: g for g in out.aggregate["guards"]}
    g3 = guards_by_name["trait_disabled_null"]
    # Should PASS (identical events → identical intake)
    assert g3["status"] == GuardStatus.PASS.value, \
        f"trait_disabled_null FAILED: {g3['reason']}"


def test_null_guard_shuffle_order_passes_with_shuffle_cfg():
    """shuffle_order guard should PASS when shuffle_creature_order=True."""
    cfg = tiny_cfg(horizon=150, shuffle_creature_order=True)
    out = run_null_guards(cfg, AXIS, SEEDS, min_pop=2)
    guards_by_name = {g["name"]: g for g in out.aggregate["guards"]}
    g7 = guards_by_name["shuffle_order"]
    assert g7["status"] == GuardStatus.PASS.value, \
        f"shuffle_order FAILED: {g7['reason']}"


def test_null_guard_population_validity_with_extinct_fraction():
    """population_validity guard: PASS when fraction < 0.5, FAIL when >= 0.5."""
    cfg = tiny_cfg(horizon=150)
    out_pass = run_null_guards(cfg, AXIS, SEEDS, pairwise_extinct_fraction=0.2)
    guards = {g["name"]: g for g in out_pass.aggregate["guards"]}
    assert guards["population_validity"]["status"] == GuardStatus.PASS.value

    out_fail = run_null_guards(cfg, AXIS, SEEDS, pairwise_extinct_fraction=0.7)
    guards2 = {g["name"]: g for g in out_fail.aggregate["guards"]}
    assert guards2["population_validity"]["status"] == GuardStatus.FAIL.value


def test_null_guard_all_pass_accounting():
    """all_pass must only count PASS/FAIL guards (NOT_IMPLEMENTED/NA are skipped)."""
    cfg = tiny_cfg(horizon=150)
    out = run_null_guards(cfg, AXIS, SEEDS, pairwise_extinct_fraction=0.1)
    guards = out.aggregate["guards"]
    terminal = {GuardStatus.PASS.value, GuardStatus.FAIL.value}
    expected_all_pass = all(
        g["status"] == GuardStatus.PASS.value
        for g in guards
        if g["status"] in terminal
    )
    assert out.aggregate["all_pass"] == expected_all_pass


# ---------------------------------------------------------------------------
# F) run_invasion_from_rarity
# ---------------------------------------------------------------------------

def test_invasion_from_rarity_verdict_and_rows():
    cfg = tiny_cfg(horizon=250)
    out = run_invasion_from_rarity(
        cfg, AXIS, SEEDS,
        win_threshold=2, lose_threshold=0, min_valid=1,
        mutant_fraction=0.05, resident_count=20,
        window=(50, 220), stride=25, min_pop=2,
    )
    assert isinstance(out, GateOutcome)
    assert out.name == "invasion_from_rarity"
    # Verdict must be a valid InvasionVerdict
    assert out.verdict in {v.value for v in InvasionVerdict}
    assert len(out.raw_rows) == len(SEEDS)


def test_invasion_from_rarity_invader_count_at_least_1():
    cfg = tiny_cfg(horizon=250)
    out = run_invasion_from_rarity(
        cfg, AXIS, SEEDS,
        win_threshold=2, lose_threshold=0, min_valid=1,
        mutant_fraction=0.05, resident_count=20,
        window=(50, 220), stride=25, min_pop=2,
    )
    assert out.aggregate["invader_count"] >= 1


def test_invasion_from_rarity_f_initial_small():
    """f_initial should be small (rare invader start) — definitely < 0.2 for 5% fraction."""
    cfg = tiny_cfg(horizon=250)
    out = run_invasion_from_rarity(
        cfg, AXIS, SEEDS,
        win_threshold=2, lose_threshold=0, min_valid=1,
        mutant_fraction=0.05, resident_count=20,
        window=(50, 220), stride=25, min_pop=2,
    )
    for row in out.raw_rows:
        f_init = row["f_initial"]
        if not math.isnan(f_init):
            assert f_init < 0.2, f"f_initial={f_init} is not small (rare start expected < 0.2)"


def test_invasion_from_rarity_required_keys():
    cfg = tiny_cfg(horizon=250)
    out = run_invasion_from_rarity(
        cfg, AXIS, SEEDS,
        win_threshold=2, lose_threshold=0, min_valid=1,
        mutant_fraction=0.05, resident_count=20,
        window=(50, 220), stride=25, min_pop=2,
    )
    required = {"gate", "seed", "h_res", "h_mut", "mutant_fraction",
                "f_initial", "f_final", "increased", "final_pop", "extinct"}
    for row in out.raw_rows:
        assert required <= set(row.keys())


# ---------------------------------------------------------------------------
# G) GATE_REGISTRY completeness
# ---------------------------------------------------------------------------

def test_gate_registry_contains_all_8_gates():
    expected = {
        "gifted_benefit",
        "monomorphic_sweep",
        "local_pairwise_gradient",
        "invasion_from_rarity",
        "density_independent_growth",
        "cost_sensitivity",
        "null_guards",
        "controller_cross_partial",
    }
    assert set(GATE_REGISTRY.keys()) == expected


def test_gate_registry_values_are_callable():
    for name, fn in GATE_REGISTRY.items():
        assert callable(fn), f"GATE_REGISTRY[{name!r}] is not callable"


# ---------------------------------------------------------------------------
# H) run_density_independent_growth (smoke)
# ---------------------------------------------------------------------------

def test_density_independent_growth_smoke():
    cfg = tiny_cfg(horizon=250)
    out = run_density_independent_growth(cfg, AXIS, SEEDS, eps=1e-6)
    assert isinstance(out, GateOutcome)
    assert out.name == "density_independent_growth"
    assert out.verdict in {v.value for v in BenefitVerdict}
    assert len(out.raw_rows) == len(SEEDS)
    for row in out.raw_rows:
        assert "delta_r" in row


# ---------------------------------------------------------------------------
# I) run_cost_sensitivity (smoke)
# ---------------------------------------------------------------------------

def test_cost_sensitivity_smoke():
    cfg = tiny_cfg(horizon=200)
    cost_values = [0.20, 0.50]
    out = run_cost_sensitivity(
        cfg, AXIS, SEEDS, cost_values,
        win_threshold=2, lose_threshold=0, min_valid=1,
        count_each=5, window=(50, 180), min_pop=2,
    )
    assert isinstance(out, GateOutcome)
    assert out.name == "cost_sensitivity"
    assert out.verdict == "", "cost_sensitivity should have no pass/fail verdict"
    assert len(out.aggregate["per_cost"]) == len(cost_values)
    for pc in out.aggregate["per_cost"]:
        assert "cost" in pc and "wins" in pc and "n_valid" in pc and "verdict" in pc


# ---------------------------------------------------------------------------
# J) run_controller_cross_partial (smoke)
# ---------------------------------------------------------------------------

def test_controller_cross_partial_smoke():
    """Smoke test with niche_weight as the controller (a field that exists on EcologyConfig)."""
    cfg = tiny_cfg(horizon=150)
    controller = ControllerSpec(
        config_field="niche_weight",
        low_value=1.0,
        high_value=8.0,
    )
    out = run_controller_cross_partial(cfg, AXIS, controller, SEEDS, window=80)
    assert isinstance(out, GateOutcome)
    assert out.name == "controller_cross_partial"
    from ecology.evolvability.verdicts import CrossPartialVerdict
    assert out.verdict in {v.value for v in CrossPartialVerdict}
    # Should have 4 corners × len(SEEDS) raw rows
    assert len(out.raw_rows) == 4 * len(SEEDS)
    for row in out.raw_rows:
        assert row["corner"] in {"ll", "hl", "lh", "hh"}
    # Aggregate should contain the 4 corner values
    for key in ("B_ll", "B_hl", "B_lh", "B_hh", "corner_effects"):
        assert key in out.aggregate


# ---------------------------------------------------------------------------
# K) GateOutcome dataclass fields
# ---------------------------------------------------------------------------

def test_gate_outcome_has_required_fields():
    """All required fields of GateOutcome are present."""
    required_fields = {
        "name", "question", "metric", "raw_rows", "per_seed",
        "aggregate", "verdict", "validity_flags", "interpretation",
    }
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(GateOutcome)}
    assert required_fields <= field_names
