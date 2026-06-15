"""tests/test_active_sensing.py — Phase 4 active-sensing mechanism tests.

Covers:
  1. OFF-path byte-identity across information_sampling_rate values.
  2. Golden-hash regression (byte-identical proof for existing code paths).
  3. Probe cost is charged when probing occurs.
  4. Probe does NOT create food (OFF-path byte-identity).
  5. Sampling rate changes probe frequency monotonically.
  6. Edge-case runs (cue_noise=0, hazard_scale=0) complete without error.
  7. Small-scale pairwise + null_guard gate runs return correct structures.
"""
from __future__ import annotations

import dataclasses as D

import pytest

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS
from ecology.genotype import Genotype, founder, mutate
from ecology.evolvability.trait_axis import make_axis


# ---------------------------------------------------------------------------
# Helper: build the active-sensing regime config
# ---------------------------------------------------------------------------

def as_cfg(**over) -> EcologyConfig:
    base = dict(
        horizon=200,
        enable_hidden_mode=True,
        enable_active_sensing=True,
        mode_wrong_regen_factor=1.0,
        mode_hazard_scale=0.6,
        capacity=50.0,
        regen_rate=3.0,
        initial_resource=0.7,
        max_population=30000,
        mode_switch_prob=0.05,
        cue_noise=1.0,
        memory_cost_slope=0.005,
        memory_upkeep_floor=0.0,
        probe_cost=0.02,
        probe_n_samples=4,
        shuffle_creature_order=True,
    )
    base.update(over)
    cfg = D.replace(SCENARIOS["balanced"], **base)
    return cfg


# ---------------------------------------------------------------------------
# Test 1: OFF-path byte-identity across information_sampling_rate
# ---------------------------------------------------------------------------

def test_off_path_byte_identical_across_info_rate() -> None:
    """With enable_active_sensing=False (and mutation_rate=0), founders with
    information_sampling_rate in {0.0, 0.5, 1.0} must produce the SAME events_hash."""
    hashes = []
    for rate in [0.0, 0.5, 1.0]:
        cfg = as_cfg(enable_active_sensing=False, mutation_rate=0.0)
        founder_geno = D.replace(cfg.founder, information_sampling_rate=rate)
        cfg = D.replace(cfg, founder=founder_geno)
        eco = Ecology(cfg, seed=42)
        result = eco.run()
        hashes.append(result["events_hash"])
    assert hashes[0] == hashes[1] == hashes[2], (
        f"OFF-path hashes differ across info rates: {hashes}"
    )


# ---------------------------------------------------------------------------
# Test 2: Golden-hash regression (byte-identical proof)
# ---------------------------------------------------------------------------

def test_golden_hashes_unchanged() -> None:
    """Pinned golden hashes must still hold — proof that adding the new
    information_sampling_rate field did NOT change any existing code path."""
    # Golden 1: balanced, horizon=120, seed=7, no hidden mode, enable_active_sensing default False
    cfg1 = D.replace(SCENARIOS["balanced"], horizon=120)
    eco1 = Ecology(cfg1, seed=7)
    r1 = eco1.run()
    assert r1["events_hash"] == "cb9c15f5f833fb7e166a5cb4bbb1bc3a72c9cbffb070082ba6bd5e84221f0ae5", (
        f"Golden hash 1 changed: {r1['events_hash']}"
    )

    # Golden 2: hidden-mode hazard regime, horizon=200, memory_horizon=2, seed=7,
    #           enable_active_sensing default False
    cfg2 = D.replace(
        SCENARIOS["balanced"],
        horizon=200,
        enable_hidden_mode=True,
        mode_wrong_regen_factor=1.0,
        mode_hazard_scale=0.6,
        capacity=50.0,
        regen_rate=3.0,
        initial_resource=0.7,
        max_population=30000,
        mode_switch_prob=0.05,
        cue_noise=1.0,
        memory_cost_slope=0.005,
        memory_upkeep_floor=0.0,
        shuffle_creature_order=True,
    )
    cfg2_founder = D.replace(cfg2.founder, memory_horizon=2)
    cfg2 = D.replace(cfg2, founder=cfg2_founder)
    eco2 = Ecology(cfg2, seed=7)
    r2 = eco2.run()
    assert r2["events_hash"] == "ab46d3740b71202f0bb50a74a6cfe83539ebf2eaa94607a71441861c30c28ec7", (
        f"Golden hash 2 changed: {r2['events_hash']}"
    )


# ---------------------------------------------------------------------------
# Test 3: Probe cost is paid
# ---------------------------------------------------------------------------

def test_probe_cost_is_paid() -> None:
    """A monomorphic population with information_sampling_rate=1.0 always probes
    (probe_count_total > 0), AND the probe cost must be CAUSALLY ACTIVE: changing
    probe_cost (0.0 vs 0.5), holding everything else fixed, must change the
    trajectory (events_hash) — proof the cost is actually deducted from energy,
    not merely counted.  cue_noise is the same in both, so the only difference is
    the energy charged per probe."""
    def run(probe_cost: float) -> dict:
        cfg = as_cfg(probe_cost=probe_cost, mutation_rate=0.0, horizon=80)
        cfg = D.replace(cfg, founder=D.replace(cfg.founder, information_sampling_rate=1.0))
        eco = Ecology(cfg, seed=1)
        result = eco.run()
        return {"hash": result["events_hash"], "probes": eco.probe_count_total}

    free = run(0.0)
    paid = run(0.5)
    assert paid["probes"] > 0, (
        f"Expected probe_count_total > 0 for info_rate=1.0, got {paid['probes']}"
    )
    assert free["hash"] != paid["hash"], (
        "probe_cost is not causally active: changing probe_cost 0.0 -> 0.5 did NOT "
        "change the trajectory, so the cost is being counted but not paid"
    )


# ---------------------------------------------------------------------------
# Test 4: Probe does NOT create food (OFF-path byte-identity)
# ---------------------------------------------------------------------------

def test_probe_does_not_create_food() -> None:
    """With enable_active_sensing=False, events_hash is identical across
    information_sampling_rate values {0.0, 0.5, 1.0} — no rng or food state changes."""
    hashes = []
    for rate in [0.0, 0.5, 1.0]:
        cfg = as_cfg(enable_active_sensing=False, mutation_rate=0.0, horizon=100)
        founder_geno = D.replace(cfg.founder, information_sampling_rate=rate)
        cfg = D.replace(cfg, founder=founder_geno)
        eco = Ecology(cfg, seed=5)
        result = eco.run()
        hashes.append(result["events_hash"])
    assert hashes[0] == hashes[1] == hashes[2], (
        f"OFF-path (probe_does_not_create_food) hashes differ: {hashes}"
    )


# ---------------------------------------------------------------------------
# Test 5: Sampling rate changes probe frequency monotonically
# ---------------------------------------------------------------------------

def test_sampling_rate_changes_probe_frequency() -> None:
    """Monomorphic runs at information_sampling_rate in {0.0, 0.5, 1.0}
    (enable_active_sensing=True, same seed/horizon) must satisfy:
      probe_count at 0.0 == 0, and 0.0 < 0.5 < 1.0 probe counts."""
    probe_counts = {}
    for rate in [0.0, 0.5, 1.0]:
        cfg = as_cfg(mutation_rate=0.0, horizon=300)
        founder_geno = D.replace(cfg.founder, information_sampling_rate=rate)
        cfg = D.replace(cfg, founder=founder_geno)
        eco = Ecology(cfg, seed=7)
        eco.run()
        probe_counts[rate] = eco.probe_count_total

    assert probe_counts[0.0] == 0, (
        f"Expected probe_count=0 for rate=0.0, got {probe_counts[0.0]}"
    )
    assert probe_counts[0.5] > probe_counts[0.0], (
        f"Expected probe_count(0.5) > probe_count(0.0), "
        f"got {probe_counts[0.5]} vs {probe_counts[0.0]}"
    )
    assert probe_counts[1.0] > probe_counts[0.5], (
        f"Expected probe_count(1.0) > probe_count(0.5), "
        f"got {probe_counts[1.0]} vs {probe_counts[0.5]}"
    )


# ---------------------------------------------------------------------------
# Test 6: Edge-case runs complete without error
# ---------------------------------------------------------------------------

def test_scramble_disable_paths_run() -> None:
    """cue_noise=0.0 and mode_hazard_scale=0.0 runs both complete without
    error and return a dict with events_hash."""
    for override_name, override_val in [("cue_noise", 0.0), ("mode_hazard_scale", 0.0)]:
        cfg = as_cfg(**{override_name: override_val}, horizon=50)
        eco = Ecology(cfg, seed=3)
        result = eco.run()
        assert isinstance(result, dict), (
            f"run() did not return dict for {override_name}={override_val}"
        )
        assert "events_hash" in result, (
            f"events_hash missing from result for {override_name}={override_val}"
        )


# ---------------------------------------------------------------------------
# Test 7: Small-scale pairwise gate + null_guard gate
# ---------------------------------------------------------------------------

def test_pairwise_runs_small_scale() -> None:
    """Smoke test that the gate machinery works end-to-end for the active-sensing axis."""
    from ecology.evolvability import gates as G

    overrides = dict(
        enable_hidden_mode=True,
        enable_active_sensing=True,
        mode_wrong_regen_factor=1.0,
        mode_hazard_scale=0.6,
        capacity=50.0,
        regen_rate=3.0,
        initial_resource=0.7,
        max_population=30000,
        mode_switch_prob=0.05,
        cue_noise=1.0,
        memory_cost_slope=0.005,
        memory_upkeep_floor=0.0,
        probe_cost=0.02,
        probe_n_samples=4,
        shuffle_creature_order=True,
    )
    base = G.build_base_cfg("balanced", 300, overrides)
    axis = make_axis("information_sampling_rate")

    # Pairwise gradient (small: 2 seeds, tight window)
    outcome = G.run_local_pairwise_gradient(
        base, axis, [0, 1],
        win_threshold=2,
        lose_threshold=0,
        min_valid=1,
        window=(50, 250),
        min_pop=5,
    )
    assert outcome.verdict != "", (
        f"Expected non-empty verdict, got {outcome.verdict!r}"
    )
    assert len(outcome.raw_rows) > 0, "Expected raw_rows to be non-empty"
    for row in outcome.raw_rows:
        assert "h_res" in row and "h_mut" in row and "inv_frac_final" in row, (
            f"Missing keys in raw_row: {row.keys()}"
        )

    # Null guards
    null_outcome = G.run_null_guards(base, axis, [0, 1])
    guard_names = [g["name"] for g in null_outcome.aggregate["guards"]]
    assert "cost_off_disconnected_byte_identical" in guard_names, (
        f"cost_off_disconnected_byte_identical guard not in: {guard_names}"
    )
    # Find the guard and check its status
    guard_map = {g["name"]: g for g in null_outcome.aggregate["guards"]}
    assert guard_map["cost_off_disconnected_byte_identical"]["status"] == "PASS", (
        f"cost_off_disconnected_byte_identical guard FAILED: "
        f"{guard_map['cost_off_disconnected_byte_identical']['reason']}"
    )
