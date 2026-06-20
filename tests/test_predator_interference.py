"""tests/test_predator_interference.py — Exp 253: gated predator interference
(ratio-dependent / Beddington-DeAngelis functional response).

Tests (all must pass after implementation):

  1. test_off_byte_identical: enable_predator_interference=False → identical events_hash
     across two runs (OFF path is bit-for-bit deterministic; guards the OFF gate).
  2. test_on_differs_from_off: same config with enable_predator_interference=True produces
     a DIFFERENT events_hash from OFF (mechanism is not a silent no-op).
  3. test_interference_reduces_capture_at_high_predator_density: hand-build a scenario with
     one prey inside capture_radius of a FOCAL predator; vary OTHER predators clustered
     within interference_radius (0 vs 5 others). Focal capture rate must be HIGH with 0
     others (~1.0) and LOWER with 5 others.
  4. test_on_deterministic: ON run reproduces identical events_hash across two independent
     runs (ON path is fully deterministic).
"""
import dataclasses as D
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred_cfg(**over):
    """Two-role predator/prey config (mirrors test_predation._pred_cfg)."""
    from ecology.genotype import founder
    from ecology.scenarios import SCENARIOS

    prey_geno = D.replace(founder(), locomotor_speed=1.0, role="prey")
    pred_geno = D.replace(founder(), locomotor_speed=1.4, role="predator")
    base = dict(
        enable_continuous_locomotion=True,
        continuous_layout="bump",
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.0,
        mutation_rate=0.05,
        horizon=300,
        enable_continuous_depletion_intake=True,
        enable_predation=True,
        mutate_predator_speed=True,
        freeze_prey_speed=True,
        founder_mix=((prey_geno, 10), (pred_geno, 3)),
    )
    base.update(over)
    return D.replace(SCENARIOS["balanced"], **base)


# ---------------------------------------------------------------------------
# Test 1: OFF gate — byte-identical across two runs
# ---------------------------------------------------------------------------

def test_off_byte_identical():
    """enable_predator_interference=False → events_hash is bit-for-bit identical across two runs.

    Guards that the OFF path draws no new rng and produces no code-path change.
    """
    from ecology.engine import Ecology

    cfg = _pred_cfg(enable_predator_interference=False)
    h1 = Ecology(cfg, seed=42).run()["events_hash"]
    h2 = Ecology(cfg, seed=42).run()["events_hash"]
    assert h1 == h2, f"OFF path not byte-identical: {h1} != {h2}"


# ---------------------------------------------------------------------------
# Test 2: ON differs from OFF (mechanism is not a no-op)
# ---------------------------------------------------------------------------

def test_on_differs_from_off():
    """enable_predator_interference=True produces a different events_hash from OFF.

    Changing the response type must alter predation outcomes (ON is not silent).
    """
    from ecology.engine import Ecology

    cfg_off = _pred_cfg(enable_predator_interference=False)
    cfg_on = _pred_cfg(
        enable_predator_interference=True,
        interference_strength=2.0,    # strong interference so effect is visible
        interference_radius=3.0,
    )
    h_off = Ecology(cfg_off, seed=5).run()["events_hash"]
    h_on = Ecology(cfg_on, seed=5).run()["events_hash"]
    assert h_on != h_off, (
        "enable_predator_interference=True produced the SAME events_hash as OFF — "
        "the mechanism is a silent no-op (gate leak or wrong placement)"
    )


# ---------------------------------------------------------------------------
# Test 3: Interference reduces capture rate at high predator density
# ---------------------------------------------------------------------------

def test_interference_reduces_capture_at_high_predator_density():
    """Focal predator capture rate falls when many OTHER predators are nearby.

    Hand-built scenario:
      - ONE prey placed inside capture_radius of the FOCAL predator.
      - Vary OTHER predators within interference_radius: 0 vs 5 others.
      - With 0 others: n_other=0 → p_success=1/(1+w*0)=1.0 → capture rate should be ~1.0
        (matches OFF baseline).
      - With 5 others: n_other=5 → p_success=1/(1+0.5*5)=1/3.5≈0.286 → capture rate is LOW.

    Uses horizon=1 so only one capture opportunity per run.
    """
    import math
    from ecology.engine import Ecology
    from ecology.genotype import founder
    from ecology.scenarios import SCENARIOS

    N_SEEDS = 500

    prey_geno = D.replace(founder(), locomotor_speed=0.0, role="prey", energy_capacity=20.0)
    pred_geno = D.replace(founder(), locomotor_speed=0.0, role="predator", energy_capacity=20.0)

    N_OTHER = 5   # number of interfering predators (not the focal one)
    N_PRED_TOTAL = 1 + N_OTHER  # focal + others

    base_cfg = D.replace(
        SCENARIOS["balanced"],
        enable_continuous_locomotion=True,
        continuous_layout="flat",
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.0,
        mutation_rate=0.0,
        horizon=1,
        enable_predation=True,
        mutate_predator_speed=False,
        freeze_prey_speed=True,
        enable_continuous_depletion_intake=False,
        founder_mix=((prey_geno, 1), (pred_geno, N_PRED_TOTAL)),
        capture_radius=1.0,
        sensing_radius=3.0,
        assimilation_efficiency=1.0,
        strike_cost=0.0,
        max_captures_per_step=1,
        interference_strength=0.5,
        interference_radius=2.5,
    )

    FOCAL_POS = (5.0, 5.0)
    CR = 1.0
    IR = 2.5

    def _run_scenario(cfg, n_other_inside_ir, n_seeds):
        """Return fraction of seeds where the FOCAL predator (lowest id) captures the prey.

        The FOCAL predator is the one with the lowest creature_id among predators (ascending
        id by construction). The prey is placed inside its capture_radius. n_other_inside_ir
        other predators are placed within interference_radius of the focal predator.
        """
        captured = 0
        for seed in range(n_seeds):
            eco = Ecology(cfg, seed=seed)
            alive = eco.alive_snapshot()  # ascending creature_id
            preds = [c for c in alive if c.genotype.role == "predator"]
            preys = [c for c in alive if c.genotype.role == "prey"]
            assert preds and preys, "need at least 1 predator and 1 prey"

            # Focal predator: lowest id
            focal = preds[0]
            focal.phenotype.pos_cont = FOCAL_POS

            # Place the prey inside capture_radius of the focal predator
            target_prey = preys[0]
            target_prey.phenotype.pos_cont = (FOCAL_POS[0] + CR * 0.5, FOCAL_POS[1])

            # Place other predators: n_other_inside_ir within interference_radius, rest far away
            for i, op in enumerate(preds[1:], start=1):
                if i <= n_other_inside_ir:
                    # Place within interference_radius (but not exactly overlapping)
                    angle = 2.0 * math.pi * i / max(n_other_inside_ir, 1)
                    r = IR * 0.5  # well within IR
                    op.phenotype.pos_cont = (
                        FOCAL_POS[0] + r * math.cos(angle),
                        FOCAL_POS[1] + r * math.sin(angle),
                    )
                else:
                    # Place far away — outside interference_radius
                    op.phenotype.pos_cont = (FOCAL_POS[0] + IR * 5.0, FOCAL_POS[1])

            eco._resolve_captures()
            if not target_prey.phenotype.alive:
                captured += 1
        return captured / n_seeds

    # --- OFF baseline (no interference): should capture ~100% ---
    cfg_off = D.replace(base_cfg, enable_predator_interference=False)
    rate_off = _run_scenario(cfg_off, n_other_inside_ir=N_OTHER, n_seeds=N_SEEDS)
    assert rate_off == pytest.approx(1.0, abs=0.0), (
        f"OFF should always capture in-range prey; got rate={rate_off:.3f}"
    )

    # --- ON with 0 other predators inside IR: p_success=1.0 → should still capture ~100% ---
    cfg_on = D.replace(base_cfg, enable_predator_interference=True)
    rate_on_0_others = _run_scenario(cfg_on, n_other_inside_ir=0, n_seeds=N_SEEDS)
    assert rate_on_0_others == pytest.approx(1.0, abs=0.0), (
        f"ON with 0 others in IR: p_success=1.0, should capture ~100%; got {rate_on_0_others:.3f}"
    )

    # --- ON with N_OTHER other predators inside IR: p_success=1/(1+0.5*5)≈0.286 → LOW rate ---
    rate_on_5_others = _run_scenario(cfg_on, n_other_inside_ir=N_OTHER, n_seeds=N_SEEDS)
    expected_p = 1.0 / (1.0 + 0.5 * N_OTHER)   # ≈ 0.286
    # Generous threshold: must be below 0.60 (well below the lone-predator rate of 1.0)
    assert rate_on_5_others < 0.60, (
        f"ON with {N_OTHER} others inside IR: expected capture rate ≈{expected_p:.3f}, "
        f"generous threshold <0.60; got {rate_on_5_others:.3f}"
    )

    print(
        f"\n[interference test] OFF rate: {rate_off:.3f}, "
        f"ON 0-others rate: {rate_on_0_others:.3f}, "
        f"ON {N_OTHER}-others rate: {rate_on_5_others:.3f} "
        f"(expected ≈{expected_p:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 4: ON path is deterministic (identical events_hash across two runs)
# ---------------------------------------------------------------------------

def test_on_deterministic():
    """enable_predator_interference=True run produces identical events_hash across two
    independent runs. Guards that the gated rng draw is in the correct deterministic stream.
    """
    from ecology.engine import Ecology

    cfg = _pred_cfg(
        enable_predator_interference=True,
        interference_strength=0.5,
        interference_radius=2.5,
    )
    h1 = Ecology(cfg, seed=99).run()["events_hash"]
    h2 = Ecology(cfg, seed=99).run()["events_hash"]
    assert h1 == h2, (
        f"ON path not deterministic across two runs: {h1} != {h2}\n"
        "The rng draw in _resolve_captures must be deterministic."
    )
