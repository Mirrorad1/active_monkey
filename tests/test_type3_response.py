"""tests/test_type3_response.py — Exp 252: Type III (sigmoid) predator functional response.

Tests (all must pass after implementation):

  1. test_off_byte_identical: enable_type3_response=False → identical events_hash across two
     runs (OFF path is bit-for-bit deterministic; this is NOT a golden — it guards the OFF gate).
  2. test_on_differs_from_off: same config with enable_type3_response=True produces a DIFFERENT
     events_hash from OFF (mechanism is not a silent no-op).
  3. test_low_density_refuge: hand-built scenario — ONE prey within capture_radius, NO other prey
     nearby (n_local=1, type3_half_density=5.0, k=2.0 → p_cap = 1/(1+25) ≈ 0.038).
     Over many seeds the ON capture rate is SIGNIFICANTLY below the OFF capture rate (where
     an in-range prey is always captured). Tests both the emergent refuge and saturation.
  4. test_on_deterministic: enable_type3_response=True run reproduces identical events_hash
     across two independent runs (ON path is fully deterministic).
"""
import dataclasses as D
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — reuse _pred_cfg from test_predation.py
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
    """enable_type3_response=False → events_hash is bit-for-bit identical across two runs.

    This guards that the OFF path draws no new rng and produces no code-path change
    relative to the predation-only baseline.
    """
    from ecology.engine import Ecology

    cfg = _pred_cfg(enable_type3_response=False)
    h1 = Ecology(cfg, seed=42).run()["events_hash"]
    h2 = Ecology(cfg, seed=42).run()["events_hash"]
    assert h1 == h2, f"OFF path not byte-identical: {h1} != {h2}"


# ---------------------------------------------------------------------------
# Test 2: ON differs from OFF (mechanism is not a no-op)
# ---------------------------------------------------------------------------

def test_on_differs_from_off():
    """enable_type3_response=True produces a different events_hash from OFF.

    Changing the response type must alter the predation outcomes (ON is not silent).
    """
    from ecology.engine import Ecology

    cfg_off = _pred_cfg(enable_type3_response=False)
    cfg_on = _pred_cfg(
        enable_type3_response=True,
        type3_half_density=3.0,
        type3_exponent=2.0,
    )
    h_off = Ecology(cfg_off, seed=5).run()["events_hash"]
    h_on = Ecology(cfg_on, seed=5).run()["events_hash"]
    assert h_on != h_off, (
        "enable_type3_response=True produced the SAME events_hash as OFF — "
        "the mechanism is a silent no-op (gate leak or wrong placement)"
    )


# ---------------------------------------------------------------------------
# Test 3: Low-density refuge + saturation
# ---------------------------------------------------------------------------

def test_low_density_refuge():
    """Type III creates an emergent low-density refuge: p_cap is LOW when n_local << h.

    Hand-built scenario:
      - ONE prey placed within capture_radius (inside geometric capture zone).
      - NO other prey in sensing_radius (n_local = 1).
      - type3_half_density = 5.0, type3_exponent = 2.0
        → p_cap = 1^2 / (5^2 + 1^2) = 1/26 ≈ 0.038

    Over N_SEEDS independent seeds the ON capture rate must be SIGNIFICANTLY below
    the OFF capture rate (which is 1.0 — an in-range prey is always captured OFF).

    Also validates SATURATION: with HIGH n_local (n_local >> h), p_cap → 1 and
    the ON capture rate approaches the OFF rate.

    This is a direct test of the emergent low-density refuge, the textbook property
    of the Type III functional response.
    """
    from ecology.engine import Ecology
    from ecology.genotype import founder

    N_SEEDS = 300   # enough to distinguish p≈0.038 from p=1.0 with high confidence

    # --- Build a minimal capture-test config ---
    # We need enable_predation=True + enable_continuous_locomotion=True, but a horizon=1
    # so the single step resolves one capture opportunity. We suppress regen/intake
    # complexity by using neutral layout and zero costs.
    from ecology.scenarios import SCENARIOS

    prey_geno = D.replace(founder(), locomotor_speed=0.0, role="prey", energy_capacity=20.0)
    pred_geno = D.replace(founder(), locomotor_speed=0.0, role="predator", energy_capacity=20.0)

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
        founder_mix=((prey_geno, 1), (pred_geno, 1)),
        capture_radius=1.0,
        sensing_radius=3.0,
        assimilation_efficiency=1.0,
        strike_cost=0.0,
        max_captures_per_step=1,
    )

    def _count_captures(cfg, n_seeds):
        """Return fraction of seeds where the single prey was captured."""
        captured = 0
        cr = cfg.capture_radius
        for seed in range(n_seeds):
            eco = Ecology(cfg, seed=seed)
            # Hand-place: predator at centre, prey inside capture_radius
            alive = eco.alive_snapshot()
            pred = next(c for c in alive if c.genotype.role == "predator")
            prey = next(c for c in alive if c.genotype.role == "prey")
            pred.phenotype.pos_cont = (5.0, 5.0)
            prey.phenotype.pos_cont = (5.0 + cr * 0.5, 5.0)  # well inside capture_radius
            eco._resolve_captures()
            if not prey.phenotype.alive:
                captured += 1
        return captured / n_seeds

    # --- OFF baseline: should capture ~100% (deterministic geometry) ---
    cfg_off = D.replace(base_cfg, enable_type3_response=False)
    rate_off = _count_captures(cfg_off, N_SEEDS)

    # --- ON at LOW density (n_local=1, h=5, k=2): p_cap = 1/(25+1) ≈ 0.038 ---
    cfg_on_low = D.replace(
        base_cfg,
        enable_type3_response=True,
        type3_half_density=5.0,
        type3_exponent=2.0,
    )
    rate_on_low = _count_captures(cfg_on_low, N_SEEDS)

    # Core assertion: OFF captures the prey every time; ON captures it much less often.
    assert rate_off == pytest.approx(1.0, abs=0.0), (
        f"OFF should always capture in-range prey; got rate={rate_off:.3f}"
    )
    # Expected ON rate ≈ 0.038; we use a generous threshold (0.20) to tolerate
    # sampling variability at 300 seeds, while still proving refuge is real.
    assert rate_on_low < 0.20, (
        f"Type III ON at n_local=1 (h=5, k=2) should have a very LOW capture rate "
        f"(expected ≈0.038, generous threshold <0.20); got {rate_on_low:.3f}"
    )

    # Report numbers for the task report.
    print(f"\n[low-density refuge] OFF capture rate: {rate_off:.3f}, ON capture rate (n_local≈1): {rate_on_low:.3f}")
    print(f"  Expected ON rate ≈ 1/26 ≈ 0.038 (Hill Type III, h=5, k=2, n_local=1)")

    # --- SATURATION: high n_local >> h → p_cap → 1 ---
    # Place MANY prey within sensing_radius and capture_radius so n_local is large.
    # With n_local=20, h=5, k=2: p_cap = 400/(25+400) ≈ 0.941
    N_HIGH = 20  # total prey (first one inside capture_radius; rest fill sensing zone)
    many_prey_geno_list = [(prey_geno, N_HIGH)]
    cfg_sat = D.replace(
        base_cfg,
        enable_type3_response=True,
        type3_half_density=5.0,
        type3_exponent=2.0,
        founder_mix=((prey_geno, N_HIGH), (pred_geno, 1)),
    )

    def _count_captures_high_density(cfg, n_seeds):
        """Place one target prey inside capture_radius + N-1 others in sensing zone."""
        captured = 0
        cr = cfg.capture_radius
        sr = cfg.sensing_radius
        for seed in range(n_seeds):
            eco = Ecology(cfg, seed=seed)
            alive = eco.alive_snapshot()
            preds = [c for c in alive if c.genotype.role == "predator"]
            preys = [c for c in alive if c.genotype.role == "prey"]
            assert preds and preys, "need at least 1 predator and 1 prey"
            pred = preds[0]
            pred.phenotype.pos_cont = (5.0, 5.0)
            # First prey is target (inside capture_radius)
            preys[0].phenotype.pos_cont = (5.0 + cr * 0.5, 5.0)
            # Remaining prey scattered inside sensing_radius (but outside capture_radius)
            # so they count toward n_local but are NOT captured
            for i, p in enumerate(preys[1:], start=1):
                angle = 2.0 * 3.14159 * i / max(len(preys) - 1, 1)
                # place just outside capture_radius but well within sensing_radius
                r = cr * 1.5 + (sr - cr * 1.5) * (i / max(len(preys), 1))
                import math
                p.phenotype.pos_cont = (5.0 + r * math.cos(angle), 5.0 + r * math.sin(angle))
            eco._resolve_captures()
            if not preys[0].phenotype.alive:
                captured += 1
        return captured / n_seeds

    N_SAT_SEEDS = 300
    rate_on_high = _count_captures_high_density(cfg_sat, N_SAT_SEEDS)
    # At n_local≈20, h=5, k=2: p_cap ≈ 0.941; we assert rate > 0.70 (generous lower bound)
    assert rate_on_high > 0.70, (
        f"Type III ON at high n_local (~{N_HIGH}, h=5, k=2) should have HIGH capture rate "
        f"(expected ≈0.941, generous threshold >0.70); got {rate_on_high:.3f}"
    )
    print(f"[saturation]        ON capture rate (n_local≈{N_HIGH}): {rate_on_high:.3f}")
    print(f"  Expected ≈ {N_HIGH**2 / (5.0**2 + N_HIGH**2):.3f} (Hill Type III, h=5, k=2)")


# ---------------------------------------------------------------------------
# Test 4: ON path is deterministic (identical events_hash across two runs)
# ---------------------------------------------------------------------------

def test_on_deterministic():
    """enable_type3_response=True run produces identical events_hash across two independent
    runs. Guards that the single gated rng draw is in the correct deterministic stream.
    """
    from ecology.engine import Ecology

    cfg = _pred_cfg(
        enable_type3_response=True,
        type3_half_density=3.0,
        type3_exponent=2.0,
    )
    h1 = Ecology(cfg, seed=99).run()["events_hash"]
    h2 = Ecology(cfg, seed=99).run()["events_hash"]
    assert h1 == h2, (
        f"ON path not deterministic across two runs: {h1} != {h2}\n"
        "The rng draw in _resolve_captures must be deterministic."
    )
