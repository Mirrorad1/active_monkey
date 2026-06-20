"""tests/test_predator_self_limit.py — Exp 254: gated predator self-limiting mortality
(density-dependent, byte-identical OFF).

Tests (all must pass after implementation):

  1. test_off_byte_identical: enable_predator_self_limit=False → identical events_hash
     across two runs (real OFF proof).
  2. test_on_differs_from_off: enable_predator_self_limit=True (low kc) differs in
     events_hash from OFF (mechanism is not a silent no-op).
  3. test_caps_predator_count: in a config where predators would otherwise grow large
     (abundant prey via decoupled births + generous capture params), turning ON predator
     self-limit with kc=20, hmax=0.2 keeps predator count bounded near/below K_P,
     whereas OFF it grows higher. Assert ON predator peak/eq < OFF.
  4. test_on_deterministic: ON reproduces identical events_hash across two runs.
"""
import dataclasses as D
import pytest


# ---------------------------------------------------------------------------
# Config helpers
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
    """enable_predator_self_limit=False → events_hash is bit-for-bit identical across two runs.

    Guards that the OFF path draws no new rng and produces no code-path change.
    """
    from ecology.engine import Ecology

    cfg = _pred_cfg(enable_predator_self_limit=False)
    h1 = Ecology(cfg, seed=42).run()["events_hash"]
    h2 = Ecology(cfg, seed=42).run()["events_hash"]
    assert h1 == h2, f"OFF path not byte-identical: {h1} != {h2}"


# ---------------------------------------------------------------------------
# Test 2: ON differs from OFF (mechanism is not a no-op)
# ---------------------------------------------------------------------------

def test_on_differs_from_off():
    """enable_predator_self_limit=True produces a different events_hash from OFF.

    A low predator_self_limit_kc (so the hazard is significant) must alter outcomes.
    """
    from ecology.engine import Ecology

    cfg_off = _pred_cfg(enable_predator_self_limit=False)
    cfg_on = _pred_cfg(
        enable_predator_self_limit=True,
        predator_self_limit_kc=5.0,   # very low K_P → high hazard even at small N_pred
        predator_self_limit_hmax=0.3,
        predator_self_limit_theta=1.0,
    )
    h_off = Ecology(cfg_off, seed=5).run()["events_hash"]
    h_on = Ecology(cfg_on, seed=5).run()["events_hash"]
    assert h_on != h_off, (
        "enable_predator_self_limit=True produced the SAME events_hash as OFF — "
        "the mechanism is a silent no-op (gate leak or wrong placement)"
    )


# ---------------------------------------------------------------------------
# Test 3: Predator self-limit caps predator count
# ---------------------------------------------------------------------------

def test_caps_predator_count():
    """With abundant prey (decoupled births), predator self-limit keeps predator count
    bounded near/below K_P, while OFF allows higher predator numbers.

    Config: generous capture_radius + max_captures_per_step + high prey_birth_rate so
    predators can boom without self-limiting mortality. ON arm: kc=20, hmax=0.2 should
    cap predators well below the OFF arm's peak.
    """
    from ecology.genotype import founder
    from ecology.scenarios import SCENARIOS
    from ecology.engine import Ecology

    prey_geno = D.replace(founder(), locomotor_speed=1.0, role="prey")
    pred_geno = D.replace(founder(), locomotor_speed=1.4, role="predator")

    base_kw = dict(
        enable_continuous_locomotion=True,
        continuous_layout="bump",
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.0,
        mutation_rate=0.0,
        horizon=400,
        enable_continuous_depletion_intake=True,
        enable_predation=True,
        mutate_predator_speed=False,
        freeze_prey_speed=True,
        founder_mix=((prey_geno, 20), (pred_geno, 5)),
        capture_radius=1.5,
        max_captures_per_step=2,
        assimilation_efficiency=0.8,
        # Abundant prey via decoupled births
        enable_decoupled_prey_birth=True,
        prey_birth_rate=0.5,
        prey_carrying_capacity=200.0,
        max_population=2000,
    )

    cfg_off = D.replace(SCENARIOS["balanced"], **base_kw,
                        enable_predator_self_limit=False)
    cfg_on = D.replace(SCENARIOS["balanced"], **base_kw,
                       enable_predator_self_limit=True,
                       predator_self_limit_kc=20.0,
                       predator_self_limit_hmax=0.2,
                       predator_self_limit_theta=1.0)

    def _peak_pred(cfg, seed):
        eco = Ecology(cfg, seed=seed)
        peak = 0
        while eco.has_alive() and not eco.exploded and eco.t < cfg.horizon:
            eco.step()
            n_pred = sum(1 for c in eco._alive_list if c.genotype.role == "predator")
            if n_pred > peak:
                peak = n_pred
        return peak

    # Run multiple seeds, compare peaks
    SEEDS = [0, 1, 2, 3, 4]
    peaks_off = [_peak_pred(cfg_off, s) for s in SEEDS]
    peaks_on = [_peak_pred(cfg_on, s) for s in SEEDS]

    avg_off = sum(peaks_off) / len(peaks_off)
    avg_on = sum(peaks_on) / len(peaks_on)

    print(
        f"\n[cap test] OFF peaks: {peaks_off} avg={avg_off:.1f}\n"
        f"           ON  peaks: {peaks_on} avg={avg_on:.1f}"
    )

    assert avg_on < avg_off, (
        f"Predator self-limit did NOT cap predator count!\n"
        f"  ON avg peak={avg_on:.1f}  OFF avg peak={avg_off:.1f}\n"
        f"  Expected ON < OFF (self-limiting mortality must suppress boom)"
    )


# ---------------------------------------------------------------------------
# Test 4: ON path is deterministic (identical events_hash across two runs)
# ---------------------------------------------------------------------------

def test_on_deterministic():
    """enable_predator_self_limit=True run produces identical events_hash across two
    independent runs. Guards that the gated rng draw is in the correct deterministic stream.
    """
    from ecology.engine import Ecology

    cfg = _pred_cfg(
        enable_predator_self_limit=True,
        predator_self_limit_kc=30.0,
        predator_self_limit_hmax=0.1,
        predator_self_limit_theta=1.0,
    )
    h1 = Ecology(cfg, seed=99).run()["events_hash"]
    h2 = Ecology(cfg, seed=99).run()["events_hash"]
    assert h1 == h2, (
        f"ON path not deterministic across two runs: {h1} != {h2}\n"
        "The rng draw in _step_one_creature (predator self-limit) must be deterministic."
    )
