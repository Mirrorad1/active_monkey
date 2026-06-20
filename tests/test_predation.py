"""tests/test_predation.py — Exp 248 Rung 0a/0b-i: Genotype.role + enable_predation gate.

Tests (all must pass after implementation; the first two FAIL before it):
  1. role defaults to "prey" and is the LAST field.
  2. role is never mutated (copied verbatim from parent, no rng draw).
  3. OFF self-consistency: with enable_predation=False (default) the per-role
     branch is never entered and the events_hash is self-consistent between
     two identical runs.

Rung 0b-i tests (Task 2):
  4. Phenotype has move_hx, move_hy, move_d fields (all float, default 0.0).
  5. With enable_predation=True (prey-only, continuous world), the run does NOT raise
     and produces a reproducible (identical) events_hash across two independent runs.
"""
import dataclasses as D
import numpy as np
import pytest

from ecology.genotype import Genotype, mutate, founder


def test_role_defaults_prey_and_is_last_field():
    g = founder()
    assert g.role == "prey"
    assert list(D.fields(Genotype))[-1].name == "role"


def test_role_never_mutates():
    rng = np.random.default_rng(0)
    g = D.replace(founder(), role="predator")
    child = mutate(g, rng, 0.2, mutate_continuous_locomotion=True)
    assert child.role == "predator"  # copied verbatim, never perturbed


def test_off_self_consistency():
    """With enable_predation=False (default), two identical runs produce the same events_hash."""
    import dataclasses as D
    from ecology.engine import Ecology
    from ecology.scenarios import SCENARIOS
    cfg = D.replace(SCENARIOS["balanced"], horizon=20)
    assert cfg.enable_predation is False
    h1 = Ecology(cfg, seed=42).run()["events_hash"]
    h2 = Ecology(cfg, seed=42).run()["events_hash"]
    assert h1 == h2, f"OFF path not deterministic: {h1} != {h2}"


# ---------------------------------------------------------------------------
# Rung 0b-i: Phenotype move-heading fields (Task 2)
# ---------------------------------------------------------------------------

def test_phenotype_has_move_heading_fields():
    """Phenotype must have move_hx, move_hy, move_d fields (float, default 0.0).

    These fields store the realized move heading for gated eat-heading reuse.
    They must NEVER appear in events_hash (they are pure telemetry).
    """
    from ecology.creature import Phenotype
    ph = Phenotype(energy=1.0, age=0, pos=0)
    assert hasattr(ph, "move_hx"), "Phenotype missing move_hx field"
    assert hasattr(ph, "move_hy"), "Phenotype missing move_hy field"
    assert hasattr(ph, "move_d"), "Phenotype missing move_d field"
    assert ph.move_hx == 0.0, f"move_hx default should be 0.0, got {ph.move_hx}"
    assert ph.move_hy == 0.0, f"move_hy default should be 0.0, got {ph.move_hy}"
    assert ph.move_d == 0.0, f"move_d default should be 0.0, got {ph.move_d}"


def test_enable_predation_on_prey_only_continuous_does_not_raise_and_is_deterministic():
    """With enable_predation=True, prey-only continuous-world run does not raise
    and produces an identical events_hash across two independent runs (determinism).

    This is the Rung 0b-i smoke test: the move-heading store and eat-heading reuse
    must plumb correctly when enable_predation=True and all founders are prey.
    """
    from ecology.engine import Ecology
    from ecology.scenarios import SCENARIOS

    # Build a minimal continuous-world config with enable_predation=True.
    # All founders default to role="prey" so no predator wiring is needed.
    cfg = D.replace(
        SCENARIOS["balanced"],
        enable_continuous_locomotion=True,
        continuous_layout="bump",
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.0,
        mutation_rate=0.0,
        horizon=30,
        enable_predation=True,          # GATE ON
        mutate_predator_speed=False,    # prey-only: no predator speed mutation
        freeze_prey_speed=False,        # default
    )
    # Run twice with same seed — must not raise and must be deterministic.
    h1 = Ecology(cfg, seed=7).run()["events_hash"]
    h2 = Ecology(cfg, seed=7).run()["events_hash"]
    assert h1 == h2, (
        f"enable_predation=True prey-only continuous run is NOT deterministic!\n"
        f"  run1: {h1}\n"
        f"  run2: {h2}"
    )
