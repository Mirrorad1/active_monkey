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


# ===========================================================================
# Rung 0b-ii: gated 3-phase predation loop + frozen snapshot + pursuit/flee +
# capture + ON golden  (Task 3)
# ===========================================================================
#
# _PREDATION_ON_GOLDEN_HASH is the determinism fingerprint of the focal two-role
# predation run (_pred_cfg(enable_predation=True), seed=5).  Pinned from the FIRST
# green run.  DO NOT REPIN WITHOUT CAUSE — a changed value means the predation
# dynamics or rng stream shifted; investigate before updating.
_PREDATION_ON_GOLDEN_HASH = "5862b4e83ac278f9ef6d6ba5d45f041bbeb158a05962d24c49be5440475e399a"  # do not repin without cause


def _pred_cfg(**over):
    """Two-role predator/prey common-garden config for the predation focal test.

    A founder_mix of breed-true prey (escape locomotor_speed=1.0) + faster predators
    (pursuit locomotor_speed=1.4), enable_predation ON, continuous world with
    depletion-aware intake, fixed mutation, horizon 300.  Roles are seeded only via
    founder_mix; predator speed mutates (the co-evolving arm), prey speed is frozen.
    """
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
        enable_continuous_depletion_intake=True,   # depletion-intake ON
        enable_predation=True,                      # GATE ON
        mutate_predator_speed=True,                 # co-evolving arm
        freeze_prey_speed=True,                     # prey breed true
        founder_mix=((prey_geno, 10), (pred_geno, 3)),
    )
    base.update(over)
    return D.replace(SCENARIOS["balanced"], **base)


def test_predation_on_differs_from_off():
    """Predation is not a silent no-op: ON and OFF produce different events_hash."""
    from ecology.engine import Ecology
    on = Ecology(_pred_cfg(enable_predation=True), seed=5).run()["events_hash"]
    off = Ecology(_pred_cfg(enable_predation=False), seed=5).run()["events_hash"]
    assert on != off  # predation is not a silent no-op


def test_predation_on_golden_hash():
    """The focal two-role predation run reproduces its pinned golden hash."""
    from ecology.engine import Ecology
    h = Ecology(_pred_cfg(enable_predation=True), seed=5).run()["events_hash"]
    assert h == _PREDATION_ON_GOLDEN_HASH


def test_predation_on_invariant_under_shuffle():
    """Frozen pre-move snapshot ⇒ headings + capture are order-independent.

    The predation branch must NOT call rng.shuffle (it would diverge the rng stream
    from the shuffle=False path); shuffle becomes inert and the hashes are identical.
    """
    from ecology.engine import Ecology
    a = Ecology(_pred_cfg(enable_predation=True, shuffle_creature_order=False), seed=5).run()["events_hash"]
    b = Ecology(_pred_cfg(enable_predation=True, shuffle_creature_order=True), seed=5).run()["events_hash"]
    assert a == b  # frozen pre-move snapshot ⇒ headings order-independent


# ---------------------------------------------------------------------------
# Step 5: anti-cheat seam tests
# ---------------------------------------------------------------------------

def test_capture_rate_invariant_to_locomotor_speed_at_fixed_positions():
    """ANTI-CHEAT: capture_radius is NEVER a function of locomotor_speed.

    With positions held fixed (we resolve captures directly), the set of captured
    prey depends ONLY on the geometric capture_radius — changing the predator's
    locomotor_speed must not change who is captured.
    """
    from ecology.engine import Ecology

    def _captured_ids(pred_speed: float):
        cfg = _pred_cfg(horizon=1)
        eco = Ecology(cfg, seed=5)
        # Hand-place: predator at id 0, two prey straddling the capture radius.
        alive = eco.alive_snapshot()
        pred = alive[0]
        pred.genotype = D.replace(pred.genotype, role="predator", locomotor_speed=pred_speed)
        pred.phenotype.pos_cont = (6.0, 6.0)
        prey_a, prey_b = alive[1], alive[2]
        prey_a.genotype = D.replace(prey_a.genotype, role="prey")
        prey_b.genotype = D.replace(prey_b.genotype, role="prey")
        cr = cfg.capture_radius
        prey_a.phenotype.pos_cont = (6.0 + cr * 0.5, 6.0)   # inside capture radius
        prey_b.phenotype.pos_cont = (6.0 + cr * 5.0, 6.0)   # well outside
        eco._resolve_captures()
        return {prey_a.creature_id: not prey_a.phenotype.alive,
                prey_b.creature_id: not prey_b.phenotype.alive}

    slow = _captured_ids(0.5)
    fast = _captured_ids(4.0)
    assert slow == fast, (
        "capture outcome changed with locomotor_speed at FIXED positions — "
        "capture_radius leaked a dependence on speed (anti-cheat violation)"
    )
    # Sanity: the near prey is captured, the far one is not (geometry, not speed).
    vals = list(slow.values())
    assert vals == [True, False]


def test_prey_heading_equals_best_heading_when_no_predator_in_range():
    """Step-5 invariant: prey flee term is EXACTLY 0.0 when no predator is in range,
    so the role-aware prey heading EQUALS best_heading (the pure foraging heading).

    This certifies the eat-heading-reuse seam reduces to the enable_predation=False
    foraging heading whenever the flee term vanishes."""
    from ecology.engine import Ecology

    cfg = _pred_cfg(horizon=1)
    eco = Ecology(cfg, seed=5)
    alive = eco.alive_snapshot()
    prey = next(c for c in alive if c.genotype.role == "prey")
    x, y = prey.phenotype.pos_cont
    # No predators anywhere near: build an empty predator snapshot.
    hx, hy = eco._role_heading(prey, x, y, prey_snap=[], pred_snap=[])
    bx, by = eco.cont_world.best_heading(x, y)
    assert hx == bx and hy == by, (
        "prey heading with no predator in range must EQUAL best_heading "
        f"(flee term not exactly zero): got ({hx},{hy}) vs ({bx},{by})"
    )
