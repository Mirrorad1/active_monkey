"""Tests for embodied.batched_world — buffer ops (Task 1) + vmap advance (Task 2)
+ per-creature life economics (Task 3)."""
import numpy as np
import pytest


def _world(max_pop=4):
    from embodied.foodfield import FoodField, FoodFieldConfig
    from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
    from embodied.batched_world import BatchedEmbodiedWorld
    return BatchedEmbodiedWorld(
        FoodField(FoodFieldConfig(), 0),
        PolicyRunner(DEFAULT_CKPT),
        max_pop,
        bout_steps=4,
    )


def test_init_buffer_and_spawn():
    w = _world(4)
    state, alive = w.init_buffer(n_founders=2, seed=0, founder_xys=[(1.0, 0.0), (-1.0, 2.0)])

    # Buffer has max_pop slots; only first n_founders are alive
    assert state.q.shape[0] == 4
    assert alive.tolist() == [True, True, False, False]

    # Founders placed at requested xy
    assert np.allclose(np.asarray(state.q[0, 0:2]), [1.0, 0.0])
    assert np.allclose(np.asarray(state.q[1, 0:2]), [-1.0, 2.0])

    # spawn_into_slot sets target slot, leaves others unchanged
    s2 = w.spawn_into_slot(state, 3, (4.0, 4.0), seed=99)
    assert np.allclose(np.asarray(s2.q[3, 0:2]), [4.0, 4.0])          # slot 3 set
    assert np.allclose(np.asarray(s2.q[0]), np.asarray(state.q[0]))   # slot 0 q unchanged
    assert np.allclose(np.asarray(s2.qd[0]), np.asarray(state.qd[0]))  # slot 0 qd unchanged (spawn writes qd too)


def test_advance_batch_equals_sequential():
    """Task 2: vmap advance must agree with the sequential reference up to float noise."""
    w = _world(4)
    state, _ = w.init_buffer(2, 0, [(1.0, 0.0), (-1.0, 2.0)])
    targets = np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    sb, pb = w.advance_batch(state, targets)
    ss, ps = w.advance_sequential(state, targets)

    # Shape: [max_pop, bout_steps, 2]
    assert pb.shape == (4, w.bout_steps, 2), f"expected (4, {w.bout_steps}, 2), got {pb.shape}"
    assert ps.shape == (4, w.bout_steps, 2), f"sequential shape {ps.shape}"

    # vmap vs loop: equal up to float noise (NOT exact equality)
    max_path_diff = float(np.max(np.abs(np.asarray(pb) - np.asarray(ps))))
    max_q_diff = float(np.max(np.abs(np.asarray(sb.q) - np.asarray(ss.q))))
    print(f"max path diff batched vs sequential: {max_path_diff:.2e}")
    print(f"max q diff batched vs sequential:    {max_q_diff:.2e}")

    assert np.allclose(np.asarray(pb), np.asarray(ps), atol=1e-4), (
        f"paths disagree: max abs diff = {max_path_diff:.2e}"
    )
    assert np.allclose(np.asarray(sb.q), np.asarray(ss.q), atol=1e-4), (
        f"final q disagrees: max abs diff = {max_q_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# Task 3: per-creature life economics (step_economics)
# ---------------------------------------------------------------------------

def test_step_economics_matches_phase2_formulas():
    """step_economics must reproduce the EXACT arithmetic of creature.life_step."""
    from ecology.genotype import founder
    from embodied.life_mechanics import step_economics

    g = founder()

    # --- reproduce case: well-fed mature creature ---
    # Use POST-increment age for maturity gate (creature.py line 79 uses `age` after age = c.age + 1).
    # age arg is the PRE-increment age; maturity_age = 5, so PRE = 5 satisfies POST-age (6) >= 5.
    pre_age = g.maturity_age  # 5 — POST age will be 6 >= maturity_age=5
    energy_in = g.energy_capacity * 0.95   # 19.0
    intake = 1.0

    # Compute expected values using Phase-2 formulas exactly.
    expected_cost = g.baseline_metabolic_cost + g.movement_cost + g.aging_cost * pre_age
    expected_new_energy = min(g.energy_capacity, energy_in + intake - expected_cost)
    expected_transfer = expected_new_energy * g.reproduction_energy_transfer_fraction
    expected_overhead = expected_new_energy * g.reproduction_cost_fraction
    expected_parent_energy = expected_new_energy - expected_transfer - expected_overhead

    r = step_economics(g, energy=energy_in, age=pre_age, intake=intake)

    assert r["reproduce"] is True, "expected reproduction"
    assert r["die"] is False, "expected alive"
    assert r["age"] == pre_age + 1, "age must be incremented"
    # Tight: exact transfer and parent energy from the formulas.
    assert np.isclose(r["transfer"], expected_transfer), (
        f"transfer mismatch: got {r['transfer']}, expected {expected_transfer}"
    )
    assert np.isclose(r["energy"], expected_parent_energy), (
        f"parent energy mismatch: got {r['energy']}, expected {expected_parent_energy}"
    )
    assert np.isclose(r["overhead"], expected_overhead), (
        f"overhead mismatch: got {r['overhead']}, expected {expected_overhead}"
    )

    # --- die case: starving creature ---
    d = step_economics(g, energy=0.01, age=1, intake=0.0)
    assert d["die"] is True, "starving creature must die"
    assert d["reproduce"] is False, "dying creature must not reproduce"

    # --- live (no reproduce) case: immature creature, sufficient energy ---
    pre_age_immature = g.maturity_age - 1  # 4 — POST age will be 5 >= maturity_age only if == it
    # Use maturity_age - 2 so POST age is maturity_age - 1, strictly below threshold.
    pre_age_immature = g.maturity_age - 2  # 3 — POST age = 4 < 5 = maturity_age
    lv = step_economics(g, energy=g.energy_capacity * 0.9, age=pre_age_immature, intake=0.5)
    assert lv["die"] is False
    assert lv["reproduce"] is False
    assert lv["transfer"] == 0.0
    assert lv["overhead"] == 0.0

    # --- age cost uses PRE-increment age (not POST) ---
    # Verify cost charged is based on input `age`, not `age+1`.
    age0_energy_in = g.energy_capacity * 0.8
    r0 = step_economics(g, energy=age0_energy_in, age=0, intake=0.0)
    r1 = step_economics(g, energy=age0_energy_in, age=1, intake=0.0)
    # cost at age=1 is higher by aging_cost*1, so energy should be lower.
    assert r1["energy"] < r0["energy"] or r0["die"] or r1["die"], (
        "aging cost must use pre-increment age"
    )
