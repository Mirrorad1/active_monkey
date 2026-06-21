import numpy as np, pytest
import jax
from pathlib import Path


def test_policy_runner_obs_and_action(tmp_path):
    import jax, jax.numpy as jnp
    from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
    pr = PolicyRunner(DEFAULT_CKPT)
    env = pr.env
    state = env.reset(jax.random.PRNGKey(0))
    obs = pr.build_obs(state.pipeline_state, (3.0, 0.0))
    assert obs.shape == (env.observation_size,)        # 29
    act = pr.act(obs)
    assert act.shape == (env.action_size,)             # 8
    assert bool(jnp.isfinite(act).all())


def test_world_advance_moves_body_deterministically():
    import numpy as np
    from embodied.foodfield import FoodField, FoodFieldConfig
    from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
    from embodied.world import EmbodiedWorld
    w = EmbodiedWorld(FoodField(FoodFieldConfig(), seed=0), PolicyRunner(DEFAULT_CKPT), bout_steps=10)
    s0 = w.env.reset(jax.random.PRNGKey(0))
    sA, pathA = w.advance(s0.pipeline_state)
    sB, pathB = w.advance(s0.pipeline_state)
    assert len(pathA) == 10
    assert pathA == pathB                                  # deterministic
    # every entry is a finite (x, y) pair
    assert all(np.isfinite(x) and np.isfinite(y) for (x, y) in pathA)


def test_foodfield_deplete_regen_nearest():
    from embodied.foodfield import FoodField, FoodFieldConfig
    f = FoodField(FoodFieldConfig(extent=6.0, cells=24, capacity=1.0, regen=0.1, n_sources=5), seed=0)
    t0 = f.total()
    # consume along a path through the arena
    got = f.consume([(-3.0, 0.0), (3.0, 0.0)], deficit=10.0)
    assert got > 0.0 and got <= 10.0
    assert f.total() < t0                       # depletion happened
    f.step_regen()
    assert f.total() > (t0 - got)               # regen restored some
    # nearest_food points at a real high-resource cell
    fx, fy = f.nearest_food_xy(0.0, 0.0)
    assert -6.0 <= fx <= 6.0 and -6.0 <= fy <= 6.0
    assert f.resource_at(fx, fy) > 0.0


# ---------------------------------------------------------------------------
# Task 4: EmbodiedCreature + life_step
# ---------------------------------------------------------------------------

def _world():
    from embodied.foodfield import FoodField, FoodFieldConfig
    from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
    from embodied.world import EmbodiedWorld
    return EmbodiedWorld(FoodField(FoodFieldConfig(), seed=0), PolicyRunner(DEFAULT_CKPT), bout_steps=8)


def test_life_step_energy_and_death():
    import numpy as np, jax
    from ecology.genotype import founder
    from embodied.creature import EmbodiedCreature, life_step
    w = _world(); rng = np.random.default_rng(0)
    s = w.env.reset(jax.random.PRNGKey(0)).pipeline_state
    g = founder()
    c = EmbodiedCreature(id=0, genotype=g, energy=0.05, age=1, body_state=s)  # near-starving
    upd, child, ev, intake = life_step(c, w, rng, next_id=1)
    # low energy + metabolism => can die; if it ate enough it lives. Assert the contract holds:
    assert ev in ("live", "reproduce", "die")
    if ev == "die":
        assert not upd.alive
    assert intake >= 0.0


def test_life_step_returns_4tuple():
    """life_step must return exactly 4 values: (parent, child_or_None, event, intake)."""
    import numpy as np, jax
    from ecology.genotype import founder
    from embodied.creature import EmbodiedCreature, life_step
    w = _world(); rng = np.random.default_rng(42)
    s = w.env.reset(jax.random.PRNGKey(1)).pipeline_state
    g = founder()
    c = EmbodiedCreature(id=0, genotype=g, energy=10.0, age=0, body_state=s)
    result = life_step(c, w, rng, next_id=1)
    assert len(result) == 4, f"expected 4-tuple, got {len(result)}-tuple"
    parent, child, event, intake = result
    assert isinstance(parent, EmbodiedCreature)
    assert child is None or isinstance(child, EmbodiedCreature)
    assert event in ("live", "reproduce", "die")
    assert isinstance(intake, float)
    assert intake >= 0.0


def test_life_step_reproduce():
    """A well-fed creature that meets reproduction criteria should eventually reproduce."""
    import numpy as np, jax
    from ecology.genotype import founder
    from embodied.creature import EmbodiedCreature, life_step
    w = _world(); rng = np.random.default_rng(7)
    s = w.env.reset(jax.random.PRNGKey(2)).pipeline_state
    g = founder()
    # Start with nearly full energy and mature age
    c = EmbodiedCreature(id=0, genotype=g, energy=19.0, age=g.maturity_age + 1, body_state=s)
    # Run up to 10 bouts until we reproduce or the creature dies
    reproduced = False
    for i in range(10):
        upd, child, ev, intake = life_step(c, w, rng, next_id=i + 1)
        if ev == "reproduce":
            assert child is not None
            assert child.id == i + 1
            assert child.energy > 0.0
            assert child.age == 0
            # Genotype is copied verbatim (no mutation in Phase 2)
            assert child.genotype is g or child.genotype == g
            reproduced = True
            break
        elif ev == "die":
            break
        c = upd
    # With nearly full energy at start, reproduction should happen quickly
    assert reproduced, "Expected reproduction but creature died or never reproduced in 10 bouts"
