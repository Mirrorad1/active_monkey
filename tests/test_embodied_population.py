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
