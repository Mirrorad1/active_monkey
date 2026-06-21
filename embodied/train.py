"""embodied.train — PPO training to teach the quadruped to walk to food.

Phase 1 (CPU smoke + GPU full): normalize_observations=False for simplicity
and robustness — no normalizer state to thread through make_inference.
Phase 2 can enable it once we have GPU checkpointing infrastructure.

JAX compatibility shim: JAX 0.10.1 removes jax.device_put_replicated
(brax 0.14.2 still calls it). We patch it at import time with the
recommended jnp.expand_dims(jax.device_put(...), 0) equivalent,
which is correct for single-CPU execution (1 device, leading axis = 1).
"""
import functools
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# ---------------------------------------------------------------------------
# JAX 0.10.x compat: restore jax.device_put_replicated removed in 0.10.1
# brax 0.14.2 calls jax.device_put_replicated(training_state, devices) to
# add a device leading-axis.  On CPU (1 device) we add a size-1 leading axis.
# ---------------------------------------------------------------------------
def _device_put_replicated_compat(x, devices):
    """Drop-in for jax.device_put_replicated on single-device CPU."""
    return jtu.tree_map(
        lambda a: jnp.expand_dims(jax.device_put(a, devices[0]), 0), x
    )

try:
    # JAX 0.10.1 REMOVED jax.device_put_replicated entirely — accessing the
    # attribute raises AttributeError. This probe triggers that (or any other
    # failure) so the single-device shim is installed ONLY when the real API is
    # unusable; a working jax.device_put_replicated is never overridden.
    _sentinel = jnp.zeros(1)
    jax.device_put_replicated(_sentinel, jax.local_devices()[:1])
except Exception:  # broad on purpose: ANY failure => fall back to the compat shim
    jax.device_put_replicated = _device_put_replicated_compat

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import model

from embodied.env import EmbodiedForageEnv

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

SMOKE = dict(
    num_timesteps=2048,
    num_envs=8,
    batch_size=8,
    num_minibatches=1,
    num_evals=1,
    episode_length=200,
    unroll_length=10,
)

FULL = dict(
    num_timesteps=30_000_000,
    num_envs=2048,
    batch_size=1024,
    num_minibatches=32,
    num_evals=5,
    episode_length=1000,
    unroll_length=20,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(num_timesteps: int, seed: int, out_dir, **overrides) -> Path:
    """Run PPO, save params, return the checkpoint path.

    Starts from SMOKE defaults so callers only need to pass num_timesteps +
    any overrides they care about.

    normalize_observations=False: Phase 1 only needs a watchable gait, not an
    optimal one; keeping this off makes the saved params a simple 3-tuple
    (normalizer_params, policy_params, value_params) with no live normalizer
    state, so make_inference is trivially reconstructable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(SMOKE)
    cfg.update(overrides)
    cfg["num_timesteps"] = num_timesteps

    env = EmbodiedForageEnv()

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(64, 64),
    )

    make_policy, params, _ = ppo.train(
        environment=env,
        seed=seed,
        network_factory=network_factory,
        num_timesteps=cfg["num_timesteps"],
        num_envs=cfg["num_envs"],
        batch_size=cfg["batch_size"],
        num_minibatches=cfg["num_minibatches"],
        num_evals=cfg["num_evals"],
        episode_length=cfg["episode_length"],
        unroll_length=cfg["unroll_length"],
        normalize_observations=False,
        use_pmap_on_reset=False,
    )

    ckpt = out_dir / "params"
    model.save_params(str(ckpt), params)
    return ckpt


def load_params(path):
    """Re-export of brax.io.model.load_params."""
    return model.load_params(str(path))


def make_inference(params):
    """Return a deterministic (obs, rng) -> (action, extras) callable.

    Reconstructs the SAME network_factory used at train time so the params
    are interpreted correctly.  normalize_observations=False means params is a
    3-tuple (normalizer_params, policy_params, value_params) with no live
    normalizer state to carry.
    """
    env = EmbodiedForageEnv()
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(64, 64),
    )
    nets = network_factory(env.observation_size, env.action_size)
    inference_factory = ppo_networks.make_inference_fn(nets)
    policy = inference_factory(params, deterministic=True)

    # Sanity-check: action shape must be (8,) for a dummy obs.
    dummy_obs = jnp.zeros(env.observation_size)
    dummy_rng = jax.random.PRNGKey(0)
    action, _extras = policy(dummy_obs, dummy_rng)
    assert action.shape == (env.action_size,), (
        f"make_inference sanity check: expected ({env.action_size},), got {action.shape}"
    )

    return policy  # callable: (obs, rng) -> (action, extras)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train PPO quadruped-forage agent.")
    p.add_argument("--smoke", action="store_true", help="Use tiny SMOKE config.")
    p.add_argument("--timesteps", type=int, default=None, help="Override num_timesteps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="embodied/checkpoints/quadruped_forage")
    a = p.parse_args()

    if a.timesteps is not None:
        n = a.timesteps
        extra = {}
    elif a.smoke:
        n = SMOKE["num_timesteps"]
        extra = {}
    else:
        n = FULL["num_timesteps"]
        extra = {k: v for k, v in FULL.items() if k != "num_timesteps"}

    t0 = time.time()
    ckpt = train(n, a.seed, a.out, **extra)
    elapsed = time.time() - t0
    print(f"checkpoint: {ckpt}  (wall time: {elapsed:.1f}s)")
