"""embodied.batched_world — MAX_POP body buffer + vmap'd advance (the GPU seam).

Task 1: batched buffer ops — init_buffer and spawn_into_slot.
Task 2+: vmap'd advance step (to be added).
"""
import jax
import jax.numpy as jnp
import numpy as np


class BatchedEmbodiedWorld:
    """Holds a [max_pop]-slot brax pipeline_state buffer for vmap'd physics.

    Attributes:
        food:      FoodField instance.
        runner:    PolicyRunner instance.
        env:       EmbodiedForageEnv (from runner).
        max_pop:   Fixed buffer capacity.
        bout_steps: Physics steps per policy bout.
    """

    def __init__(self, food_field, runner, max_pop, bout_steps=8):
        self.food = food_field
        self.runner = runner
        self.env = runner.env
        self.max_pop = int(max_pop)
        self.bout_steps = int(bout_steps)
        self._key = jax.random.PRNGKey(0)  # deterministic policy ignores it

    # ------------------------------------------------------------------
    # Buffer construction
    # ------------------------------------------------------------------

    def init_buffer(self, n_founders, seed, founder_xys):
        """Return (batched_state, alive) for a freshly-seeded buffer.

        Args:
            n_founders:  Number of living founder slots (must be <= max_pop).
            seed:        Integer RNG seed for body initialisation.
            founder_xys: List of (x, y) tuples; first n_founders entries used.

        Returns:
            batched_state: brax pipeline_state whose array leaves have shape
                           (max_pop, ...).  All max_pop slots hold valid poses;
                           only the first n_founders have the requested xy.
            alive: numpy bool array [max_pop], True for the first n_founders.
        """
        keys = jax.random.split(jax.random.PRNGKey(seed), self.max_pop)
        batched = jax.vmap(lambda k: self.env.reset(k).pipeline_state)(keys)

        # Place founders at their requested root xy positions.
        q = batched.q
        for i, (x, y) in enumerate(list(founder_xys)[:n_founders]):
            q = q.at[i, 0:2].set(jnp.array([float(x), float(y)]))
        batched = batched.replace(q=q)

        alive = np.zeros(self.max_pop, dtype=bool)
        alive[:n_founders] = True
        return batched, alive

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def spawn_into_slot(self, batched_state, idx, xy, seed):
        """Write a fresh body into slot *idx* at position *xy*.

        Resets q (qpos) and qd (qvel) for that slot; all other slots are
        unchanged.  The fresh body is drawn from env.reset so joint angles
        and velocities start from a valid resting pose.

        Args:
            batched_state: Current (max_pop, ...) batched pipeline_state.
            idx:           Integer slot index to overwrite (0 <= idx < max_pop).
            xy:            (x, y) tuple for the root position.
            seed:          Integer RNG seed for the new body reset.

        Returns:
            Updated batched_state with slot *idx* replaced.
        """
        fresh = self.env.reset(jax.random.PRNGKey(int(seed))).pipeline_state

        # Build fresh q with the requested xy substituted in.
        fresh_q = fresh.q.at[0:2].set(jnp.array([float(xy[0]), float(xy[1])]))

        # Write q and qd into the requested slot; leave all other slots intact.
        q = batched_state.q.at[idx].set(fresh_q)
        qd = batched_state.qd.at[idx].set(fresh.qd)
        return batched_state.replace(q=q, qd=qd)

    # ------------------------------------------------------------------
    # Advance: vmap'd bout (GPU seam) + sequential reference
    # ------------------------------------------------------------------

    def _single_bout(self, state, target):
        """One body's bout: bout_steps control steps toward a fixed target.

        Obs matches policy_runner.PolicyRunner.build_obs exactly:
            concat(q[2:], qd, target - q[0:2])
        where target plays the role of food_xy in production.

        Args:
            state:  A single (un-batched) brax pipeline_state.
            target: jnp array of shape (2,) — the navigation target xy.

        Returns:
            (final_state, path) where path has shape (bout_steps, 2).
        """
        def step(s, _):
            obs = jnp.concatenate([s.q[2:], s.qd, jnp.asarray(target) - s.q[0:2]])
            action, _ = self.runner._infer(obs, self._key)   # deterministic mean action
            ns = self.env.pipeline_step(s, action)
            return ns, ns.q[0:2]

        final, path = jax.lax.scan(step, state, None, length=self.bout_steps)
        return final, path   # path: [bout_steps, 2]

    def advance_batch(self, batched_state, targets):
        """vmap + jit'd bout over all max_pop slots simultaneously.

        Args:
            batched_state: brax pipeline_state with leading max_pop axis.
            targets:       Array-like of shape (max_pop, 2).

        Returns:
            (new_batched_state, paths) where paths has shape (max_pop, bout_steps, 2).
        """
        if not hasattr(self, "_adv"):
            self._adv = jax.jit(jax.vmap(self._single_bout))
        return self._adv(batched_state, jnp.asarray(targets))

    def advance_sequential(self, batched_state, targets):
        """Reference: identical math via a plain Python loop over slots.

        Not used in production; exists to validate advance_batch via the
        equivalence gate in tests.

        Args:
            batched_state: brax pipeline_state with leading max_pop axis.
            targets:       Array-like of shape (max_pop, 2).

        Returns:
            (new_batched_state, paths) where paths has shape (max_pop, bout_steps, 2).
        """
        targets = jnp.asarray(targets)
        finals, paths = [], []
        for i in range(self.max_pop):
            s_i = jax.tree_util.tree_map(lambda a, _i=i: a[_i], batched_state)
            f_i, p_i = self._single_bout(s_i, targets[i])
            finals.append(f_i)
            paths.append(p_i)
        final = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *finals)
        return final, jnp.stack(paths)
