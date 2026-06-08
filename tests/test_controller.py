import jax
import jax.numpy as jnp
import numpy as np

from active_loop.controller import Controller, Action


def test_controller_constructs_and_steps():
    ctrl = Controller(seed=0)
    ctrl.reset()
    out = ctrl.step([1, 2, 0, 0])  # success, high conf, no errors, no feedback
    assert out.neg_efe.shape == (1, 3)
    assert out.action in (Action.ACT, Action.ASK, Action.SWITCH)
    assert np.all(np.isfinite(np.asarray(out.neg_efe)))


def test_action_choice_responds_to_observation():
    # The policy posterior must differ between a confident-success observation and a
    # failing-uncertain one: the controller is genuinely conditioning on what it sees.
    ctrl = Controller(seed=0)
    ctrl.reset()
    q_pi_conf = np.asarray(ctrl.step([1, 2, 0, 0]).q_pi)
    ctrl.reset()
    q_pi_fail = np.asarray(ctrl.step([0, 0, 2, 0]).q_pi)
    assert not np.allclose(q_pi_conf, q_pi_fail, atol=1e-3)


def test_ask_is_a_reachable_choice():
    # Across a small grid of seeds/observations, ASK is selected at least once,
    # confirming asking is a live option the EFE can favor. Calibrating ASK to fire
    # specifically under uncertainty is the job of the M2 free-energy loop, not of
    # these hand-set baseline priors — so M1 only asserts reachability, not calibration.
    obs_grid = [[0, 0, 2, 0], [0, 1, 1, 0], [1, 1, 1, 0], [1, 2, 0, 0]]
    chosen = []
    for seed in range(8):
        for obs in obs_grid:
            ctrl = Controller(seed=seed)
            ctrl.reset()
            chosen.append(ctrl.step(obs).action)
    assert Action.ASK in chosen


def test_belief_entropy_decreases_after_informative_obs():
    ctrl = Controller(seed=0)
    ctrl.reset()
    out = ctrl.step([1, 2, 0, 0])
    qs = np.asarray(out.qs[0]).reshape(-1)
    qs = qs / qs.sum()
    entropy = -np.sum(qs * np.log(qs + 1e-12))
    max_entropy = np.log(len(qs))
    assert entropy < max_entropy  # posterior is more peaked than uniform
