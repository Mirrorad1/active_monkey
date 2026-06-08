import jax
import jax.numpy as jnp
import numpy as np

from active_loop.controller import Controller, Action


def test_controller_constructs_and_steps():
    ctrl = Controller(seed=0)
    ctrl.reset()
    out = ctrl.step([1, 2, 0, 0])
    assert out.neg_efe.shape == (1, 3)
    assert out.action in (Action.ACT, Action.ASK, Action.SWITCH)
    assert np.all(np.isfinite(np.asarray(out.neg_efe)))


def test_efe_prefers_ask_under_uncertainty_vs_act_under_confidence():
    ctrl = Controller(seed=0)
    ctrl.reset()
    out_uncertain = ctrl.step([0, 1, 1, 0])
    efe_act_u = -float(out_uncertain.neg_efe[0, Action.ACT])
    efe_ask_u = -float(out_uncertain.neg_efe[0, Action.ASK])
    ctrl.reset()
    out_conf = ctrl.step([1, 2, 0, 0])
    efe_act_c = -float(out_conf.neg_efe[0, Action.ACT])
    efe_ask_c = -float(out_conf.neg_efe[0, Action.ASK])
    assert (efe_ask_u - efe_act_u) < (efe_ask_c - efe_act_c)


def test_belief_entropy_decreases_after_informative_obs():
    ctrl = Controller(seed=0)
    ctrl.reset()
    out = ctrl.step([1, 2, 0, 0])
    qs = np.asarray(out.qs[0]).reshape(-1)
    qs = qs / qs.sum()
    entropy = -np.sum(qs * np.log(qs + 1e-12))
    max_entropy = np.log(len(qs))
    assert entropy < max_entropy
