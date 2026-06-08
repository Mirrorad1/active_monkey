import numpy as np

from active_loop.controller import Controller, Action
from active_loop.worker import MockWorker
from active_loop.task_env import TaskEnv
from active_loop.oracle import Oracle
from active_loop.hybrid import run_episode, Trajectory


def test_run_episode_returns_trajectory_of_right_length():
    env = TaskEnv(seed=0, num_steps=6)
    traj = run_episode(Controller(seed=0), MockWorker(seed=0), env, Oracle(seed=0))
    assert isinstance(traj, Trajectory)
    assert len(traj.actions) == 6
    assert len(traj.neg_efe) == 6
    assert len(traj.successes) == 6
    assert all(np.all(np.isfinite(np.asarray(e))) for e in traj.neg_efe)


def test_ask_rate_is_a_fraction():
    env = TaskEnv(seed=0, num_steps=10)
    traj = run_episode(Controller(seed=0), MockWorker(seed=0), env, Oracle(seed=0))
    assert 0.0 <= traj.ask_rate() <= 1.0


def test_learning_changes_A_matrix():
    ctrl = Controller(seed=0)
    A_before = np.asarray(ctrl.agent.A[0]).copy()
    env = TaskEnv(seed=0, num_steps=10)
    traj = run_episode(ctrl, MockWorker(seed=0), env, Oracle(seed=0))
    ctrl.learn(traj)
    A_after = np.asarray(ctrl.agent.A[0])
    assert not np.allclose(A_before, A_after)
