from active_loop.oracle import Oracle
from active_loop.task_env import TaskEnv


def test_oracle_gives_correcting_hint_when_help_needed_no_noise():
    env = TaskEnv(seed=0, num_steps=10)
    oracle = Oracle(seed=0, noise=0.0)
    step = next(i for i in range(10) if env.needs_help(i))
    hint, feedback = oracle.answer(env, step)
    assert hint == "correct"
    assert feedback in ("right", "wrong")


def test_oracle_noise_can_mislead():
    env = TaskEnv(seed=1, num_steps=10)
    oracle = Oracle(seed=1, noise=1.0)
    step = next(i for i in range(10) if env.needs_help(i))
    hint, _ = oracle.answer(env, step)
    assert hint == "misleading"


def test_oracle_is_deterministic_per_seed():
    env = TaskEnv(seed=2, num_steps=10)
    o1 = Oracle(seed=5, noise=0.3)
    o2 = Oracle(seed=5, noise=0.3)
    assert [o1.answer(env, i) for i in range(10)] == [o2.answer(env, i) for i in range(10)]
