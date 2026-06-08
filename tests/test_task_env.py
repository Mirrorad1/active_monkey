from active_loop.task_env import TaskEnv


def test_episode_has_fixed_length():
    env = TaskEnv(seed=0, num_steps=8)
    assert env.num_steps == 8


def test_needs_help_is_deterministic_per_seed():
    env_a = TaskEnv(seed=2, num_steps=10)
    env_b = TaskEnv(seed=2, num_steps=10)
    assert [env_a.needs_help(i) for i in range(10)] == [env_b.needs_help(i) for i in range(10)]


def test_some_steps_need_help():
    env = TaskEnv(seed=0, num_steps=20)
    helps = [env.needs_help(i) for i in range(20)]
    assert any(helps) and not all(helps)
