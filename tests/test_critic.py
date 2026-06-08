from active_loop.critic import MockCritic, Verdict


def test_mock_critic_approves():
    v = MockCritic(approve=True).review("some diff", repo=".")
    assert isinstance(v, Verdict) and v.approved is True
    assert isinstance(v.reason, str)


def test_mock_critic_rejects():
    v = MockCritic(approve=False).review("some diff", repo=".")
    assert v.approved is False
