from active_loop.worker import MockWorker
from active_loop.signals import WorkerSignal


def test_mock_worker_is_deterministic():
    w1 = MockWorker(seed=3)
    w2 = MockWorker(seed=3)
    sigs1 = [w1.do_step(step=i, hint=None) for i in range(5)]
    sigs2 = [w2.do_step(step=i, hint=None) for i in range(5)]
    assert sigs1 == sigs2


def test_mock_worker_signal_fields_in_range():
    w = MockWorker(seed=1)
    sig = w.do_step(step=0, hint=None)
    assert isinstance(sig, WorkerSignal)
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.error_count >= 0


def test_hint_improves_success_probability():
    def success_rate(hint):
        succ = 0
        n = 50
        for s in range(n):
            w = MockWorker(seed=s)
            succ += int(w.do_step(step=0, hint=hint).succeeded)
        return succ / n
    assert success_rate("correct") > success_rate(None)
