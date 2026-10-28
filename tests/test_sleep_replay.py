"""Replay must improve inference without manufacturing additional observations."""
import numpy as np
import pytest

from active_loop.sleep_replay import Encounter, ReplayModel, refit, objective
from active_loop.state import AgentState


def model(k=1):
    rng = np.random.default_rng(1)
    p0 = 1 + rng.random((2, k, 2)) if k > 1 else np.ones((2, 1, 2))
    p1 = 1 + rng.random((3, k, 2)) if k > 1 else np.ones((3, 1, 2))
    return ReplayModel(p0 / p0.sum(0), p1 / p1.sum(0), p0, p1, np.ones(k) / k)


EVENTS = (Encounter('a', 0, 0, 0, 2), Encounter('b', 1, 1, 1, 0))


def test_hand_computed_counts_use_lived_time_and_unique_evidence():
    anchor = model()
    fitted, receipt = refit(anchor, anchor, EVENTS, passes=4, decay=.5, learning_rate=2.)
    np.testing.assert_allclose(fitted.p0[:, 0, :], [[1.25, .25], [.25, 2.25]])
    expected = np.full((3, 1, 2), .25)
    expected[2, 0, 0] += 1
    expected[0, 0, 1] += 2
    np.testing.assert_allclose(fitted.p1, expected)
    assert receipt['metrics']['unique_encounters'] == 2
    assert receipt['metrics']['evidence_mass'] == pytest.approx(3.)
    assert receipt['metrics']['fit_posterior_evaluations'] == 8
    assert receipt['metrics']['objective_posterior_evaluations'] == 10
    assert receipt['metrics']['posterior_evaluations'] == 18
    assert receipt['scope'] == 'creature'


def test_extra_passes_cannot_increase_total_evidence_or_mutate_source():
    anchor, start = model(2), model(2)
    before = start.to_state().content_hash()
    one, _ = refit(anchor, start, EVENTS, passes=1)
    many, _ = refit(anchor, start, EVENTS, passes=25)
    np.testing.assert_allclose(one.p0.sum(), many.p0.sum(), atol=1e-12)
    np.testing.assert_allclose(one.p1.sum(), many.p1.sum(), atol=1e-12)
    assert start.to_state().content_hash() == before
    assert objective(many, anchor, EVENTS) >= objective(one, anchor, EVENTS) - 1e-10


def test_joint_regularized_objective_is_monotone():
    anchor = model(2)
    events = tuple(Encounter(str(i), i, i % 2, i % 2, (i // 2) % 3) for i in range(20))
    _, receipt = refit(anchor, anchor, events, passes=30)
    values = receipt['metrics']['objective']
    assert np.all(np.diff(values) >= -1e-9)


def test_chronological_refitting_restarts_from_anchor_each_pass():
    anchor = model(2)
    first, _ = refit(anchor, anchor, EVENTS, passes=1, mode='chronological')
    last, _ = refit(anchor, model(2), EVENTS, passes=20, mode='chronological')
    np.testing.assert_array_equal(first.p0, last.p0)
    np.testing.assert_array_equal(first.p1, last.p1)


@pytest.mark.parametrize('events', [EVENTS + (EVENTS[0],),
    (Encounter('a', 1, 0, 0, 0), Encounter('b', 0, 0, 0, 0)),
    (Encounter('a', 0, 9, 0, 0),), (Encounter('a', 0, 0, -1, 0),),
    (Encounter('a', 0, 0, 0, 9),), (Encounter('a', -1, 0, 0, 0),)])
def test_invalid_or_duplicate_observed_records_are_rejected(events):
    with pytest.raises(ValueError):
        refit(model(), model(), events)


@pytest.mark.parametrize('kwargs', [dict(passes=-1), dict(passes=1.5), dict(decay=0),
                                  dict(learning_rate=float('nan')), dict(mode='magic')])
def test_invalid_learning_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        refit(model(), model(), EVENTS, **kwargs)


def test_noop_and_checkpoint_roundtrip(tmp_path):
    anchor = model(2)
    asleep, _ = refit(anchor, anchor, EVENTS, passes=0)
    assert asleep.to_state().content_hash() == anchor.to_state().content_hash()
    asleep, _ = refit(anchor, anchor, (), passes=10)
    assert asleep.to_state().content_hash() == anchor.to_state().content_hash()
    fitted, _ = refit(anchor, anchor, EVENTS)
    state = fitted.to_state()
    state.save(tmp_path)
    loaded = AgentState.load(tmp_path)
    assert loaded.content_hash() == state.content_hash()
    restored = ReplayModel.from_state(loaded)
    np.testing.assert_array_equal(restored.p0, fitted.p0)


def test_invalid_model_shapes_and_nonfinite_inputs_are_rejected():
    a = model(2)
    with pytest.raises(ValueError):
        ReplayModel(a.a0[:, :1], a.a1, a.p0, a.p1, a.prior)
    with pytest.raises(ValueError):
        ReplayModel(a.a0 * np.nan, a.a1, a.p0, a.p1, a.prior)
