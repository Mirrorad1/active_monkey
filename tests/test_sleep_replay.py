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


def test_incremental_first_pass_equals_joint_and_preserves_inputs():
    anchor, start = model(2), model(2)
    before = (anchor.to_state().content_hash(), start.to_state().content_hash(), EVENTS)
    joint, _ = refit(anchor, start, EVENTS, passes=1)
    incremental, receipt = refit(anchor, start, EVENTS, passes=1, mode='incremental')
    np.testing.assert_array_equal(incremental.p0, joint.p0)
    np.testing.assert_array_equal(incremental.p1, joint.p1)
    assert receipt['metrics']['fit_posterior_evaluations'] == 2
    assert receipt['metrics']['objective_posterior_evaluations'] == 4
    assert (anchor.to_state().content_hash(), start.to_state().content_hash(), EVENTS) == before


@pytest.mark.parametrize('passes', [1, 2, 25])
def test_incremental_replaces_old_responsibilities_without_adding_evidence(passes):
    anchor = model()
    fitted, receipt = refit(anchor, anchor, EVENTS, passes=passes, mode='incremental',
                            decay=.5, learning_rate=2.)
    np.testing.assert_allclose(fitted.p0[:, 0, :], [[1.25, .25], [.25, 2.25]])
    np.testing.assert_allclose(fitted.p1[:, 0, :], [[.25, 2.25], [.25, .25], [1.25, .25]])
    assert receipt['metrics']['evidence_mass'] == 3.
    assert receipt['metrics']['fit_posterior_evaluations'] == 2 * passes
    assert receipt['metrics']['objective_posterior_evaluations'] == 2 * (passes + 1)
    assert receipt['metrics']['posterior_evaluations'] == 4 * passes + 2


def test_incremental_refreshes_later_events_using_latest_model():
    # Two identical events: initial responsibilities are (16/17, 1/17).
    # After the first batch, selected probabilities are (49/66, 19/36).
    # Sequential replacements give q0=.664294..., then q1=.606251....
    # A frozen batch update or a restart would give different final counts.
    anchor = ReplayModel(np.full((2, 2, 1), .5), np.full((2, 2, 1), .5),
                         np.ones((2, 2, 1)), np.ones((2, 2, 1)), np.full(2, .5))
    start = ReplayModel(np.array([[[.8], [.2]], [[.2], [.8]]]),
                        np.array([[[.8], [.2]], [[.2], [.8]]]),
                        np.ones((2, 2, 1)), np.ones((2, 2, 1)), np.full(2, .5))
    events = (Encounter('a', 0, 0, 0, 0), Encounter('b', 1, 0, 0, 0))
    fitted, _ = refit(anchor, start, events, passes=2, mode='incremental',
                      decay=1., learning_rate=1.)
    expected = np.array([[[2.270545740151], [1.729454259849]], [[1.], [1.]]])
    np.testing.assert_allclose(fitted.p0, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(fitted.p1, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(fitted.prior, anchor.prior)


def test_incremental_is_deterministic_and_keeps_all_source_arrays():
    anchor, start = model(2), model(2)
    before = (anchor.to_state().content_hash(), start.to_state().content_hash())
    first, receipt1 = refit(anchor, start, EVENTS, passes=9, mode='incremental')
    second, receipt2 = refit(anchor, start, EVENTS, passes=9, mode='incremental')
    assert first.to_state().content_hash() == second.to_state().content_hash()
    assert receipt1 == receipt2
    assert (anchor.to_state().content_hash(), start.to_state().content_hash()) == before


def test_incremental_receipt_matches_actual_posterior_work(monkeypatch):
    import active_loop.sleep_replay as replay

    actual_posterior = replay._posterior
    event_evaluations = []

    def counted_posterior(fitted, events):
        event_evaluations.append(len(events))
        return actual_posterior(fitted, events)

    monkeypatch.setattr(replay, '_posterior', counted_posterior)
    _, receipt = refit(model(2), model(2), EVENTS, passes=3, mode='incremental')
    # Initial monitor, initial fit/monitor, then two sequential sweeps/monitors.
    assert event_evaluations == [2, 2, 2, 1, 1, 2, 1, 1, 2]
    assert receipt['metrics']['posterior_evaluations'] == sum(event_evaluations) == 14


@pytest.mark.parametrize('kwargs', [dict(passes=-1), dict(passes=True), dict(decay=0),
                                  dict(learning_rate=float('nan'))])
def test_incremental_rejects_invalid_learning_configuration(kwargs):
    with pytest.raises(ValueError, match='configuration'):
        refit(model(), model(), EVENTS, mode='incremental', **kwargs)


def test_incremental_cannot_change_latent_prior_or_capacity():
    anchor, start = model(2), model(2)
    start.prior[:] = [.25, .75]
    with pytest.raises(ValueError, match='retain the latent prior'):
        refit(anchor, start, EVENTS, mode='incremental')
    with pytest.raises(ValueError, match='dimensions differ'):
        refit(anchor, model(), EVENTS, mode='incremental')


@pytest.mark.parametrize('events,passes,fit,monitor', [(EVENTS, 0, 0, 2), ((), 20, 0, 0)])
def test_incremental_noop_keeps_start_and_accounts_only_actual_work(events, passes, fit, monitor):
    anchor, start = model(2), model(2)
    fitted, receipt = refit(anchor, start, events, passes=passes, mode='incremental')
    assert fitted.to_state().content_hash() == start.to_state().content_hash()
    assert receipt['metrics']['fit_posterior_evaluations'] == fit
    assert receipt['metrics']['objective_posterior_evaluations'] == monitor


@pytest.mark.parametrize('mode', ['incremental20', None, 0])
def test_invalid_incremental_mode_spellings_remain_rejected(mode):
    with pytest.raises(ValueError, match='configuration'):
        refit(model(), model(), EVENTS, mode=mode)
