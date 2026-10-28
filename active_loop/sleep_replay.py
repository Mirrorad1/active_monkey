"""Once-weighted offline inference for the toy direct-head dyad.

The learner sees observed (code, response, outcome) only. No correct mapping or simulated
unchosen rewards enter this module. Joint EM optimizes a regularized mixture likelihood;
its prior pseudo-counts are a penalty, not extra observed events.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from active_loop.sleep import make_receipt
from active_loop.state import AgentState


@dataclass(frozen=True)
class Encounter:
    event_id: str
    t: int
    code: int
    response: int
    outcome: int


@dataclass
class ReplayModel:
    a0: np.ndarray
    a1: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    prior: np.ndarray

    def __post_init__(self):
        for name in ('a0', 'a1', 'p0', 'p1', 'prior'):
            setattr(self, name, np.array(getattr(self, name), dtype=np.float64, copy=True))
        if (self.a0.ndim != 3 or self.a1.ndim != 3 or self.prior.ndim != 1 or
            self.p0.shape != self.a0.shape or self.p1.shape != self.a1.shape or
            self.a0.shape[1] != len(self.prior) or self.a1.shape[1] != len(self.prior) or
            self.a0.shape[2] != self.a1.shape[2]):
            raise ValueError('incompatible direct-head model shapes')
        for array in (self.a0, self.a1, self.p0, self.p1, self.prior):
            if not array.size or not np.all(np.isfinite(array)) or np.any(array <= 0):
                raise ValueError('model entries must be finite and strictly positive')
        for array in (self.a0, self.a1, self.prior):
            if not np.allclose(array.sum(axis=0), 1., atol=1e-6):
                raise ValueError('probabilities must be normalized along the observation axis')

    def copy(self):
        return ReplayModel(self.a0, self.a1, self.p0, self.p1, self.prior)

    def to_state(self):
        return AgentState(architecture_id='sleep-direct-head-v1', agent_class='ReplayModel',
                          tensors={name: getattr(self, name) for name in
                                   ('a0', 'a1', 'p0', 'p1', 'prior')})

    @classmethod
    def from_state(cls, state):
        if state.architecture_id != 'sleep-direct-head-v1':
            raise ValueError('incompatible replay checkpoint')
        return cls(**{name: state.tensors[name] for name in ('a0', 'a1', 'p0', 'p1', 'prior')})


def _validate(anchor, start, events, passes, decay, learning_rate, mode):
    if (type(passes) is not int or passes < 0 or not np.isfinite(decay) or not 0 < decay <= 1 or
        not np.isfinite(learning_rate) or learning_rate <= 0 or
        mode not in ('joint', 'chronological')):
        raise ValueError('invalid replay configuration')
    if anchor.a0.shape != start.a0.shape or anchor.a1.shape != start.a1.shape:
        raise ValueError('anchor/start dimensions differ')
    if not np.array_equal(anchor.prior, start.prior):
        raise ValueError('replay must retain the latent prior')
    ids, last = set(), -1
    for event in events:
        if (not isinstance(event.event_id, str) or not event.event_id or event.event_id in ids or
            type(event.t) is not int or event.t <= last or
            type(event.code) is not int or not 0 <= event.code < anchor.a0.shape[0] or
            type(event.response) is not int or not 0 <= event.response < anchor.a1.shape[2] or
            type(event.outcome) is not int or not 0 <= event.outcome < anchor.a1.shape[0]):
            raise ValueError('records must have unique IDs, ordered lived times and valid observations')
        ids.add(event.event_id)
        last = event.t


def _posterior(model, events):
    code = np.array([e.code for e in events])
    response = np.array([e.response for e in events])
    outcome = np.array([e.outcome for e in events])
    joint = model.a0[code, :, response] * model.a1[outcome, :, response] * model.prior
    return joint / joint.sum(axis=1, keepdims=True), joint.sum(axis=1)


def _counts_model(p0, p1, prior):
    return ReplayModel(p0 / p0.sum(axis=0), p1 / p1.sum(axis=0), p0, p1, prior)


def _weights(events, decay, learning_rate):
    horizon = events[-1].t + 1 if events else 0
    return horizon, np.array([learning_rate * decay ** (horizon - 1 - e.t) for e in events])


def objective(model, anchor, events, decay=.999, learning_rate=4.):
    events = tuple(events)
    _validate(anchor, model, events, 0, decay, learning_rate, 'joint')
    horizon, weights = _weights(events, decay, learning_rate)
    likelihood = float(weights @ np.log(_posterior(model, events)[1])) if events else 0.
    penalty = decay ** horizon * (np.sum(anchor.p0 * np.log(model.a0)) +
                                  np.sum(anchor.p1 * np.log(model.a1)))
    return float(likelihood + penalty)


def refit(anchor, start, events, *, passes=20, mode='joint', decay=.999, learning_rate=4.):
    """Re-estimate assignments; never accumulate replay passes as independent observations.

    Anchor is the immutable checkpoint BEFORE the recorded window. Start is the model after
    wake (used to initialize joint inference). The chronological control deliberately repeats
    identical, restarted passes; its redundant passes match posterior-evaluation work.
    """
    events = tuple(events)
    _validate(anchor, start, events, passes, decay, learning_rate, mode)
    fitted = start.copy()
    horizon, weights = _weights(events, decay, learning_rate)
    values = [objective(fitted, anchor, events, decay, learning_rate)]
    if events:
        for _ in range(passes):
            if mode == 'joint':
                q, _ = _posterior(fitted, events)
                p0, p1 = anchor.p0 * decay ** horizon, anchor.p1 * decay ** horizon
                for event, responsibility, weight in zip(events, q, weights):
                    p0[event.code, :, event.response] += weight * responsibility
                    p1[event.outcome, :, event.response] += weight * responsibility
                fitted = _counts_model(p0, p1, anchor.prior)
            else:
                fitted, previous_t = anchor.copy(), -1
                for event in events:
                    q, _ = _posterior(fitted, (event,))
                    factor = decay ** (event.t - previous_t)
                    p0, p1 = fitted.p0 * factor, fitted.p1 * factor
                    p0[event.code, :, event.response] += learning_rate * q[0]
                    p1[event.outcome, :, event.response] += learning_rate * q[0]
                    fitted = _counts_model(p0, p1, anchor.prior)
                    previous_t = event.t
            values.append(objective(fitted, anchor, events, decay, learning_rate))
    used_passes = passes if events else 0
    receipt = make_receipt('creature', start.to_state().content_hash(),
        fitted.to_state().content_hash(), [e.event_id for e in events],
        dict(passes=used_passes, mode=mode, decay=decay, learning_rate=learning_rate),
        dict(unique_encounters=len(events), lived_steps=horizon, evidence_mass=float(weights.sum()),
             fit_posterior_evaluations=used_passes * len(events),
             objective_posterior_evaluations=len(values) * len(events),
             posterior_evaluations=(used_passes + len(values)) * len(events), objective=values), 'candidate')
    return fitted, receipt
