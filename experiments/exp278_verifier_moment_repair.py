"""Exp 278: verifier-revision repair of Adam history (toy mechanistic screen).

PREDECLARATION (written before implementation):
Hypothesis: Following a localized verifier correction, replacing Adam's historical first-
and second-moment contributions with gradients rescored under the repaired reward,
on the fixed actual parameter path, improves short-horizon true-reward recovery
versus retaining or resetting optimizer state. This is NOT LLM evidence.

Frozen design: NumPy contextual three-action softmax bandit, 16 fixed contexts,
8 shared row-L2-normalized features, argmax(X @ teacher) targets, N(0, .03)
initial weights, 8 sampled actions per context per update. Binary rewards are
normalized by each context's full group's population mean/std (zero advantage
when std == 0). Loss gradient is -X.T @ mean[A*(onehot(a)-p)] / n_contexts.
Adam: lr=.03, beta1=.9, beta2=.999, eps=1e-8. Train 60 clean updates, then 20
updates with first 4 (sparse) or 12 (dense) verifier targets cycled +1 mod 3;
repair at update 80; evaluate after each of 30 clean recovery updates.
No-change control has no corrupt labels. Confirmatory seeds: 9100..9107, all
reported. Smoke seed 9099 is excluded, and no tuning is licensed by its result.

At s=61..80, log actual probabilities, actions, features, and old gradients.
Recompute rewards AND whole-group advantages at these same logged probabilities.
Replace moments using m*=m80+sum((1-b1)*b1**(80-s)*(g_new-g_old)); likewise
v*=v80+sum((1-b2)*b2**(80-s)*(g_new**2-g_old**2)), with theta and t=80 fixed.
This is fixed-path moment repair, not the counterfactual clean-training trajectory.

Arms: keep; reset_all (m=v=0,t=0); reset_m (m=0,v unchanged,t=80);
reset_moments_keep_clock (m=v=0,t=80); reset_clock (m,v unchanged,t=0,
timestep-only baseline inspired by Adam-Rel, not its full reproduction);
repair_m; repair_mv; refresh_current. The last replaces historical moment
contributions with a clipped-importance-weighted score gradient evaluated at
theta80: clip(p_current(a)/p_logged(a),.8,1.2) * normalized repaired advantage
* grad_current log p(a). The importance weights are treated as constants in
this explicit score-gradient estimator. This is NOT exact PPO or its clipped
surrogate gradient. Its extra replay computation is reported separately.

Recovery arms use the same uniform random variates, sampling from each arm's
own probabilities. Evaluation is exact expected true success, including subgroup
scores. Primary utility is mean success over the 30 POST-update recovery points.
PASS requires, separately in BOTH sparse and dense, repair_mv's paired mean
advantage >= .01 over EACH of keep/reset_all/reset_m/reset_moments_keep_clock/
reset_clock/refresh_current AND positive differences on >= 6 of 8 seeds against
each. repair_m is a descriptive mechanism ablation. No-change repair_mv must
equal keep to <=1e-12. Falsifier: any failed scientific conjunct => NEGATIVE; failed
instrument checks => INVALID. Smoke receives no scientific verdict.

Instrument checks: replacement formula equals forward replay from prefix moments
using rescored fixed-path gradients (absolute error <=1e-12); intervention leaves
theta unchanged; v remains nonnegative; zero reward revision is a no-op; the
declared first 4/12 labels actually change; and sampled changed-reward count >0
in corrupt regimes. The last two gates inspect inputs, never gradient magnitude.

All features, labels, intervention states, logged batches, raw score trajectories,
seeds, configuration/source hashes, runtime, and work counts are emitted as JSON.
Use --self-test for deterministic invariants, --smoke for the excluded seed, and
--workers 2 (default) for bounded independent-seed threads. No paid calls or GPUs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


N_CONTEXTS, N_FEATURES, N_ACTIONS, GROUP = 16, 8, 3, 8
WARMUP, CORRUPT, RECOVERY = 60, 20, 30
BETA1, BETA2, EPS, LR = 0.9, 0.999, 1e-8, 0.03
TOL = 1e-12
CONFIRMATORY_SEEDS = tuple(range(9100, 9108))
SMOKE_SEED = 9099
REGIMES = {"sparse": 4, "dense": 12, "no_change": 0}
ARMS = (
    "keep", "reset_all", "reset_m", "reset_moments_keep_clock",
    "reset_clock", "repair_m", "repair_mv", "refresh_current",
)
BASELINES = (
    "keep", "reset_all", "reset_m", "reset_moments_keep_clock",
    "reset_clock", "refresh_current",
)


@dataclass
class State:
    theta: np.ndarray
    m: np.ndarray
    v: np.ndarray
    t: int = 0

    def copy(self) -> State:
        return State(self.theta.copy(), self.m.copy(), self.v.copy(), self.t)


def probabilities(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    logits = features @ theta
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def sample_actions(p: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(p, axis=1)
    cdf[:, -1] = 1.0
    return (uniforms[:, :, None] >= cdf[:, None, :]).sum(axis=2)


def rewards_and_advantages(
    actions: np.ndarray, targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = (actions == targets[:, None]).astype(np.float64)
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    advantages = np.divide(centered, std, out=np.zeros_like(centered), where=std > 0)
    return rewards, advantages


def score_gradient(
    features: np.ndarray, p: np.ndarray, actions: np.ndarray,
    advantages: np.ndarray, weights: np.ndarray | None = None,
) -> np.ndarray:
    """Gradient of negative score-weighted log likelihood, weights held fixed."""
    coefficients = advantages if weights is None else advantages * weights
    score = np.eye(N_ACTIONS)[actions] - p[:, None, :]
    return -(features.T @ (coefficients[:, :, None] * score).mean(axis=1)) / len(features)


def gradient(
    features: np.ndarray, p: np.ndarray, actions: np.ndarray, targets: np.ndarray,
) -> np.ndarray:
    return score_gradient(features, p, actions, rewards_and_advantages(actions, targets)[1])


def refreshed_gradient(
    features: np.ndarray, p_current: np.ndarray, p_logged: np.ndarray,
    actions: np.ndarray, targets: np.ndarray,
) -> np.ndarray:
    row = np.arange(len(features))[:, None]
    ratio = p_current[row, actions] / p_logged[row, actions]
    weights = np.clip(ratio, 0.8, 1.2)
    return score_gradient(
        features, p_current, actions, rewards_and_advantages(actions, targets)[1], weights,
    )


def adam_update(state: State, g: np.ndarray) -> None:
    state.t += 1
    state.m = BETA1 * state.m + (1 - BETA1) * g
    state.v = BETA2 * state.v + (1 - BETA2) * g * g
    mhat = state.m / (1 - BETA1 ** state.t)
    vhat = state.v / (1 - BETA2 ** state.t)
    state.theta -= LR * mhat / (np.sqrt(vhat) + EPS)


def replace_moments(
    state: State, old_gradients: list[np.ndarray], new_gradients: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the actual parameter path; replace only historical moments."""
    if len(old_gradients) != len(new_gradients) or len(old_gradients) > state.t:
        raise ValueError("Moment history lengths must match and fit the state clock")
    m, v = state.m.copy(), state.v.copy()
    last_index = len(old_gradients) - 1
    for index, (old, new) in enumerate(zip(old_gradients, new_gradients, strict=True)):
        age = last_index - index
        m += (1 - BETA1) * BETA1 ** age * (new - old)
        v += (1 - BETA2) * BETA2 ** age * (new * new - old * old)
    return m, v


def forward_moments(prefix: State, gradients: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    m, v = prefix.m.copy(), prefix.v.copy()
    for g in gradients:
        m = BETA1 * m + (1 - BETA1) * g
        v = BETA2 * v + (1 - BETA2) * g * g
    return m, v


def intervened_states(
    state: State, fixed: tuple[np.ndarray, np.ndarray],
    refreshed: tuple[np.ndarray, np.ndarray],
) -> dict[str, State]:
    arms = {name: state.copy() for name in ARMS}
    for name in ("reset_all", "reset_m", "reset_moments_keep_clock"):
        arms[name].m.fill(0)
    for name in ("reset_all", "reset_moments_keep_clock"):
        arms[name].v.fill(0)
    for name in ("reset_all", "reset_clock"):
        arms[name].t = 0
    arms["repair_m"].m = fixed[0].copy()
    arms["repair_mv"].m, arms["repair_mv"].v = (value.copy() for value in fixed)
    arms["refresh_current"].m, arms["refresh_current"].v = (value.copy() for value in refreshed)
    return arms


def score(
    features: np.ndarray, state: State, targets: np.ndarray, affected: int, step: int,
) -> dict[str, Any]:
    successes = probabilities(features, state.theta)[np.arange(len(features)), targets]
    return {
        "update": step,
        "true_reward": float(successes.mean()),
        "affected_true_reward": float(successes[:affected].mean()) if affected else None,
        "unaffected_true_reward": float(successes[affected:].mean()),
        "first4_true_reward": float(successes[:4].mean()),
        "first12_true_reward": float(successes[:12].mean()),
        "per_context_true_reward": successes.tolist(),
    }


def state_json(state: State) -> dict[str, Any]:
    return {"theta": state.theta.tolist(), "m": state.m.tolist(), "v": state.v.tolist(), "t": state.t}


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def run_seed(job: tuple[int, str]) -> dict[str, Any]:
    seed, regime = job
    started = time.perf_counter()
    affected = REGIMES[regime]
    # Regimes share the context/teacher/initial state and exogenous uniforms for a seed.
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(N_CONTEXTS, N_FEATURES))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    teacher = rng.normal(size=(N_FEATURES, N_ACTIONS))
    targets = np.argmax(features @ teacher, axis=1)
    verifier_targets = targets.copy()
    verifier_targets[:affected] = (targets[:affected] + 1) % N_ACTIONS
    initial_theta = rng.normal(0, 0.03, size=(N_FEATURES, N_ACTIONS))
    state = State(initial_theta.copy(), np.zeros_like(initial_theta), np.zeros_like(initial_theta))
    uniforms = rng.random((WARMUP + CORRUPT + RECOVERY, N_CONTEXTS, GROUP))
    history = [score(features, state, targets, affected, 0)]
    ledger: list[dict[str, Any]] = []
    old_gradients: list[np.ndarray] = []
    new_gradients: list[np.ndarray] = []
    prefix = state.copy()
    prefix_min_v = 0.0
    changed_reward_count = 0
    for step in range(1, WARMUP + CORRUPT + 1):
        p = probabilities(features, state.theta)
        actions = sample_actions(p, uniforms[step - 1])
        used_targets = targets if step <= WARMUP else verifier_targets
        old_rewards, old_advantages = rewards_and_advantages(actions, used_targets)
        g = score_gradient(features, p, actions, old_advantages)
        if step > WARMUP:
            revised_rewards, revised_advantages = rewards_and_advantages(actions, targets)
            revised_g = score_gradient(features, p, actions, revised_advantages)
            changed_reward_count += int(np.count_nonzero(old_rewards != revised_rewards))
            old_gradients.append(g.copy())
            new_gradients.append(revised_g)
            ledger.append({
                "update": step, "features_reference": "features", "p_logged": p.copy(),
                "actions": actions, "old_rewards": old_rewards,
                "revised_rewards": revised_rewards, "old_advantages": old_advantages,
                "revised_advantages": revised_advantages, "old_gradient": g.copy(),
                "revised_fixed_path_gradient": revised_g,
            })
        adam_update(state, g)
        prefix_min_v = min(prefix_min_v, float(state.v.min()))
        history.append(score(features, state, targets, affected, step))
        if step == WARMUP:
            prefix = state.copy()

    p_current = probabilities(features, state.theta)
    refreshed_gradients = []
    for row in ledger:
        refresh = refreshed_gradient(features, p_current, row["p_logged"], row["actions"], targets)
        row["refreshed_current_gradient"] = refresh
        refreshed_gradients.append(refresh)
    fixed = replace_moments(state, old_gradients, new_gradients)
    refreshed = replace_moments(state, old_gradients, refreshed_gradients)
    fixed_forward = forward_moments(prefix, new_gradients)
    refreshed_forward = forward_moments(prefix, refreshed_gradients)
    old_forward = forward_moments(prefix, old_gradients)
    no_op = replace_moments(state, old_gradients, old_gradients)
    errors = {
        "fixed_m_forward_max_abs_error": max_error(fixed[0], fixed_forward[0]),
        "fixed_v_forward_max_abs_error": max_error(fixed[1], fixed_forward[1]),
        "refresh_m_forward_max_abs_error": max_error(refreshed[0], refreshed_forward[0]),
        "refresh_v_forward_max_abs_error": max_error(refreshed[1], refreshed_forward[1]),
        "old_m_forward_max_abs_error": max_error(state.m, old_forward[0]),
        "old_v_forward_max_abs_error": max_error(state.v, old_forward[1]),
        "zero_revision_m_max_abs_error": max_error(state.m, no_op[0]),
        "zero_revision_v_max_abs_error": max_error(state.v, no_op[1]),
    }
    arms = intervened_states(state, fixed, refreshed)
    initial_states = {name: state_json(arm) for name, arm in arms.items()}
    errors["theta_intervention_max_abs_error"] = max(max_error(arm.theta, state.theta) for arm in arms.values())
    minimum_v = min(prefix_min_v, *(float(arm.v.min()) for arm in arms.values()))
    null_state_error = max(max_error(arms["repair_mv"].m, state.m), max_error(arms["repair_mv"].v, state.v))
    arm_results: dict[str, Any] = {}
    for name, arm in arms.items():
        trajectory = []
        for recovery_step in range(1, RECOVERY + 1):
            step = WARMUP + CORRUPT + recovery_step
            p = probabilities(features, arm.theta)
            actions = sample_actions(p, uniforms[step - 1])
            adam_update(arm, gradient(features, p, actions, targets))
            minimum_v = min(minimum_v, float(arm.v.min()))
            trajectory.append(score(features, arm, targets, affected, step))
        arm_results[name] = {
            "intervention_state": initial_states[name],
            "trajectory": trajectory,
            "utility": float(np.mean([row["true_reward"] for row in trajectory])),
            "final_state": state_json(arm),
        }
    null_trajectory_error = max(
        max_error(np.asarray(left["per_context_true_reward"]), np.asarray(right["per_context_true_reward"]))
        for left, right in zip(arm_results["keep"]["trajectory"], arm_results["repair_mv"]["trajectory"], strict=True)
    ) if affected == 0 else None
    flags = {
        "moment_and_theta_errors_within_tolerance": all(value <= TOL for value in errors.values()),
        "second_moments_nonnegative": minimum_v >= 0.0,
        "all_outputs_finite": all(np.isfinite(value) for value in errors.values()) and all(
            np.isfinite(arm.theta).all() and np.isfinite(arm.m).all() and np.isfinite(arm.v).all()
            for arm in arms.values()
        ),
        "label_revision_matches_declared_indices": np.array_equal(
            np.flatnonzero(verifier_targets != targets), np.arange(affected),
        ),
        "sampled_changed_reward_input_occurred": changed_reward_count > 0 if affected else changed_reward_count == 0,
        "no_change_moments_noop": null_state_error <= TOL if affected == 0 else True,
    }
    return {
        "seed": seed, "regime": regime, "affected_contexts": list(range(affected)),
        "features": features, "teacher": teacher, "true_targets": targets,
        "verifier_targets_during_revision_window": verifier_targets,
        "initial_theta": initial_theta, "uniforms": uniforms,
        "clean_prefix_state": state_json(prefix), "before_intervention_state": state_json(state),
        "prefix_trajectory": history, "logged_batches": ledger, "arms": arm_results,
        "instrument_checks": {"flags": flags, "errors": errors, "minimum_v": minimum_v,
                              "changed_reward_count": changed_reward_count,
                              "no_change_state_max_abs_error": null_state_error if affected == 0 else None},
        "no_change_trajectory_max_abs_error": null_trajectory_error,
        "wall_seconds": time.perf_counter() - started,
    }


def summarize(results: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    all_conjuncts = []
    for regime in ("sparse", "dense"):
        rows = [row for row in results if row["regime"] == regime]
        comparisons[regime] = {}
        for baseline in BASELINES:
            differences = [row["arms"]["repair_mv"]["utility"] - row["arms"][baseline]["utility"] for row in rows]
            mean = float(np.mean(differences))
            positive_count = sum(value > 0 for value in differences)
            passed = mean >= 0.01 and positive_count >= 6 and len(rows) == 8
            all_conjuncts.append(passed)
            comparisons[regime][baseline] = {
                "seeds": [row["seed"] for row in rows], "paired_differences": differences,
                "mean_paired_difference": mean, "positive_seeds": positive_count,
                "n_seeds": len(rows), "effect_threshold": 0.01, "positive_seed_threshold": 6,
                "passes_confirmatory_conjunct": passed if not smoke else None,
            }
    null_errors = [row["no_change_trajectory_max_abs_error"] for row in results if row["regime"] == "no_change"]
    null_pass = len(null_errors) == (1 if smoke else 8) and all(error <= TOL for error in null_errors)
    valid = all(all(row["instrument_checks"]["flags"].values()) for row in results)
    declared_jobs = {(seed, regime) for seed in ((SMOKE_SEED,) if smoke else CONFIRMATORY_SEEDS) for regime in REGIMES}
    valid = valid and {(row["seed"], row["regime"]) for row in results} == declared_jobs and len(results) == len(declared_jobs)
    verdict = "INVALID" if not valid else "SMOKE_ONLY" if smoke else "POSITIVE" if all(all_conjuncts) and null_pass else "NEGATIVE"
    return {
        "verdict": verdict, "scope": "toy mechanistic screen; not evidence of LLM improvement or novelty",
        "primary_comparisons": comparisons,
        "utility_by_regime_arm": {
            regime: {arm: float(np.mean([row["arms"][arm]["utility"] for row in results if row["regime"] == regime])) for arm in ARMS}
            for regime in REGIMES
        },
        "instrument_valid": valid, "no_change_trajectory_pass": null_pass,
        "no_change_max_trajectory_error": max(null_errors),
        "repair_m_role": "descriptive ablation; excluded from primary success criteria",
    }


def configuration() -> dict[str, Any]:
    return {
        "experiment": 278, "contexts": N_CONTEXTS, "features": N_FEATURES, "actions": N_ACTIONS,
        "samples_per_context": GROUP, "warmup_updates": WARMUP, "corrupted_updates": CORRUPT,
        "recovery_updates": RECOVERY, "lr": LR, "beta1": BETA1, "beta2": BETA2, "eps": EPS,
        "initial_weight_std": 0.03, "feature_normalization": "row L2",
        "advantage_normalization": "whole context group population std; zero if std=0",
        "refresh_importance_clip": [0.8, 1.2], "refresh_weight_derivative": "stop gradient",
        "confirmatory_seeds": CONFIRMATORY_SEEDS, "smoke_seed_excluded": SMOKE_SEED,
        "regime_affected_counts": REGIMES, "arms": ARMS, "primary_baselines": BASELINES,
        "primary_utility": "mean exact true reward at updates 81..110 inclusive",
        "paired_mean_advantage_threshold": 0.01, "positive_seed_count_threshold": 6,
        "invariant_atol": TOL, "maximum_workers": 2,
        "reset_clock_note": "timestep-only baseline inspired by Adam-Rel, not a full reproduction",
    }


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def self_test() -> dict[str, Any]:
    """Algorithmic invariants, independent of the confirmatory seeds."""
    rng = np.random.default_rng(2718)
    x = rng.normal(size=(4, N_FEATURES))
    theta = rng.normal(0, 0.1, size=(N_FEATURES, N_ACTIONS))
    p = probabilities(x, theta)
    actions = np.array([[0, 1, 2, 0], [2, 2, 1, 0], [1, 1, 0, 2], [0, 0, 0, 0]])
    targets = np.array([0, 1, 2, 1])
    rewards, advantages = rewards_and_advantages(actions, targets)
    assert np.all(advantages[-1] == 0), "constant reward group must have zero advantage"
    assert np.allclose(advantages.mean(axis=1), 0, atol=TOL)
    assert np.allclose(advantages[:3].std(axis=1), 1, atol=TOL)
    analytic = gradient(x, p, actions, targets)
    def fixed_advantage_loss(w: np.ndarray) -> float:
        logp = np.log(probabilities(x, w)[np.arange(4)[:, None], actions])
        return float(-(advantages * logp).mean())
    finite_diff = np.zeros_like(theta)
    delta = 1e-6
    for index in np.ndindex(theta.shape):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += delta
        minus[index] -= delta
        finite_diff[index] = (fixed_advantage_loss(plus) - fixed_advantage_loss(minus)) / (2 * delta)
    assert max_error(analytic, finite_diff) < 1e-8, "policy-gradient sign/scale failed finite differences"
    assert np.allclose(refreshed_gradient(x, p, p, actions, targets), analytic, atol=TOL)
    # Freeze the ratio weights at this current state, as the declared estimator does.
    theta_current = theta + rng.normal(0, 1, size=theta.shape)
    p_current = probabilities(x, theta_current)
    importance = p_current[np.arange(4)[:, None], actions] / p[np.arange(4)[:, None], actions]
    clipped = np.clip(importance, 0.8, 1.2)
    assert np.any(importance < 0.8) and np.any(importance > 1.2), "both clipping boundaries must be exercised"
    refresh_fd = np.zeros_like(theta)
    def frozen_weight_loss(w: np.ndarray) -> float:
        logp = np.log(probabilities(x, w)[np.arange(4)[:, None], actions])
        return float(-(clipped * advantages * logp).mean())
    for index in np.ndindex(theta.shape):
        plus, minus = theta_current.copy(), theta_current.copy()
        plus[index] += delta
        minus[index] -= delta
        refresh_fd[index] = (frozen_weight_loss(plus) - frozen_weight_loss(minus)) / (2 * delta)
    refresh_analytic = refreshed_gradient(x, p_current, p, actions, targets)
    assert max_error(refresh_fd, refresh_analytic) < 1e-8
    assert np.all(gradient(x, p, np.zeros_like(actions), targets) == 0)
    assert np.array_equal(sample_actions(np.array([[0.2, 0.3, 0.5]]), np.array([[0, 0.1999, 0.2, 0.4999, 0.5, 0.9999]])), np.array([[0, 0, 1, 1, 2, 2]]))
    # An arbitrary valid prefix, old gradients and unrelated revisions exercise both moments.
    prefix = State(theta.copy(), rng.normal(size=theta.shape), rng.random(theta.shape), 60)
    old = [rng.normal(size=theta.shape) for _ in range(CORRUPT)]
    revised = [rng.normal(size=theta.shape) for _ in range(CORRUPT)]
    old_m, old_v = forward_moments(prefix, old)
    state = State(theta.copy(), old_m, old_v, 80)
    repaired = replace_moments(state, old, revised)
    replayed = forward_moments(prefix, revised)
    assert all(max_error(a, b) <= TOL for a, b in zip(repaired, replayed, strict=True))
    # g_new^2-g_old^2 can be negative; (g_new-g_old)^2 loses the cross term.
    scalar = State(np.zeros((1, 1)), np.array([[0.3]]), np.array([[0.009]]), 1)
    _, scalar_v = replace_moments(scalar, [np.array([[3.0]])], [np.array([[1.0]])])
    assert max_error(scalar_v, np.array([[0.001]])) <= TOL
    assert not np.allclose(scalar_v, scalar.v + (1 - BETA2) * (1 - 3) ** 2)
    assert all(np.array_equal(a, b) for a, b in zip(replace_moments(state, old, old), (state.m, state.v), strict=True))
    arms = intervened_states(state, repaired, repaired)
    assert all(np.array_equal(arm.theta, theta) for arm in arms.values())
    assert np.all(arms["reset_m"].m == 0) and np.array_equal(arms["reset_m"].v, state.v)
    assert arms["reset_m"].t == arms["repair_mv"].t == arms["reset_moments_keep_clock"].t == 80
    assert arms["reset_all"].t == arms["reset_clock"].t == 0
    assert np.array_equal(arms["reset_clock"].m, state.m) and np.array_equal(arms["reset_clock"].v, state.v)
    assert np.array_equal(arms["repair_m"].v, state.v)
    arms["keep"].theta[0, 0] += 1
    assert np.array_equal(arms["repair_mv"].theta, theta), "arms must not share writable state"
    first = State(theta.copy(), np.zeros_like(theta), np.zeros_like(theta))
    g = rng.normal(size=theta.shape)
    adam_update(first, g)
    assert max_error(first.theta, theta - LR * g / (np.abs(g) + EPS)) < TOL
    assert first.t == 1 and np.all(first.v >= 0)
    return {"passed": True, "finite_difference_max_abs_error": max_error(analytic, finite_diff),
            "refreshed_gradient_finite_difference_max_abs_error": max_error(refresh_fd, refresh_analytic),
            "moment_replay_max_abs_error": max(max_error(a, b) for a, b in zip(repaired, replayed, strict=True)),
            "checks": ["whole-group normalization", "degenerate reward groups", "gradient finite differences",
                       "refresh identity and clipped-weight finite differences", "inverse-CDF sampling", "moment forward replay", "squared-gradient cross term", "revision no-op",
                       "theta unchanged", "reset clocks/moments", "independent arm copies", "first Adam step"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="only excluded seed 9099, all three regimes")
    parser.add_argument("--self-test", action="store_true", help="run deterministic algorithmic invariant checks only")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, help="JSON artifact path; defaults to experiments/outputs/exp278.txt (exp278_smoke.json for smoke)")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    started = time.perf_counter()
    seeds = (SMOKE_SEED,) if args.smoke else CONFIRMATORY_SEEDS
    jobs = [(seed, regime) for seed in seeds for regime in REGIMES]
    update_count = len(jobs) * (WARMUP + CORRUPT + len(ARMS) * RECOVERY)
    counts = {
        "independent_seed_regime_jobs": len(jobs), "workers": args.workers,
        "updates_per_job_including_all_recovery_arms": WARMUP + CORRUPT + len(ARMS) * RECOVERY,
        "optimizer_updates_total": update_count,
        "sampled_action_decisions": update_count * N_CONTEXTS * GROUP,
        "common_random_uniform_draws": len(jobs) * (WARMUP + CORRUPT + RECOVERY) * N_CONTEXTS * GROUP,
        "logged_batches": len(jobs) * CORRUPT,
        "revised_reward_evaluations": len(jobs) * CORRUPT * N_CONTEXTS * GROUP,
        "extra_fixed_path_gradient_evaluations": len(jobs) * CORRUPT,
        "extra_refresh_current_gradient_evaluations": len(jobs) * CORRUPT,
        "paired_prefix_training_shared_across_arms": True,
        "paid_model_calls": 0, "gpu_hours": 0,
    }
    print(json.dumps({"preflight": counts, "active_seeds": seeds, "smoke": args.smoke}), file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_seed, jobs))
    config = configuration()
    payload = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": sys.version, "numpy": np.__version__, "executor": "ThreadPoolExecutor"},
        "run_mode": "excluded_seed_smoke" if args.smoke else "confirmatory", "active_seeds": seeds,
        "source_counts": counts, "results": results, "summary": summarize(results, args.smoke),
        "wall_seconds": time.perf_counter() - started,
    }
    # Serialize with allow_nan=False before opening a file so invalid numerics cannot become a valid-looking artifact.
    serialized = json.dumps(payload, indent=2, default=json_default, allow_nan=False)
    output = args.output or Path("experiments/outputs/exp278_smoke.json" if args.smoke else "experiments/outputs/exp278.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n")
    print(json.dumps({"output": str(output.resolve()), "summary": payload["summary"], "wall_seconds": payload["wall_seconds"]}, indent=2))
    return 1 if payload["summary"]["verdict"] == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
