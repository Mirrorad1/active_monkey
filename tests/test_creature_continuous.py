"""Tests for active_loop.creature_continuous — ContinuousPlace Gaussian belief."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.linalg import inv

from active_loop.continuous import gaussian_product
from active_loop.creature_continuous import ContinuousPlace

# Arena used for predict_clamped_moments tests: (xmin, xmax, ymin, ymax) = (0, 3, 0, 3)
_ARENA_3 = (0.0, 3.0, 0.0, 3.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_psd(d: int, rng: np.random.Generator, eps: float = 0.5) -> np.ndarray:
    A = rng.standard_normal((d, d))
    return A @ A.T + eps * np.eye(d)


# ---------------------------------------------------------------------------
# test_predict_clamps_mean_inside_arena
# ---------------------------------------------------------------------------

def test_predict_clamps_mean_inside_arena() -> None:
    """predict() keeps the mean inside the arena even when delta pushes it out;
    Sigma grows by Q each step."""
    arena = (0.0, 3.0, 0.0, 3.0)
    rng = np.random.default_rng(17)
    Q = 0.05 ** 2 * np.eye(2)

    # Start near the top-right wall
    mu0 = np.array([2.9, 2.9])
    Sigma0 = 0.1 * np.eye(2)
    cp = ContinuousPlace(mu0, Sigma0, arena)

    # Delta pushes toward +x, +y (outside the arena)
    delta = np.array([0.5, 0.5])
    Sigma_expected = Sigma0.copy()

    for step in range(10):
        cp.predict(delta, Q)
        Sigma_expected = Sigma_expected + Q

        mu = cp.mu
        xmin, xmax, ymin, ymax = arena

        # Mean must stay strictly within arena bounds
        assert xmin <= mu[0] <= xmax, (
            f"step {step}: x={mu[0]} outside [{xmin}, {xmax}]"
        )
        assert ymin <= mu[1] <= ymax, (
            f"step {step}: y={mu[1]} outside [{ymin}, {ymax}]"
        )

        # Sigma grows by Q each step
        np.testing.assert_allclose(
            cp.Sigma, Sigma_expected, atol=1e-12,
            err_msg=f"Sigma mismatch at step {step}"
        )

    # Also test clamping toward negative direction
    mu_neg = np.array([0.1, 0.1])
    Sigma_neg = 0.1 * np.eye(2)
    cp2 = ContinuousPlace(mu_neg, Sigma_neg, arena)
    delta_neg = np.array([-0.5, -0.5])
    for _ in range(5):
        cp2.predict(delta_neg, Q)
        mu2 = cp2.mu
        assert xmin <= mu2[0] <= xmax
        assert ymin <= mu2[1] <= ymax


# ---------------------------------------------------------------------------
# test_update_matches_gaussian_product
# ---------------------------------------------------------------------------

def test_update_matches_gaussian_product() -> None:
    """update() matches the reference gaussian_product on random PSD inputs."""
    rng = np.random.default_rng(42)
    arena = (-10.0, 10.0, -10.0, 10.0)

    for trial in range(8):
        mu1 = rng.standard_normal(2)
        Sigma1 = _random_psd(2, rng)
        mu2 = rng.standard_normal(2)
        Sigma2 = _random_psd(2, rng)

        # Reference via gaussian_product
        mu_ref, Sigma_ref = gaussian_product(mu1, Sigma1, mu2, Sigma2)

        # ContinuousPlace path
        cp = ContinuousPlace(mu1, Sigma1, arena)
        cp.update(mu2, Sigma2)

        np.testing.assert_allclose(
            cp.mu, mu_ref, atol=1e-10,
            err_msg=f"trial {trial}: mu mismatch"
        )
        np.testing.assert_allclose(
            cp.Sigma, Sigma_ref, atol=1e-10,
            err_msg=f"trial {trial}: Sigma mismatch"
        )


# ---------------------------------------------------------------------------
# test_no_wall_predict_update_kalman
# ---------------------------------------------------------------------------

def test_no_wall_predict_update_kalman() -> None:
    """Away from walls, predict(delta, Q) then update equals the textbook Kalman step."""
    rng = np.random.default_rng(99)
    arena = (-100.0, 100.0, -100.0, 100.0)  # large arena: clamping never activates

    for trial in range(6):
        mu0 = rng.standard_normal(2)
        Sigma0 = _random_psd(2, rng)
        delta = rng.standard_normal(2) * 0.3
        Q = 0.05 ** 2 * np.eye(2)

        # Observation
        mu_k = rng.standard_normal(2)
        Sigma_k = _random_psd(2, rng)

        # Hand-computed Kalman step
        # predict
        mu_pred = mu0 + delta
        Sigma_pred = Sigma0 + Q
        # update (Gaussian product)
        mu_kf, Sigma_kf = gaussian_product(mu_pred, Sigma_pred, mu_k, Sigma_k)

        # ContinuousPlace path
        cp = ContinuousPlace(mu0, Sigma0, arena)
        cp.predict(delta, Q)
        cp.update(mu_k, Sigma_k)

        np.testing.assert_allclose(
            cp.mu, mu_kf, atol=1e-10,
            err_msg=f"trial {trial}: mu mismatch vs Kalman"
        )
        np.testing.assert_allclose(
            cp.Sigma, Sigma_kf, atol=1e-10,
            err_msg=f"trial {trial}: Sigma mismatch vs Kalman"
        )


# ---------------------------------------------------------------------------
# test_coverage_95
# ---------------------------------------------------------------------------

def test_coverage_95() -> None:
    """At mu with Sigma=I, a point at distance sqrt(5.991) on an axis is the boundary.

    Slightly less -> inside (True); slightly more -> outside (False).
    """
    arena = (-100.0, 100.0, -100.0, 100.0)
    mu = np.array([0.0, 0.0])
    Sigma = np.eye(2)
    cp = ContinuousPlace(mu, Sigma, arena)

    threshold = np.sqrt(5.991)  # Mahalanobis distance at 95% boundary (Sigma=I)

    # Point exactly inside (slightly less)
    p_inside = np.array([threshold - 1e-9, 0.0])
    assert cp.coverage_95(p_inside) is True, (
        f"Expected True for point slightly inside boundary; "
        f"Maha^2 = {(threshold - 1e-9)**2:.6f}, threshold = 5.991"
    )

    # Point exactly outside (slightly more)
    p_outside = np.array([threshold + 1e-9, 0.0])
    assert cp.coverage_95(p_outside) is False, (
        f"Expected False for point slightly outside boundary; "
        f"Maha^2 = {(threshold + 1e-9)**2:.6f}, threshold = 5.991"
    )

    # Non-diagonal Sigma: verify the formula still holds
    rng = np.random.default_rng(7)
    Sigma2 = _random_psd(2, rng)
    mu2 = rng.standard_normal(2)
    cp2 = ContinuousPlace(mu2, Sigma2, arena)

    # Construct a point at Maha distance exactly 5.991 via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(Sigma2)
    # Move along first eigenvector by sqrt(5.991 * eigvals[0])
    direction = eigvecs[:, 0]
    dist = np.sqrt(5.991 * eigvals[0])
    p_boundary_in = mu2 + direction * (dist - 1e-8)
    p_boundary_out = mu2 + direction * (dist + 1e-8)

    assert cp2.coverage_95(p_boundary_in) is True, "Non-diagonal: inside boundary should be True"
    assert cp2.coverage_95(p_boundary_out) is False, "Non-diagonal: outside boundary should be False"


# ---------------------------------------------------------------------------
# test_clamped_moments_match_monte_carlo
# ---------------------------------------------------------------------------

def _mc_clamped_moments(
    mu: np.ndarray,
    sigma_diag: np.ndarray,
    delta: np.ndarray,
    Q_diag: np.ndarray,
    arena: tuple,
    n_samples: int = 200_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo reference: clamp(N(mu+delta, diag(sigma_diag+Q_diag))) moments."""
    rng = np.random.default_rng(seed)
    m = mu + delta
    v = sigma_diag + Q_diag
    xmin, xmax, ymin, ymax = arena
    lo = np.array([xmin, ymin])
    hi = np.array([xmax, ymax])
    samples = rng.standard_normal((n_samples, 2)) * np.sqrt(v) + m
    samples = np.clip(samples, lo, hi)
    mc_mean = samples.mean(axis=0)
    mc_var = samples.var(axis=0)
    return mc_mean, mc_var


def test_clamped_moments_match_monte_carlo() -> None:
    """predict_clamped_moments matches 200k-sample Monte Carlo: mean within 0.01,
    variance within 0.01 for several (mu, Sigma_diag, delta) cases including
    (i) far from walls, (ii) mean pushed past a wall, (iii) straddling a wall."""
    arena = _ARENA_3
    Q_diag = np.array([0.05 ** 2, 0.05 ** 2])
    Q = np.diag(Q_diag)

    # Each case: (mu, sigma_diag, delta, description)
    cases = [
        # (i) far from walls: center of arena, small delta
        (np.array([1.5, 1.5]), np.array([0.3, 0.3]), np.array([0.1, 0.1]), "far_from_walls"),
        # (ii) mean pushed past upper-right wall
        (np.array([2.8, 2.8]), np.array([0.4, 0.4]), np.array([0.8, 0.8]), "mean_past_wall"),
        # (iii) straddling: mean near lower-left wall
        (np.array([0.3, 0.3]), np.array([0.5, 0.5]), np.array([-0.5, -0.5]), "straddle_lo_wall"),
        # (iv) large variance near upper wall on x only
        (np.array([2.5, 1.5]), np.array([0.8, 0.2]), np.array([0.4, 0.0]), "large_var_near_wall"),
    ]

    for seed_offset, (mu, sigma_diag, delta, desc) in enumerate(cases):
        Sigma = np.diag(sigma_diag)
        cp = ContinuousPlace(mu.copy(), Sigma.copy(), arena)
        cp.predict_clamped_moments(delta, Q)
        analytic_mean = cp.mu
        analytic_var = np.diag(cp.Sigma)

        mc_mean, mc_var = _mc_clamped_moments(mu, sigma_diag, delta, Q_diag, arena, seed=seed_offset)

        np.testing.assert_allclose(
            analytic_mean, mc_mean, atol=0.01,
            err_msg=f"case={desc}: mean mismatch (analytic={analytic_mean}, mc={mc_mean})",
        )
        np.testing.assert_allclose(
            analytic_var, mc_var, atol=0.01,
            err_msg=f"case={desc}: variance mismatch (analytic={analytic_var}, mc={mc_var})",
        )


# ---------------------------------------------------------------------------
# test_clamped_moments_no_wall_equals_naive
# ---------------------------------------------------------------------------

def test_clamped_moments_no_wall_equals_naive() -> None:
    """Far from walls, predict_clamped_moments == naive predict (mu+delta, Sigma+Q) to 1e-6."""
    # Use a large arena so the mean + delta is nowhere near a wall
    arena = (0.0, 100.0, 0.0, 100.0)
    mu = np.array([50.0, 50.0])
    sigma_diag = np.array([0.4, 0.6])
    Sigma = np.diag(sigma_diag)
    delta = np.array([0.3, -0.2])
    Q_diag = np.array([0.05 ** 2, 0.05 ** 2])
    Q = np.diag(Q_diag)

    cp_clamped = ContinuousPlace(mu.copy(), Sigma.copy(), arena)
    cp_clamped.predict_clamped_moments(delta, Q)

    cp_naive = ContinuousPlace(mu.copy(), Sigma.copy(), arena)
    cp_naive.predict(delta, Q)

    np.testing.assert_allclose(
        cp_clamped.mu, cp_naive.mu, atol=1e-6,
        err_msg="Far-from-wall: clamped-moments mean should equal naive mean",
    )
    np.testing.assert_allclose(
        np.diag(cp_clamped.Sigma), np.diag(cp_naive.Sigma), atol=1e-6,
        err_msg="Far-from-wall: clamped-moments diagonal variance should equal naive variance",
    )


# ---------------------------------------------------------------------------
# test_clamped_moments_shrinks_variance_at_wall
# ---------------------------------------------------------------------------

def test_clamped_moments_shrinks_variance_at_wall() -> None:
    """Mean pushed well past a wall -> post variance on that axis < pre variance + Q
    (wall information gained through moment-matched clamping)."""
    arena = _ARENA_3
    # Push x well past the upper wall: mu_x=2.9, delta_x=+1.5 -> pre-clamp m_x=4.4 >> xmax=3.0
    mu = np.array([2.9, 1.5])
    sigma_diag = np.array([0.5, 0.5])
    Sigma = np.diag(sigma_diag)
    delta = np.array([1.5, 0.0])
    Q_diag = np.array([0.05 ** 2, 0.05 ** 2])
    Q = np.diag(Q_diag)

    pre_var_x = sigma_diag[0] + Q_diag[0]  # naive pre-clamp variance on x

    cp = ContinuousPlace(mu.copy(), Sigma.copy(), arena)
    cp.predict_clamped_moments(delta, Q)
    post_var_x = cp.Sigma[0, 0]

    assert post_var_x < pre_var_x, (
        f"Expected variance to shrink at wall (wall information gained): "
        f"post_var_x={post_var_x:.6f} should be < pre_var_x={pre_var_x:.6f}"
    )


# ---------------------------------------------------------------------------
# ContinuousCreature tests
# ---------------------------------------------------------------------------

import tempfile
from active_loop.creature_continuous import ContinuousCreature


def test_continuous_creature_roundtrip() -> None:
    """birth -> live(200, seed=5) -> save -> load -> state_hash identical;
    two fresh loads each living 50 further steps with the same explicit seed
    produce identical hashes (resume determinism)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = tmpdir + "/nira_test"

        # Birth and live
        c = ContinuousCreature.birth("nira_test", seed=42)
        c.live(200, seed=5)
        hash_before = c.state_hash()

        # Save and reload
        c.save(state_dir)
        c2 = ContinuousCreature.load(state_dir)
        assert c2.state_hash() == hash_before, (
            f"Round-trip hash mismatch: before={hash_before[:12]} "
            f"loaded={c2.state_hash()[:12]}"
        )

        # Resume determinism: two fresh loads, same explicit seed -> same hash
        c3 = ContinuousCreature.load(state_dir)
        c4 = ContinuousCreature.load(state_dir)
        c3.live(50, seed=9)
        c4.live(50, seed=9)
        assert c3.state_hash() == c4.state_hash(), (
            "Resume nondeterminism: two fresh loads with same seed produced "
            f"different hashes after live(50): {c3.state_hash()[:12]} vs {c4.state_hash()[:12]}"
        )


def test_continuous_creature_biography_appends() -> None:
    """live() called twice -> biography has 2 live events with increasing ages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = tmpdir + "/bio_test"

        c = ContinuousCreature.birth("bio_test", seed=7)
        # Bind a state dir so biography is written
        import pathlib
        pathlib.Path(state_dir).mkdir(parents=True, exist_ok=True)
        c._state_dir = pathlib.Path(state_dir)

        c.live(100)
        age1 = c.age_steps
        c.live(50)
        age2 = c.age_steps

        # Read biography
        import json
        bio_path = pathlib.Path(state_dir) / "BIOGRAPHY.jsonl"
        events = [json.loads(line) for line in bio_path.read_text().splitlines() if line.strip()]
        live_events = [e for e in events if e.get("event") == "live"]

        assert len(live_events) == 2, f"Expected 2 live events, got {len(live_events)}"
        assert live_events[0]["age_steps"] == age1, (
            f"First live event age {live_events[0]['age_steps']} != {age1}"
        )
        assert live_events[1]["age_steps"] == age2, (
            f"Second live event age {live_events[1]['age_steps']} != {age2}"
        )
        assert age2 > age1, f"Age did not increase: {age1} -> {age2}"
