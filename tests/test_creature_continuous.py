"""Tests for active_loop.creature_continuous — ContinuousPlace Gaussian belief."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.linalg import inv

from active_loop.continuous import gaussian_product
from active_loop.creature_continuous import ContinuousPlace


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
