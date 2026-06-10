"""Tests for active_loop.continuous — precision-accumulation Gaussian belief and NIW."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.linalg import inv

from active_loop.continuous import (
    GaussianBelief,
    NIW,
    gaussian_product,
    predictive_word_logprobs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_psd(d: int, rng: np.random.Generator, eps: float = 0.5) -> np.ndarray:
    A = rng.standard_normal((d, d))
    return A @ A.T + eps * np.eye(d)


# ---------------------------------------------------------------------------
# test_gaussian_product_matches_natural_params
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 5])
def test_gaussian_product_matches_natural_params(d: int) -> None:
    rng = np.random.default_rng(42 + d)
    mu1 = rng.standard_normal(d)
    Sigma1 = _random_psd(d, rng)
    mu2 = rng.standard_normal(d)
    Sigma2 = _random_psd(d, rng)

    # Reference: textbook closed form
    mu_ref, Sigma_ref = gaussian_product(mu1, Sigma1, mu2, Sigma2)

    # GaussianBelief path: start at (mu1, Sigma1), then observe (mu2, Lambda2)
    Lambda2 = inv(Sigma2)
    belief = GaussianBelief(mu1, Sigma1)
    belief.observe(mu2, Lambda2)

    np.testing.assert_allclose(belief.mu, mu_ref, atol=1e-8,
                               err_msg=f"mu mismatch at d={d}")
    np.testing.assert_allclose(belief.Sigma, Sigma_ref, atol=1e-8,
                               err_msg=f"Sigma mismatch at d={d}")


# ---------------------------------------------------------------------------
# test_precision_monotone
# ---------------------------------------------------------------------------

def test_precision_monotone() -> None:
    rng = np.random.default_rng(7)
    d = 3
    Sigma0 = _random_psd(d, rng)
    mu0 = rng.standard_normal(d)
    belief = GaussianBelief(mu0, Sigma0)

    prev_prec_trace = belief.precision_trace
    prev_sigma_trace = belief.trace_sigma

    for _ in range(20):
        mu_k = rng.standard_normal(d)
        Sigma_k = _random_psd(d, rng)
        Lambda_k = inv(Sigma_k)
        belief.observe(mu_k, Lambda_k)

        cur_prec_trace = belief.precision_trace
        cur_sigma_trace = belief.trace_sigma

        assert cur_prec_trace > prev_prec_trace, (
            f"precision_trace not strictly increasing: {prev_prec_trace} -> {cur_prec_trace}"
        )
        assert cur_sigma_trace <= prev_sigma_trace + 1e-12, (
            f"trace_sigma increased: {prev_sigma_trace} -> {cur_sigma_trace}"
        )
        prev_prec_trace = cur_prec_trace
        prev_sigma_trace = cur_sigma_trace


# ---------------------------------------------------------------------------
# test_niw_requires_proper_nu
# ---------------------------------------------------------------------------

def test_niw_requires_proper_nu() -> None:
    d = 4
    m = np.zeros(d)
    S = np.eye(d)
    with pytest.raises(ValueError):
        NIW(m=m, kappa=1.0, nu=d + 1, S=S)


# ---------------------------------------------------------------------------
# test_niw_limits
# ---------------------------------------------------------------------------

def test_niw_limits() -> None:
    rng = np.random.default_rng(123)
    d = 3

    # (a) Large kappa0: expected_mu should barely move from m0 after 10 obs
    m0 = np.array([1.0, -0.5, 2.0])
    kappa0 = 1e8
    nu0 = d + 2
    niw_a = NIW(m=m0, kappa=kappa0, nu=float(nu0), S=np.eye(d))
    data = rng.standard_normal((10, d)) * 5.0  # data far from m0
    niw_a_updated = niw_a.update_batch(data)
    # displacement of data centroid from m0
    data_displacement = float(np.linalg.norm(data.mean(axis=0) - m0))
    mu_displacement = float(np.linalg.norm(niw_a_updated.expected_mu() - m0))
    assert mu_displacement < 1e-4 * data_displacement, (
        f"Large kappa0: mu moved too much ({mu_displacement:.6f}) relative to data "
        f"displacement ({data_displacement:.6f})"
    )

    # (b) Small kappa0, weak prior -> expected_mu should track empirical mean
    rng2 = np.random.default_rng(456)
    target = np.array([3.0, -1.0, 0.5])
    kappa0_weak = 1e-3
    nu0_weak = d + 2
    niw_b = NIW(m=np.zeros(d), kappa=kappa0_weak, nu=float(nu0_weak), S=np.eye(d))
    draws = rng2.multivariate_normal(target, 0.1 * np.eye(d), size=500)
    niw_b_updated = niw_b.update_batch(draws)
    emp_mean = draws.mean(axis=0)
    err = float(np.linalg.norm(niw_b_updated.expected_mu() - emp_mean))
    assert err < 0.05, (
        f"Weak prior: expected_mu={niw_b_updated.expected_mu()} far from "
        f"empirical mean={emp_mean}, err={err:.4f}"
    )

    # (c) update(x) == update_batch([x]) exactly
    rng3 = np.random.default_rng(789)
    m_c = rng3.standard_normal(d)
    niw_c1 = NIW(m=m_c, kappa=2.0, nu=float(d + 2), S=np.eye(d))
    niw_c2 = NIW(m=m_c.copy(), kappa=2.0, nu=float(d + 2), S=np.eye(d))
    x = rng3.standard_normal(d)
    niw_c1_u = niw_c1.update(x)
    niw_c2_u = niw_c2.update_batch(x.reshape(1, -1))
    np.testing.assert_allclose(niw_c1_u.expected_mu(), niw_c2_u.expected_mu(),
                               atol=1e-12, err_msg="update vs update_batch: mu mismatch")
    np.testing.assert_allclose(niw_c1_u.expected_Sigma(), niw_c2_u.expected_Sigma(),
                               atol=1e-12, err_msg="update vs update_batch: Sigma mismatch")


# ---------------------------------------------------------------------------
# test_predictive_logprobs_normalized
# ---------------------------------------------------------------------------

def test_predictive_logprobs_normalized() -> None:
    rng = np.random.default_rng(99)
    d = 2
    M = 6

    # Build M word Gaussians
    word_mus = [rng.standard_normal(d) for _ in range(M)]
    word_Sigmas = [_random_psd(d, rng) for _ in range(M)]

    # Arbitrary posterior
    mu_post = rng.standard_normal(d)
    Sigma_post = _random_psd(d, rng)

    log_probs = predictive_word_logprobs(mu_post, Sigma_post, word_mus, word_Sigmas)

    # Must be normalised: logsumexp == 0
    log_sum = float(np.logaddexp.reduce(log_probs))
    assert abs(log_sum) < 1e-10, f"logsumexp={log_sum} (expected 0)"

    # Posterior sitting exactly on word 0's centre with tiny Sigma_post
    mu_post_0 = word_mus[0].copy()
    Sigma_post_tiny = 1e-8 * np.eye(d)
    log_probs_peaked = predictive_word_logprobs(
        mu_post_0, Sigma_post_tiny, word_mus, word_Sigmas
    )
    assert np.argmax(log_probs_peaked) == 0, (
        f"Expected word 0 to dominate; got argmax={np.argmax(log_probs_peaked)}, "
        f"log_probs={log_probs_peaked}"
    )


def test_log_categorical_posterior_order_independent_no_ratchet():
    """Exp 134 guard: the log-space filter must be order-independent and immune to
    the underflow ratchet that breaks multiply-then-renormalize filters."""
    from active_loop.continuous import log_categorical_posterior

    rng = np.random.default_rng(0)
    # Extreme-separation table: off-state likelihoods ~ exp(-500)
    logA = np.array([[0.0, -500.0], [-500.0, 0.0]])
    logA = logA - np.logaddexp.reduce(logA, axis=0, keepdims=True)
    # 60 of word 0, 40 of word 1 -> majority state 0 must win
    words = np.array([0] * 60 + [1] * 40)
    logq = log_categorical_posterior(logA, words)
    assert np.isclose(np.logaddexp.reduce(logq), 0.0, atol=1e-10)
    assert int(np.argmax(logq)) == 0
    # order independence: any permutation gives the identical posterior
    perm = rng.permutation(len(words))
    logq_perm = log_categorical_posterior(logA, words[perm])
    assert np.allclose(logq, logq_perm, atol=1e-9)
    # the naive prob-space filter ratchets on this input (documents WHY the
    # guard exists): state 0's entry hits exact float 0 on the first word-1
    # run-in and never recovers
    q = np.ones(2) / 2
    A = np.exp(logA)
    for w in np.concatenate([[1] * 2, [0] * 98]):  # 2 early minority words
        q = q * A[w, :]
        s = q.sum()
        q = np.ones(2) / 2 if s < 1e-300 else q / s
    # exact Bayes on the same stream says state 0 (98 vs 2)
    logq2 = log_categorical_posterior(logA, np.concatenate([[1] * 2, [0] * 98]))
    assert int(np.argmax(logq2)) == 0
