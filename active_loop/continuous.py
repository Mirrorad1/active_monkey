"""Problem-2 continuous-substrate math: precision-accumulation Gaussian belief and NIW
conjugate prior for continuous latent concepts.

This module implements the closed-form Bayesian building blocks for the continuous-
substrate direction (Problem 2 in the active-loop research programme).  The central
design choice is natural-parameter (precision) form for Gaussian beliefs, enabling O(d^2)
online updates without matrix inversions at each step.  NIW conjugacy provides the
foundation for emission-parameter learning (rung 3 and above).  See
docs/research/problem2-continuous-substrate.md for the full research context and the
relationship to the tabular (discrete state-space) twin.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky, solve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cho_inv(A: np.ndarray) -> np.ndarray:
    """Invert a symmetric positive-definite matrix via Cholesky."""
    d = A.shape[0]
    L = cholesky(A)
    L_inv = solve(L, np.eye(d))
    return L_inv.T @ L_inv


def _cho_logdet(A: np.ndarray) -> float:
    """log|A| for symmetric PSD A via Cholesky (2 * sum log diag L)."""
    L = cholesky(A)
    return 2.0 * float(np.sum(np.log(np.diag(L))))


# ---------------------------------------------------------------------------
# GaussianBelief
# ---------------------------------------------------------------------------

class GaussianBelief:
    """Natural-parameter Gaussian belief over a latent state s in R^d.

    Internal representation: precision matrix Lambda = Sigma^{-1} and
    natural parameter h = Lambda @ mu.  All updates are O(d^2) additions —
    no matrix inversion at update time.

    Parameters
    ----------
    mu : array_like, shape (d,)
        Prior mean.
    Sigma : array_like, shape (d, d)
        Prior covariance (symmetric positive definite).
    """

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray) -> None:
        mu = np.asarray(mu, dtype=float)
        Sigma = np.asarray(Sigma, dtype=float)
        self._Lambda = _cho_inv(Sigma)  # (d, d)
        self._h = self._Lambda @ mu     # (d,)

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def observe(self, mu_k: np.ndarray, Lambda_k: np.ndarray) -> None:
        """Accumulate an observation in natural-parameter form.

        Precision accumulation: Lambda += Lambda_k; h += Lambda_k @ mu_k.
        This is the conjugate update for a Gaussian likelihood with known
        precision Lambda_k centred at mu_k.

        Parameters
        ----------
        mu_k : array_like, shape (d,)
            Likelihood centre (e.g., the word embedding or emission mean).
        Lambda_k : array_like, shape (d, d)
            Likelihood precision (Sigma_k^{-1}).
        """
        Lambda_k = np.asarray(Lambda_k, dtype=float)
        mu_k = np.asarray(mu_k, dtype=float)
        self._Lambda = self._Lambda + Lambda_k
        self._h = self._h + Lambda_k @ mu_k

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def mu(self) -> np.ndarray:
        """Posterior mean: solve(Lambda, h)."""
        return solve(self._Lambda, self._h)

    @property
    def Sigma(self) -> np.ndarray:
        """Posterior covariance: Lambda^{-1} (via Cholesky)."""
        return _cho_inv(self._Lambda)

    @property
    def trace_sigma(self) -> float:
        """tr(Sigma) — posterior uncertainty volume."""
        return float(np.trace(self.Sigma))

    @property
    def precision_trace(self) -> float:
        """tr(Lambda) — accumulated precision."""
        return float(np.trace(self._Lambda))

    @property
    def entropy(self) -> float:
        """Differential entropy of the Gaussian: 0.5 * log|2*pi*e*Sigma|."""
        d = len(self._h)
        log_det_Sigma = -_cho_logdet(self._Lambda)  # log|Sigma| = -log|Lambda|
        return 0.5 * (d * (1.0 + np.log(2.0 * np.pi)) + log_det_Sigma)


# ---------------------------------------------------------------------------
# Closed-form Gaussian product (reference implementation)
# ---------------------------------------------------------------------------

def gaussian_product(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Textbook product of two unnormalised Gaussians.

    Computes::

        Sigma = (Sigma1^{-1} + Sigma2^{-1})^{-1}
        mu    = Sigma @ (Sigma1^{-1} @ mu1 + Sigma2^{-1} @ mu2)

    Used as an independent reference in tests (both agents update from the
    same two-observation stream; the result must equal GaussianBelief's output).

    Parameters
    ----------
    mu1, mu2 : array_like, shape (d,)
    Sigma1, Sigma2 : array_like, shape (d, d)

    Returns
    -------
    mu : np.ndarray, shape (d,)
    Sigma : np.ndarray, shape (d, d)
    """
    Sigma1 = np.asarray(Sigma1, dtype=float)
    Sigma2 = np.asarray(Sigma2, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)

    L1 = _cho_inv(Sigma1)
    L2 = _cho_inv(Sigma2)
    Sigma_out = _cho_inv(L1 + L2)
    mu_out = Sigma_out @ (L1 @ mu1 + L2 @ mu2)
    return mu_out, Sigma_out


# ---------------------------------------------------------------------------
# Normal-Inverse-Wishart conjugate prior
# ---------------------------------------------------------------------------

class NIW:
    """Normal-Inverse-Wishart conjugate prior over (mu, Sigma) of one Gaussian.

    Parameterised by (m, kappa, nu, S) following the standard convention:
    - m : prior mean of mu (d-vector)
    - kappa : prior pseudo-count for mu (scalar > 0)
    - nu : prior degrees-of-freedom for Sigma (must be >= d + 2 for finite
           expected covariance; enforced at construction)
    - S : prior scatter matrix (d x d, positive definite)

    update(x) and update_batch([x]) are numerically identical.
    """

    def __init__(
        self,
        m: np.ndarray,
        kappa: float,
        nu: float,
        S: np.ndarray,
    ) -> None:
        m = np.asarray(m, dtype=float)
        S = np.asarray(S, dtype=float)
        d = m.shape[0]
        if nu < d + 2:
            raise ValueError(
                f"NIW requires nu >= d + 2 for finite expected covariance; "
                f"got nu={nu}, d={d} (need nu >= {d + 2})"
            )
        self._m = m.copy()
        self._kappa = float(kappa)
        self._nu = float(nu)
        self._S = S.copy()
        self._d = d

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(self, x: np.ndarray) -> "NIW":
        """Return a new NIW updated with a single observation x.

        Standard NIW posterior update (T=1 batch formula applied directly)::

            kappa_T = kappa + 1
            nu_T    = nu + 1
            m_T     = (kappa * m + x) / kappa_T
            S_T     = S + (x - m)(x - m)^T * kappa / kappa_T

        Parameters
        ----------
        x : array_like, shape (d,)

        Returns
        -------
        NIW
            New NIW instance with updated parameters.
        """
        return self.update_batch(np.asarray(x, dtype=float).reshape(1, -1))

    def update_batch(self, X: np.ndarray) -> "NIW":
        """Return a new NIW updated with a batch of observations X (shape T x d).

        Standard NIW batch update::

            xbar    = mean(X, axis=0)
            scatter = sum_t (x_t - xbar)(x_t - xbar)^T
            kappa_T = kappa + T
            nu_T    = nu + T
            m_T     = (kappa * m + T * xbar) / kappa_T
            S_T     = S + scatter + (kappa * T / kappa_T) * (xbar - m)(xbar - m)^T

        Parameters
        ----------
        X : array_like, shape (T, d)

        Returns
        -------
        NIW
            New NIW instance with updated parameters.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        T = X.shape[0]

        xbar = X.mean(axis=0)
        diff = X - xbar
        scatter = diff.T @ diff

        kappa_T = self._kappa + T
        nu_T = self._nu + T
        m_T = (self._kappa * self._m + T * xbar) / kappa_T
        outer = np.outer(xbar - self._m, xbar - self._m)
        S_T = self._S + scatter + (self._kappa * T / kappa_T) * outer

        # nu_T is guaranteed >= d+2 since we only add positive T and started >= d+2
        new = NIW.__new__(NIW)
        new._m = m_T
        new._kappa = kappa_T
        new._nu = nu_T
        new._S = S_T
        new._d = self._d
        return new

    def update_moments(self, mu_q: np.ndarray, Sigma_q: np.ndarray) -> "NIW":
        """Return a new NIW updated with one observation known only as N(mu_q, Sigma_q).

        This is moment-matched update for a soft observation: the data point's
        location is uncertain with posterior N(mu_q, Sigma_q).  The mean update and
        kappa/nu increments are identical to update(x=mu_q), but the scatter term
        absorbs the observation uncertainty::

            kappa_T = kappa + 1
            nu_T    = nu + 1
            m_T     = (kappa * m + mu_q) / kappa_T
            S_T     = S + Sigma_q + (kappa / kappa_T) * outer(mu_q - m, mu_q - m)

        The Sigma_q -> 0 limit recovers update(x=mu_q) exactly: the outer-product
        term equals kappa/(kappa+1) * outer(x-m, x-m), which is the standard NIW
        scatter for T=1, and Sigma_q vanishes.

        Parameters
        ----------
        mu_q : array_like, shape (d,)
            Mean of the observation posterior.
        Sigma_q : array_like, shape (d, d)
            Covariance of the observation posterior.  Pass zeros((d,d)) to recover
            the hard-observation update(x=mu_q).

        Returns
        -------
        NIW
            New NIW instance with updated parameters.
        """
        mu_q = np.asarray(mu_q, dtype=float)
        Sigma_q = np.asarray(Sigma_q, dtype=float)

        kappa_T = self._kappa + 1.0
        nu_T = self._nu + 1.0
        m_T = (self._kappa * self._m + mu_q) / kappa_T
        diff = mu_q - self._m
        outer = np.outer(diff, diff)
        S_T = self._S + Sigma_q + (self._kappa / kappa_T) * outer

        new = NIW.__new__(NIW)
        new._m = m_T
        new._kappa = kappa_T
        new._nu = nu_T
        new._S = S_T
        new._d = self._d
        return new

    # ------------------------------------------------------------------
    # Expected parameters
    # ------------------------------------------------------------------

    def expected_mu(self) -> np.ndarray:
        """E[mu] = m."""
        return self._m.copy()

    def expected_Sigma(self) -> np.ndarray:
        """E[Sigma] = S / (nu - d - 1).

        Valid because nu >= d + 2 is enforced at construction.
        """
        return self._S / (self._nu - self._d - 1)


# ---------------------------------------------------------------------------
# Log-space categorical filtering (the tabular twin's safe update)
# ---------------------------------------------------------------------------

def log_categorical_posterior(
    logA: np.ndarray,
    words: np.ndarray,
    log_prior: np.ndarray | None = None,
) -> np.ndarray:
    """Exact static-state categorical posterior in log space.

    Computes log q_c = log_prior_c + sum_t logA[words_t, c], normalised.  This is
    the order-independent product posterior; computing it in probability space with
    per-step renormalisation lets individual entries underflow to exact float 0,
    after which they can never recover — an order-dependent ratchet (found in
    Exp 134: 7/46 argmax anomalies, all artifacts of the floored filter).  Tabular
    twins in this direction MUST use this instead.

    Parameters
    ----------
    logA : array_like, shape (M, C)
        Log emission table, log p(word=k | state=c).
    words : array_like of int, shape (T,)
        Observed word indices.
    log_prior : array_like, shape (C,), optional
        Log prior over states (default: uniform).

    Returns
    -------
    logq : np.ndarray, shape (C,)
        Normalised log posterior (logsumexp(logq) == 0).
    """
    logA = np.asarray(logA, dtype=float)
    logq = (
        np.zeros(logA.shape[1])
        if log_prior is None
        else np.asarray(log_prior, dtype=float).copy()
    )
    for w in np.asarray(words, dtype=int):
        logq = logq + logA[w, :]
    logq = logq - float(np.logaddexp.reduce(logq))
    return logq


# ---------------------------------------------------------------------------
# Predictive word log-probabilities
# ---------------------------------------------------------------------------

def predictive_word_logprobs(
    mu_post: np.ndarray,
    Sigma_post: np.ndarray,
    word_mus: list[np.ndarray],
    word_Sigmas: list[np.ndarray],
) -> np.ndarray:
    """Normalised log-probs over M word types given posterior Gaussian belief.

    The predictive probability for word k is proportional to the Gaussian
    footprint integral::

        p_k ∝ sqrt(det(2*pi*Sigma_k)) * N(mu_post; mu_k, Sigma_k + Sigma_post)

    where N(.; mu, C) is the normalised Gaussian density.  In log form::

        log_unnorm_k = 0.5 * log|2*pi*Sigma_k|
                       - 0.5 * (mu_post - mu_k)^T (Sigma_k + Sigma_post)^{-1} (mu_post - mu_k)
                       - 0.5 * log|2*pi*(Sigma_k + Sigma_post)|

    The det(Sigma_k) factor is kept so word-types with different covariances are
    correctly compared.  The result is log-softmax normalised over k (numerically
    stable via logsumexp).

    Parameters
    ----------
    mu_post : array_like, shape (d,)
    Sigma_post : array_like, shape (d, d)
    word_mus : list of M array_like, each shape (d,)
    word_Sigmas : list of M array_like, each shape (d, d)

    Returns
    -------
    log_probs : np.ndarray, shape (M,)
        Normalised log-probabilities (log-softmax over k); sums to 0 in
        exp-space, i.e. logsumexp(log_probs) == 0.
    """
    mu_post = np.asarray(mu_post, dtype=float)
    Sigma_post = np.asarray(Sigma_post, dtype=float)
    M = len(word_mus)
    log_unnorm = np.empty(M, dtype=float)

    for k in range(M):
        mu_k = np.asarray(word_mus[k], dtype=float)
        Sigma_k = np.asarray(word_Sigmas[k], dtype=float)
        C = Sigma_k + Sigma_post  # (d, d)
        diff = mu_post - mu_k
        # log N(mu_post; mu_k, C) = -0.5*log|2*pi*C| - 0.5*diff^T C^{-1} diff
        log_det_C = _cho_logdet(C)
        log_det_Sk = _cho_logdet(Sigma_k)
        d = mu_k.shape[0]
        maha = float(diff @ solve(C, diff))
        # log unnorm: 0.5*log|2*pi*Sigma_k| + log N(mu_post; mu_k, C)
        #           = 0.5*log|2*pi*Sigma_k| - 0.5*log|2*pi*C| - 0.5*maha
        #           = 0.5*(d*log(2*pi) + log|Sigma_k|) - 0.5*(d*log(2*pi) + log|C|) - 0.5*maha
        #           = 0.5*(log|Sigma_k| - log|C|) - 0.5*maha
        log_unnorm[k] = 0.5 * (log_det_Sk - log_det_C) - 0.5 * maha

    # log-softmax: subtract logsumexp for normalisation
    log_sum = float(np.logaddexp.reduce(log_unnorm))
    return log_unnorm - log_sum
