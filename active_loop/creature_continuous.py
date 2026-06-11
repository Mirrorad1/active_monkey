"""Continuous-substrate place machinery for the continuous-creature migration.

Direction card: loop/directions/continuous-creature.md
Migration directive "A" (loop/IDEAS.md, 2026-06-10).

ADDITIVE module — the tabular creature (creature.py) is untouched.

Implements a Gaussian place belief over s in R^2 on a clamped rectangular arena,
using mean/covariance form (d=2 is tiny; clarity over precision-form here).

Declared approximation: predict() clamps the posterior mean to the arena but does
not adjust the covariance to reflect the wall boundary.  The true wall posterior is
non-Gaussian; the experiment (Exp 141) measures the calibration cost of this
approximation via coverage_95() tracking.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import inv


class ContinuousPlace:
    """Gaussian place belief over s in R^2 on a clamped rectangular arena.

    The state is represented in mean/covariance form.  The arena is an axis-aligned
    rectangle [xmin, xmax] x [ymin, ymax].

    Parameters
    ----------
    mu : array_like, shape (2,)
        Prior mean position.
    Sigma : array_like, shape (2, 2)
        Prior covariance (symmetric positive definite).
    arena : tuple of four floats (xmin, xmax, ymin, ymax)
        Inclusive bounds of the rectangular arena.
    """

    def __init__(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        arena: tuple,
    ) -> None:
        self._mu = np.asarray(mu, dtype=float).copy()
        self._Sigma = np.asarray(Sigma, dtype=float).copy()
        self._arena = tuple(float(v) for v in arena)  # (xmin, xmax, ymin, ymax)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clamp(self, mu: np.ndarray) -> np.ndarray:
        """Clamp a 2-vector to the arena (per-axis)."""
        xmin, xmax, ymin, ymax = self._arena
        out = mu.copy()
        out[0] = float(np.clip(out[0], xmin, xmax))
        out[1] = float(np.clip(out[1], ymin, ymax))
        return out

    # ------------------------------------------------------------------
    # Belief updates
    # ------------------------------------------------------------------

    def predict(self, delta: np.ndarray, Q: np.ndarray) -> None:
        """Motor update for known displacement delta with process noise Q.

        DECLARED approximation: only the mean is clamped to the arena; the
        covariance is not adjusted for the wall.  The true wall posterior is
        non-Gaussian.  Exp 141 measures the calibration cost.

        In-place update.  mu is guaranteed to stay inside the arena after each call.

        Parameters
        ----------
        delta : array_like, shape (2,)
            Known motor displacement in (x, y).
        Q : array_like, shape (2, 2)
            Process noise covariance (symmetric positive semi-definite).
        """
        delta = np.asarray(delta, dtype=float)
        Q = np.asarray(Q, dtype=float)
        self._mu = self._clamp(self._mu + delta)
        self._Sigma = self._Sigma + Q

    def update(self, mu_k: np.ndarray, Sigma_k: np.ndarray) -> None:
        """Conjugate observation update (product of Gaussians, textbook form).

        Fuses the current belief N(mu, Sigma) with an observation likelihood
        N(mu_k, Sigma_k) using the standard Gaussian product::

            Lam       = Sigma^{-1}
            Lam_k     = Sigma_k^{-1}
            Sigma_new = (Lam + Lam_k)^{-1}
            mu_new    = Sigma_new @ (Lam @ mu + Lam_k @ mu_k)

        In-place update.

        Parameters
        ----------
        mu_k : array_like, shape (2,)
            Observation likelihood centre.
        Sigma_k : array_like, shape (2, 2)
            Observation likelihood covariance.
        """
        mu_k = np.asarray(mu_k, dtype=float)
        Sigma_k = np.asarray(Sigma_k, dtype=float)

        Lam = inv(self._Sigma)
        Lam_k = inv(Sigma_k)
        Sigma_new = inv(Lam + Lam_k)
        mu_new = Sigma_new @ (Lam @ self._mu + Lam_k @ mu_k)

        self._mu = mu_new
        self._Sigma = Sigma_new

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage_95(self, true_pos: np.ndarray) -> bool:
        """Return True iff true_pos lies within the 95% ellipse of the belief.

        Uses the chi-squared threshold for d=2: chi2_2(0.95) = 5.991.
        A point p is inside the 95% ellipse iff::

            (p - mu)^T Sigma^{-1} (p - mu) <= 5.991

        Parameters
        ----------
        true_pos : array_like, shape (2,)
            Ground-truth position to test.

        Returns
        -------
        bool
        """
        true_pos = np.asarray(true_pos, dtype=float)
        diff = true_pos - self._mu
        maha2 = float(diff @ inv(self._Sigma) @ diff)
        return maha2 <= 5.991

    # ------------------------------------------------------------------
    # Properties (copies — callers cannot mutate internals)
    # ------------------------------------------------------------------

    @property
    def mu(self) -> np.ndarray:
        """Posterior mean (copy)."""
        return self._mu.copy()

    @property
    def Sigma(self) -> np.ndarray:
        """Posterior covariance (copy)."""
        return self._Sigma.copy()
