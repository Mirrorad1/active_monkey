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

import math

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

    def predict_clamped_moments(self, delta: np.ndarray, Q: np.ndarray) -> None:
        """Moment-matched clamped predict for DIAGONAL covariance only.

        Computes the exact moments of Y = clamp(X, lo, hi) per axis where
        X ~ N(m_i, v_i) with m_i = mu_i + delta_i, v_i = Sigma_ii + Q_ii.
        This is the truncated-Gaussian moment formula applied per axis, which
        is exact for diagonal Sigma (axes independent).

        Requires diagonal Sigma: asserts abs(off-diagonal) < 1e-9 after adding Q.
        The tabular twin preserves wall information by construction; the naive
        mean-clamp discards it (variance does not shrink at walls).  This
        variant recovers that information via analytic truncated-Gaussian moments,
        shrinking per-axis variance when the pre-clamp mean is near or past a wall.

        Off-diagonals are set to 0 (the conjugate Gaussian place update with
        isotropic emission keeps diagonality; this method enforces it explicitly).

        Closed-form moments per axis i (a = (lo-m)/s, b = (hi-m)/s, s = sqrt(v)):
          E[Y_i]   = lo*Phi(a) + hi*(1-Phi(b)) + m*(Phi(b)-Phi(a)) + s*(phi(a)-phi(b))
          E[Y_i^2] = lo^2*Phi(a) + hi^2*(1-Phi(b))
                     + m^2*(Phi(b)-Phi(a)) + 2*m*s*(phi(a)-phi(b))
                     + v*((Phi(b)-Phi(a)) + a*phi(a) - b*phi(b))
          Var[Y_i] = E[Y_i^2] - E[Y_i]^2, floored at 1e-8.

        Parameters
        ----------
        delta : array_like, shape (2,)
            Known motor displacement in (x, y).
        Q : array_like, shape (2, 2)
            Process noise covariance (diagonal, symmetric positive semi-definite).
        """
        delta = np.asarray(delta, dtype=float)
        Q = np.asarray(Q, dtype=float)

        xmin, xmax, ymin, ymax = self._arena
        lo = np.array([xmin, ymin])
        hi = np.array([xmax, ymax])

        # Pre-clamp parameters: X_i ~ N(m_i, v_i)
        m = self._mu + delta
        Sigma_pre = self._Sigma + Q

        # Assert diagonal: off-diagonals must be < 1e-9 in absolute value.
        off_diag_max = abs(Sigma_pre[0, 1])
        assert off_diag_max < 1e-9, (
            f"predict_clamped_moments requires diagonal Sigma; "
            f"max off-diagonal = {off_diag_max:.3e} >= 1e-9. "
            f"Use predict() for non-diagonal covariance."
        )

        mu_new = np.empty(2)
        sigma_diag_new = np.empty(2)

        for i in range(2):
            mi = m[i]
            vi = Sigma_pre[i, i]
            si = math.sqrt(vi)
            li = lo[i]
            hi_i = hi[i]

            a = (li - mi) / si
            b = (hi_i - mi) / si

            # CDF values: Phi(a), Phi(b)
            Phi_a = 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))
            Phi_b = 0.5 * (1.0 + math.erf(b / math.sqrt(2.0)))

            # PDF values: phi(a), phi(b)
            phi_a = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
            phi_b = math.exp(-0.5 * b * b) / math.sqrt(2.0 * math.pi)

            # Middle mass: Phi(b) - Phi(a)
            mid = Phi_b - Phi_a
            # P_lo = Phi(a), P_hi = 1 - Phi(b)
            P_lo = Phi_a
            P_hi = 1.0 - Phi_b

            # E[Y_i] = lo*P_lo + hi*P_hi + m*(Phi(b)-Phi(a)) + s*(phi(a)-phi(b))
            EY = li * P_lo + hi_i * P_hi + mi * mid + si * (phi_a - phi_b)

            # E[Y_i^2]: derive via E[X^2; a<Z<b] where Z=(X-m)/s
            # E[X^2; a<Z<b] = integral_{a}^{b} (m + s*z)^2 phi(z) dz
            #               = m^2*(Phi(b)-Phi(a)) + 2*m*s*(phi(a)-phi(b))
            #                 + v*((Phi(b)-Phi(a)) + a*phi(a) - b*phi(b))
            # (last term uses integral z^2 phi(z) dz = -(z*phi(z))' + Phi evaluated at limits)
            EX2_mid = (mi * mi * mid
                       + 2.0 * mi * si * (phi_a - phi_b)
                       + vi * (mid + a * phi_a - b * phi_b))

            EY2 = li * li * P_lo + hi_i * hi_i * P_hi + EX2_mid

            VarY = EY2 - EY * EY
            VarY = max(VarY, 1e-8)

            mu_new[i] = EY
            sigma_diag_new[i] = VarY

        self._mu = mu_new
        # Off-diagonals set to 0: diagonal Sigma maintained explicitly.
        self._Sigma = np.diag(sigma_diag_new)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """Differential entropy of the Gaussian: 0.5 * log|2*pi*e*Sigma|.

        For d=2: 0.5 * (d*(1 + log(2*pi)) + log|Sigma|).
        Used for diagnostics to track place-belief uncertainty over time.
        """
        d = 2
        sign, logdet = np.linalg.slogdet(self._Sigma)
        return 0.5 * (d * (1.0 + math.log(2.0 * math.pi)) + logdet)

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
