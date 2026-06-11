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


# ---------------------------------------------------------------------------
# ContinuousCreature — persistent packaged creature with M1-M5 machinery
# ---------------------------------------------------------------------------

import copy as _copy
import hashlib as _hashlib
import json as _json
import math as _math
import time as _time
from pathlib import Path as _Path
from typing import Optional as _Optional

from active_loop.continuous import NIW as _NIW, predictive_word_logprobs as _predictive_word_logprobs


# World constants — non-aliased identity 4x4 grid
_ROWS = 4
_COLS = 4
_N_CELLS = _ROWS * _COLS        # 16
_N_COLORS = 16
_ARENA = (0.0, float(_COLS - 1), 0.0, float(_ROWS - 1))   # (0, 3, 0, 3)
_ARENA_CENTER = np.array([1.5, 1.5])

# Cell (r,c) -> x=c, y=r
_CELL_CENTERS = np.array(
    [[float(c), float(r)] for r in range(_ROWS) for c in range(_COLS)]
)   # shape (16, 2)

# cmap is identity
_CMAP = list(range(_N_CELLS))

# NIW prior constants (from exp142/exp149)
_D = 2
_KAPPA0 = 1.0
_NU0 = 4.0
_S0_SCALE = 0.35 ** 2 * (_NU0 - _D - 1)   # = 0.35^2 * 1 = 0.1225
_S0 = _S0_SCALE * np.eye(_D)

# Process noise
_Q_SCALE = 0.05
_Q = _Q_SCALE ** 2 * np.eye(_D)

# Faithful valence evaluation (M4c)
_SIGMA_EVAL = 0.01 * np.eye(_D)

# Action deltas: 0=up(y-1), 1=down(y+1), 2=left(x-1), 3=right(x+1)
_ACTION_DELTA = {
    0: np.array([0.0, -1.0]),
    1: np.array([0.0, +1.0]),
    2: np.array([-1.0, 0.0]),
    3: np.array([+1.0, 0.0]),
}


def _move_cell(cell: int, action: int) -> int:
    """Wall-clamped grid move."""
    r, c = divmod(cell, _COLS)
    if action == 0:
        r = max(0, r - 1)
    elif action == 1:
        r = min(_ROWS - 1, r + 1)
    elif action == 2:
        c = max(0, c - 1)
    else:
        c = min(_COLS - 1, c + 1)
    return r * _COLS + c


def _clamp_pos(pos: np.ndarray) -> np.ndarray:
    xmin, xmax, ymin, ymax = _ARENA
    out = pos.copy()
    out[0] = float(np.clip(out[0], xmin, xmax))
    out[1] = float(np.clip(out[1], ymin, ymax))
    return out


def _niws_to_arrays(niws: list) -> tuple:
    """Extract (m, kappa, nu, S) arrays from a list of NIW objects."""
    K = len(niws)
    m = np.array([n.m for n in niws], dtype=float)           # (K, 2)
    kappa = np.array([n.kappa for n in niws], dtype=float)   # (K,)
    nu = np.array([n.nu for n in niws], dtype=float)         # (K,)
    S = np.array([n.S for n in niws], dtype=float)           # (K, 2, 2)
    return m, kappa, nu, S


def _niws_from_arrays(m: np.ndarray, kappa: np.ndarray, nu: np.ndarray,
                      S: np.ndarray) -> list:
    """Reconstruct a list of NIW objects from parameter arrays."""
    K = m.shape[0]
    niws = []
    for k in range(K):
        niws.append(_NIW(m=m[k].copy(), kappa=float(kappa[k]),
                         nu=float(nu[k]), S=S[k].copy()))
    return niws


def _predictive_logprobs_at_point(pos_mu: np.ndarray, niws: list) -> np.ndarray:
    """Predictive color log-probs at (pos_mu, SIGMA_EVAL) — faithful M4c rule."""
    word_mus = [niws[k].expected_mu() for k in range(_N_COLORS)]
    word_Sigmas = []
    for k in range(_N_COLORS):
        Sk_full = niws[k].expected_Sigma()
        Sk_diag = np.diag(np.diag(Sk_full)) + 1e-6 * np.eye(_D)
        word_Sigmas.append(Sk_diag)
    return _predictive_word_logprobs(pos_mu, _SIGMA_EVAL, word_mus, word_Sigmas)


def _value_field(cp_mu: np.ndarray, niws: list, value_share: np.ndarray) -> float:
    """V(s) = sum_k value_share_k * p(o=k | s) evaluated at s=cp_mu via SIGMA_EVAL."""
    log_probs = _predictive_logprobs_at_point(cp_mu, niws)
    probs = np.exp(log_probs)
    return float(np.dot(value_share, probs))


class ContinuousCreature:
    """Persistent continuous-substrate creature: M1-M5 machinery in a packaged life.

    The creature lives in the half-noisy 16-color 4x4 world introduced in M4/M5.
    Its state persists across sessions via save()/load(), with an append-only biography
    and hash-verified round-trip determinism.

    Design mirrors Creature (creature.py) conventions exactly:
    - Belief (mu, Sigma) is NEVER reset after birth.
    - All randomness is derived from (_seed, rng_counter): resumed lives are deterministic.
    - Biography (BIOGRAPHY.jsonl) is append-only — the honest log of a life.

    Parameters
    ----------
    name : str — unique identifier.
    noisy_half : 'left' | 'right' | None — which grid half has noisy observations.
    noise_p : float — reliable observation probability in the noisy half (default 0.6).
    mu : np.ndarray, shape (2,) — current place belief mean.
    Sigma_diag : np.ndarray, shape (2,) — diagonal of place belief covariance.
    niw_m : np.ndarray, shape (16, 2) — per-color NIW mean parameters.
    niw_kappa : np.ndarray, shape (16,) — per-color NIW kappa.
    niw_nu : np.ndarray, shape (16,) — per-color NIW nu.
    niw_S : np.ndarray, shape (16, 2, 2) — per-color NIW scatter matrices.
    value_counts : np.ndarray, shape (16,) — grounded valence accumulators.
    vocab : dict[str, np.ndarray] — word -> color-count array.
    true_pos : int — ground-truth cell index.
    age_steps : int — total steps lived.
    lineage : list[str] — ancestry chain.
    rng_counter : int — monotone counter for deterministic seed derivation.
    _seed : int — birth seed.
    _state_dir : Path | None — bound when loaded/saved.
    """

    # World spec (fixed for this species line)
    rows: int = _ROWS
    cols: int = _COLS
    n_colors: int = _N_COLORS
    cmap: list = _CMAP

    def __init__(
        self,
        name: str,
        noisy_half: _Optional[str],
        noise_p: float,
        mu: np.ndarray,
        Sigma_diag: np.ndarray,
        niw_m: np.ndarray,
        niw_kappa: np.ndarray,
        niw_nu: np.ndarray,
        niw_S: np.ndarray,
        value_counts: np.ndarray,
        vocab: dict,
        true_pos: int,
        age_steps: int,
        lineage: list,
        rng_counter: int,
        _seed: int,
        _state_dir: _Optional[_Path] = None,
    ) -> None:
        self.name = name
        self.noisy_half = noisy_half
        self.noise_p = float(noise_p)

        self.mu = np.asarray(mu, dtype=float).copy()           # shape (2,)
        self.Sigma_diag = np.asarray(Sigma_diag, dtype=float).copy()  # shape (2,)

        # NIW parameters stored as arrays for persistence; also kept as NIW list
        self.niw_m = np.asarray(niw_m, dtype=float).copy()          # (16, 2)
        self.niw_kappa = np.asarray(niw_kappa, dtype=float).copy()  # (16,)
        self.niw_nu = np.asarray(niw_nu, dtype=float).copy()        # (16,)
        self.niw_S = np.asarray(niw_S, dtype=float).copy()          # (16, 2, 2)

        self.value_counts = np.asarray(value_counts, dtype=float).copy()  # (16,)
        self.vocab = {w: np.asarray(v, dtype=float).copy() for w, v in vocab.items()}

        self.true_pos = int(true_pos)
        self.age_steps = int(age_steps)
        self.lineage = list(lineage)
        self.rng_counter = int(rng_counter)
        self._seed = int(_seed)
        self._state_dir = _state_dir

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def birth(
        cls,
        name: str,
        seed: int = 0,
        noisy_half: str = 'left',
        noise_p: float = 0.6,
    ) -> "ContinuousCreature":
        """Birth a fresh continuous creature with broad priors.

        Place prior: N(arena_center, 4*I_diag).
        NIW priors: m0=center, kappa0=1, nu0=4, S0=0.1225*I (from exp149).
        value_counts: zeros.
        cmap: identity (non-aliased 16-color 4x4).
        """
        # Broad place prior
        mu = _ARENA_CENTER.copy()
        Sigma_diag = np.array([4.0, 4.0])

        # NIW priors per color
        niw_m = np.tile(_ARENA_CENTER, (_N_COLORS, 1))         # (16, 2)
        niw_kappa = np.full(_N_COLORS, _KAPPA0)                 # (16,)
        niw_nu = np.full(_N_COLORS, _NU0)                       # (16,)
        niw_S = np.tile(_S0, (_N_COLORS, 1, 1))                 # (16, 2, 2)

        value_counts = np.zeros(_N_COLORS)
        vocab: dict = {}

        return cls(
            name=name,
            noisy_half=noisy_half,
            noise_p=noise_p,
            mu=mu,
            Sigma_diag=Sigma_diag,
            niw_m=niw_m,
            niw_kappa=niw_kappa,
            niw_nu=niw_nu,
            niw_S=niw_S,
            value_counts=value_counts,
            vocab=vocab,
            true_pos=0,
            age_steps=0,
            lineage=[],
            rng_counter=0,
            _seed=seed,
        )

    # ------------------------------------------------------------------
    # Internals: rng, hash, helpers
    # ------------------------------------------------------------------

    def _derive_rng(self) -> np.random.Generator:
        """Derive a reproducible RNG from (birth seed, rng_counter).

        Mirror of Creature._derive_rng: resumed life (load -> live) with the same
        rng_counter produces the same sequence as an uninterrupted run.
        """
        combined_seed = (self._seed * 1_000_003 + self.rng_counter) & 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(combined_seed)

    def _niws(self) -> list:
        """Reconstruct a list of NIW objects from the stored parameter arrays."""
        return _niws_from_arrays(self.niw_m, self.niw_kappa, self.niw_nu, self.niw_S)

    def _sync_niws(self, niws: list) -> None:
        """Write back NIW list parameters into the stored arrays."""
        self.niw_m, self.niw_kappa, self.niw_nu, self.niw_S = _niws_to_arrays(niws)

    def _sample_obs(self, true_cell: int, rng: np.random.Generator) -> int:
        """Draw an observation with noise model matching exp149."""
        true_color = int(_CMAP[true_cell])
        col = true_cell % _COLS
        if self.noisy_half == 'left':
            is_noisy = col <= 1
        elif self.noisy_half == 'right':
            is_noisy = col >= 2
        else:
            is_noisy = False
        if is_noisy:
            if rng.random() < self.noise_p:
                return true_color
            else:
                return int(rng.integers(0, _N_COLORS))
        return true_color

    def _reliable_colors(self) -> set:
        """Return the set of color indices in the deterministic (reliable) half."""
        result = set()
        for cell in range(_N_CELLS):
            col = cell % _COLS
            if self.noisy_half == 'left':
                if col >= 2:
                    result.add(int(_CMAP[cell]))
            elif self.noisy_half == 'right':
                if col <= 1:
                    result.add(int(_CMAP[cell]))
            else:
                result.add(int(_CMAP[cell]))
        return result

    def state_hash(self) -> str:
        """SHA-256 over the concatenated raw bytes of all persisted arrays + scalars.

        Deterministic order mirrors Creature._state_hash.
        """
        h = _hashlib.sha256()
        # Numeric arrays in declaration order
        for arr in [self.mu, self.Sigma_diag,
                    self.niw_m, self.niw_kappa, self.niw_nu, self.niw_S,
                    self.value_counts]:
            h.update(arr.tobytes())
        # Scalars as little-endian int64
        h.update(self.true_pos.to_bytes(8, 'little'))
        h.update(self.age_steps.to_bytes(8, 'little'))
        h.update(self.rng_counter.to_bytes(8, 'little'))
        # Vocab in sorted word order
        for word in sorted(self.vocab):
            h.update(self.vocab[word].tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Core life
    # ------------------------------------------------------------------

    def live(self, steps: int, seed: _Optional[int] = None) -> None:
        """Wander for ``steps`` steps with the M4c machinery.

        Phase-A semantics (uniform-random wander):
        - Draw obs with noise model (noisy half: true color w.p. noise_p, else uniform).
        - Place update (Kalman update with learned emissions).
        - Faithful valence rule (M4c): weight = exp(-H(predictive at (mu_post, SIGMA_EVAL))).
        - value_counts[obs] += weight.
        - NIW moment-matched update with (mu_post, Sigma_post).
        - Random action, wall-clamped move.
        - predict_clamped_moments(delta, Q).

        RNG derivation mirrors Creature: explicit seed overrides; else _derive_rng
        from (_seed, rng_counter); rng_counter += 1 per live() call.

        Biography appended on each call.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self._derive_rng()

        # Reconstruct NIW list for step-by-step updates
        niws = self._niws()

        # Working place belief (mean/covariance form)
        cp = ContinuousPlace(self.mu.copy(), np.diag(self.Sigma_diag), _ARENA)

        for _ in range(steps):
            obs = self._sample_obs(self.true_pos, rng)

            # Place update
            mu_k = niws[obs].expected_mu()
            Sk_full = niws[obs].expected_Sigma()
            Sigma_k = np.diag(np.diag(Sk_full)) + 1e-6 * np.eye(_D)
            cp.update(mu_k, Sigma_k)

            # M4c faithful valence: predictive at (mu_post, SIGMA_EVAL)
            post_mu = cp.mu
            log_pred = _predictive_logprobs_at_point(post_mu, niws)
            H = float(-np.sum(np.exp(log_pred) * log_pred))
            weight = _math.exp(-H)
            self.value_counts[obs] += weight

            # NIW moment-matched update
            niws[obs] = niws[obs].update_moments(cp.mu, cp.Sigma)

            # Random action, move, predict
            action = int(rng.integers(0, 4))
            self.true_pos = _move_cell(self.true_pos, action)
            delta = _ACTION_DELTA[action]
            cp.predict_clamped_moments(delta, _Q)

        # Write back continuous place state
        self.mu = cp.mu
        self.Sigma_diag = np.diag(cp.Sigma)

        # Write back NIW arrays
        self._sync_niws(niws)

        self.age_steps += steps
        self.rng_counter += 1

        # Biography
        n_formed = self.map_formed_count()
        loc_err = self._localization_error()
        rel_share = self.reliable_share()
        summary = (
            f"lived {steps} steps; map_formed={n_formed}/{_N_COLORS}; "
            f"loc_err={loc_err:.3f}; reliable_share={rel_share:.4f}"
        )
        self._bio_append({
            "event": "live",
            "age_steps": self.age_steps,
            "summary": summary,
            "state_hash": self.state_hash(),
        })

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def map_formed_count(self) -> int:
        """Number of colors whose learned center is within 0.5 of the true cell center."""
        count = 0
        for k in range(_N_COLORS):
            center_k = _CELL_CENTERS[k]   # identity cmap: color k -> cell k
            dist = float(np.linalg.norm(self.niw_m[k] - center_k))
            if dist <= 0.5:
                count += 1
        return count

    def _localization_error(self) -> float:
        """L2 distance from current mu to the true cell center."""
        r, c = divmod(self.true_pos, _COLS)
        true_xy = np.array([float(c), float(r)])
        return float(np.linalg.norm(self.mu - true_xy))

    def reliable_share(self) -> float:
        """Value share of the reliable (deterministic) half."""
        rel = self._reliable_colors()
        total = self.value_counts.sum()
        if total == 0:
            return 0.0
        return float(sum(self.value_counts[k] for k in rel) / total)

    def favorite(self) -> int:
        """Return the color index with highest accumulated value."""
        return int(np.argmax(self.value_counts))

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------

    def teach_word(self, word: str, color_idx: int, n: int = 8) -> None:
        """Associate a word with a color via n few-shot examples.

        Mirrors creature.py teach_word exactly:
            vocab[word] = ones(n_colors) * 0.1  (if new)
            vocab[word][color_idx] += n
        Biography appended.
        """
        if word not in self.vocab:
            self.vocab[word] = np.ones(_N_COLORS) * 0.1
        self.vocab[word][color_idx] += n

        self._bio_append({
            "event": "teach_word",
            "age_steps": self.age_steps,
            "summary": f"taught word '{word}' -> color {color_idx} (n={n})",
            "state_hash": self.state_hash(),
        })

    def answer_what_do_you_like(self) -> str:
        """Answer 'what do you like?' in taught words.

        Mirrors creature.py answer_what_do_you_like exactly.
        """
        fav = self.favorite()
        word = self._word_for_color(fav)
        if word is not None:
            return f"I like {word}"
        return f"I like color-{fav} (no word taught yet)"

    def answer_do_you_like(self, word: str) -> str:
        """Answer 'do you like <word>?' using self-formed values and taught labels.

        Mirrors creature.py answer_do_you_like exactly (threshold = 1/n_colors).
        """
        color = self._color_for_word(word)
        if color is None:
            return f"I don't know what '{word}' means"
        total = self.value_counts.sum()
        if total == 0:
            return "I haven't experienced enough to say"
        val_frac = float(self.value_counts[color] / total)
        threshold = 1.0 / _N_COLORS   # = 0.5 / n_colors * 2.0, exactly creature.py line 539
        if val_frac > threshold:
            return f"I like {word}"
        else:
            return f"{word} unsettles me"

    def _word_for_color(self, color: int) -> _Optional[str]:
        """Best word for a color (mirrors creature.py)."""
        if not self.vocab:
            return None
        best_word, best_score = None, -1.0
        for word, counts in self.vocab.items():
            total = counts.sum()
            if total > 0:
                score = float(counts[color] / total)
                if score > best_score:
                    best_score, best_word = score, word
        return best_word

    def _color_for_word(self, word: str) -> _Optional[int]:
        """Most associated color for a word (mirrors creature.py)."""
        if word not in self.vocab:
            return None
        return int(np.argmax(self.vocab[word]))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _bio_path(self) -> _Optional[_Path]:
        if self._state_dir is None:
            return None
        return self._state_dir / "BIOGRAPHY.jsonl"

    def _bio_append(self, record: dict) -> None:
        """Append one event to the biography file (append-only)."""
        path = self._bio_path()
        if path is None:
            return
        with path.open("a") as fh:
            fh.write(_json.dumps(record) + "\n")

    def save(self, dir_path) -> None:
        """Save state to ``dir_path``.

        Creates:
          - ``arrays.npz``: all numeric state arrays.
          - ``manifest.json``: scalars + world config + state_hash.
          - Appends a BIOGRAPHY event.

        Round-trip guarantee: load(dir_path).state_hash() == self.state_hash().
        """
        dir_path = _Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self._state_dir = dir_path

        # Pack vocab arrays
        vocab_keys = sorted(self.vocab.keys())
        vocab_arrays = {f"vocab__{w}": self.vocab[w] for w in vocab_keys}

        np.savez(
            dir_path / "arrays.npz",
            mu=self.mu,
            Sigma_diag=self.Sigma_diag,
            niw_m=self.niw_m,
            niw_kappa=self.niw_kappa,
            niw_nu=self.niw_nu,
            niw_S=self.niw_S,
            value_counts=self.value_counts,
            **vocab_arrays,
        )

        hash_ = self.state_hash()
        manifest = {
            "name": self.name,
            "lineage": self.lineage,
            "age_steps": self.age_steps,
            "true_pos": self.true_pos,
            "rng_counter": self.rng_counter,
            "seed": self._seed,
            "noisy_half": self.noisy_half,
            "noise_p": self.noise_p,
            "world": {
                "rows": self.rows,
                "cols": self.cols,
                "n_colors": self.n_colors,
                "cmap": self.cmap,
            },
            "state_hash": hash_,
            "saved_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }
        (dir_path / "manifest.json").write_text(_json.dumps(manifest, indent=2) + "\n")

        self._bio_append({
            "event": "save",
            "age_steps": self.age_steps,
            "summary": f"saved to {dir_path}",
            "state_hash": hash_,
        })

    @classmethod
    def load(cls, dir_path) -> "ContinuousCreature":
        """Load creature from ``dir_path``.  Verifies state_hash integrity."""
        dir_path = _Path(dir_path)
        manifest = _json.loads((dir_path / "manifest.json").read_text())
        arrs = np.load(dir_path / "arrays.npz", allow_pickle=False)

        vocab = {}
        for key in arrs.files:
            if key.startswith("vocab__"):
                word = key[len("vocab__"):]
                vocab[word] = arrs[key].copy()

        c = cls(
            name=manifest["name"],
            noisy_half=manifest["noisy_half"],
            noise_p=float(manifest["noise_p"]),
            mu=arrs["mu"].copy(),
            Sigma_diag=arrs["Sigma_diag"].copy(),
            niw_m=arrs["niw_m"].copy(),
            niw_kappa=arrs["niw_kappa"].copy(),
            niw_nu=arrs["niw_nu"].copy(),
            niw_S=arrs["niw_S"].copy(),
            value_counts=arrs["value_counts"].copy(),
            vocab=vocab,
            true_pos=int(manifest["true_pos"]),
            age_steps=int(manifest["age_steps"]),
            lineage=list(manifest["lineage"]),
            rng_counter=int(manifest["rng_counter"]),
            _seed=int(manifest["seed"]),
            _state_dir=dir_path,
        )

        computed = c.state_hash()
        stored = manifest.get("state_hash", "")
        if stored and computed != stored:
            raise ValueError(
                f"state_hash mismatch for '{manifest['name']}': "
                f"stored={stored[:12]}... computed={computed[:12]}..."
            )
        return c

    def __repr__(self) -> str:
        return (
            f"ContinuousCreature(name={self.name!r}, age={self.age_steps}, "
            f"map_formed={self.map_formed_count()}/{_N_COLORS}, "
            f"noisy_half={self.noisy_half!r})"
        )
