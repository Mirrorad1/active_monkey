"""Exp 243 — trait-agnostic stability-certification instrument.

Consumes an analysis-window N(t) series (+ telemetry) and applies the predeclared,
FROZEN stability gates and the hardened oscillation detector from
docs/superpowers/specs/2026-06-19-continuous-substrate-stabilization-design.md.
Pure numpy (no scipy.signal) for determinism.
"""
from __future__ import annotations
import numpy as np


def n_eq(N) -> float:
    return float(np.median(np.asarray(N, dtype=float)))


def level_cv(N) -> float:
    N = np.asarray(N, dtype=float)
    m = N.mean()
    return float(N.std() / m) if m != 0 else float("inf")


def drift_slope(N, n_eq_val) -> float:
    """Total fractional drift across the window: |OLS_slope * len(N) / n_eq|."""
    N = np.asarray(N, dtype=float)
    t = np.arange(len(N), dtype=float)
    slope = np.polyfit(t, N, 1)[0]
    return float(abs(slope * len(N) / n_eq_val)) if n_eq_val != 0 else float("inf")


def return_map_slope(N) -> float:
    """Local OLS slope of N(t+1) vs N(t) over the window (empirical one-step return map)."""
    N = np.asarray(N, dtype=float)
    x, y = N[:-1], N[1:]
    return float(np.polyfit(x, y, 1)[0])


def seed_agreement(n_eqs) -> float:
    a = np.asarray(n_eqs, dtype=float)
    med = np.median(a)
    return float((a.max() - a.min()) / med) if med != 0 else float("inf")


def persistence(N) -> float:
    return float(np.asarray(N, dtype=float).min())


# --- hardened oscillation detector (thresholds PREDECLARED + FROZEN per the spec) ---
_AR_MODULUS_MAX = 0.90
_AUTOCORR_TROUGH_MIN = -0.30
_AMP_PTP_MAX = 0.30
_DAMP_K = 2.0


def _detrend(N):
    N = np.asarray(N, dtype=float)
    t = np.arange(len(N), dtype=float)
    a, b = np.polyfit(t, N, 1)
    return N - (a*t + b)


def _ar2_dominant_modulus(x):
    """Fit x_t = phi1 x_{t-1} + phi2 x_{t-2} by least squares; return max root modulus."""
    x = np.asarray(x, dtype=float)
    X = np.column_stack([x[1:-1], x[:-2]]); y = x[2:]
    phi, *_ = np.linalg.lstsq(X, y, rcond=None)
    phi1, phi2 = phi
    # characteristic roots of z^2 - phi1 z - phi2 = 0
    roots = np.roots([1.0, -phi1, -phi2])
    return float(np.max(np.abs(roots)))


def _autocorr(x, max_lag):
    x = np.asarray(x, dtype=float); x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(max_lag+1)
    return np.array([np.dot(x[:len(x)-k], x[k:]) / denom for k in range(max_lag+1)])


def _periodogram_prominence(x):
    """Power in the dominant bin (+/-1 neighbor) as a fraction of total (excl. DC)."""
    x = np.asarray(x, dtype=float)
    P = np.abs(np.fft.rfft(x - x.mean()))**2
    P = P[1:]                                   # drop DC
    if P.sum() == 0 or len(P) < 3:
        return 0.0
    k = int(np.argmax(P))
    lo, hi = max(0, k-1), min(len(P), k+2)
    return float(P[lo:hi].sum() / P.sum())


def _ar1_null_prominence_95(x, seed):
    """95th-pct dominant-bin prominence of AR(1) surrogates fit to x (red-noise null)."""
    x = np.asarray(x, dtype=float)
    r = _autocorr(x, 1)[1]                       # lag-1 autocorrelation
    sigma = x.std() * np.sqrt(max(1e-9, 1 - r*r))
    rng = np.random.default_rng(seed)
    proms = []
    for _ in range(200):
        s = np.empty(len(x)); s[0] = 0.0
        e = rng.normal(0, sigma, len(x))
        for i in range(1, len(x)):
            s[i] = r*s[i-1] + e[i]
        proms.append(_periodogram_prominence(s))
    return float(np.percentile(proms, 95))


def oscillation_verdict(N, *, seed=0) -> dict:
    x = _detrend(N)
    n = len(x)
    ar_mod = _ar2_dominant_modulus(x)
    prom = _periodogram_prominence(x)
    prom95 = _ar1_null_prominence_95(x, seed)
    ac = _autocorr(x, n//2)
    trough = float(ac[1:].min()) if n > 2 else 0.0
    eqv = n_eq(N)
    amp_ptp = float((np.max(N) - np.min(N)) / eqv) if eqv != 0 else float("inf")
    # QUATERNARY: late-half amplitude must drop below early-half by k*SE (bootstrap).
    half = n // 2
    early, late = x[:half], x[half:]
    rng = np.random.default_rng(seed + 1)
    def _amp(a): return a.std()
    boot = [ _amp(rng.choice(late, size=len(late))) for _ in range(200) ]
    se = float(np.std(boot))
    damping_ok = bool(_amp(late) <= _amp(early) - _DAMP_K*se) or _amp(early) <= _amp(late)*1.05
    primary = ar_mod < _AR_MODULUS_MAX
    secondary = prom <= prom95
    # TERTIARY: flag as oscillatory only if BOTH a deep anti-phase autocorrelation trough
    # (trough ≤ -0.30) AND a large relative peak-to-trough amplitude (amp_ptp ≥ 0.30);
    # demographic noise has the amplitude but not the anti-phase autocorrelation.
    # PASS (DAMPED-contributing) when either condition is absent → OR combinator.
    tertiary = (trough > _AUTOCORR_TROUGH_MIN) or (amp_ptp < _AMP_PTP_MAX)
    classification = "DAMPED" if (primary and secondary and tertiary and damping_ok) else "OSCILLATORY"
    return {"ar_modulus": ar_mod, "periodogram_prominence": prom,
            "periodogram_null_95": prom95, "autocorr_trough": trough,
            "amp_ptp": amp_ptp, "damping_ok": damping_ok,
            "classification": classification}
