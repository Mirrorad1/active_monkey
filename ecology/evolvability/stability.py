"""Exp 243 — trait-agnostic stability-certification instrument.

Consumes an analysis-window N(t) series (+ telemetry) and applies the predeclared,
FROZEN stability gates and the hardened oscillation detector from
docs/superpowers/specs/2026-06-19-continuous-substrate-stabilization-design.md.
Pure numpy (no scipy.signal) for determinism.
"""
from __future__ import annotations
import numpy as np
from ecology.engine import _density_mortality_p


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
    if np.ptp(x) == 0:      # constant series: one-step return map slope is 0 by convention
        return 0.0
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
    damping_ok = (_amp(late) <= _amp(early) - _DAMP_K*se) or (_amp(early) <= _amp(late)*1.05)
    primary = ar_mod < _AR_MODULUS_MAX
    secondary = prom <= prom95
    # TERTIARY (DAMPED-contributing witness): tertiary=True keeps the series DAMPED.
    # It is True when the series LACKS at least one cycle signature — i.e. the autocorr
    # trough is shallow (> -0.30) OR the peak-to-trough amplitude is small (< 0.30).
    # A sustained cycle has BOTH a deep anti-phase trough (<= -0.30) AND large amplitude
    # (>= 0.30); demographic noise has large amplitude but a shallow trough, so it passes
    # via the trough clause -> DAMPED. Only when BOTH cycle signatures hold is tertiary
    # False -> OSCILLATORY.
    tertiary = (trough > _AUTOCORR_TROUGH_MIN) or (amp_ptp < _AMP_PTP_MAX)
    classification = "DAMPED" if (primary and secondary and tertiary and damping_ok) else "OSCILLATORY"
    return {"ar_modulus": ar_mod, "periodogram_prominence": prom,
            "periodogram_null_95": prom95, "autocorr_trough": trough,
            "amp_ptp": amp_ptp, "damping_ok": damping_ok,
            "classification": classification}


# --- per-run gates, cell certification, non-degeneracy, band verdict (FROZEN thresholds) ---
_PERSIST_FLOOR = 30
_LEVEL_CV_MED_MAX = 0.15
_LEVEL_CV_SEED_MAX = 0.25
_DRIFT_MAX = 0.10
_RETURN_SLOPE_MAX = 1.0
_MARGINAL_BRAKE_MAX = 0.5
_SEED_AGREE_MAX = 0.25
_AVAIL_LO, _AVAIL_HI = 0.05, 0.85
_BOUNDARY_MAX = 0.5


def _marginal_brake(n_eq_val, params):
    """Compute the crowding-brake strength at n_eq.

    Uses _density_mortality_p (canonical engine helper) for the hazard p,
    then computes the derivative p' analytically for the theta-logistic.
    Returns |p + n_eq * p'|.
    """
    h, Kc, th = params["hmax"], params["Kc"], params["theta"]
    p = _density_mortality_p(n_eq_val, h, Kc, th)
    pprime = h * th * (n_eq_val / Kc) ** (th - 1) / Kc if n_eq_val > 0 else 0.0
    return abs(p + n_eq_val * pprime)


def non_degeneracy_ok(run):
    """Return (bool, reasons) — True iff run passes all non-degeneracy checks."""
    reasons = []
    if run["exploded"]:
        reasons.append("exploded")
    if not (_PERSIST_FLOOR <= run["n_eq"]):
        reasons.append("below_floor")
    if not (_AVAIL_LO < run["availability_mean"] < _AVAIL_HI):
        reasons.append("degenerate_depletion(availability=%.3f)" % run["availability_mean"])
    if run["boundary_frac"] >= _BOUNDARY_MAX:
        reasons.append("wall_clamped")
    if run["interbump_flux"] <= 0.0:
        reasons.append("static_spatial_mosaic")
    return (len(reasons) == 0, reasons)


def certify_run(run, params):
    """Apply all per-run stability gates. Returns dict with 'passes' bool and per-check detail."""
    N = np.asarray(run["N"], dtype=float)
    eqv = run.get("n_eq", n_eq(N))
    checks = {}
    checks["persistence"] = persistence(N) >= _PERSIST_FLOOR and not run["exploded"]
    checks["level_cv"] = level_cv(N) <= _LEVEL_CV_SEED_MAX
    checks["drift"] = drift_slope(N, eqv) <= _DRIFT_MAX
    checks["return_map"] = (abs(return_map_slope(N)) < _RETURN_SLOPE_MAX
                            and _marginal_brake(eqv, params) < _MARGINAL_BRAKE_MAX)
    bp, cp = run["births_per_step"], run["crowding_per_step"]
    checks["birth_pulse"] = (cp == 0.0 and bp == 0.0) or (cp > 0 and 0.5 <= bp / cp <= 2.0)
    checks["oscillation"] = oscillation_verdict(N)["classification"] == "DAMPED"
    nd_ok, nd_reasons = non_degeneracy_ok(run)
    checks["non_degenerate"] = nd_ok
    passes = all(checks.values())
    return {"passes": passes, "checks": checks, "nd_reasons": nd_reasons}


def certify_cell(runs, params, seeds):
    """Certify a cell across multiple seed runs.

    Passes iff >=ceil(0.75*seeds) runs pass AND seed_agreement <= _SEED_AGREE_MAX.
    """
    per = [certify_run(r, params) for r in runs]
    n_pass = sum(1 for p in per if p["passes"])
    need = int(np.ceil(0.75 * seeds))
    n_eqs = [r.get("n_eq", n_eq(r["N"])) for r in runs]
    agree = seed_agreement(n_eqs) <= _SEED_AGREE_MAX
    return {"certified": n_pass >= need and agree, "n_pass": n_pass, "need": need,
            "seed_agreement_ok": agree, "per_seed": per}


def band_verdict(cell_stable, expressed_speeds):
    """GO iff a contiguous run of >=3 stable speeds overlaps expressed_speeds.

    cell_stable: {speed: bool}
    expressed_speeds: set of speeds observed in the expressed population.
    """
    speeds = sorted(cell_stable)
    best = []
    run = []
    for s in speeds:
        if cell_stable[s]:
            run.append(s)
        else:
            if len(run) > len(best):
                best = run
            run = []
    if len(run) > len(best):
        best = run
    overlap = [s for s in best if s in expressed_speeds]
    go = len(best) >= 3 and len(overlap) >= 1
    return {"verdict": "GO" if go else "NO-GO", "band": best, "overlap": overlap}
