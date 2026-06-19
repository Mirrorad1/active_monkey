import numpy as np
from ecology.evolvability import stability as S

def test_n_eq_is_median():
    assert S.n_eq(np.array([10, 12, 11, 100])) == 11.5

def test_level_cv_flat_is_low():
    N = 150 + np.zeros(500)
    assert S.level_cv(N) == 0.0

def test_level_cv_cycle_is_high():
    t = np.arange(1000)
    N = 150 + 40*np.sin(2*np.pi*t/50)
    assert S.level_cv(N) > 0.15

def test_drift_slope_flat_is_zero():
    N = 150 + np.zeros(500)
    assert abs(S.drift_slope(N, 150.0)) < 1e-9

def test_drift_slope_trending():
    N = np.linspace(100, 200, 1000)          # +100 over window, n_eq~150 -> ~0.67
    assert S.drift_slope(N, 150.0) > 0.5

def test_return_map_slope_damped_below_one():
    # AR(1) with phi=0.5 -> return-map slope ~0.5
    rng = np.random.default_rng(0); N = np.zeros(2000); N[0] = 150
    for i in range(1, 2000):
        N[i] = 150 + 0.5*(N[i-1]-150) + rng.normal(0, 1)
    assert abs(S.return_map_slope(N)) < 1.0

def test_seed_agreement():
    assert S.seed_agreement([100, 110, 120]) == (120-100)/110


def _damped_series(seed=0, n=1600, phi=0.5, n0=150.0, sigma=2.0):
    rng = np.random.default_rng(seed); N = np.empty(n); N[0] = n0
    for i in range(1, n):
        N[i] = n0 + phi*(N[i-1]-n0) + rng.normal(0, sigma)
    return N

def _limit_cycle(seed=0, n=1600, n0=150.0, amp=25.0, period=40.0, sigma=3.0):
    rng = np.random.default_rng(seed); t = np.arange(n)
    return n0 + amp*np.sin(2*np.pi*t/period) + rng.normal(0, sigma, n)

def test_detector_passes_damped():
    v = S.oscillation_verdict(_damped_series())
    assert v["classification"] == "DAMPED"

def test_detector_flags_limit_cycle():
    v = S.oscillation_verdict(_limit_cycle())
    assert v["classification"] == "OSCILLATORY"

def test_detector_does_not_false_fail_flat_noise():
    # truly-flat + demographic noise must NOT be misread as a sustained cycle.
    rng = np.random.default_rng(1); N = 150 + rng.normal(0, np.sqrt(150), 1600)
    assert S.oscillation_verdict(N)["classification"] == "DAMPED"

def test_detector_catches_low_amplitude_cycle():
    # the failure mode the candidate detector missed: a noisy ~12-24% sustained cycle.
    v = S.oscillation_verdict(_limit_cycle(amp=22.0, sigma=6.0))
    assert v["classification"] == "OSCILLATORY"


def test_certify_run_passes_clean_damped():
    N = _damped_series(seed=2, n0=150.0)
    run = dict(N=N, births_per_step=1.5, crowding_per_step=1.4, p_hazard_mean=0.04,
               exploded=False, availability_mean=0.4, boundary_frac=0.1, interbump_flux=0.3,
               n_eq=float(np.median(N)))
    res = S.certify_run(run, params=dict(hmax=0.04, Kc=60.0, theta=1.0))
    assert res["passes"] is True

def test_certify_run_fails_extinct_floor():
    N = 8 + np.zeros(1600)                       # below the persistence floor of 30
    run = dict(N=N, births_per_step=0.0, crowding_per_step=0.0, p_hazard_mean=0.0,
               exploded=False, availability_mean=0.9, boundary_frac=0.1, interbump_flux=0.0,
               n_eq=8.0)
    assert S.certify_run(run, params=dict(hmax=0.04, Kc=60.0, theta=1.0))["passes"] is False

def test_non_degeneracy_flags_static_mosaic():
    run = dict(N=150+np.zeros(1600), interbump_flux=0.0, availability_mean=0.4,
               boundary_frac=0.1, exploded=False, n_eq=150.0)
    ok, reasons = S.non_degeneracy_ok(run)
    assert ok is False and any("mosaic" in r for r in reasons)

def test_band_verdict_requires_overlap():
    # stable at slow {0.25,0.5,0.75} but expressed only at fast {2,3,4} -> NO-GO (no overlap)
    cell = {0.25: True, 0.5: True, 0.75: True, 1.0: False, 2.0: False, 3.0: False}
    v = S.band_verdict(cell, expressed_speeds={2.0, 3.0, 4.0})
    assert v["verdict"] == "NO-GO"
    cell2 = {1.0: True, 1.5: True, 2.0: True}
    v2 = S.band_verdict(cell2, expressed_speeds={1.0, 1.5, 2.0})
    assert v2["verdict"] == "GO" and v2["band"] == [1.0, 1.5, 2.0]


def test_marginal_brake_agrees_with_density_mortality_p():
    """_marginal_brake's p-term must agree with ecology.engine._density_mortality_p."""
    from ecology.engine import _density_mortality_p
    params = dict(hmax=0.05, Kc=80.0, theta=2.0)
    n_eq_val = 60.0
    # compute p via the canonical engine helper
    p_engine = _density_mortality_p(n_eq_val, params["hmax"], params["Kc"], params["theta"])
    # compute p as used internally by _marginal_brake (h * clamp((N/Kc)^th, 0, 1))
    h, Kc, th = params["hmax"], params["Kc"], params["theta"]
    f = min(1.0, max(0.0, (n_eq_val / Kc) ** th))
    p_local = h * f
    assert abs(p_engine - p_local) < 1e-12, f"p_engine={p_engine} p_local={p_local}"
    # also verify _marginal_brake's OUTPUT matches a reference built from _density_mortality_p
    brake_val = S._marginal_brake(n_eq_val, params)
    pprime = h * th * (n_eq_val / Kc) ** (th - 1) / Kc
    expected = abs(p_engine + n_eq_val * pprime)
    assert abs(brake_val - expected) < 1e-12, f"brake_val={brake_val} expected={expected}"
