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
