"""Behavioral guards for the finite-flux analytic verification instrument."""

import math
from pathlib import Path

import numpy as np
import pytest


def _module():
    source = Path(__file__).resolve().parents[1] / "experiments/exp282_finite_flux_page.py"
    assert source.exists(), "the bounded finite-flux verification runner is missing"
    # Import through the package so dataclasses and subprocesses use the same module.
    from experiments import exp282_finite_flux_page

    return exp282_finite_flux_page


@pytest.mark.parametrize("a,b,R", [
    (0, 1, 2), (1, -1, 2), (1, 1, 1), (1, 1, .9),
    (math.nan, 1, 2), (1, math.inf, 2), (1, 1, math.inf),
])
def test_rejects_invalid_caps_and_height(a, b, R):
    with pytest.raises(ValueError):
        _module().optimal_parameters(a, b, R)


def test_profile_stationary_endpoints_and_c1_switches():
    pulse = _module().optimal_parameters(1, 2, 2)
    w, v, acc = pulse.profile(np.array([-1, 0, pulse.tau, pulse.duration,
                                       pulse.duration + 1]))
    np.testing.assert_allclose(w, [1, 1, 2, 1, 1], atol=1e-13)
    np.testing.assert_allclose(v, 0, atol=1e-13)
    np.testing.assert_allclose(acc[[0, -1]], 0)
    for switch in (pulse.t1, pulse.tau, pulse.duration - pulse.t1):
        left = np.array(pulse.profile(switch - 1e-9))[:2]
        right = np.array(pulse.profile(switch + 1e-9))[:2]
        np.testing.assert_allclose(left, right, atol=2e-8)
    # Independent matching fixture for a=1, b=2, R=2.
    w_switch, v_switch, _ = pulse.profile(pulse.t1)
    assert w_switch == pytest.approx(math.sqrt(3.4), abs=1e-13)
    assert v_switch == pytest.approx(math.sqrt(2.4), abs=1e-13)


def test_equal_cap_inverse_and_time_scaling():
    runner = _module()
    pulse = runner.optimal_parameters(1, 1, math.sqrt(2))
    assert pulse.duration == pytest.approx(2.3641554481214142, abs=1e-13)
    for R in (math.exp(.1), math.exp(2)):
        cap = runner.equal_cap_peak(R)
        assert runner.optimal_parameters(math.sqrt(cap), math.sqrt(cap), R).duration == pytest.approx(1)
    pulse = runner.optimal_parameters(.8, 2.1, 3)
    scaled = runner.optimal_parameters(2.4, 6.3, 3)
    assert scaled.duration == pytest.approx(pulse.duration / 3)
    assert scaled.energy_over_K == pytest.approx(3 * pulse.energy_over_K)


def test_multi_turnover_baseline_uses_actual_caps():
    runner = _module()
    t = np.array([0, .25, .5, .75, 1])
    w, v, _ = runner.sinusoidal_profile(2, 2, t)
    np.testing.assert_allclose(w, [1, math.exp(2), 1, math.exp(2), 1], atol=1e-12)
    np.testing.assert_allclose(v, 0, atol=1e-12)
    assert runner.sinusoidal_caps(2, 2) == pytest.approx((20 * math.pi**2, 16 * math.pi**2))
    assert runner.sinusoidal_caps(.5, 3) == pytest.approx((9 * math.pi**2, 9 * math.pi**2))
    row = runner.sinusoidal_check(2, 2)
    assert row["peaks"] == 2
    assert row["duration_over_minimum"] >= 1 - 2e-7


def test_asymmetric_polynomial_caps_cover_interior_extrema():
    runner = _module()
    for x in (.1, 2, 5):
        row = runner.polynomial_check(x)
        g = runner.asymmetric_shape()
        assert g(1 / 3) == pytest.approx(1, abs=1e-13)
        assert g(0) == pytest.approx(0)
        assert g(1) == pytest.approx(0, abs=1e-13)
        grid = np.linspace(0, 1, 10001)
        q = x * g.deriv(2)(grid) + x * x * g.deriv()(grid)**2
        assert max(q) <= row["positive_cap"] + 1e-8
        assert min(q) >= -row["negative_cap"] - 1e-8
        assert row["duration_over_minimum"] >= 1 - 2e-7
        assert row["critical_points"]


def test_switched_ode_is_an_independent_endpoint_and_energy_witness():
    row = _module().verify_case(1.4, .7, 3)
    assert row["ode"]["success"]
    assert row["ode"]["nfev"] > 0
    np.testing.assert_allclose(row["ode"]["states"][2][:2], [3, 0], atol=2e-8)
    np.testing.assert_allclose(row["ode"]["states"][-1][:2], [1, 0], atol=2e-8)
    assert row["energy_ode"] == pytest.approx(row["energy_formula"], rel=2e-7, abs=2e-7)
    assert all(row["checks"].values())


def test_profile_obeys_both_speed_envelopes_and_flux_caps():
    runner = _module()
    for a, b, R in ((1, .1, 1.01), (1, 3, 10), (2, 2, 2)):
        pulse = runner.optimal_parameters(a, b, R)
        w, v, acc = pulse.profile(np.linspace(0, pulse.duration, 1001))
        assert min(w) >= 1 - 1e-12
        assert max(w) <= R + 1e-12
        assert np.all(v*v <= a*a*(w*w - 1) + 1e-9)
        assert np.all(v*v <= b*b*(R*R - w*w) + 1e-9)
        assert np.all(acc / w <= a*a + 1e-12)
        assert np.all(acc / w >= -b*b - 1e-12)
        quadrature = runner.envelope_quadrature(pulse)
        assert quadrature["duration"] == pytest.approx(pulse.duration, rel=2e-8, abs=2e-8)


def test_exact_symbolic_identities_have_zero_residuals():
    result = _module().symbolic_checks()
    assert result["checks"]
    assert all(result["checks"].values())
    assert all(value == "0" for value in result["residuals"].values())


def test_gap_inverse_has_required_domain_root_and_derivative():
    row = _module().gap_inverse_check()
    assert row["large_b_limit"] < 3.5 < row["initial_duration"]
    assert row["bcrit"] > 1
    assert row["root_residual"] < 1e-9
    assert row["derivative_analytic"] < 0
    assert row["derivative_finite_difference"] == pytest.approx(row["derivative_analytic"], abs=2e-7)
    assert all(row["checks"].values())
    # Diagnostic only: no scientific acceptance threshold was declared for this limit.
    assert [sample["b"] for sample in row["inverse_asymptotic_diagnostic"]] == [10, 20, 50]


def test_raw_outputs_preserve_existing_evidence(tmp_path):
    runner = _module()
    prefix = tmp_path / "verification"
    result = {"scope": "fixture", "checks": {"fixture": False}, "all_checks_pass": False}
    runner.write_outputs(prefix, result)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["verification.json", "verification.txt"]
    before = (tmp_path / "verification.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.write_outputs(prefix, result | {"all_checks_pass": True})
    assert (tmp_path / "verification.json").read_bytes() == before

