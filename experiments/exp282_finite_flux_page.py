"""Exp 282: deterministic verification of a finite-flux entropy-excursion bound.

HYPOTHESIS: for 1<=w<=R, w(0)=w(T)=1, stationary endpoints, max(w)=R>1,
and -b^2 <= w''/w <= a^2, the proposed minimum duration is
2[acosh(y)/a + acos(y/R)/b], y^2=(b^2 R^2+a^2)/(a^2+b^2).
PREDICTION: symbolic identities vanish; independent ODE endpoint/apex/matching
residuals and the integrated-flux identity are <=2e-7 after the declared
normalizations; endpoint-regularized quadrature agrees within 2e-8; every
tested regular bounded-flux profile satisfies T/Tmin >=1-2e-7 under its actual caps.
FALSIFIER: a valid counterexample, symbolic error, or numerical discrepancy
beyond its declared tolerance defeats the corresponding claim. All records,
including failed checks, are retained. These checks are analytic verification,
not a proof of optimality, a blind discovery experiment, or a novelty claim.
The baseline examples with constant continuation are C1 and W^{2,infinity},
piecewise smooth, not globally C-infinity. Smooth sharpness is a separate
mollification argument; these numerical examples do not establish it.

The separate QNEC-gap inverse is an analytic follow-up chosen before its
numerical check. The parent derived it independently and a reviewer supplied
numeric values: its prediction is not blind. It asserts no state realizability.
The equal-cap formula has prior art in Stefanatos, arXiv:1301.5137.

Run from the repository root with uv run --no-sync --python .venv python
experiments/exp282_finite_flux_page.py --smoke. No files are written unless
--output-prefix is supplied; that option creates PREFIX.json and PREFIX.txt
without overwriting earlier evidence. The full batch contains 30 ODE cases.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from numpy.polynomial import Polynomial
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq


HEIGHTS = (.01, .1, .5, 1., 2., 5.)
RATIOS = (.1, .3, 1., 3., 10.)
RESIDUAL_TOL = 2e-7
QUADRATURE_TOL = 2e-8
ODE_RTOL = 2e-12
ODE_ATOL = 2e-13


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _residual(left, right):
    return abs(left - right) / max(1., abs(left), abs(right))


@dataclass(frozen=True)
class Pulse:
    a: float
    b: float
    R: float
    y: float
    t1: float
    t2: float

    @property
    def tau(self):
        return self.t1 + self.t2

    @property
    def duration(self):
        return 2 * self.tau

    @property
    def energy_over_K(self):
        return 2 * self.a**2 * self.t1 - 2 * self.b**2 * self.t2

    def profile(self, t):
        """Analytic (w,w',w''); w'' is one-sided at the cap switches.

        The extension is w=1 outside [0,T]. At the endpoints w'' is assigned
        its exterior value zero; the discontinuous acceleration is defined a.e.
        """
        t = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(t)):
            raise ValueError("profile times must be finite")
        inside = (t > 0) & (t < self.duration)
        s = np.where(inside, np.minimum(t, self.duration - t), 0.)
        accelerating = s <= self.t1
        w = np.where(accelerating, np.cosh(self.a*s),
                     self.R*np.cos(self.b*(self.tau - s)))
        v = np.where(accelerating, self.a*np.sinh(self.a*s),
                     self.b*self.R*np.sin(self.b*(self.tau - s)))
        v *= np.where(t <= self.tau, 1., -1.)
        acc = np.where(accelerating, self.a**2*w, -self.b**2*w)
        result = (np.where(inside, w, 1.), np.where(inside, v, 0.),
                  np.where(inside, acc, 0.))
        return tuple(float(z) for z in result) if t.ndim == 0 else result


def optimal_parameters(a, b, R):
    for value, name in ((a, "a"), (b, "b"), (R, "R")):
        _positive(value, name)
    if R <= 1:
        raise ValueError("R must exceed one")
    # Scaling first avoids squaring both cap frequencies in y's formula.
    scale = math.hypot(a, b)
    y = math.hypot((b / scale)*R, a / scale)
    t1 = math.acosh(max(1., y)) / a
    t2 = math.acos(min(1., y / R)) / b
    if not math.isfinite(t1 + t2) or t1 <= 0 or t2 <= 0:
        raise ValueError("parameters exceed the representable nondegenerate range")
    return Pulse(float(a), float(b), float(R), y, t1, t2)


def envelope_quadrature(pulse):
    """Integrate dw/min(a sqrt(w²-1), b sqrt(R²-w²)) independently.

    The quadratic maps w=1+(y-1)s² and w=R-(R-y)s² remove the endpoint
    singularities algebraically, without evaluating any inverse trig function.
    """
    d1, d2 = pulse.y - 1, pulse.R - pulse.y
    first, err1 = quad(lambda s: 2*math.sqrt(d1) /
                       (pulse.a*math.sqrt(2 + d1*s*s)), 0, 1,
                       epsabs=1e-12, epsrel=1e-12)
    second, err2 = quad(lambda s: 2*math.sqrt(d2) /
                        (pulse.b*math.sqrt(2*pulse.R - d2*s*s)), 0, 1,
                        epsabs=1e-12, epsrel=1e-12)
    return {"duration": 2*(first + second), "rising_acceleration": first,
            "rising_braking": second, "estimated_error": 2*(err1 + err2),
            "substitutions": ["w=1+(y-1)s^2", "w=R-(R-y)s^2"]}


def integrate_switched(pulse):
    """Independent state propagation, never resetting to analytic switch states."""
    start = time.perf_counter()
    state = np.array([1., 0., 0.])
    states, residuals, evaluations = [state.tolist()], [], 0
    elapsed = 0.
    norm = max(1., pulse.R, pulse.a*pulse.R, pulse.b*pulse.R)
    for duration, q in ((pulse.t1, pulse.a**2), (pulse.t2, -pulse.b**2),
                        (pulse.t2, -pulse.b**2), (pulse.t1, pulse.a**2)):
        def rhs(_t, z):
            return (z[1], q*z[0], (z[1]/z[0])**2)

        times = np.linspace(0, duration, 65)
        sol = solve_ivp(rhs, (0., duration), state, method="DOP853",
                        rtol=ODE_RTOL, atol=ODE_ATOL, t_eval=times,
                        max_step=duration/16)
        evaluations += sol.nfev
        if not sol.success:
            return {"success": False, "message": sol.message, "states": states,
                    "nfev": evaluations, "runtime_seconds": time.perf_counter()-start}
        reference = np.array(pulse.profile(elapsed + times))[:2]
        residuals.append(float(np.max(np.abs(sol.y[:2] - reference))) / norm)
        state = sol.y[:, -1]
        states.append(state.tolist())
        elapsed += duration
    return {"success": True, "states": states, "nfev": evaluations,
            "sample_count": 4*65, "segment_profile_residuals": residuals,
            "runtime_seconds": time.perf_counter()-start}


def verify_case(a, b, R):
    pulse = optimal_parameters(a, b, R)
    norm = max(1., R, a*R, b*R)
    matching = max(abs(math.cosh(a*pulse.t1) - R*math.cos(b*pulse.t2)),
                   abs(a*math.sinh(a*pulse.t1) - b*R*math.sin(b*pulse.t2))) / norm
    ode = integrate_switched(pulse)
    quadrature = envelope_quadrature(pulse)
    row = {"a": a, "b": b, "R": R, "log_R": math.log(R), "b_over_a": b/a,
           "parameters": asdict(pulse), "duration": pulse.duration,
           "normalization": norm, "matching_residual": matching, "ode": ode,
           "envelope_quadrature": quadrature, "energy_formula": pulse.energy_over_K}
    checks = {"ode_succeeded": ode["success"], "matching": matching <= RESIDUAL_TOL,
              "envelope_quadrature": _residual(quadrature["duration"], pulse.duration) <= QUADRATURE_TOL}
    if ode["success"]:
        endpoint = max(abs(ode["states"][-1][0]-1), abs(ode["states"][-1][1])) / norm
        apex = max(abs(ode["states"][2][0]-R), abs(ode["states"][2][1])) / norm
        energy = ode["states"][-1][2]
        energy_residual = _residual(energy, pulse.energy_over_K)
        row.update(endpoint_residual=endpoint, apex_residual=apex, energy_ode=energy,
                   energy_residual=energy_residual,
                   envelope_residual=_residual(quadrature["duration"], pulse.duration))
        checks.update(endpoint=endpoint <= RESIDUAL_TOL, apex=apex <= RESIDUAL_TOL,
                      profile=max(ode["segment_profile_residuals"]) <= RESIDUAL_TOL,
                      energy=energy_residual <= RESIDUAL_TOL)
    row["checks"] = checks
    return row


def sinusoidal_profile(x, n, t):
    _positive(x, "x")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    t = np.asarray(t, dtype=float)
    g = np.sin(n*math.pi*t)**2
    gp = n*math.pi*np.sin(2*n*math.pi*t)
    gpp = 2*(n*math.pi)**2*np.cos(2*n*math.pi*t)
    w = np.exp(x*g)
    return w, w*x*gp, w*(x*gpp + x*x*gp*gp)


def sinusoidal_caps(x, n):
    _positive(x, "x")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    factor = (n*math.pi)**2
    return (factor*(2*x if x <= 1 else x*x + 1), 2*factor*x)


def _baseline_record(x, qplus, qminus):
    minimum = optimal_parameters(math.sqrt(qplus), math.sqrt(qminus), math.exp(x)).duration
    ratio = 1 / minimum
    return {"log_R": x, "R": math.exp(x), "duration": 1.,
            "regularity": "C1 and W^{2,infinity} with constant continuation; piecewise smooth, not globally C-infinity",
            "positive_cap": qplus, "negative_cap": qminus, "minimum_duration": minimum,
            "duration_over_minimum": ratio,
            "checks": {"duration_bound": ratio >= 1-RESIDUAL_TOL}}


def sinusoidal_check(x, n):
    row = _baseline_record(x, *sinusoidal_caps(x, n))
    row.update(profile="w=exp(x sin^2(n pi t))", peaks=n,
               cap_method="analytic extrema of q as a concave quadratic in cos(2 n pi t)")
    return row


def asymmetric_shape():
    """g=(729/16)t²(1-t)^4; its unique interior maximum is g(1/3)=1."""
    return (729/16)*Polynomial([0., 0., 1., -4., 6., -4., 1.])


def polynomial_check(x):
    _positive(x, "x")
    g = asymmetric_shape()
    q = x*g.deriv(2) + x*x*g.deriv()*g.deriv()
    roots = q.deriv().roots()
    interior = sorted(float(z.real) for z in roots
                      if abs(z.imag) < 1e-8 and 0 < z.real < 1)
    points = [0., *interior, 1.]
    values = [float(q(t)) for t in points]
    row = _baseline_record(x, max(values), -min(values))
    row.update(profile="w=exp(x (729/16)t^2(1-t)^4)", peak_time=1/3,
               cap_method="all real derivative-polynomial roots plus endpoints, evaluated in float64",
               q_coefficients=q.coef.tolist(), derivative_roots=[[float(z.real), float(z.imag)] for z in roots],
               critical_points=points, critical_flux_values=values)
    return row


def equal_cap_peak(R, duration=1.):
    _positive(duration, "duration")
    if not math.isfinite(R) or R <= 1:
        raise ValueError("R must be finite and exceed one")
    h = .5*math.acosh(R*R) + math.atan(math.sqrt((R*R-1)/(R*R+1)))
    return 4*h*h / duration**2


def equal_cap_check(x):
    R = math.exp(x)
    peak = equal_cap_peak(R)
    sine_peak = max(sinusoidal_caps(x, 1))
    inverse = optimal_parameters(math.sqrt(peak), math.sqrt(peak), R).duration
    return {"log_R": x, "R": R, "optimal_peak": peak, "sinusoidal_peak": sine_peak,
            "sinusoidal_over_optimal": sine_peak/peak, "inverse_duration": inverse,
            "small_height_leading": 16*x,
            "small_height_ratio": peak/(16*x),
            "large_height_asymptotic": 4*(x + .5*math.log(2) + math.pi/4)**2,
            "large_height_ratio": peak/(4*(x + .5*math.log(2) + math.pi/4)**2),
            "checks": {"inverse": _residual(inverse, 1) <= QUADRATURE_TOL,
                       "sinusoidal_comparison": sine_peak/peak >= 1-RESIDUAL_TOL}}


def _poly_add(*polys):
    result = {}
    for poly in polys:
        for term, coefficient in poly.items():
            result[term] = result.get(term, Fraction(0)) + coefficient
    return {term: coefficient for term, coefficient in result.items() if coefficient}


def _poly_scale(poly, coefficient):
    return {term: Fraction(coefficient)*value for term, value in poly.items() if coefficient*value}


def _poly_mul(left, right):
    return _poly_add(*({tuple(sorted(a+b)): x*y} for a, x in left.items() for b, y in right.items()))


def symbolic_checks():
    """Exact symbolic polynomial checks; no optional SymPy installation required.

    Each dictionary is a polynomial in independent formal variables with rational
    coefficients. Positive denominators are cleared before comparing coefficients.
    """
    symbol = lambda name: {(name,): Fraction(1)}
    one = {(): Fraction(1)}
    A, B, U, rho, sigma = (symbol(s) for s in ("A", "B", "U", "rho", "sigma"))
    add, mul, scale = _poly_add, _poly_mul, _poly_scale
    rho2 = mul(rho, rho)
    # L=(log p')'=-2rho; L'=-2(sigma-rho²); {p,u}=L'-(1/2)L².
    Lprime = add(scale(sigma, -2), scale(rho2, 2))
    schwarzian = add(Lprime, scale(rho2, -2))
    D, N = add(A, B), add(mul(B, U), A)
    rising_v2_numerator = mul(A, add(N, scale(D, -1)))
    braking_v2_numerator = mul(B, add(mul(U, D), scale(N, -1)))
    residuals = {
        "schwarzian_equals_minus_two_wpp_over_w": add(schwarzian, scale(sigma, 2)),
        "minus_c_over_24pi_schwarzian_equals_K_wpp_over_w": add(scale(schwarzian, Fraction(-1, 2)), scale(sigma, -1)),
        "switch_velocities_squared_match": add(rising_v2_numerator, scale(braking_v2_numerator, -1)),
        "acceleration_displacement_identity": add(N, scale(D, -1), scale(mul(B, add(U, scale(one, -1))), -1)),
        "braking_displacement_identity": add(mul(U, D), scale(N, -1), scale(mul(A, add(U, scale(one, -1))), -1)),
        "equal_cap_cosh_double_angle": add(add(U, one), scale(one, -1), scale(U, -1)),
        "equal_cap_braking_tangent_squared": add(scale(U, 2), scale(add(U, one), -1), scale(add(U, scale(one, -1)), -1)),
    }
    return {"backend": "exact rational-coefficient formal polynomial algebra (stdlib fractions)",
            "definitions": "A=a^2, B=b^2, U=R^2, y^2=(BU+A)/(A+B), rho=w'/w, sigma=w''/w",
            "derivation": "p'=w^-2 => L=-2rho; rho'=sigma-rho^2; {p,u}=L'-L^2/2=-2sigma; F=-c{p,u}/(24pi)=c sigma/(12pi)",
            "checks": {name: not value for name, value in residuals.items()},
            "residuals": {name: "0" if not value else str(value) for name, value in residuals.items()}}


def gap_inverse_check():
    """Disclosed nonblind follow-up; invert a duration bound, not a quantum state."""
    a, b0, R, target = 1., 1., math.e, 3.5
    limit = 2*math.acosh(R)/a
    initial = optimal_parameters(a, b0, R).duration
    objective = lambda b: optimal_parameters(a, b, R).duration - target
    high = 2*b0
    while objective(high) > 0 and high < 2**20*b0:
        high *= 2
    bcrit = brentq(objective, b0, high, xtol=1e-13, rtol=1e-14)
    pulse = optimal_parameters(a, bcrit, R)
    derivative = -2*math.acos(pulse.y/R)/bcrit**2
    step = 1e-5*bcrit
    numeric = (objective(bcrit+step) - objective(bcrit-step))/(2*step)
    root_residual = abs(objective(bcrit))
    limit_values = [{"b": b, "duration": optimal_parameters(a, b, R).duration,
                     "excess_above_limit": optimal_parameters(a, b, R).duration-limit}
                    for b in (10., 100., 1000.)]
    coefficient = a*math.sqrt(1-R**-2)
    asymptotic = []
    for b in (10., 20., 50.):
        scaled = b*b*(optimal_parameters(a, b, R).duration-limit)
        asymptotic.append({"b": b, "b_squared_duration_excess": scaled,
                           "predicted_coefficient": coefficient,
                           "difference": scaled-coefficient})
    return {"provenance": "analytic follow-up chosen before numerical execution; independently derived by parent, reviewer supplied numbers; not blind",
            "scope": "inverse cap bound only; no state-realizability claim",
            "a": a, "b0": b0, "R": R, "target_duration": target,
            "initial_duration": initial, "large_b_limit": limit, "limit_samples": limit_values,
            "inverse_asymptotic_diagnostic": asymptotic,
            "asymptotic_status": "diagnostic only; no post-hoc acceptance threshold",
            "root_bracket": [b0, high], "bcrit": bcrit, "D_over_K": bcrit*bcrit-b0*b0,
            "root_residual": root_residual, "derivative_analytic": derivative,
            "derivative_finite_difference": numeric, "difference_step": step,
            "derivative_residual": _residual(derivative, numeric),
            "checks": {"preconditions": limit < target < initial, "larger_required_cap": bcrit > b0,
                       "root": root_residual < 1e-9,
                       "derivative": _residual(derivative, numeric) <= RESIDUAL_TOL,
                       "one_sided_limit": all(item["excess_above_limit"] > 0 for item in limit_values)
                       and all(left["duration"] > right["duration"] for left, right in zip(limit_values, limit_values[1:]))}}


def run_verification(smoke=False):
    start = time.perf_counter()
    heights = (.5,) if smoke else HEIGHTS
    cases = [(1., ratio, math.exp(x)) for x in heights for ratio in ((.3,) if smoke else RATIOS)]
    workers = 1 if smoke else 4
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ode_rows = list(pool.map(lambda args: verify_case(*args), cases))
    sine_rows = [sinusoidal_check(x, n) for x in heights for n in (1, 2, 3)]
    polynomial_rows = [polynomial_check(x) for x in heights]
    equal_rows = [equal_cap_check(x) for x in heights]
    symbolic, gap = symbolic_checks(), gap_inverse_check()
    checks = {"symbolic": all(symbolic["checks"].values()),
              "ode_and_quadrature": all(all(r["checks"].values()) for r in ode_rows),
              "sinusoidal_baselines": all(all(r["checks"].values()) for r in sine_rows),
              "polynomial_baselines": all(all(r["checks"].values()) for r in polynomial_rows),
              "equal_cap": all(all(r["checks"].values()) for r in equal_rows),
              "qnec_gap_followup": all(gap["checks"].values())}
    root = Path(__file__).resolve().parents[1]
    paths = ("experiments/exp282_finite_flux_page.py", "tests/test_exp282_finite_flux.py",
             "docs/research/hep-th-finite-flux/analysis-plan.md")
    hashes = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    try:
        sympy_version = importlib.metadata.version("sympy")
    except importlib.metadata.PackageNotFoundError:
        sympy_version = None
    return {"experiment": 282, "kind": "deterministic analytic verification", "smoke": smoke,
            "scope": "conformal-vacuum calculation; no proof of novelty, general QNEC states, gravitational islands, or four-dimensional backreaction",
            "predeclaration": __doc__, "parameters": {"heights": list(heights), "ratios": [.3] if smoke else list(RATIOS),
                "a": 1., "ode_rtol": ODE_RTOL, "ode_atol": ODE_ATOL,
                "residual_tolerance": RESIDUAL_TOL, "quadrature_tolerance": QUADRATURE_TOL,
                "normalization": "position/velocity: max(1,R,aR,bR); energy/time: max(1,abs(left),abs(right))",
                "ode_cases": len(cases), "executor": "ThreadPoolExecutor", "max_workers": workers},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sympy_installed": sympy_version},
            "source_sha256": hashes, "symbolic": symbolic, "ode_cases": ode_rows,
            "sinusoidal_baselines": sine_rows, "polynomial_baselines": polynomial_rows,
            "equal_cap": equal_rows, "qnec_gap_followup": gap,
            "runtime_seconds": time.perf_counter()-start,
            "projected_serial_30_ode_seconds": sum(row["ode"]["runtime_seconds"] for row in ode_rows)*30/len(ode_rows),
            "checks": checks, "all_checks_pass": all(checks.values())}


def render_text(result):
    lines = ["Exp 282 — deterministic analytic verification", result["scope"],
             "PASS is a statement about listed checks, not a publication or novelty verdict."]
    for key, value in result.items():
        lines.append(f"\n{key}:")
        if isinstance(value, list):
            lines.extend(json.dumps(row, sort_keys=True, allow_nan=False) for row in value)
        else:
            lines.append(json.dumps(value, sort_keys=True, allow_nan=False))
    return "\n".join(lines) + "\n"


def write_outputs(prefix, result):
    json_path, text_path = Path(f"{prefix}.json"), Path(f"{prefix}.txt")
    if json_path.exists() or text_path.exists():
        raise FileExistsError("refusing to overwrite an existing verification output")
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report = render_text(result)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x") as handle:
        handle.write(payload)
    with text_path.open("x") as handle:
        handle.write(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="one independent ODE case, plus small analytic checks")
    parser.add_argument("--output-prefix", type=Path, help="write PREFIX.json and PREFIX.txt; refuse overwrites")
    args = parser.parse_args(argv)
    result = run_verification(smoke=args.smoke)
    if args.output_prefix is not None:
        write_outputs(args.output_prefix, result)
    print(render_text(result), end="")
    return 0 if result["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
