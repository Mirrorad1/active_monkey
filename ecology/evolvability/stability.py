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
