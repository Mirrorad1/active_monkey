"""tests/test_predation.py — Exp 248 Rung 0a: Genotype.role + enable_predation gate.

Tests (all must pass after implementation; the first two FAIL before it):
  1. role defaults to "prey" and is the LAST field.
  2. role is never mutated (copied verbatim from parent, no rng draw).
  3. OFF self-consistency: with enable_predation=False (default) the per-role
     branch is never entered and the events_hash is self-consistent between
     two identical runs.
"""
import dataclasses as D
import numpy as np
import pytest

from ecology.genotype import Genotype, mutate, founder


def test_role_defaults_prey_and_is_last_field():
    g = founder()
    assert g.role == "prey"
    assert list(D.fields(Genotype))[-1].name == "role"


def test_role_never_mutates():
    rng = np.random.default_rng(0)
    g = D.replace(founder(), role="predator")
    child = mutate(g, rng, 0.2, mutate_continuous_locomotion=True)
    assert child.role == "predator"  # copied verbatim, never perturbed


def test_off_self_consistency():
    """With enable_predation=False (default), two identical runs produce the same events_hash."""
    import dataclasses as D
    from ecology.engine import Ecology
    from ecology.scenarios import SCENARIOS
    cfg = D.replace(SCENARIOS["balanced"], horizon=20)
    assert cfg.enable_predation is False
    h1 = Ecology(cfg, seed=42).run()["events_hash"]
    h2 = Ecology(cfg, seed=42).run()["events_hash"]
    assert h1 == h2, f"OFF path not deterministic: {h1} != {h2}"
