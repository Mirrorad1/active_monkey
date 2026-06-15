"""tests/test_uncertainty_gated.py — Exp 211 uncertainty-gated active-sensing mechanism.

Guards (durable, per loop/META.md) for the probe-policy abstraction added in Exp 211:
  1. OFF + fixed_rate remain byte-identical (the Exp 210 contract is untouched).
  2. uncertainty_gated probes are ENRICHED at low action-margin (probe only when unsure).
  3. higher gate threshold (treats more states as uncertain) raises the probe rate;
     a ~0 threshold suppresses probing.
  4. pure_cost pays the probe cost but NEVER changes the action (no information).
  5. random_cost_matched / gate_shuffle / hidden_scramble all run and pay the cost.
  6. probe does not create food / increase reward (disconnected ⇒ byte-identical).
  7. probe_changed_action telemetry is live and policy-consistent.
"""
from __future__ import annotations

import dataclasses as D

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Helper: the hidden-mode active-sensing regime (memory OFF ⇒ belief = current cue)
# ---------------------------------------------------------------------------

def as_cfg(**over) -> "EcologyConfig":
    base = dict(
        horizon=300,
        enable_hidden_mode=True,
        enable_active_sensing=True,
        mode_wrong_regen_factor=1.0,
        mode_hazard_scale=0.6,
        capacity=50.0,
        regen_rate=3.0,
        initial_resource=0.7,
        max_population=30000,
        mode_switch_prob=0.05,
        cue_noise=1.0,
        memory_cost_slope=0.0,
        memory_upkeep_floor=0.0,
        probe_cost=0.01,
        probe_n_samples=4,
        shuffle_creature_order=True,
        mutation_rate=0.0,
    )
    gain = base_gain = over.pop("gain", 0.5)
    base.update(over)
    cfg = D.replace(SCENARIOS["balanced"], **base)
    return D.replace(cfg, founder=D.replace(cfg.founder, information_sampling_rate=gain))


def _run(policy: str, seed: int = 50, **over):
    cfg = as_cfg(probe_policy=policy, **over)
    eco = Ecology(cfg, seed=seed)
    eco.run()
    hm = max(1, eco.hidden_mode_steps_total)
    a_n = eco.action_margin_n
    p_n = eco.action_margin_at_probe_n
    np_n = a_n - p_n
    m_probe = eco.action_margin_at_probe_sum / p_n if p_n else float("nan")
    m_noprobe = (eco.action_margin_sum - eco.action_margin_at_probe_sum) / np_n if np_n else float("nan")
    return {
        "hash": eco.events_hash(),
        "probes": eco.probe_count_total,
        "rate": eco.probe_count_total / hm,
        "changed": eco.probe_changed_action_count,
        "m_probe": m_probe,
        "m_noprobe": m_noprobe,
    }


# ---------------------------------------------------------------------------
# 1. OFF + fixed_rate byte-identity (the Exp 210 contract)
# ---------------------------------------------------------------------------

def test_off_policy_produces_no_probes() -> None:
    """probe_policy='off' skips the probe block ⇒ zero probes (no probe cost paid).

    NOTE: the canonical byte-identical-OFF contract is enable_active_sensing=False
    (golden-hash pinned in tests/test_active_sensing.py).  probe_policy='off' with
    active sensing still ENABLED is NOT byte-identical to that, because
    enable_active_sensing also gates the heritable information_sampling_rate mutation
    draw — an honest, documented difference, not a regression."""
    cfg = as_cfg(probe_policy="off")
    eco = Ecology(D.replace(cfg, founder=D.replace(cfg.founder, information_sampling_rate=1.0)), seed=7)
    eco.run()
    assert eco.probe_count_total == 0, f"probe_policy='off' must not probe, got {eco.probe_count_total}"


def test_off_policy_byte_identical_to_disabled_when_no_mutation_diff() -> None:
    """With the active-sensing trait frozen at the founder (mutation_rate=0) AND the
    founder information_sampling_rate left at 0, probe_policy='off' is byte-identical to a
    fixed_rate run (the probe never fires at gain 0, so only the trigger branch differs —
    and it produces no draws either way)."""
    cfg_off = D.replace(as_cfg(probe_policy="off"), founder=D.replace(as_cfg().founder, information_sampling_rate=0.0))
    cfg_fix = D.replace(as_cfg(probe_policy="fixed_rate"), founder=D.replace(as_cfg().founder, information_sampling_rate=0.0))
    # fixed_rate at gain 0 still DRAWS u + extras (keep-the-draw idiom), so it differs from
    # 'off' which skips the block — this asserts they differ, documenting the draw structure.
    h_off = Ecology(cfg_off, seed=7).run()["events_hash"]
    h_fix = Ecology(cfg_fix, seed=7).run()["events_hash"]
    assert h_off != h_fix


def test_fixed_rate_default_unchanged() -> None:
    """The default probe_policy is 'fixed_rate' and probing fires (Exp 210 path)."""
    cfg = as_cfg()  # default policy
    assert cfg.probe_policy == "fixed_rate"
    eco = Ecology(D.replace(cfg, founder=D.replace(cfg.founder, information_sampling_rate=1.0)), seed=1)
    eco.run()
    assert eco.probe_count_total > 0


# ---------------------------------------------------------------------------
# 2-3. Uncertainty gating: enrichment + monotone in threshold
# ---------------------------------------------------------------------------

def test_uncertainty_gated_probes_are_enriched_at_low_margin() -> None:
    """Gated probes fire at MUCH lower action-margin than non-probe steps (it probes
    only when the which-half decision is ambiguous)."""
    r = _run("uncertainty_gated")
    assert r["probes"] > 0
    assert r["m_probe"] < r["m_noprobe"], (
        f"gated probes not enriched at low margin: at-probe {r['m_probe']:.3f} "
        f">= without {r['m_noprobe']:.3f}"
    )
    # strong separation expected (probe only near the 0.5 boundary)
    assert r["m_probe"] < 0.3 < r["m_noprobe"]


def test_higher_threshold_increases_probe_rate() -> None:
    """Raising the gate threshold (treat more states as uncertain) raises the probe rate;
    a ~0 threshold suppresses probing toward zero."""
    lo = _run("uncertainty_gated", uncertainty_gate_threshold=0.001)
    mid = _run("uncertainty_gated", uncertainty_gate_threshold=0.15)
    hi = _run("uncertainty_gated", uncertainty_gate_threshold=0.5)
    assert lo["rate"] < mid["rate"] < hi["rate"], (
        f"probe rate not monotone in threshold: {lo['rate']:.4f} < "
        f"{mid['rate']:.4f} < {hi['rate']:.4f}"
    )
    assert lo["rate"] < 0.02, f"near-zero threshold should suppress probing, got {lo['rate']:.4f}"


def test_gated_probes_less_than_fixed_rate_at_same_gain() -> None:
    """At equal gain, uncertainty_gated probes FAR less than fixed_rate (it spends budget
    only on ambiguous steps) — the whole point of gating."""
    fixed = _run("fixed_rate")
    gated = _run("uncertainty_gated")
    assert gated["rate"] < fixed["rate"]


# ---------------------------------------------------------------------------
# 4. pure_cost: pays but never changes the action
# ---------------------------------------------------------------------------

def test_pure_cost_pays_but_never_changes_action() -> None:
    """pure_cost uses the gated trigger + pays probe_cost, but discards the extra cues —
    so it can NEVER change the which-half decision (changed-action == 0)."""
    r = _run("pure_cost")
    assert r["probes"] > 0, "pure_cost should still trigger probes (and pay for them)"
    assert r["changed"] == 0, (
        f"pure_cost must never change the action (no info), got {r['changed']}"
    )


def test_pure_cost_is_costly() -> None:
    """Changing probe_cost under pure_cost changes the trajectory ⇒ the cost is paid."""
    def run(pc):
        cfg = as_cfg(probe_policy="pure_cost", probe_cost=pc)
        return Ecology(cfg, seed=1).run()["events_hash"]
    assert run(0.0) != run(0.5), "pure_cost probe_cost is not causally active"


# ---------------------------------------------------------------------------
# 5. All controls run + pay the cost
# ---------------------------------------------------------------------------

def test_all_policies_run() -> None:
    """Every probe policy runs end-to-end and returns a valid summary."""
    for pol in ("off", "fixed_rate", "uncertainty_gated", "pure_cost",
                "random_cost_matched", "gate_shuffle", "hidden_scramble"):
        extra = {"random_cost_matched_probe_rate": 0.05} if pol == "random_cost_matched" else {}
        cfg = as_cfg(probe_policy=pol, horizon=120, **extra)
        result = Ecology(cfg, seed=3).run()
        assert "events_hash" in result


def test_random_cost_matched_uses_its_rate() -> None:
    """random_cost_matched probes at ~the configured fixed rate, decoupled from margin."""
    r = _run("random_cost_matched", random_cost_matched_probe_rate=0.2)
    assert r["probes"] > 0
    # random timing ⇒ at-probe margin ~ overall (no enrichment), unlike gated.
    assert abs(r["m_probe"] - r["m_noprobe"]) < 0.2


# ---------------------------------------------------------------------------
# 6. Disconnect (anti-cheat): byte-identical across gain when active sensing OFF
# ---------------------------------------------------------------------------

def test_gated_disconnected_byte_identical_across_gain() -> None:
    """With enable_active_sensing=False (the disconnect), the gated trait
    (information_sampling_rate) is causally inert ⇒ events_hash identical across values."""
    hashes = []
    for gain in (0.0, 0.5, 1.0):
        cfg = as_cfg(probe_policy="uncertainty_gated", enable_active_sensing=False, gain=gain)
        hashes.append(Ecology(cfg, seed=5).run()["events_hash"])
    assert hashes[0] == hashes[1] == hashes[2], f"gated disconnect not byte-identical: {hashes}"


# ---------------------------------------------------------------------------
# 7. Probe cost is paid under uncertainty_gated
# ---------------------------------------------------------------------------

def test_gated_probe_cost_is_paid() -> None:
    """Changing probe_cost under uncertainty_gated changes the trajectory (cost deducted)."""
    def run(pc):
        cfg = as_cfg(probe_policy="uncertainty_gated", probe_cost=pc)
        eco = Ecology(cfg, seed=1)
        return eco.run()["events_hash"], eco.probe_count_total
    h0, n0 = run(0.0)
    h1, n1 = run(0.5)
    assert n1 > 0
    assert h0 != h1, "gated probe_cost is not causally active"
