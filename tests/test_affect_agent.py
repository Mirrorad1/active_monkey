"""tests/test_affect_agent.py — M4a AffectAgent wiring guards (Exp 214 / increment 1c).

Durable guards (loop/META.md) institutionalised after Exp 214: the affect EFE/policy and the
1c timing re-wire must keep working, because a silent regression there would turn every affect
experiment into a FALSE NEGATIVE (the agent appears not to learn when really the instrument
broke).  These tests do NOT assert the agent learns (it does not, at this scale — that is the
honest Exp 214 result); they assert the MACHINERY is sound.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import jax.numpy as jnp

from active_loop.affect_spec import build_dyad_model, build_direct_head_model, LV, NEU, POS, U, R, V
from active_loop.affect_agent import AffectAgent, DirectHeadAgent


def test_efe_liveness_on_gifted_model():
    """INSTRUMENT SOUNDNESS (the Exp 214 control): given a DISCRIMINATIVE model — intent 0
    emits POS strongly and response 0 (GREET) drives every intent -> intent 0 — the EFE
    policy must PREFER response 0.  If this fails, the EFE/policy is disconnected from A/B
    and any 'agent does not learn' result is a wiring bug, not science."""
    m = build_dyad_model(0)
    A1 = np.full((3, 4), 0.1); A1[POS, 0] = 5.0; A1 = A1 / A1.sum(0, keepdims=True)
    m["A"][1] = jnp.array(A1[None]); m["pA"][1] = jnp.array((A1 * 1 + 0.1)[None])
    B = np.array(m["B"][0])[0]; B[:, :, 0] = 0.0; B[0, :, 0] = 1.0
    m["B"][0] = jnp.array(B[None]); m["pB"][0] = jnp.array((B * 1 + 0.1)[None])
    ag = AffectAgent(m, lv=LV, seed=0)
    for code in range(3):
        qs = ag.agent.infer_states([jnp.array([code]), jnp.array([NEU])], ag._prior)
        q_pi, _ = ag.agent.infer_policies(qs)
        q0 = float(np.array(q_pi).reshape(-1)[0])
        assert q0 > 0.4, f"EFE does not prefer the POS-reaching response (q_pi[0]={q0:.3f}); wiring bug"


def test_perceive_uses_utterance_alone():
    """1c timing fix: perceive() takes ONLY the code (no valence carry) and returns a proper
    intent posterior.  Guards against re-introducing the stale-valence leak (Exp 128)."""
    params = list(inspect.signature(AffectAgent.perceive).parameters)
    assert params == ["self", "code"], f"perceive signature regressed: {params}"
    ag = AffectAgent(build_dyad_model(1), lv=LV, seed=1)
    qs = ag.perceive(3)
    assert qs.shape[0] == 4 and abs(qs.sum() - 1.0) < 1e-5 and not np.isnan(qs).any()


def test_window_arithmetic_and_session_runs():
    """A short session runs end-to-end and the LV-decay window arithmetic is exact (Exp 125 P4):
    pA[0].sum() == init*LV^n + sum_{k<n} LV^k.  Guards the Dirichlet window wiring."""
    ag = AffectAgent(build_dyad_model(2), lv=LV, seed=2)
    init = ag._init_pA0_sum
    n = 20
    rng = np.random.default_rng(2)
    for t in range(n):
        code = int(rng.integers(0, 6))
        ag.perceive(code)
        r = ag.act()
        val = POS if r == (code % 4) else NEU
        ag.observe_feedback(code, val)
    expected = init * (LV ** n) + sum(LV ** k for k in range(n))
    assert abs(ag.pA0_sum() - expected) < 0.5, f"window arithmetic off: {ag.pA0_sum():.3f} vs {expected:.3f}"
    assert ag.pB0_sum() > 0  # B-learning accumulated mass


# ---------------------------------------------------------------------------
# Increment 1d: DirectHeadAgent (2-factor direct response->valence head) wiring guards
# ---------------------------------------------------------------------------

def test_direct_head_efe_liveness():
    """INSTRUMENT SOUNDNESS (Exp 215 control): with a CLEANLY-gifted direct-head model —
    code0→intent0 (A0) and A1[intent0, resp2]→POS — the 2-factor EFE must PREFER resp2.
    Guards the direct head + the [1,R] control wiring against a silent disconnect."""
    import jax.numpy as jnp
    from pymdp.agent import Agent
    m = build_direct_head_model(0)
    A0 = np.array(m["A"][0])[0]; A0[:] = 0.02; A0[0, 0, :] = 0.9
    A0 = A0 / A0.sum(0, keepdims=True); m["A"][0] = jnp.array(A0[None]); m["pA"][0] = jnp.array((A0 + 0.1)[None])
    A1 = np.array(m["A"][1])[0]; A1[:, 0, 2] = [0.05, 0.05, 0.9]
    A1 = A1 / A1.sum(0, keepdims=True); m["A"][1] = jnp.array(A1[None]); m["pA"][1] = jnp.array((A1 + 0.1)[None])
    ag = Agent(A=m["A"], B=m["B"], C=m["C"], D=m["D"], pA=m["pA"], pB=m["pB"],
               num_controls=[1, R], policy_len=1, action_selection="stochastic", sampling_mode="full",
               inference_algo="fpi", batch_size=1, learn_A=True, learn_B=False)
    qs = ag.infer_states([jnp.array([0]), jnp.array([NEU])], ag.D)
    qp = np.array(ag.infer_policies(qs)[0]).reshape(-1)
    assert int(np.argmax(qp)) == 2 and qp[2] > 0.4, f"direct-head EFE wiring bug: q_pi={np.round(qp,3)}"


def test_direct_head_agent_runs_and_learns_A1():
    """A short DirectHeadAgent session runs end-to-end and the valence head A1 accumulates mass
    (learn_A on the 2-factor model); the chosen response is a valid index."""
    ag = DirectHeadAgent(build_direct_head_model(3), lv=LV, seed=3)
    rng = np.random.default_rng(3)
    for _ in range(8):
        code = int(rng.integers(0, U))
        ag.perceive(code)
        r = ag.act()
        assert 0 <= r < R
        ag.observe_feedback(code, POS if r == (code % 4) else NEU)
    assert ag.pA1_sum() > ag._init_pA1_sum - 1.0  # the head ledger is live


# ---------------------------------------------------------------------------
# Exp 217: HONEST exploration scaffolds — fast guards (no 300-turn sessions)
# ---------------------------------------------------------------------------

def test_directhead_eps_off_byte_identical():
    """OFF guard: two DirectHeadAgents on the same model+seed, one with all defaults and one with
    explicit eps0=0.0, optimism=0.0, must produce BYTE-IDENTICAL action sequences over 12 turns.
    Guards that the OFF path consumes ZERO extra RNG (a silent extra split would break fixed-seed
    determinism and corrupt all future regression checks)."""
    CODES = [0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 5]
    model = build_direct_head_model(0, k=6)
    ag_default = DirectHeadAgent(model, lv=LV, seed=0)
    ag_explicit = DirectHeadAgent(model, lv=LV, seed=0, eps0=0.0, eps_min=0.0, eps_turns=0,
                                   optimism=0.0)
    acts_default, acts_explicit = [], []
    for code in CODES:
        ag_default.perceive(code); ag_explicit.perceive(code)
        acts_default.append(ag_default.act()); acts_explicit.append(ag_explicit.act())
        ag_default.observe_feedback(code, NEU); ag_explicit.observe_feedback(code, NEU)
    assert acts_default == acts_explicit, (
        f"eps OFF must be byte-identical to defaults; got {acts_default} vs {acts_explicit}")


def test_directhead_eps_greedy_explores():
    """eps=1.0 (always explore): over 60 acts on a fixed code, ALL R responses must appear at
    least once, and no single response must dominate (>70%).  Guards that the exploration override
    actually fires and samples uniformly across responses."""
    import collections
    model = build_direct_head_model(0, k=6)
    ag = DirectHeadAgent(model, lv=LV, seed=42, eps0=1.0, eps_min=1.0, eps_turns=0)
    counts: dict[int, int] = collections.Counter()
    n = 60
    for _ in range(n):
        ag.perceive(0)
        r = ag.act()
        counts[r] += 1
        ag.observe_feedback(0, NEU)
    assert len(counts) == R, (
        f"eps=1 exploration must cover all {R} responses; only saw {sorted(counts.keys())}")
    for r in range(R):
        assert counts[r] > 0, f"response {r} never appeared in {n} fully-exploring acts"
        frac = counts[r] / n
        assert frac < 0.70, f"response {r} too dominant ({frac:.2f}); exploration is not uniform"


def test_directhead_optimism_is_uniform_across_responses():
    """Honesty guard: optimism=5.0 must add EXACTLY +5.0 to the POS row of pA1 uniformly across
    ALL (intent, response) pairs, and 0.0 elsewhere.  Any response-specific increment would
    encode the answer and violate the HONEST constraint."""
    model = build_direct_head_model(0, k=6)
    ag_base = DirectHeadAgent(model, lv=LV, seed=0, optimism=0.0)
    ag_opt  = DirectHeadAgent(model, lv=LV, seed=0, optimism=5.0)
    pA1_base = np.array(ag_base.agent.pA[1])  # (1, V, k, R)
    pA1_opt  = np.array(ag_opt.agent.pA[1])   # (1, V, k, R)
    diff = pA1_opt - pA1_base
    # POS row: every entry must be exactly +5.0
    pos_diff = diff[:, POS, :, :]
    assert np.allclose(pos_diff, 5.0, atol=1e-5), (
        f"optimism +5 must add exactly 5.0 to every POS entry; got min={pos_diff.min():.4f} "
        f"max={pos_diff.max():.4f}")
    # All other rows (NEG, NEU): must be 0.0 (no response-specific leak)
    other_diff = np.delete(diff, POS, axis=1)
    assert np.allclose(other_diff, 0.0, atol=1e-5), (
        f"optimism must leave NEG/NEU rows unchanged; got max abs diff {np.abs(other_diff).max():.4g}")


def test_directhead_optimism_off_byte_identical():
    """OFF guard: default DirectHeadAgent vs optimism=0.0 must produce BYTE-IDENTICAL actions over
    12 turns.  Guards that the zero-optimism path does not mutate the model_dict even slightly."""
    CODES = [3, 0, 5, 1, 4, 2, 3, 5, 0, 4, 1, 2]
    model = build_direct_head_model(7, k=6)
    ag_default = DirectHeadAgent(model, lv=LV, seed=7)
    ag_zero    = DirectHeadAgent(model, lv=LV, seed=7, optimism=0.0)
    acts_default, acts_zero = [], []
    for code in CODES:
        ag_default.perceive(code); ag_zero.perceive(code)
        acts_default.append(ag_default.act()); acts_zero.append(ag_zero.act())
        ag_default.observe_feedback(code, NEU); ag_zero.observe_feedback(code, NEU)
    assert acts_default == acts_zero, (
        f"optimism=0 must be byte-identical to defaults; got {acts_default} vs {acts_zero}")


def test_exp218_cells_well_formed():
    """Guard: Exp 218 CELLS dict is structurally sound (Exp 218 integrity sentinel).

    No sessions; pure config check.  Ensures CELLS can be imported and every entry
    satisfies the predeclared constraints so a typo in the constant table cannot
    silently corrupt a long-running experiment.
    """
    import importlib.util as _u
    import pathlib as _pl
    _spec = _u.spec_from_file_location(
        "exp218", _pl.Path(__file__).parent.parent / "experiments" / "exp218_m4a_ratchet.py"
    )
    _mod = _u.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cells = _mod.CELLS

    assert set(cells) == {"anchor", "gamma4", "gamma1", "K5", "K4", "turns100", "realistic"}, \
        f"CELLS keys mismatch: {set(cells)}"

    for name, cfg in cells.items():
        assert cfg["gamma"] > 0,             f"{name}: gamma must be >0, got {cfg['gamma']}"
        assert 4 <= cfg["K"] <= 6,           f"{name}: K must be in [4,6], got {cfg['K']}"
        assert cfg["turns"] in {100, 300},   f"{name}: turns must be 100 or 300, got {cfg['turns']}"

    # boundary sentinels
    anc = cells["anchor"]
    assert anc["gamma"] == 8.0 and anc["K"] == 6 and anc["turns"] == 300, \
        f"anchor must be gamma=8/K=6/turns=300, got {anc}"
    real = cells["realistic"]
    assert real["gamma"] == 1.0 and real["K"] == 4 and real["turns"] == 100, \
        f"realistic must be gamma=1/K=4/turns=100, got {real}"


def test_constant_response_ceiling_is_one_third_and_exceeds_ignition_threshold():
    """Exp 218 metric guard (blind-verified): a last-third POS-rate of ~0.33 does NOT prove
    per-intent learning — a degenerate constant-response policy already reaches it.

    For the standard M4a map (CORRECT[c]=c%4, U=6, R=5) the constant-response ceiling is 1/3,
    which is ABOVE the ignition threshold (last>=0.30) used in Exp 216/217/218. So that
    threshold under-discriminates: future M4a ignition claims must clear 1/3 (or report a
    per-intent correct-select readout). This test pins the ceiling so the flaw can't be
    silently re-introduced.
    """
    from active_loop.affect_spec import constant_response_ceiling, U, R

    correct = {c: c % 4 for c in range(U)}
    ceil = constant_response_ceiling(correct, R)
    assert abs(ceil - 1 / 3) < 1e-9, f"constant-response ceiling expected 1/3, got {ceil}"
    # the uniform-random null (1/R) is LOWER — and is the wrong null for this map
    assert ceil > 1 / R, "constant ceiling must exceed the uniform-random ceiling 1/R"
    # the documented flaw: the 0.30 ignition threshold sits BELOW this ceiling
    assert ceil > 0.30, "constant ceiling must be above the 0.30 ignition threshold (the flaw)"


def test_correct_select_high_on_gifted_discriminative_model():
    """Exp 219 probe guard: a gifted K=6 direct-head agent should discriminate."""
    correct = {c: c % 4 for c in range(U)}
    m = build_direct_head_model(0, k=6)

    A0 = np.array(m["A"][0])[0]
    A0[:] = 0.01
    for c in range(U):
        A0[c, c, :] = 0.95
    A0 = A0 / A0.sum(0, keepdims=True)
    m["A"][0] = jnp.array(A0[None])
    m["pA"][0] = jnp.array((A0 + 0.1)[None])

    A1 = np.array(m["A"][1])[0]
    A1[:] = 0.1
    for c, r in correct.items():
        A1[POS, c, r] = 0.9
    A1 = A1 / A1.sum(0, keepdims=True)
    m["A"][1] = jnp.array(A1[None])
    m["pA"][1] = jnp.array((A1 + 0.1)[None])

    ag = DirectHeadAgent(m, seed=0, lr_pA=4.0, lv=LV)

    assert ag.correct_select(correct) >= 0.5


def test_correct_select_near_chance_on_untrained():
    """Exp 219 probe guard: an untrained direct-head agent must not look discriminative."""
    ag = DirectHeadAgent(build_direct_head_model(0, k=6), seed=0)
    assert ag.correct_select({c: c % 4 for c in range(U)}) <= 0.5


def test_exp219_cells_and_genuine_rule():
    """Exp 219 config guard: cells and constant-response ceiling match the predeclaration."""
    import importlib.util as _u
    import pathlib as _pl
    import pytest
    from active_loop.affect_spec import constant_response_ceiling

    _spec = _u.spec_from_file_location(
        "exp219", _pl.Path(__file__).parent.parent / "experiments" / "exp219_m4a_discriminate.py"
    )
    _mod = _u.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    correct = {c: c % 4 for c in range(U)}
    assert set(_mod.CELLS) == {"g1_100", "g1_300", "g4_100", "g4_300", "g4_600",
                               "ctrl_anchor", "ctrl_g4K6"}
    for name, cfg in _mod.CELLS.items():
        assert cfg["gamma"] > 0, f"{name}: gamma must be >0, got {cfg['gamma']}"
        assert cfg["K"] in {4, 6}, f"{name}: K must be 4 or 6, got {cfg['K']}"
        assert cfg["turns"] in {100, 300, 600}, f"{name}: unexpected turns {cfg['turns']}"

    assert constant_response_ceiling(correct, R) == pytest.approx(1 / 3)
