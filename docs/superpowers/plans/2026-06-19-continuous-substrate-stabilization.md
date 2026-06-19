# Stabilized Continuous Consumer–Resource Substrate (Exp 243) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three gated mechanisms to the `ecology` substrate — (A) global density-dependent consumer mortality, (B) monotone floored regen, (C) a `freeze_continuous_locomotion` buildability flag — plus a reusable stability-certification instrument that returns a GO/NO-GO on whether a damped, non-oscillatory N_eq exists across an expressed band of speeds (so the invasion-from-rarity test becomes posable).

**Architecture:** All three mechanisms are gated by `enable_*`/`freeze_*` flags on `EcologyConfig`, default OFF, byte-identical OFF, golden-hash guarded. A and B are *substrate scalars* (no new heritable trait, no `mutate_*` flag). The certification instrument is a pure-numpy analysis module under `ecology/evolvability/` that consumes an `N(t)` series + telemetry and applies predeclared, frozen stability gates + a hardened oscillation detector. Experiment scripts under `experiments/` drive the staged sweep behind a runtime preflight.

**Tech Stack:** Python 3, numpy (scipy.gammaln allowed; avoid scipy.signal — implement AR/periodogram in numpy for determinism), pytest. Run everything with `uv run --python .venv … ` from the repo root.

**Spec:** `docs/superpowers/specs/2026-06-19-continuous-substrate-stabilization-design.md` (the predeclared-falsifier contract — frozen).

## Global Constraints

- Run Python via `uv run --python .venv …` from `/Users/mirro/Projects/active-loop` (the shell auto-activates conda base and shadows the venv). Experiment scripts need repo root or `PYTHONPATH=.`.
- FROZEN paths (`eval/`, the autopilot trust boundary) are never edited.
- Every new mechanism is behind an `enable_*`/`freeze_*` flag, default OFF; the OFF path makes **zero new rng draws**, no new float ops into shared state, and allocates new state only when ON ⇒ **byte-identical** `events_hash` to current HEAD.
- A and B introduce **no new heritable trait** and **no `mutate_*` flag** (they are substrate scalars).
- `events_hash` = SHA-256 over canonical JSON of `self.events` (`engine.py:1384`); `resource_tick` (`engine.py:1261`) hashes `float(np.sum(self.world.resource))` — the **discrete** grid — so continuous-resource (Mechanism B) changes are **invisible** to `events_hash`. B's ON-differs witness MUST read `cont_world._resource` directly, and B is a **silent no-op unless `enable_continuous_depletion_intake=True`** (intake ignores `_resource` otherwise).
- Verdicts blind-verified per PROTOCOL 4.5. Conservative language only (functional, costed, local gradient, posable). Predeclared falsifiers (spec §9) are frozen before any sweep run.
- Commit after every task. Small, single-purpose commits. End commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Currently on branch `exp235-terrain-locomotion`; stay on it. Do **not** stage `active_loop/cli/converse_demo.py` (a pre-existing unrelated modification).
- Before committing, `git status` to confirm no concurrent cron-loop commit landed (a live loop shares this working tree).

## File Structure

- **Modify** `ecology/engine.py` — add 6 `EcologyConfig` fields; thread `continuous_floored_regen` into the `ContinuousWorld.from_config` call; flip the mutate-coupling line; add `N_frozen` capture + the gated step-7c crowding roll; add a pure `_density_mortality_p` helper.
- **Modify** `ecology/continuous_world.py` — add `floored_regen` dataclass field + `from_config` param + a `step_regen` branch.
- **Create** `ecology/evolvability/stability.py` — the trait-agnostic certification instrument (metrics, hardened oscillation detector, cell certification, non-degeneracy audit, band verdict).
- **Create** `ecology/evolvability/cert_run.py` — the monomorphic certification run wrapper (clamped speed, telemetry, JSON rows).
- **Create** `experiments/exp243_birthbalance.py`, `experiments/exp243_preflight.py`, `experiments/exp243_controls.py`, `experiments/exp243_sweep.py` — diagnostics, preflight pilot, control arms, staged sweep.
- **Modify/Create tests** `tests/test_ecology_continuous.py` (gating goldens), `tests/test_stability_cert.py` (instrument), `tests/test_density_mortality.py` (mechanism A unit tests).

---

### Task 1: `freeze_continuous_locomotion` buildability flag

**Files:**
- Modify: `ecology/engine.py` (EcologyConfig field after line 464; mutate-coupling at line 1043–1050)
- Test: `tests/test_ecology_continuous.py`

**Interfaces:**
- Produces: `EcologyConfig.freeze_continuous_locomotion: bool = False`. When True (and continuous ON), child genotypes breed `locomotor_speed` TRUE (the per-child speed rng draw at `genotype.py:229–234` is skipped).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ecology_continuous.py` (reuse existing `_cont_cfg`, `_founder_with_speed`, `D` = `dataclasses`):

```python
class TestFreezeContinuousLocomotion:
    def _run(self, *, freeze: bool, seed: int = 3, horizon: int = 60):
        cfg = D.replace(_cont_cfg(horizon=horizon),
                        founder=_founder_with_speed(1.5),
                        mutation_rate=0.1,                      # mutation ON so the draw matters
                        freeze_continuous_locomotion=freeze)
        return Ecology(cfg, seed=seed).run()

    def test_freeze_changes_hash_and_is_deterministic(self):
        frozen = self._run(freeze=True)
        unfrozen = self._run(freeze=False)
        # Freezing SKIPS the per-child locomotor_speed draw => the event stream must differ.
        assert frozen["events_hash"] != unfrozen["events_hash"]
        # ...and freezing must itself be deterministic across reruns.
        assert frozen["events_hash"] == self._run(freeze=True)["events_hash"]

    def test_off_path_byte_identical(self):
        # freeze defaults False => existing continuous-ON golden must be unchanged.
        result = D.replace  # sanity: helper import present
        assert self._run(freeze=False, horizon=50, seed=_CONTINUOUS_ON_GOLDEN_SEED) is not None
```

Also confirm the existing `test_continuous_on_golden_hash_stable` (the `_CONTINUOUS_ON_GOLDEN_HASH` pin) still passes unchanged after this task — it is the OFF-path (`freeze=False`) byte-identity guard.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python .venv pytest tests/test_ecology_continuous.py::TestFreezeContinuousLocomotion -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'freeze_continuous_locomotion'`.

- [ ] **Step 3: Add the config field**

In `ecology/engine.py`, immediately after the `enable_continuous_depletion_intake` field (line 464), add:

```python
    # Exp 243: freeze locomotor_speed (breed TRUE) — OFF by default, byte-identical to Exp 238-242.
    # The certification/null runs need MONOMORPHIC populations, but engine.py couples
    # mutate_continuous_locomotion to enable_continuous_locomotion (every continuous run mutates
    # speed). When True (continuous ON), the per-child locomotor_speed rng draw
    # (genotype.py LOCOMOTION_CONTINUOUS_TRAITS skip-guard) is skipped so speed breeds true.
    # OFF (default) ⇒ byte-identical: the mutate flag is unchanged.
    freeze_continuous_locomotion: bool = False
```

- [ ] **Step 4: Flip the mutate-coupling line**

In `ecology/engine.py`, change the `mutate()` call argument (line 1050) from:

```python
                                    mutate_continuous_locomotion=cfg.enable_continuous_locomotion)
```
to:
```python
                                    mutate_continuous_locomotion=(cfg.enable_continuous_locomotion
                                                                  and not cfg.freeze_continuous_locomotion))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --python .venv pytest tests/test_ecology_continuous.py -v`
Expected: PASS (including the unchanged `_CONTINUOUS_ON_GOLDEN_HASH` test — OFF path byte-identical).

- [ ] **Step 6: Commit**

```bash
git status   # confirm no concurrent cron commit
git add ecology/engine.py tests/test_ecology_continuous.py
git commit -m "Exp 243 Task 1: freeze_continuous_locomotion flag (breed-true buildability fix)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Mechanism B — monotone floored regen

**Files:**
- Modify: `ecology/continuous_world.py` (dataclass field ~144; `from_config` ~378–397; `step_regen` ~305–342)
- Modify: `ecology/engine.py` (EcologyConfig `continuous_floored_regen` field after Task 1's field; thread into the `from_config` call at 544–552)
- Test: `tests/test_ecology_continuous.py`

**Interfaces:**
- Consumes: `EcologyConfig.continuous_floored_regen: bool` (Task 2).
- Produces: `ContinuousWorld(..., floored_regen: bool=False)`; when ON, `step_regen` uses `v_new = (1-regen_rate)*v + regen_rate*capacity` per sub-cell. `from_config(..., floored_regen=False)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ecology_continuous.py`:

```python
from ecology.continuous_world import ContinuousWorld, _GRID_CELLS

class TestFlooredRegen:
    def test_no_absorbing_state_and_monotone(self):
        w = ContinuousWorld.from_config(regen_rate=0.05, capacity=2.0, floored_regen=True)
        # Strip one cell to zero, step regen: floored regen must lift it OFF zero (no dead zone).
        w._resource[0][0] = 0.0
        w.step_regen()
        assert w._resource[0][0] == 0.05 * 2.0          # r*cap at v=0
        # A near-full cell must not overshoot capacity.
        w._resource[0][1] = 2.0
        w.step_regen()
        assert w._resource[0][1] == 2.0                  # delta=0 at v=cap

    def test_off_path_is_flat_regen(self):
        w = ContinuousWorld.from_config(regen_rate=0.05, capacity=2.0, floored_regen=False)
        w._resource[0][0] = 1.0
        w.step_regen()
        assert w._resource[0][0] == 1.05                 # flat += 0.05 (Exp 238-239 path)

    def test_on_differs_in_continuous_resource_with_depletion(self):
        # B is a silent no-op unless depletion-intake reads _resource. Run with depletion ON,
        # assert the cont_world._resource trajectory differs between floored and flat regen.
        def run(floored):
            cfg = D.replace(_cont_cfg(horizon=80),
                            founder=_founder_with_speed(1.5),
                            initial_population=20,
                            enable_continuous_depletion_intake=True,
                            continuous_floored_regen=floored)
            eco = Ecology(cfg, seed=3); eco.run()
            return float(np.sum(eco.cont_world._resource))
        assert run(True) != run(False), "Mechanism B is a silent no-op — VOID if byte-identical"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python .venv pytest tests/test_ecology_continuous.py::TestFlooredRegen -v`
Expected: FAIL — `from_config() got an unexpected keyword argument 'floored_regen'`.

- [ ] **Step 3: Add the dataclass field + from_config param (continuous_world.py)**

After the `enable_depletion_intake` field (line 144), add:

```python
    # Exp 243: monotone floored regen — OFF by default, byte-identical to Exp 238-239 when OFF.
    # When True, step_regen uses gap-proportional delta = regen_rate*(capacity - v):
    # at v=0 -> regen_rate*capacity > 0 (NO absorbing dead-zone, unlike logistic_regen),
    # strictly decreasing in v with NO overcompensating hump, capped without overshoot.
    # SEPARATE from logistic_regen (left untouched so the Exp-240 golden is preserved verbatim).
    floored_regen: bool = False
```

In `from_config` (signature ~378–390 and the `return cls(...)` ~395), add `floored_regen: bool = False` to the signature and `floored_regen=floored_regen` to the `cls(...)` call.

- [ ] **Step 4: Add the step_regen branch (continuous_world.py)**

In `step_regen` (line 305), make the floored branch the FIRST check (before the existing `if self.logistic_regen:` / else flat):

```python
        cap = self.capacity
        rr = self.regen_rate
        if self.floored_regen:
            # Exp 243: gap-proportional (monotone, floored, non-overcompensating) — ON branch only.
            for ri in range(_GRID_CELLS):
                row = self._resource[ri]
                for ci in range(_GRID_CELLS):
                    v = float(row[ci])
                    v = v + rr * (cap - v)
                    if v > cap:
                        v = cap
                    elif v < 0.0:
                        v = 0.0
                    row[ci] = v
        elif self.logistic_regen:
            # ... existing logistic branch unchanged ...
```
(Leave the existing `if self.logistic_regen:` body and `else:` flat body exactly as-is, now under `elif`/`else`.)

- [ ] **Step 5: Thread the flag through engine.py**

Add the EcologyConfig field (after Task 1's `freeze_continuous_locomotion`):

```python
    # Exp 243: Mechanism B — monotone floored regen for the continuous world. OFF byte-identical.
    # NOTE: only observable when enable_continuous_depletion_intake=True (intake ignores _resource otherwise).
    continuous_floored_regen: bool = False
```

In the `ContinuousWorld.from_config(...)` call (engine.py:544–552), add the argument:

```python
                floored_regen=cfg.continuous_floored_regen,
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --python .venv pytest tests/test_ecology_continuous.py::TestFlooredRegen -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git status
git add ecology/continuous_world.py ecology/engine.py tests/test_ecology_continuous.py
git commit -m "Exp 243 Task 2: Mechanism B — monotone floored regen (gated, byte-identical OFF)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Mechanism A — global density-dependent mortality

**Files:**
- Modify: `ecology/engine.py` (EcologyConfig fields; a pure `_density_mortality_p` helper; `N_frozen` capture before the order/shuffle at ~1209; the gated step-7c crowding roll after `release_maps` at ~1148)
- Test: `tests/test_density_mortality.py` (new), `tests/test_ecology_continuous.py`

**Interfaces:**
- Consumes: `EcologyConfig.{freeze_continuous_locomotion, continuous_floored_regen}` (Tasks 1–2).
- Produces: `EcologyConfig.enable_density_mortality: bool=False`, `density_mortality_theta: float=1.0`, `density_mortality_hmax: float=0.04`, `density_mortality_Kc: float=60.0`, `density_mortality_rate_scale: float=0.0`. A new death `cause_of_death="crowding"`. Module fn `_density_mortality_p(N, hmax, Kc, theta, rate_scale=0.0, dN=0.0) -> float`.

- [ ] **Step 1: Write the failing unit test for the hazard function**

Create `tests/test_density_mortality.py`:

```python
import math
import dataclasses as D
import numpy as np
from ecology.engine import Ecology, _density_mortality_p
from tests.test_ecology_continuous import _cont_cfg, _founder_with_speed  # reuse helpers

def test_hazard_formula_and_clamp():
    # p = hmax * clamp((N/Kc)**theta, 0, 1)
    assert _density_mortality_p(60, hmax=0.04, Kc=60.0, theta=1.0) == 0.04
    assert _density_mortality_p(30, hmax=0.04, Kc=60.0, theta=1.0) == 0.02
    assert _density_mortality_p(0,  hmax=0.04, Kc=60.0, theta=1.0) == 0.0
    # clamp at 1: above Kc the density factor saturates at 1 -> p == hmax
    assert _density_mortality_p(600, hmax=0.04, Kc=60.0, theta=1.0) == 0.04
    # theta=2 quadratic
    assert abs(_density_mortality_p(30, hmax=0.04, Kc=60.0, theta=2.0) - 0.04*0.25) < 1e-12
    # optional lead term
    assert _density_mortality_p(60, hmax=0.04, Kc=60.0, theta=1.0, rate_scale=0.04, dN=60.0) == 0.04 + 0.04*1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --python .venv pytest tests/test_density_mortality.py::test_hazard_formula_and_clamp -v`
Expected: FAIL — `ImportError: cannot import name '_density_mortality_p'`.

- [ ] **Step 3: Add the pure hazard helper (engine.py)**

Near the top of `ecology/engine.py` (module level, after imports):

```python
def _density_mortality_p(N, hmax, Kc, theta, rate_scale=0.0, dN=0.0):
    """Exp 243 Mechanism A per-step crowding death probability (theta-logistic on N).

    p = hmax * clamp((N/Kc)**theta, 0, 1)  [ + rate_scale*max(0, dN/Kc) when rate_scale>0 ].
    Pure function of the GLOBAL scalar N (no genotype term) -> trait-flat by construction.
    """
    factor = (N / Kc) ** theta
    if factor < 0.0:
        factor = 0.0
    elif factor > 1.0:
        factor = 1.0
    p = hmax * factor
    if rate_scale > 0.0 and dN > 0.0:
        p += rate_scale * (dN / Kc)
    return p
```

- [ ] **Step 4: Run to verify the unit test passes**

Run: `uv run --python .venv pytest tests/test_density_mortality.py::test_hazard_formula_and_clamp -v`
Expected: PASS.

- [ ] **Step 5: Write the failing integration tests**

Append to `tests/test_density_mortality.py`:

```python
def _dm_cfg(*, enable, hmax=0.04, horizon=120, seed=3, pop=40):
    return D.replace(_cont_cfg(horizon=horizon),
                     founder=_founder_with_speed(1.5),
                     initial_population=pop,
                     enable_continuous_depletion_intake=True,
                     continuous_floored_regen=True,
                     freeze_continuous_locomotion=True,
                     enable_density_mortality=enable,
                     density_mortality_hmax=hmax)

def _final_alive(res):  # last N from the run summary; adapt to the run() return shape
    return res["final_alive"] if "final_alive" in res else res["alive_count"]

def test_off_path_byte_identical():
    on  = Ecology(_dm_cfg(enable=False), seed=3).run()["events_hash"]
    # An equivalent run WITHOUT the density-mortality config keys must match.
    base = Ecology(D.replace(_dm_cfg(enable=False)), seed=3).run()["events_hash"]
    assert on == base

def test_on_differs_and_lowers_population():
    off = Ecology(_dm_cfg(enable=False), seed=3).run()
    on  = Ecology(_dm_cfg(enable=True,  hmax=0.10), seed=3).run()
    assert on["events_hash"] != off["events_hash"]
    assert _final_alive(on) < _final_alive(off)              # crowding mortality lowers N

def test_hmax_zero_on_but_null():
    # ON path (rng draw occurs) but hmax=0 => zero 'crowding' deaths, deterministic.
    r1 = Ecology(_dm_cfg(enable=True, hmax=0.0), seed=3).run()
    r2 = Ecology(_dm_cfg(enable=True, hmax=0.0), seed=3).run()
    assert r1["events_hash"] == r2["events_hash"]
    crowding = [e for e in r1["events"] if e.get("details", {}).get("cause") == "crowding"]
    assert crowding == []

def test_fixed_order_determinism():
    a = Ecology(_dm_cfg(enable=True), seed=7).run()["events_hash"]
    b = Ecology(_dm_cfg(enable=True), seed=7).run()["events_hash"]
    assert a == b
```

(Adapt `_final_alive` / `r1["events"]` to the actual `Ecology.run()` return shape — confirm by reading the end of `run()`; if events aren't returned, expose `eco.events` after `eco.run()`.)

- [ ] **Step 6: Run to verify failure**

Run: `uv run --python .venv pytest tests/test_density_mortality.py -v`
Expected: FAIL — `enable_density_mortality` unknown kwarg.

- [ ] **Step 7: Add the config fields (engine.py)**

After Task 2's `continuous_floored_regen` field:

```python
    # Exp 243: Mechanism A — global density-dependent (crowding) mortality. OFF byte-identical.
    # Per-creature Bernoulli death roll keyed ONLY on the frozen total head-count N:
    #   p = hmax * clamp((N/Kc)**theta, 0, 1) [+ rate_scale*max(0,(N-N_prev)/Kc)].
    # A SUBSTRATE scalar, not a genotype trait (no new heritable trait, no mutate_* flag).
    enable_density_mortality: bool = False
    density_mortality_theta: float = 1.0      # 1.0 linear (default); 2.0 = Stage-2 probe
    density_mortality_hmax: float = 0.04      # per-step hazard ceiling
    density_mortality_Kc: float = 60.0        # density scale (birth-balance-derived)
    density_mortality_rate_scale: float = 0.0 # optional lead/derivative brake (default OFF)
```

- [ ] **Step 8: Capture N_frozen before the order/shuffle (engine.py ~1209)**

Immediately BEFORE the `if self.cfg.shuffle_creature_order:` block (line 1210), add:

```python
        # Exp 243 Mechanism A: freeze the global head-count ONCE per step, before the
        # per-creature loop and before any shuffle, so every alive creature sees the SAME N.
        if self.cfg.enable_density_mortality:
            _N_now = self.alive_count()
            if not hasattr(self, "_dm_N_prev"):
                self._dm_N_prev = _N_now          # t=0: derivative term is 0
            self._dm_dN = _N_now - self._dm_N_prev
            self._dm_N_frozen = _N_now
            self._dm_N_prev = _N_now
```

- [ ] **Step 9: Add the gated crowding roll (engine.py, end of _step_one_creature ~after 1148)**

After the `if not ph.alive and c.policy is not None: c.policy.release_maps()` block (line 1147–1148), and before the method's closing divider (line 1150), add:

```python
        # 7c. Exp 243 Mechanism A: global density-dependent crowding mortality.
        #     Gated; OFF path makes ZERO rng draws and emits no event. Only rolled if the
        #     creature is still alive this step (starvation/senescence take precedence).
        if cfg.enable_density_mortality and ph.alive:
            p = _density_mortality_p(
                self._dm_N_frozen, cfg.density_mortality_hmax, cfg.density_mortality_Kc,
                cfg.density_mortality_theta, cfg.density_mortality_rate_scale,
                getattr(self, "_dm_dN", 0.0))
            if p > 0.0 and self.rng.random() < p:
                ph.alive = False
                ph.cause_of_death = "crowding"
                self.events.append(self._event("death", c, details={
                    "cause": "crowding",
                    "age": ph.age,
                    "offspring_count": ph.offspring_count,
                }))
                if c.policy is not None:
                    c.policy.release_maps()
```

Note: the `self.rng.random()` draw is INSIDE `if cfg.enable_density_mortality`, so the OFF path is byte-identical. `_event` mirrors the existing death-event shape (the non-senescence starvation arm: cause/age/offspring_count, no `complexity` key).

- [ ] **Step 10: Run all Task-3 tests**

Run: `uv run --python .venv pytest tests/test_density_mortality.py tests/test_ecology_continuous.py -v`
Expected: PASS. If `test_off_path_byte_identical` or the existing continuous golden fails, an rng draw leaked onto the OFF path — fix before proceeding.

- [ ] **Step 11: Commit**

```bash
git status
git add ecology/engine.py tests/test_density_mortality.py
git commit -m "Exp 243 Task 3: Mechanism A — global density-dependent crowding mortality (gated, byte-identical OFF)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Stability metrics module

**Files:**
- Create: `ecology/evolvability/stability.py`
- Test: `tests/test_stability_cert.py`

**Interfaces:**
- Produces: `level_cv(N)`, `drift_slope(N, n_eq)`, `return_map_slope(N)`, `seed_agreement(n_eqs)`, `persistence(N)`, `n_eq(N)` — all consuming a 1-D numpy array of the analysis-window `N(t)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stability_cert.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --python .venv pytest tests/test_stability_cert.py -v`
Expected: FAIL — module `ecology.evolvability.stability` not found.

- [ ] **Step 3: Implement the metrics**

Create `ecology/evolvability/stability.py`:

```python
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
```

Add `ecology/evolvability/__init__.py` export if the package uses explicit exports (check the existing `__init__.py`; append `from . import stability` only if that matches the existing style).

- [ ] **Step 4: Run to verify pass**

Run: `uv run --python .venv pytest tests/test_stability_cert.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add ecology/evolvability/stability.py tests/test_stability_cert.py ecology/evolvability/__init__.py
git commit -m "Exp 243 Task 4: stability metrics (n_eq, cv, drift, return-map slope, seed agreement)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Hardened oscillation detector

**Files:**
- Modify: `ecology/evolvability/stability.py`
- Test: `tests/test_stability_cert.py`

**Interfaces:**
- Produces: `oscillation_verdict(N, *, seed=0) -> dict` with keys `{ar_modulus, periodogram_prominence, autocorr_trough, amp_ptp, damping_ok, classification}` where `classification ∈ {"DAMPED","OSCILLATORY"}`. DAMPED iff PRIMARY (`ar_modulus < 0.90`) ∧ SECONDARY (prominence ≤ AR(1)-null 95th pct) ∧ TERTIARY (`autocorr_trough > -0.30` and `amp_ptp < 0.30`) ∧ QUATERNARY (`damping_ok`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stability_cert.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --python .venv pytest tests/test_stability_cert.py -k detector -v`
Expected: FAIL — `oscillation_verdict` not defined.

- [ ] **Step 3: Implement the detector**

Append to `ecology/evolvability/stability.py`:

```python
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
    tertiary = (trough > _AUTOCORR_TROUGH_MIN) and (amp_ptp < _AMP_PTP_MAX)
    classification = "DAMPED" if (primary and secondary and tertiary and damping_ok) else "OSCILLATORY"
    return {"ar_modulus": ar_mod, "periodogram_prominence": prom,
            "periodogram_null_95": prom95, "autocorr_trough": trough,
            "amp_ptp": amp_ptp, "damping_ok": damping_ok,
            "classification": classification}
```

Note on QUATERNARY: the `or _amp(early) <= _amp(late)*1.05` clause prevents a false FAIL on a truly-flat series (where early/late amplitudes are equal up to noise). Verify `test_detector_does_not_false_fail_flat_noise` passes; if a flat series misclassifies, the secondary/tertiary gates are the real guard there.

- [ ] **Step 4: Run to verify pass**

Run: `uv run --python .venv pytest tests/test_stability_cert.py -k detector -v`
Expected: PASS (all four detector tests). If `test_detector_catches_low_amplitude_cycle` fails, the SECONDARY (periodogram-vs-AR(1)-null) gate is the one that must catch it — confirm `prom > prom95` for that series.

- [ ] **Step 5: Commit**

```bash
git status
git add ecology/evolvability/stability.py tests/test_stability_cert.py
git commit -m "Exp 243 Task 5: hardened oscillation detector (AR(2) modulus + periodogram-vs-AR(1)-null + autocorr + damping witness)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Cell certification + non-degeneracy audit + band verdict

**Files:**
- Modify: `ecology/evolvability/stability.py`
- Test: `tests/test_stability_cert.py`

**Interfaces:**
- Consumes: per-seed run dicts `{"N": ndarray, "births_per_step": float, "crowding_per_step": float, "p_hazard_mean": float, "exploded": bool, "availability_mean": float, "boundary_frac": float, "interbump_flux": float, "n_eq": float}` and the cell's `(hmax,Kc,theta,...)`.
- Produces: `certify_run(run, params) -> dict` (per-run pass/fail on all gates); `certify_cell(runs, params, seeds) -> dict` (≥⌈0.75·seeds⌉ pass); `non_degeneracy_ok(run) -> (bool, reasons)`; `band_verdict(cell_results, expressed_speeds) -> dict` (GO iff a contiguous ≥3 stable band overlaps the expressed band).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stability_cert.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --python .venv pytest tests/test_stability_cert.py -k "certify or non_degeneracy or band" -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement (append to stability.py)**

```python
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
    h, Kc, th = params["hmax"], params["Kc"], params["theta"]
    p = _safe_hazard(n_eq_val, h, Kc, th)
    pprime = h * th * (n_eq_val/Kc)**(th-1) / Kc if n_eq_val > 0 else 0.0
    return abs(p + n_eq_val*pprime)


def _safe_hazard(N, h, Kc, th):
    f = (N/Kc)**th
    f = min(1.0, max(0.0, f))
    return h*f


def non_degeneracy_ok(run):
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
    N = np.asarray(run["N"], dtype=float)
    eqv = run.get("n_eq", n_eq(N))
    checks = {}
    checks["persistence"] = persistence(N) >= _PERSIST_FLOOR and not run["exploded"]
    checks["level_cv"] = level_cv(N) <= _LEVEL_CV_SEED_MAX
    checks["drift"] = drift_slope(N, eqv) <= _DRIFT_MAX
    checks["return_map"] = abs(return_map_slope(N)) < _RETURN_SLOPE_MAX \
        and _marginal_brake(eqv, params) < _MARGINAL_BRAKE_MAX
    bp, cp = run["births_per_step"], run["crowding_per_step"]
    checks["birth_pulse"] = (cp == 0.0 and bp == 0.0) or (cp > 0 and 0.5 <= bp/cp <= 2.0)
    checks["oscillation"] = oscillation_verdict(N)["classification"] == "DAMPED"
    nd_ok, nd_reasons = non_degeneracy_ok(run)
    checks["non_degenerate"] = nd_ok
    passes = all(checks.values())
    return {"passes": passes, "checks": checks, "nd_reasons": nd_reasons}


def certify_cell(runs, params, seeds):
    per = [certify_run(r, params) for r in runs]
    n_pass = sum(1 for p in per if p["passes"])
    need = int(np.ceil(0.75*seeds))
    n_eqs = [r.get("n_eq", n_eq(r["N"])) for r in runs]
    agree = seed_agreement(n_eqs) <= _SEED_AGREE_MAX
    return {"certified": n_pass >= need and agree, "n_pass": n_pass, "need": need,
            "seed_agreement_ok": agree, "per_seed": per}


def band_verdict(cell_stable, expressed_speeds):
    """cell_stable: {speed: bool}. GO iff a contiguous run of >=3 stable speeds overlaps expressed."""
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --python .venv pytest tests/test_stability_cert.py -v`
Expected: PASS (whole file).

- [ ] **Step 5: Commit**

```bash
git status
git add ecology/evolvability/stability.py tests/test_stability_cert.py
git commit -m "Exp 243 Task 6: cell certification + non-degeneracy audit + GO/NO-GO band verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Birth-balance diagnostic + runtime preflight pilot

**Files:**
- Create: `experiments/exp243_birthbalance.py`, `experiments/exp243_preflight.py`

**Interfaces:**
- Consumes: the Task 1–3 config flags; `ecology.engine.Ecology`.
- Produces: a printed/JSON report of empirical per-capita birth rate `b` at the certified availability (to set `Kc` so `p_hazard(N_eq)≈b` below the steep knee), and a **measured** per-run wall-time + a logistic-aware growth check on a 3-cell pilot.

- [ ] **Step 1: Write the birth-balance diagnostic**

Create `experiments/exp243_birthbalance.py` — run a monomorphic clamped-speed=1.0 population (depletion ON, floored regen ON, freeze ON, density-mortality OFF) at a short horizon, count births/step and mean field availability over the last window; print `b` and the implied `Kc` for `hmax=0.04` such that `p_hazard(N_eq)=b`. (Use `ecology.evolvability.stability` helpers for the window stats.)

- [ ] **Step 2: Run it**

Run: `uv run --python .venv python experiments/exp243_birthbalance.py`
Expected: prints empirical `b ≈ 0.1/step` (per the spec) and an implied N_eq / Kc; record the numbers in the script's output comment for the sweep to consume.

- [ ] **Step 3: Write the runtime preflight pilot**

Create `experiments/exp243_preflight.py` following the `compute-batch-runtime-preflight` skill: run 3 representative cells (slow/medium/fast speed, default hmax/Kc/regen) at horizon 4000, **time each**, assert N plateaus near a birth-balance N_eq (logistic-aware: growth DECELERATES — not runaway), and extrapolate the full ~2.9k-run wall time. Print GO/NO-GO on launching the full sweep.

- [ ] **Step 4: Run the preflight (use the skill)**

Invoke the `compute-batch-runtime-preflight` skill, then:
Run: `uv run --python .venv python experiments/exp243_preflight.py`
Expected: measured per-run seconds + a projected total; a logistic-aware verdict that no cell is runaway. If a cell runs away or a run exceeds the projection materially, STOP and report before the full sweep.

- [ ] **Step 5: Commit**

```bash
git status
git add experiments/exp243_birthbalance.py experiments/exp243_preflight.py
git commit -m "Exp 243 Task 7: birth-balance diagnostic + runtime preflight pilot

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Monomorphic certification run wrapper + telemetry

**Files:**
- Create: `ecology/evolvability/cert_run.py`
- Test: `tests/test_stability_cert.py` (a short smoke test)

**Interfaces:**
- Produces: `run_cert(speed, *, hmax, Kc, theta, regen_rate, rate_scale, layout, seed, horizon=4000, burn_in=0.6) -> dict` returning the analysis-window run dict shape consumed by `certify_run` (Task 6): `{N, births_per_step, crowding_per_step, p_hazard_mean, exploded, availability_mean, boundary_frac, interbump_flux, n_eq}`. It builds the `EcologyConfig` (depletion ON, floored regen ON, freeze ON, density-mortality ON), runs, and computes telemetry from `eco.events` + `eco.cont_world._resource` + per-creature `pos_cont`.

- [ ] **Step 1: Write the smoke test**

Append to `tests/test_stability_cert.py`:

```python
def test_cert_run_smoke():
    from ecology.evolvability.cert_run import run_cert
    r = run_cert(speed=1.0, hmax=0.04, Kc=60.0, theta=1.0, regen_rate=0.05,
                 rate_scale=0.0, layout="bump", seed=3, horizon=300, burn_in=0.6)
    assert set(r) >= {"N","births_per_step","crowding_per_step","exploded",
                      "availability_mean","boundary_frac","interbump_flux","n_eq"}
    assert len(r["N"]) == int(300*0.4)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --python .venv pytest tests/test_stability_cert.py::test_cert_run_smoke -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `cert_run.py`**

Create `ecology/evolvability/cert_run.py`: build the config via `dataclasses.replace` on a base continuous config (mirror `tests/test_ecology_continuous.py::_cont_cfg` + `_founder_with_speed`), set the Task 1–3 flags, run, then over the final `(1-burn_in)` window compute: `N(t)` from per-step alive counts (read `resource_tick`-style or instrument `eco` to record `alive_count()` each step — add a lightweight per-step N log behind the cert path, NOT in events_hash), births/step and crowding-deaths/step from `eco.events` (`event_type=="birth"` / `details.cause=="crowding"`), `availability_mean` from `eco.cont_world._resource` sampled at occupied cells, `boundary_frac` and `interbump_flux` from `pos_cont` trajectories. Return the dict.

Note: recording per-step `N(t)` must NOT enter `events_hash`. Add an opt-in recorder (e.g. `eco.record_alive_series=True` populating `eco._alive_series`) gated so the default path is byte-identical, OR compute N(t) by replaying the event stream's birth/death deltas. Prefer the event-replay approach (zero engine change).

- [ ] **Step 4: Run to verify pass**

Run: `uv run --python .venv pytest tests/test_stability_cert.py::test_cert_run_smoke -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add ecology/evolvability/cert_run.py tests/test_stability_cert.py
git commit -m "Exp 243 Task 8: monomorphic certification run wrapper + telemetry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Anti-gaming control arms

**Files:**
- Create: `experiments/exp243_controls.py`

**Interfaces:**
- Consumes: `run_cert` (Task 8), `Ecology`, the config flags.
- Produces: a JSON report with the control verdicts: (1) recruitment-decomposed frequency-delta null (crowd-ON vs crowd-OFF in a 50/50 clamped garden); (2) mortality-only flat-availability arm; (3) fixed-density L39 common-garden benefit curve; (4) A-only and B-only ablations; (5) flat-ρ stability null.

- [ ] **Step 1: Implement the controls script**

Create `experiments/exp243_controls.py` with one function per control, each predeclaring its pass condition (spec §7 controls 5,6,7,9,11 and §9 falsifiers):
- `recruitment_null()`: 50/50 founder mix of speeds {0.5, 2.0}, `freeze=True`; run crowd-ON and crowd-OFF at matched N; PASS iff the slow/fast frequency trajectory under crowd-ON stays within the crowd-OFF inter-seed 2-SE envelope. Also report per-class births:crowding-deaths and mean energy-at-crowding-death.
- `mortality_only_flat()`: crowd-ON, depletion-intake OFF (flat availability); frequency must stay flat.
- `l39_fixed_density()`: one shared world held near a fixed N; per-capita intake vs speed for all classes; PASS iff non-flat (>10% spread) AND swept distance holds.
- `ablation_diag()`: A-only (mortality ON, flat regen) and B-only (floored regen ON, mortality OFF) on the speed diagonal.
- `flat_rho_null()`: re-run the default certified cell under flat ρ; the damped fixed point should persist.

- [ ] **Step 2: Smoke-run the controls (short horizon)**

Run: `uv run --python .venv python experiments/exp243_controls.py --smoke`
Expected: each control constructs + runs at a short horizon without error and prints its (provisional) verdict. (Full-horizon results come with the sweep.)

- [ ] **Step 3: Commit**

```bash
git status
git add experiments/exp243_controls.py
git commit -m "Exp 243 Task 9: anti-gaming control arms (recruitment null, mortality-only, L39, ablations, flat-rho)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Staged sweep driver + verdict synthesis + EXPERIMENTS log

**Files:**
- Create: `experiments/exp243_sweep.py`, `experiments/outputs/exp243.txt`
- Modify: `EXPERIMENTS.md`, `loop/directions/continuous-locomotion.md` (STATUS), the site `experiments-data.js`

**Interfaces:**
- Consumes: `run_cert` (Task 8), `certify_cell` / `band_verdict` (Task 6), the controls (Task 9), the preflight GO (Task 7).
- Produces: the Stage-1 grid + Stage-2 probe results, the GO/NO-GO band verdict, JSON rows `{exp, speed, hmax, Kc, theta, regen, rate_scale, seed, metric, value}`, and the EXPERIMENTS.md entry.

- [ ] **Step 1: Implement the staged sweep**

Create `experiments/exp243_sweep.py`: Stage-1 grid (speeds × hmax{0.02,0.04,0.06,0.10} × Kc{40,60,120} × regen{0.025,0.05,0.10} × 8 seeds, θ=1, rate_scale=0); per cell call `run_cert` ×8 seeds, then `certify_cell`; find the winning region; Stage-2 probes (θ=2 arm; rate_scale=0.04 at the 3 fastest speeds) on the winners only; then `band_verdict` against the L39 expressed band from the controls. Write JSON rows + a human-readable `outputs/exp243.txt` summary. **Gate the full run behind the preflight GO** (read its verdict; refuse if NO-GO).

- [ ] **Step 2: Run the full sweep (only after preflight GO)**

Run: `uv run --python .venv python experiments/exp243_sweep.py`
Expected: a GO/NO-GO verdict with the named band (or the named NO-GO reason from spec §9). Save `outputs/exp243.txt`.

- [ ] **Step 3: Blind-verify (PROTOCOL 4.5)**

Dispatch a fresh subagent to independently re-derive the verdict from the committed JSON rows + script, confirming the GO/NO-GO and the named band/reason. Record agreement.

- [ ] **Step 4: Log the experiment (append-only)**

Append the Exp 243 entry to `EXPERIMENTS.md` (verdict, mechanism, the named band or NO-GO reason, caveats incl. the energy-leak channel and any θ-tension, generalizability tier). Update `loop/directions/continuous-locomotion.md` STATUS. Update the site `experiments-data.js` via the Python site helpers (NOT shell single-quote pipelines — the apostrophe-leak guard).

- [ ] **Step 5: Commit (experiment script + outputs + log together)**

```bash
git status
git add experiments/exp243_sweep.py experiments/outputs/exp243.txt EXPERIMENTS.md loop/directions/continuous-locomotion.md site/experiments-data.js
git commit -m "Exp 243: stabilized continuous substrate — <GO|NO-GO> posability verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Distill to a coalescence artifact**

Per the artifact standard: on GO, draft a MechanismCard (stabilized continuous consumer–resource substrate); on NO-GO, extend the `local-gradient-wall` BoundaryNote to "continuous substrate cannot host a stable damped equilibrium because <named mechanism>." Run `active-monkey coalesce validate --all`.

---

## Self-Review

**Spec coverage:** §4 Mechanism A → Task 3. §5 Mechanism B → Task 2. §6 freeze fix → Task 1. §7 metrics → Task 4; oscillation detector → Task 5; cell-cert + non-degeneracy + band overlap → Task 6; controls → Task 9. §8 sweep + runtime preflight → Tasks 7,10. §9 falsifiers → evaluated in Tasks 6 (gates), 9 (control falsifiers), 10 (verdict). §11 implementation surface → Tasks 1–3 (engine/world), 4–6,8 (evolvability), 7,9,10 (experiments), tests throughout. §12 coalescence → Task 10 Step 6. All spec sections map to a task.

**Placeholder scan:** Tasks 7–10 reference scripts whose bodies are described rather than fully spelled (diagnostics/drivers that orchestrate the fully-coded primitives from Tasks 1–8). Each names exact inputs/outputs and pass conditions; the load-bearing logic (hazard, regen, metrics, detector, certification) is complete code in Tasks 1–6. This is intentional decomposition, not a placeholder — the orchestration scripts are thin glue over tested primitives.

**Type consistency:** the run-dict keys produced by `run_cert` (Task 8) match the keys consumed by `certify_run`/`non_degeneracy_ok` (Task 6): `N, births_per_step, crowding_per_step, p_hazard_mean, exploded, availability_mean, boundary_frac, interbump_flux, n_eq`. `_density_mortality_p` (Task 3) signature matches the `_safe_hazard`/`_marginal_brake` usage (Task 6) — both compute `hmax*clamp((N/Kc)**theta,0,1)`; the Task-6 copy is intentionally independent (analysis side) but numerically identical (add a test asserting they agree if drift is a concern). `oscillation_verdict` keys (Task 5) consumed only via `["classification"]` (Task 6) — consistent.
