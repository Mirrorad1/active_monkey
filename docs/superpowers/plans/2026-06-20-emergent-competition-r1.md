# Emergent Intraspecific Competition — R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a gated, costed, **unrewarded** intraspecific-contest affordance (heritable `aggr`) + lineage tracking to `ecology/patchmosaic.py`, then run the R1 experiment testing whether prey-vs-prey aggression is genuinely SELECTED (invades from rarity + positive local gradient) under food SCARCITY but NOT abundance — or hits the local-gradient WALL.

**Architecture:** Extend `Critter` with two data fields (`aggr`, `lineage`). Add a gated contest phase to `_patch_dynamics` that runs BEFORE the prey-birth draws and redistributes *reproduction opportunity under crowding* (the substrate's real scarce currency — no energy buffer is introduced) from losers to winners, costed by `aggr`. The prize scales with crowding (`N/K`), so contesting pays under scarcity and is worthless under abundance. Lineage ids are inherited (observation-only). The experiment reuses the Exp-260 local-gradient instrument adapted to `aggr`.

**Tech Stack:** Python, numpy, `ecology/patchmosaic.py`, `ecology/evolvability/metrics.py`, pytest. Run via `PYTHONPATH=. python3` from the worktree root.

## Global Constraints

- **Permissive + UNREWARDED:** no reward/score for contesting; the only driver is reproduction-opportunity-under-scarcity. The substrate NEVER ranks/selects/teleports — the evaluator only records (anti-cheat law). Verbatim spec: "affordances are available, costed, and NEVER rewarded — energy-preservation is the ONLY driver."
- **Byte-identical OFF:** with `enable_contest=False` (default) the rng draw stream is unchanged. The ring golden `d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5` (T10/T15) AND the trait-evolution golden `790a8499be51644f255f3e431ac0488612dd625047d696d1f646b7919fef7623` (mutation_rate=0.1, seed=1, h=200) MUST be preserved. New config defaults must make the feature inert.
- **Inert drift-null:** `enable_contest=True` with `aggr=0` everywhere and no aggr-mutation must ALSO be byte-identical to OFF (contest only draws rng for creatures with `aggr>0`).
- **No energy buffer:** do NOT add a per-creature energy state (it destabilized the continuous substrate). Currency = reproduction opportunity under crowding.
- **Deterministic:** one `self.rng`; per-step draw order fixed; lineage tracking adds zero rng draws.
- **Measurement discipline:** verdict by selection-vs-drift (7/8-strict `default_thresholds`) + `mean-of-opposites-guard` (per-seed dispersion, never a pooled mean); controller re-runs the decisive numbers itself.

---

### Task 1: `Critter` carries `aggr` + `lineage` (data only, byte-identical)

**Files:**
- Modify: `ecology/patchmosaic.py:180` (the `Critter` dataclass)
- Test: `tests/test_patchmosaic.py`

**Interfaces:**
- Produces: `Critter(role, trait, cid, aggr=0.0, lineage=-1)` — `aggr` and `lineage` are keyword fields with defaults, so every existing positional call `Critter("prey", trait, cid)` is unchanged.

- [ ] **Step 1: Write the failing test** (events_hash unchanged by the new fields; default Critter has aggr=0/lineage=-1)
```python
def test_critter_new_fields_default_and_byte_identical():
    from ecology.patchmosaic import Critter, PatchMosaicConfig, PatchMosaicSim
    c = Critter("prey", 1.0, 0)
    assert c.aggr == 0.0 and c.lineage == -1
    h = PatchMosaicSim(PatchMosaicConfig(horizon=200), seed=1).run()["events_hash"]
    assert h == "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"
```
- [ ] **Step 2: Run it, expect FAIL** (`TypeError: unexpected keyword` or AttributeError): `PYTHONPATH=. python3 -m pytest tests/test_patchmosaic.py::test_critter_new_fields_default_and_byte_identical -v`
- [ ] **Step 3: Implement** — add to the `Critter` dataclass after `cid: int`:
```python
    aggr: float = 0.0      # intraspecific-contest propensity (R1); only read when enable_contest
    lineage: int = -1      # founder-lineage id, inherited by offspring (observation-only)
```
- [ ] **Step 4: Run, expect PASS.** Also run the full golden set: `PYTHONPATH=. python3 -m pytest tests/test_patchmosaic.py -q` (all green — adding defaulted fields changes no counts).
- [ ] **Step 5: Commit** (`git add -A && git commit -m "patchmosaic: Critter carries aggr+lineage (data only, byte-identical)"`)

---

### Task 2: Config fields for the contest affordance + lineage tracking (all inert by default)

**Files:**
- Modify: `ecology/patchmosaic.py:145` (insert after `attack_baseline`, inside the gated trait-evolution block)
- Test: `tests/test_patchmosaic.py`

**Interfaces:**
- Produces config fields: `enable_contest: bool=False`, `aggr0: float=0.0`, `aggr_mutation_sd: float=0.05`, `contest_cost: float=0.10`, `contest_seize: float=0.50`, `contest_dissipation: float=0.0`, `track_lineages: bool=False`.

- [ ] **Step 1: Write the failing test** (defaults exist and are inert):
```python
def test_contest_config_defaults_inert():
    from ecology.patchmosaic import PatchMosaicConfig
    c = PatchMosaicConfig()
    assert c.enable_contest is False and c.aggr0 == 0.0 and c.track_lineages is False
    assert c.contest_cost == 0.10 and c.contest_seize == 0.50 and c.contest_dissipation == 0.0
```
- [ ] **Step 2: Run, expect FAIL** (AttributeError).
- [ ] **Step 3: Implement** — insert after `attack_baseline: float = 1.0`:
```python
    # ---- Intraspecific contest (R1; gated, default OFF -> byte-identical) ----
    enable_contest: bool = False        # the contest affordance; OFF => no contest phase, no new draws
    aggr0: float = 0.0                  # founder prey aggression propensity
    aggr_mutation_sd: float = 0.05      # gaussian sd for aggr inheritance (when enable_trait_evolution)
    contest_cost: float = 0.10          # fecundity cost coefficient: birth *= (1 - contest_cost*aggr)
    contest_seize: float = 0.50         # max share of the crowding-prize transferred winner<-loser
    contest_dissipation: float = 0.0    # fraction of the seized prize lost (0 = zero-sum transfer)
    track_lineages: bool = False        # record per-lineage aggr distribution (observation-only, no rng)
```
- [ ] **Step 4: Run, expect PASS**; full golden set still green.
- [ ] **Step 5: Commit** (`patchmosaic: gated contest + lineage config fields (inert defaults)`).

---

### Task 3: Founders seed `aggr0` + lineage ids (byte-identical)

**Files:**
- Modify: `ecology/patchmosaic.py:237` (founder loop) and `ecology/patchmosaic.py:218` (`__init__`)
- Test: `tests/test_patchmosaic.py`

**Interfaces:**
- Produces: each founder prey gets `aggr=cfg.aggr0` and a `lineage` id (one per founding patch, `lineage = patch_index`); predators unchanged.

- [ ] **Step 1: Write the failing test:**
```python
def test_founders_seed_aggr_and_lineage_byte_identical():
    from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
    sim = PatchMosaicSim(PatchMosaicConfig(aggr0=0.3, n_patches=4), seed=1)
    assert all(c.aggr == 0.3 for p in sim.patches for c in p.prey)
    assert all(c.lineage == p.idx for p in sim.patches for c in p.prey)
    # aggr0 alone (contest OFF) must not change the golden
    h = PatchMosaicSim(PatchMosaicConfig(horizon=200), seed=1).run()["events_hash"]
    assert h == "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"
```
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** — in the founder loop at line ~235, change the prey append to:
```python
                patch.prey.append(Critter("prey", cfg.prey_escape, self._next_cid,
                                          aggr=cfg.aggr0, lineage=i))
```
(`i` is the patch index in the founder loop; predators keep `Critter("pred", cfg.pred_attack, self._next_cid)`.)
- [ ] **Step 4: Run, expect PASS**; full golden set green (founder fields are data only).
- [ ] **Step 5: Commit** (`patchmosaic: founders seed aggr0 + per-patch lineage (byte-identical)`).

---

### Task 4: Gated contest phase in `_patch_dynamics` (the core mechanic)

**Files:**
- Modify: `ecology/patchmosaic.py:453-466` (prey-birth section `(b)`)
- Test: `tests/test_patchmosaic.py`

**Interfaces:**
- Consumes: `cfg.enable_contest`, `cfg.contest_cost/seize/dissipation`, each prey's `aggr`.
- Produces: a per-prey birth multiplier applied to the existing birth draw; rng draws ONLY for prey with `aggr>0` and only when `enable_contest`.

**Mechanic (scarcity-dependent contest competition).** Before the prey-birth draw loop, when `enable_contest`: compute `crowd_prize = contest_seize * min(1.0, N_prey / K_prey_local)` (≈0 under abundance, ≈`contest_seize` under scarcity). Build `bmult = [1.0]*N_prey`. For each prey `i` (cid order) with `aggr>0`: pay cost `bmult[i] *= max(0, 1 - contest_cost*aggr_i)`; if `N_prey>=2`, draw a target `j = rng.integers(N_prey)`; if `j!=i`, win with prob `aggr_i/(aggr_i+aggr_j+1e-9)` (one `rng.random()`); on win, `bmult[i] += crowd_prize*(1-dissipation)` and `bmult[j] = max(0, bmult[j]-crowd_prize)`. The birth draw then uses `birth_p * bmult[i] * <escape-cost term>`.

- [ ] **Step 1: Write the failing tests** — (a) byte-identical OFF; (b) inert when aggr=0 even with contest ON; (c) cost reduces a lone aggressor's birth; (d) contest is LIVE (changes the hash) when aggr>0:
```python
def test_contest_byte_identical_off_and_inert():
    from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
    GOLD = "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"
    # OFF (default)
    assert PatchMosaicSim(PatchMosaicConfig(horizon=200), 1).run()["events_hash"] == GOLD
    # ON but aggr0=0 and no aggr mutation => no creature contests => byte-identical
    cfg_inert = PatchMosaicConfig(horizon=200, enable_contest=True, aggr0=0.0)
    assert PatchMosaicSim(cfg_inert, 1).run()["events_hash"] == GOLD

def test_contest_live_changes_hash():
    from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
    GOLD = "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"
    cfg = PatchMosaicConfig(horizon=200, enable_contest=True, aggr0=0.5)
    assert PatchMosaicSim(cfg, 1).run()["events_hash"] != GOLD  # contest draws + transfers fire
```
- [ ] **Step 2: Run, expect FAIL** (contest not implemented; `test_contest_live` fails because hash still == GOLD).
- [ ] **Step 3: Implement** — restructure section `(b)`. Replace the existing prey-birth loop (lines 453-466) with: compute `birth_p` as now; if `cfg.enable_contest`, build `bmult` per the Mechanic above; then in the per-prey loop multiply the effective birth prob by `bmult[i]` (defaulting to 1.0 when contest OFF). Full code:
```python
        # (b) Prey logistic births (+ async); optional gated intraspecific contest redistributes
        #     reproduction-opportunity-under-crowding from losers to winners, costed by aggr.
        new_prey_children: List[Critter] = []
        birth_p = self._prey_birth_prob(patch.idx, N_prey, t)
        bmult = [1.0] * N_prey
        if cfg.enable_contest and N_prey > 0:
            crowd_prize = cfg.contest_seize * min(1.0, N_prey / cfg.K_prey_local)
            for i, p in enumerate(patch.prey):           # ascending cid
                if p.aggr <= 0.0:
                    continue
                bmult[i] *= max(0.0, 1.0 - cfg.contest_cost * p.aggr)
                if N_prey < 2:
                    continue
                j = int(rng.integers(N_prey))
                if j == i:
                    continue
                q = patch.prey[j]
                p_win = p.aggr / (p.aggr + q.aggr + 1e-9)
                if rng.random() < p_win:
                    bmult[i] += crowd_prize * (1.0 - cfg.contest_dissipation)
                    bmult[j] = max(0.0, bmult[j] - crowd_prize)
        for i, p in enumerate(patch.prey):               # ascending cid
            if cfg.enable_trait_evolution:
                bp = birth_p * bmult[i] * max(0.0, 1.0 - cfg.escape_cost * max(0.0, p.trait - cfg.escape_baseline))
                draw_ok = rng.random() < bp
                child_trait = p.trait if cfg.freeze_prey_trait else self._mutate(p.trait)
                child_aggr = self._mutate_aggr(p.aggr)
            else:
                draw_ok = rng.random() < (birth_p * bmult[i])
                child_trait = p.trait
                child_aggr = p.aggr
            if draw_ok:
                new_prey_children.append(Critter("prey", child_trait, self._next_cid,
                                                 aggr=child_aggr, lineage=p.lineage))
                self._next_cid += 1
```
NOTE: when `enable_contest=False`, `bmult` stays all `1.0` and no contest rng is drawn, so `birth_p*bmult[i] == birth_p` and the draw stream is identical to the pre-change code — byte-identical. `_mutate_aggr` is added in Task 5; for THIS task stub it as `child_aggr = p.aggr` (no aggr mutation yet) so the test passes, then Task 5 wires mutation.
- [ ] **Step 4: Run, expect PASS** (all four contest tests + full golden set: OFF and inert both == GOLD, live != GOLD).
- [ ] **Step 5: Commit** (`patchmosaic: gated intraspecific contest phase (scarcity-prize redistribution, byte-identical OFF)`).

---

### Task 5: Heritable `aggr` (mutation) + lineage inheritance

**Files:**
- Modify: `ecology/patchmosaic.py:377` (trait-evolution helpers; add `_mutate_aggr`)
- Test: `tests/test_patchmosaic.py`

**Interfaces:**
- Produces: `_mutate_aggr(parent_aggr) -> float` — gated like `_mutate`; clamps to `[0, 1]`; only draws rng when `enable_trait_evolution and mutation_rate>0`.

- [ ] **Step 1: Write the failing test** (aggr drifts under mutation; frozen otherwise; lineage inherited):
```python
def test_aggr_heritable_and_lineage_inherited():
    from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
    cfg = PatchMosaicConfig(horizon=300, n_patches=4, enable_trait_evolution=True,
                            enable_contest=True, aggr0=0.5, mutation_rate=0.5, aggr_mutation_sd=0.1)
    sim = PatchMosaicSim(cfg, 7); sim.run()
    aggrs = [c.aggr for p in sim.patches for c in p.prey]
    if aggrs:
        assert not all(a == 0.5 for a in aggrs)          # aggr drifted
        assert all(0.0 <= a <= 1.0 for a in aggrs)        # clamped
        assert all(c.lineage in range(4) for p in sim.patches for c in p.prey)  # lineage preserved
```
- [ ] **Step 2: Run, expect FAIL** (aggr never changes — Task 4 stubbed it to `p.aggr`).
- [ ] **Step 3: Implement** — add the helper near `_mutate` (line ~377):
```python
    def _mutate_aggr(self, parent_aggr: float) -> float:
        """Offspring aggression; gated like _mutate. Draws rng only when mutation fires."""
        cfg = self.cfg
        if cfg.mutation_rate > 0.0 and self.rng.random() < cfg.mutation_rate:
            delta = self.rng.normal(0.0, cfg.aggr_mutation_sd)
            return max(0.0, min(1.0, parent_aggr + delta))
        return parent_aggr
```
And in Task 4's birth loop, replace the stub `child_aggr = p.aggr` (in the `enable_trait_evolution` branch) with `child_aggr = p.aggr if cfg.freeze_prey_trait else self._mutate_aggr(p.aggr)`.
CRITICAL byte-identity: `_mutate_aggr` shares the SAME rng as `_mutate`. Adding a second per-birth mutation draw CHANGES the trait-evolution stream — so the `790a8499...` golden (which has NO aggr mutation path) is only preserved if `_mutate_aggr` is NOT called when contest is off. Guard: only call `_mutate_aggr` when `cfg.enable_contest` (aggr is meaningless without contest). Re-confirm `test_trait_evolution_*` (790a8499) stay green.
- [ ] **Step 4: Run, expect PASS**; re-run T15/T16/T20 + the 790a8499 golden — all green.
- [ ] **Step 5: Commit** (`patchmosaic: heritable aggr (mutation, gated) + lineage inheritance`).

---

### Task 6: Lineage / aggr-distribution recording in `run()` (observation-only)

**Files:**
- Modify: `ecology/patchmosaic.py:684` (`run()`), `ecology/patchmosaic.py:626` (`step()` returns are fine)
- Test: `tests/test_patchmosaic.py`

**Interfaces:**
- Produces: when `cfg.track_lineages`, `run()` result gains `aggr_mean_series` (global mean aggr per step) and `lineage_aggr_final` (`{lineage_id: (count, mean_aggr)}` at end). Zero rng draws.

- [ ] **Step 1: Write the failing test:**
```python
def test_lineage_recording_present_and_no_rng_change():
    from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
    GOLD = "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"
    r = PatchMosaicSim(PatchMosaicConfig(horizon=200, track_lineages=True), 1).run()
    assert "aggr_mean_series" in r and "lineage_aggr_final" in r
    assert r["events_hash"] == GOLD            # recording adds no rng draws
```
- [ ] **Step 2: Run, expect FAIL** (keys absent).
- [ ] **Step 3: Implement** — in `run()`, when `cfg.track_lineages`, append the global mean `aggr` each step and compute the per-lineage summary at the end; add both keys to the returned dict (guard so the default path is unchanged). Complete code to add into the run loop + result dict:
```python
        aggr_mean_series = []  # near the other *_series inits
        ...
        # inside the while loop, after stepping:
        if cfg.track_lineages:
            prey = [c.aggr for p in self.patches for c in p.prey]
            aggr_mean_series.append(float(np.mean(prey)) if prey else 0.0)
        ...
        # in the result dict:
        result["aggr_mean_series"] = aggr_mean_series if cfg.track_lineages else []
        if cfg.track_lineages:
            from collections import defaultdict
            agg = defaultdict(list)
            for p in self.patches:
                for c in p.prey:
                    agg[c.lineage].append(c.aggr)
            result["lineage_aggr_final"] = {k: (len(v), float(np.mean(v))) for k, v in agg.items()}
        else:
            result["lineage_aggr_final"] = {}
```
- [ ] **Step 4: Run, expect PASS** (keys present, golden preserved).
- [ ] **Step 5: Commit** (`patchmosaic: observation-only lineage + aggr-distribution recording`).

---

### Task 7: R1 experiment — does aggression EMERGE under scarcity (not abundance)?

**Files:**
- Create: `experiments/exp263_intraspecific_contest_emergence.py`
- Create: `experiments/outputs/exp263.txt` (written by the run)

**Interfaces:**
- Consumes: the contest substrate (Tasks 1–6), `ecology.evolvability.metrics` (`selection_coefficient_freq`, `count_wins`, `default_thresholds`).

This experiment mirrors the Exp-260/261 instrument, focal = PREY aggression, predator frozen. Two measurements per regime, both reading PER-SEED dispersion (mean-of-opposites-guard):
1. **Local-gradient preflight (breed-true 50/50 common garden):** prey split 50/50 into resident `aggr=a` and small-ε mutant `aggr=a+0.1` (breed-true: `freeze_prey_trait=True`, `mutation_rate=0`, `enable_contest=True`); window 400; `s = selection_coefficient_freq(0.5, f1, 400)`; win = s>0; 8 seeds (600-607); 7/8-strict. Run at resident `a=0.0` and a few anchors, at SCARCE vs ABUNDANT food.
2. **Invasion-from-rarity:** inject a rare `aggr=0.5` breed-true mutant into an `aggr=0` resident at equilibrium; does it rise? static (no aggr mutation) vs the same under abundance.

**Scarcity knob:** food scarcity = crowding `N/K`. Make it scarce by LOWERING `K_prey_local` (e.g. scarce `K_prey_local=120`, abundant `K_prey_local=600`) so the resident prey sit near vs far from K. Predator frozen/weak so prey persist; `enable_contest=True`, `contest_cost=0.10`, `contest_seize=0.50`.

**Controls (binding):** (a) DRIFT-NULL — `enable_contest=True` but make aggression causally inert by setting `contest_seize=0.0` (cost-only) AND a separate cost-null (`contest_cost=0`): the inert null must be NEUTRAL (4/8-ish); (b) ABUNDANCE arm must NOT select aggression (proves scarcity-driven); (c) byte-identity isolation already covered by Task 4 tests.

**Verdict labels (predeclared):** `AGGRESSION_EMERGES` (positive local gradient + invasion under SCARCITY, neutral under ABUNDANCE, drift-null neutral); `WALL` (no positive local gradient under scarcity — costed aggression cannot invade); `NO_VERDICT` (drift-null fires / populations collapse / per-seed bimodal un-gated).

- [ ] **Step 1:** Write the experiment script with module docstring containing a HYPOTHESIS/PREDICTION and a PREDECLARED FALSIFIER (verifier-floor requirement), the scarcity sweep, both measurements, the controls, and RAW per-seed output (sd(D), regime split per the mean-of-opposites-guard). Reuse the Exp-260 `gradient_probe` shape (focal='prey', but split on `aggr` not escape).
- [ ] **Step 2:** Smoke-run (`--smoke`, 2 seeds) to validate; confirm the drift-null (contest_seize=0) is byte-identical / neutral and the live arm differs.
- [ ] **Step 3:** Full run (8 seeds preflight + 10 seeds invasion), write `experiments/outputs/exp263.txt`.
- [ ] **Step 4:** CONTROLLER adjudication (NOT delegated): apply the 7/8-strict gate + mean-of-opposites-guard to the raw rows yourself; classify AGGRESSION_EMERGES / WALL / NO_VERDICT.
- [ ] **Step 5: Commit** the script + output.

---

### Task 8: Log Exp 263 to the journey + verifier-floor compliance

**Files:**
- Modify: `EXPERIMENTS.md` (append entry), `site/data/experiments-data.js` (journey entry + tally), `loop/directions/emergent-intraspecific-competition.md` (STATUS update), regenerate `site/data/lab-status.js`.

- [ ] **Step 1:** Append the Exp 263 entry (Plain / Consolidation / Question+FALSIFIER / Setup / Result / Implication / Honest caveat / Verdict / Verifier / Next / trace). If POSITIVE, include a `POSITIVE-SINGLE` self-grade; create `experiments/outputs/exp263.txt`.
- [ ] **Step 2:** Add the journey entry + bump `AM_TALLY` (total+1; kind by verdict). Caveat field = the MD caveat's first sentence (test_site_data sync).
- [ ] **Step 3:** Update the direction-card STATUS (≤800 chars on one line); regenerate lab-status (`python -m active_loop.site_data --lab-status`).
- [ ] **Step 4:** Run `tests/test_patchmosaic.py tests/test_check_iteration.py tests/test_experiments_parser.py tests/test_docs_consistency.py tests/test_site_data.py tests/test_status_line_length.py tests/test_lab_status.py` — all green.
- [ ] **Step 5: Commit**, then merge to main via PR.

---

## Self-Review

- **Spec coverage:** permissive/unrewarded contest ✓ (T2–T4, no reward term); energy=reproduction-under-crowding ✓ (T4 `crowd_prize`); lineage tracking ✓ (T1,T3,T6); scarcity-dependence ✓ (T7 K sweep); selection-vs-drift gate + mean-of-opposites-guard ✓ (T7); byte-identical OFF + goldens ✓ (T1–T6 tests); verdict labels + falsifier ✓ (T7). R2–R4 are out of scope for this plan (one rung).
- **Placeholder scan:** none — every code step has complete code; the one stub (`child_aggr = p.aggr` in T4) is explicitly resolved in T5.
- **Type consistency:** `Critter(role,trait,cid,aggr,lineage)` consistent T1/T3/T4; `_mutate_aggr` defined T5 used T4-as-amended; `bmult` local to T4; `aggr_mean_series`/`lineage_aggr_final` defined+returned T6.
- **Byte-identity risk (the load-bearing one):** the aggr-mutation draw in `_mutate_aggr` must be gated behind `enable_contest` so the existing trait-evolution golden (790a8499) is preserved — called out in T5 Step 3.
