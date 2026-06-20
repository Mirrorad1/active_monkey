# Spec — Stabilized continuous consumer–resource substrate (Exp 243)

**Date:** 2026-06-19 · **Direction:** continuous-locomotion (reopened) · **Status:** design approved, pre-registered

> This is both a brainstorming design doc and the experiment's **predeclared-falsifier contract**
> (`loop/VALIDATION.md` is binding). Falsifiers below are frozen before any run. Hardened by a
> 9-agent design + adversarial-audit workflow (3 designers → synthesize → 4 adversarial lenses →
> harden); all four lenses returned "needs-fix" and the fixes are folded in.

---

## 1. Premise & reopen condition

The continuous-locomotion chapter closed at **Exp 242 (CAN'T POSE)**. The toy `ContinuousWorld`
could not host a clean locomotion-evolvability verdict: it **ran away** when intake ignored
depletion (the foundational Exp-238–241 bug — `line_integral_intake` read the non-depleting `rho()`
field), and fell into an **oscillatory limit cycle** when intake depleted a regenerating field
(`enable_continuous_depletion_intake`, Exp 242: a 45-cell speed sweep found **0 stable fixed
points**; slow speeds went extinct). No stable N_eq for any speed ⇒ invasion-from-rarity is not posable.

The recorded reopen condition: *a human word **and** a fundamentally better continuous substrate that
yields stable equilibria across a range of speeds (proper, non-oscillatory density regulation) before
any evolvability claim is even testable.* Both are now satisfied: the human chose **"build a better
continuous substrate"**, by **extending `ContinuousWorld` ourselves** (not Brax/an engine — the
blocker is consumer–resource *population dynamics*, not movement *kinematics*; a physics engine
solves the wrong problem and breaks the determinism the honesty machinery requires).

## 2. Goal & scope

**Goal.** Convert the Exp-242 oscillator into a substrate with a **damped, non-oscillatory N_eq across
a usable band of speeds where the locomotion trait is behaviorally expressed**, so the
invasion-from-rarity test becomes **posable**.

**Scope of THIS build (Exp 243):** the substrate (mechanisms A + B + a buildability fix, all gated)
**plus** a stability-certification instrument that produces a **GO / NO-GO on posability**.

**Out of scope:** the invasion-from-rarity / evolvability test itself (the *next* experiment). Even a
clean GO certifies *posability only* (a stable, expressed, non-degenerate, layout-robust N_eq over an
overlapping band) — it does **not** establish that `locomotor_speed` is evolvable (L41).

## 3. Diagnosis (why Exp 242 oscillated)

Lagged, synchronized consumer–resource feedback:
- **Large consumer energy buffer** (`energy_capacity=20` vs upkeep ~0.5–0.8/step ≈ 30-step buffer)
  + **slow resource refill** (`regen_rate=0.05` on `capacity=2.0` ≈ 40 steps), with the **only brake
  being all-or-nothing starvation** firing *after* the buffer empties → overshoot → synchronized
  crash → boom → repeat (classic lagged-logistic / Ricker route to a stable limit cycle).
- **No consumer self-limitation** at all (death is starvation-only; `max_population` is a safety halt,
  not a regulator).
- The optional **logistic regen** `Δv = r·v·(1−v/cap)` makes it worse two ways: an **absorbing state
  at v=0** (a stripped cell regens 0 forever → permanent dead zones) and an **overcompensating hump**
  (max regen at v=cap/2 → discrete-time period-doubling, synchronized recovery).

The fix is **ecological, not kinematic**: add the missing consumer self-limitation (A) and replace the
pathological regen with a well-posed one (B).

## 4. Mechanism A — global density-dependent mortality (primary)

New gate `enable_density_mortality: bool = False`. A per-creature **Bernoulli death roll** each step,
keyed **only** on the frozen total head-count N (θ-logistic / Gilpin–Ayala immediate density dependence):

```
p_hazard(N) = density_mortality_hmax · clamp( (N / density_mortality_Kc) ** density_mortality_theta , 0.0, 1.0 )
```

Optional anticipatory (lead) term, default OFF:
```
when density_mortality_rate_scale > 0:
    p_hazard += density_mortality_rate_scale · max(0.0, (N_frozen − N_prev) / Kc)   # before the clamp
```

**Placement & determinism.**
- `N_frozen = self.alive_count()` (= `len(self._alive_list)`, O(1), insertion-ordered, **no float
  reduction / dict ordering**) captured **once** at the top of `step()` **before** the `rng.shuffle`
  of `order` (engine.py:1210–1214), frozen exactly like `class_occ_prev` (engine.py:1228) — every
  alive creature this step sees the **same N**.
- The roll is a **new gated branch, step 7c**, inserted in `_step_one_creature` **after** the existing
  starvation/senescence death-precedence block (engine.py:1112–1138) and **before** `release_maps`
  (engine.py:1147) — a creature already dead this step is never rolled.
- Per alive-and-not-already-dead creature, in the loop's `order`: draw `u = self.rng.random()`
  (**exactly one draw**, only inside `if cfg.enable_density_mortality:`); if `u < p_hazard(N_frozen)`
  set `alive=False`, `cause_of_death='crowding'`, append a `crowding` death event (emitted ON-path only).
- Pure per-creature Bernoulli **hazard** — never a cull / sort / top-K / ranking. The only
  cross-creature term is the frozen scalar N. **A is a substrate scalar, not a genotype trait → no new
  heritable trait, no `mutate_*` flag.**

**Parameters & audit-driven defaults.**

| param | default | range / notes |
|---|---|---|
| `enable_density_mortality` | `False` | master gate; OFF = zero new rng draws, no N_frozen, no `crowding` event, no state |
| `density_mortality_theta` | **1.0** | `{1.0, 2.0}`. **Default 1.0 (linear)**: `p′=hmax/Kc` constant so `N·p′(N_eq)=hmax·N_eq/Kc` stays bounded, far from the λ=−1 flip. θ=2 demoted to a Stage-2 probe (steep-knee/flip risk + quadratic suppression at low N) |
| `density_mortality_hmax` | **0.04** | `[0.02, 0.20]` per-step hazard ceiling. Brake timescale = `1/p_hazard(N_eq)` at the *empirical* equilibrium, not `1/hmax`. Sweep `{0.02,0.04,0.06,0.10}`; GO must find a damped cell at the **low** hmax end |
| `density_mortality_Kc` | **60.0** | `[40,200]`. **Birth-balance-derived** (lowered 120→60) so `p_hazard(N_eq)≈b` with N_eq **below** the steep knee and `|p+N·p′|<0.5`. **Crossed with `regen_rate`** in the main grid |
| `density_mortality_rate_scale` | `0.0` | `[0,0.08]` optional lead brake on `(N_frozen−N_prev)`; `N_prev` initialized to `N_frozen` on step 0 (term = 0 at t=0), allocated only when `>0`. Swept `{0.0,0.04}` at the fast end |

**Why it adds no direct speed selection.** `p_hazard` is a pure function of the **global** scalar N
(and substrate scalars) — no genotype field, identical threshold for every alive creature ⇒ direct
crowding-death **rate** is trait-flat *by construction*. **This makes a per-class death-rate null
degenerate (passes by construction — the L40 trap).** Trait-neutrality is therefore proven by the
controls in §7, not by the rate-equality check.

**Honest characterization (retracted overclaim).** Because N is frozen at step-top and children enter
`_alive_list` only **after** the loop (engine.py:1242–1247), **A is a ≥1-step lagged brake with an
in-step birth-pulse blind spot** — the candidate's "zero buffer lag" was mechanically false. A
unit-delayed level brake is exactly the structure that flips to period-2 as strength rises; this drives
several design choices (land GO at low hmax; the `rate_scale` lead term; the hardened detector).

## 5. Mechanism B — monotone, floored regen (companion)

New gate `enable_floored_regen: bool = False`, **separate** from the existing `logistic_regen` (left
untouched so the Exp-240 golden is preserved verbatim). New branch in `ContinuousWorld.step_regen`,
checked before the existing logistic/flat selection. Per sub-cell per step when ON:

```
delta = regen_rate * (capacity - v)
v_new = min(capacity, v + delta)          #  ==  (1 - regen_rate)*v + regen_rate*capacity
```

- At `v=0`: `delta = regen_rate·capacity > 0` → **no absorbing dead-zone**. At `v=cap`: `delta=0` (no
  overshoot). `delta` strictly decreasing in v, strictly positive for `v<cap` → **no overcompensating
  hump**. Per-cell map slope `|dv′/dv| = 1−regen_rate ∈ (0,1)` → a contraction toward `cap`.
- Reuses `regen_rate` (no new resource param), no heritable trait. ON-path may be vectorized
  (`v = (1−r)*v + r*cap` on the numpy array).

**Plumbing (must not be a silent no-op).** Thread `enable_floored_regen: bool=False` through
`ContinuousWorld.__init__` **and** `from_config` (continuous_world.py:377–397, default-arg discipline
mirroring `logistic_regen`), wired from a new `EcologyConfig.continuous_floored_regen` flag at the
engine continuous-world construction site (engine.py:544–550) — **explicitly threaded**.

**Supply-inflation caveat (audit).** Floored regen refills a fully-stripped cell at `0.10/step` vs flat
`0.05` — **twice as fast at the bottom** — raising the depleted-cell floor and effective field capacity,
which can push N_eq onto A's steep curve. Therefore: the sweep adds `regen_rate=0.025`
(timescale-matched *down* to flat, not supply-matched up); Kc is crossed with `regen_rate`; and a
**supply-budget invariant** (§7) requires B not raise mean time-integrated availability >15% vs A+flat.
Per-cell contraction does **not** by itself establish coupled resource–consumer damping; B's claim is
scoped narrowly to "removes the resource-side absorbing/hump pathology and raises the depleted-cell floor."

## 6. Buildability fix — `freeze_continuous_locomotion` (critical, audit-caught)

The monomorphic "breeds-true" premise was **unbuildable**: `engine.py:1050` hard-wires
`mutate_continuous_locomotion = cfg.enable_continuous_locomotion`, so every continuous run mutates
`locomotor_speed` (and draws the per-child speed rng). **Fix:** add `freeze_continuous_locomotion:
bool = False` to `EcologyConfig` and change engine.py:1050 to
`mutate_continuous_locomotion = cfg.enable_continuous_locomotion and not cfg.freeze_continuous_locomotion`.
The genotype.py:232 skip-guard then skips the per-child speed draw when frozen. Used by **all**
certification + null runs. OFF path byte-identical; ON must produce a *different* events_hash than OFF
(a draw is skipped) and be deterministic across reruns.

## 7. Stability certification protocol

**Run.** Monomorphic founder population clamped to a single `locomotor_speed`, `freeze_continuous_locomotion=True`,
**H = 4000**, ≥8 seeds. Record `N(t) = alive_count()` every step. **Burn-in = first 60% (2400
steps); analysis window W = final 40% (1600 steps)** (≫ the ~30/40-step time constants).
`N_eq` (per run) = **median** of `N(t)` over W; certified `N_eq` (per speed) = median of per-seed
`N_eq` across seeds passing all gates. A run is scorable only if **neither extinct nor exploded**
(`N(t)>0` ∀t∈W AND `exploded` never set). A pre-sweep **birth-balance diagnostic** (per-capita birth
rate `b` at the certified availability) sets the expected N_eq and chooses Kc so `p_hazard(N_eq)≈b`
below the steep knee.

**Per-run gates (ALL must pass):**

| metric | definition | pass threshold |
|---|---|---|
| persistence | `min_{t∈W} N(t)`, `exploded` | `≥ 30` AND `exploded == False` (floor raised 5→30: a pop riding ~8–12 shows low CV only because births≈deaths≈0 near quasi-extinction, and can't host a "rare" invader) |
| level_cv | `std/mean` of N over W, per seed | median-across-seeds `≤ 0.15` AND no seed `> 0.25` |
| drift_slope | `|OLS slope · len(W) / N_eq|` (total fractional drift) | `≤ 0.10` every seed |
| return_map_slope | local OLS slope of `N(t+1)` vs `N(t)` at N_eq; plus `−[p_hazard(N_eq)+N_eq·p_hazard′(N_eq)]` | `|slope| < 1` every seed (catches the flip/period-2 exit λ<−1) AND `|p+N·p′| < 0.5` |
| birth_pulse_vs_removal | mean births/step & max `|ΔN|/N` over W vs mean crowding-removals/step | births/step ≈ crowding-removals/step within ~2× AND `max |ΔN|/N < 0.15` (else enable `rate_scale=0.04`, re-certify) |
| seed_agreement | `(max N_eq − min N_eq)/median N_eq` across seeds | `≤ 0.25` |
| oscillation detector | see below | passes PRIMARY ∧ SECONDARY ∧ TERTIARY ∧ QUATERNARY |

**Oscillation detector (hardened; all thresholds predeclared & frozen).** On OLS-detrended N(t) over W:
- **PRIMARY** — AR(1)-vs-AR(2) **dominant-root-modulus** test: PASS iff fitted dominant root **modulus
  < 0.90** (a damped fixed point + noise fits a real root <1; a sustained cycle needs complex AR(2)
  roots near the unit circle).
- **SECONDARY** — periodogram **peak-prominence** (dominant bin ±1 neighbor) vs an **AR(1) red-noise
  null** fit to the same series: FAIL if prominence exceeds the AR(1)-null 95th percentile (catches a
  noisy cycle that spreads power across adjacent bins).
- **TERTIARY** — detrended-autocorrelation trough `r_min` over lags 1..|W|/2: FLAG if `min_k r(k) ≤
  −0.30` OR relative peak-to-trough `A_ptp = (maxN−minN)/N_eq ≥ 0.30`.
- **QUATERNARY** — noise-scaled damping witness: `late_amp ≤ early_amp − k·SE` (k=2, SE = bootstrap std
  of the amplitude estimate) — so it doesn't coin-flip on a truly-flat series.

**A speed is CERTIFIED-STABLE iff ≥ ⌈0.75·seeds⌉ (≥6/8) seeds pass ALL per-run gates.**

**Anti-gaming controls (binding):**
1. **OFF-path golden `events_hash` regression** — with all three new flags OFF, byte-identical to the
   committed pre-A+B goldens across the **cross** of (continuous-depletion baseline × discrete EXP194 ×
   terrain-ON). Silent-no-op guard.
2. **`freeze_continuous_locomotion` build-correctness** — freeze=True (continuous ON) skips the
   genotype.py:232 speed draw ⇒ different hash than freeze=False AND deterministic.
3. **Fix-actually-changes-behavior (ON-differs) witnesses** — `enable_density_mortality=True (hmax>0)`
   ⇒ different hash AND lower N_eq than OFF. For **B**, the witness asserts on a **continuous-resource-
   derived** quantity (`cont_world._resource` time series / intake/crowding stream), **not** the discrete
   `resource_tick` (engine.py:1262 reads `self.world.resource` and may be hash-invariant to continuous
   regen); predeclared VOID if byte-identical.
4. **`hmax=0` ON-but-null arm** — gate ON, hmax=0: rng draw still occurs (differs from OFF), pin a
   golden, require deterministic and **zero `crowding` deaths**.
5. **Recruitment-decomposed speed-neutrality null (Mechanism A trait-flatness)** — 50/50 clamped
   common garden (`s_lo=0.5, s_hi=2.0`, frozen), run crowding-ON vs crowding-OFF at matched N. **PASS
   iff turning crowding ON does not shift the slow/fast frequency trajectory beyond the OFF baseline's
   inter-seed 2-SE envelope.** Also report per-class births:crowding-deaths ratio (must be equal) and
   mean energy-at-crowding-death. (The cheap death-rate equality is kept only as a trivial sanity
   assertion — it passes by construction.)
6. **Mortality-only flat-availability arm** — crowding ON, depletion-intake OFF / uniform availability
   so the starvation gradient on speed is neutralized; the slow/fast frequency **must stay flat** — any
   drift indicts the mortality mechanism's intrinsic trait-coupling directly.
7. **A-only / B-only ablation diagonals** — neither alone should certify the full band (A-only retains
   regen dead-zones/lag; B-only lacks the brake). If either alone certifies, the "both needed" framing
   is FALSIFIED → adopt the simpler mechanism (honest DOWNGRADE). B-only also measures B's supply inflation.
8. **L40 non-degenerate-equilibrium audit (expanded)** per certified cell: (a) N_eq interior
   (`30 ≤ N_eq < 0.5·max_population`); (b) quantitative turnover above a predeclared floor; (c)
   `exploded` never fired; (d) non-degenerate depletion (mean occupied-bump availability ∈ (0.05, 0.85)
   — grazed, not reverted to non-depleting physics, not dead zones); (e) not wall-clamped (Exp-239:
   boundary-step fraction < 0.5, non-trivial position variance); (f) **spatial-mixing gate** —
   per-bump occupancy + inter-bump flux > 0; NO-GO if the equilibrium is a static spatial mosaic
   (invasion would be a priority effect); (g) **supply-budget invariant** — A+B mean availability ≤
   A+flat + 15%, else lower `regen_rate` to 0.025 and report N_eq as B-supply-specific.
9. **Fixed-density L39 expressibility precheck (common-garden)** — measure per-capita intake vs speed
   in **one shared world held near a fixed N** (all classes experience the same field); require
   **non-flat (max−min spread > 10% of mean)** AND swept distance `d=speed·dt` holds (not wall-clamped).
   (Across-equilibrium intake is a separate descriptive readout, not the gate.)
10. **Stable-band ⊇ expressed-band gate** — the GO requires a single speed (or contiguous ≥3-wide band)
    that **simultaneously** (i) is certified damped-stable, (ii) sits where the fixed-density benefit
    curve is non-flat (>10% **local** spread AT that speed), (iii) passes the non-degeneracy audit. The
    GO statement must name the overlapping band.
11. **Flat-ρ stability null** — the damped fixed point should persist under flat ρ (stability is a
    demographic property of A+B, layout-robust). If not, narrow the GO to the bump layout.
12. **Exp-239 nav-isolation layout control** — bump+nav vs flat+nav vs neutral (+ cost-neutralized
    coverage); a set-up control for the later invasion test, not a stability gate.
13. **Determinism / order-independence** — under fixed order (shuffle OFF) two same-seed ON runs are
    byte-identical; under shuffle ON only the **aggregate** crowding-death tally is preserved (N frozen),
    NOT per-creature identity or events_hash → the shuffle check is an aggregate-tally test.

## 8. Sweep grid & staging

- **Speeds (monomorphic, clamped):** `{0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}` — includes the slow
  speeds that went extinct in Exp-242 so the certification locates the stable band's lower edge.
- **Stage-1 primary grid** (θ=1, rate_scale=0): 8 speeds × hmax `{0.02,0.04,0.06,0.10}` × Kc
  `{40,60,120}` × regen `{0.025,0.05,0.10}` × 8 seeds ≈ **2304 runs**.
- **Stage-2 probes** on the Stage-1 winning region only: θ=2 arm (8 speeds × best cell × 8 ≈ 64);
  rate_scale=0.04 at the 3 fastest speeds (≈ 24).
- **Ablations** (A-only, B-only): 2 × 8 × 8 = 128. **Controls** (recruitment-decomposed null,
  mortality-only flat arm, fixed-density L39 garden, spatial-mixing/supply telemetry, hmax=0 null,
  determinism/shuffle, flat-ρ null): ≈ 400. **Total ≈ 2.9k runs @ H=4000.**
- **Runtime preflight first** (`compute-batch-runtime-preflight`): the candidate's "order seconds" is
  optimistic — `step_regen` is an O(576) Python double-loop per step (fixed cost, N-independent) plus
  the per-creature K=16 line integral. Run a staged 3-cell pilot, report **measured** per-run wall time
  (logistic-aware growth check: confirm N plateaus near N_eq, not runaway) before the full launch;
  vectorize the ON-path floored-regen if the regen loop dominates. Trivially parallel across seeds/cells.

## 9. Predeclared falsifiers (binding — frozen before any run)

- **NO-GO (substrate-still-can't-pose):** no `(hmax,θ,Kc,regen,rate_scale)` cell yields a
  CERTIFIED-STABLE result (all gates incl. return_map_slope + birth_pulse_vs_removal + the full
  oscillation detector, ≥6/8 seeds) for a **contiguous band of ≥3 adjacent speeds** at a single regime
  **that also overlaps the expressed-trait band**.
- **NO-GO (stable but not expressed — L39/L41 vacuous-GO):** a contiguous damped band exists but does
  not overlap the speeds where the fixed-density benefit curve is non-flat (>10% local spread).
- **NO-GO (still oscillatory):** level/CV gates pass but the hardened detector FAILS (AR(2) modulus ≥
  0.90, OR periodogram prominence > AR(1)-null 95th pct, OR autocorr trough ≤ −0.30, OR `A_ptp ≥ 0.30`,
  OR the noise-scaled damping witness fails).
- **NO-GO (period-2/flip):** CV/persistence pass but `|return-map slope at N_eq| ≥ 1` OR
  `|p_hazard(N_eq)+N_eq·p_hazard′(N_eq)| ≥ 0.5`.
- **NO-GO (over-damped to extinction at the slow edge):** the only (hmax,θ) that kill the fast-end
  oscillation drive all slow speeds (≤0.75) extinct.
- **NO-GO (under-damped-at-low-N, decoupled from over-damping):** a slow-speed seed FAILS the detector
  AND realized `p_hazard` over W is `< 0.01/step` (brake suppressed because N_eq ≪ Kc → the Exp-242
  starvation cliff still in control). θ=2's rare-invader-protection-vs-slow-edge-damping tension is
  reported as a substrate finding if no common Kc solves both.
- **NO-GO (mechanism-A invalid / selects on speed):** the recruitment-decomposed null shows crowding-ON
  shifts the slow/fast frequency trajectory beyond the crowd-OFF 2-SE envelope, OR births:crowding-deaths
  differs across classes, OR the mortality-only flat-availability arm shows frequency drift. (NOT the
  degenerate death-rate equality.) → A is not trait-neutral → disqualified, redesign.
- **NO-GO (degenerate stability, L40 expanded):** a cell passes the numeric bar but fails the
  non-degeneracy audit — N_eq < 30 / boundary-pinned, sub-floor turnover, stability via the
  max_population halt, availability ≥ 0.85 (reverted to non-depleting physics) or ≤ 0.05 (dead zones),
  wall-clamped (>50% boundary steps), a static spatial mosaic, or B-supply-inflation > 15% with A+flat
  NOT certifying.
- **NO-GO (inexpressible, L39 fixed-density):** at a fixed density the per-capita intake-vs-speed curve
  is flat (<10% spread).
- **NO-GO (seed-fragile):** certified N_eq disagrees across seeds beyond `(max−min)/median > 0.25`.
- **NO-GO (layout-dependent stability):** the damped fixed point does not persist under flat ρ →
  narrow the GO to the bump layout (or NO-GO if narrowing empties the expressed band).
- **DOWNGRADE / report honestly:** if A-only OR B-only already certifies the full overlapping band, the
  "both needed" framing is wrong — adopt the single sufficient mechanism and re-state.
- **GATING FAILURE (block merge):** OFF-path `events_hash` not byte-identical to the committed golden
  (full cross), OR any ON path leaves the hash unchanged on the lever it changes (silent no-op — for B,
  on a continuous-resource-derived quantity), OR `freeze=True` doesn't change the hash vs `freeze=False`,
  OR the `hmax=0` ON-but-null arm emits any `crowding` death.
- **VERDICT IS POSABILITY ONLY:** even a clean GO certifies the invasion test is now *posable* (stable,
  expressed, non-degenerate, layout-robust N_eq over a named band); it does not establish evolvability
  (L41). The GO must be stated narrowly and name the band, the energy-leak caveat (§10), and any θ-tension.

## 10. Risks / honest expectations

NO-GO is a **real, valid outcome** — this is conservative-minimal A+B. Named risks:
- **Period-2 wobble** from the snapshot-N lag + in-step birth-pulse blind spot (A is a ≥1-step lagged
  brake). Mitigated by low-hmax GO, the `rate_scale` lead term, and the AR(2)/return-map detector; a
  low-amplitude period-2 wobble near the operating point is the most likely residual failure.
- **θ tension at the slow edge:** θ=2 protects rare invaders but is quadratically weak at low N; θ=1
  damps but offers less invader protection. There may be **no common solution** — itself a reportable
  substrate finding.
- **B supply inflation** interacts with A's tuning (handled by Kc×regen cross + the supply-budget
  invariant + the 0.025 timescale-matched option + B-only ablation).
- **Static spatial mosaic** could look damped globally (guarded by the spatial-mixing gate).
- **Energy-leak channel:** crowding removes creatures at full energy without returning it to the field;
  fast movers carry more energy, so equal death-rate still imposes a larger absolute energy-investment
  leak on the fast class. Named, measured (energy-at-crowding-death per class); the future invasion test
  must control for it. (All mortality forms incl. starvation have state-dependent costs — a substrate
  property to name, not a disqualifier.)
- **Empty damped basin** is a NO-GO, not a tuning bug.
- **Two simultaneous changes (A+B)** don't self-attribute — the ablation diagonals resolve this.

## 11. Implementation surface (gated, byte-identical OFF, golden-hash guarded)

- `ecology/genotype.py` — (no new heritable trait) verify the `mutate_continuous_locomotion` skip-guard
  (≈ line 232) honors the new freeze flag.
- `ecology/engine.py` — `EcologyConfig`: add `enable_density_mortality`, `density_mortality_{theta,hmax,Kc,rate_scale}`,
  `continuous_floored_regen`, `freeze_continuous_locomotion`. Capture `N_frozen` at `step()` top
  (≈1210); change the mutate-coupling line (≈1050); add the step-7c crowding roll (after ≈1138, before
  ≈1147); thread `continuous_floored_regen` into the continuous-world construction (≈544–550).
- `ecology/continuous_world.py` — `enable_floored_regen` field + `from_config`/`__init__` arg; new
  branch in `step_regen` (before the logistic/flat selection).
- `ecology/evolvability/` — the certification instrument (stability metrics, the hardened oscillation
  detector, the controls + falsifier evaluation) as a reusable trait-agnostic stability-cert module.
- `experiments/exp243_*.py` (+ outputs, + `exp243.txt`) — the staged sweep, the birth-balance
  diagnostic, the runtime preflight pilot, blind-verified per PROTOCOL 4.5.
- `tests/` — OFF-path + ON-differs + freeze + hmax=0-null golden-hash guards.

## 12. After this build

Distill the verdict into coalescence artifacts: a `MechanismCard` (stabilized continuous
consumer–resource substrate) or extend the `local-gradient-wall` BoundaryNote. On a **GO**, the next
experiment is the **invasion-from-rarity** evolvability test on the certified band (with the energy-leak
control). On a **NO-GO**, the continuous chapter stays closed with a sharper boundary note (named
mechanism). Either way: EXPERIMENTS.md entry + direction-card STATUS update + `experiments-data.js`.
