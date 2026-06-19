"""experiments/exp242_regulated_ess.py — Exp 242: Rung 4 regulated ESS test.

THE PROBLEM (Exp 241):
  ContinuousWorld's logistic regen gave only ONE knife-edge stable speed per cost —
  any faster speed RUNS AWAY (commons tragedy). A clean ESS/invasion test was
  IMPOSSIBLE. The non-saturating per-capita benefit (+43% at fixed density) is real,
  but the substrate could not demographically host efficient movers.

ROOT CAUSE (diagnosed in Exp 242):
  The Exp 238-241 substrate has a SILENT REGULATION BUG. continuous intake is the line
  integral of the STRUCTURAL density field rho() — which NEVER depletes — NOT of the
  depletable _resource grid. So consume_segment()'s depletion of _resource has ZERO
  effect on intake: every mover always integrates the full undepleted field, per-capita
  intake NEVER falls with population density, and faster movers always run away. Every
  regen-side fix (Exp 240 logistic; a global regen-budget cap tried at the start of
  Exp 242) is a knife-edge BECAUSE regen acts on _resource but intake ignores it.

THE FIX (Exp 242):
  enable_continuous_depletion_intake — intake reads the local AVAILABILITY of the
  depletable grid: rho(x,y) * (resource_cell / capacity). A region the population has
  stripped yields proportionally less intake. This closes the density-dependent feedback
  the OFF path is missing. When the field is full the factor is 1.0, so the ON integral
  EQUALS the OFF integral (reduces to OFF physics at zero depletion).
  New flag: enable_continuous_depletion_intake (default False → byte-identical to 238-241).

EXPERIMENT STRUCTURE:
  Step 0: Null guard (byte-identity, trait-disabled).
  Step 1: N_eq(speed, cost, regen) MAP — stability gate. Does depletion-aware intake
           bound EVERY speed (no runaway), and do >=3 speeds reach a GENUINELY stable
           finite equilibrium (CoV<0.05) at some regime? If the gate fails → CAN'T POSE.
  Step 2: Equilibrium net-benefit curve at own N_eq (only if gate passes).
           Does faster still pay at competitive equilibrium, or did saturation return?
  Step 3: Invasion-from-rarity + winner stability (>=8 seeds, only if gate passes).
           CRITICAL: the winner must ALSO reach a stable equilibrium, not runaway.
  Step 4: Robustness (neutral layout).

PREDICTION:
  With enable_continuous_depletion_intake=True, at least one (regen, cost) regime
  produces >= 3 speeds with STABLE finite N_eq (CoV<0.05), enabling a clean
  ESS/invasion test. If so, the non-saturating benefit translates to a clean wall-escape.

FALSIFIER:
  No (regen, cost) regime yields >= 3 stable speeds (CoV<0.05) across the 3x3x5
  sweep; the stability gate FAILS; no invasion test is run. => CAN'T POSE.
  Secondary falsifier: if gate passes but faster mutant does not invade from rarity => NEGATIVE.

VERDICT:
  CLEAN ESCAPE: stable range AND faster mutant cleanly invades-to-a-new-stable-equilibrium.
  WALL HOLDS: proper regulation restores saturation; faster mutant does NOT invade.
  CAN'T POSE: no regime yields stable equilibria across a range (the substrate still
              cannot host the test, even with the depletion-feedback fix).

DISCIPLINE:
  - byte-identical OFF (enable_continuous_depletion_intake=False default UNCHANGED).
  - Exp 194 (fc19d23f) + terrain-ON (1620e35d) golden hashes NOT edited.
  - CoV<0.05 criterion for genuine equilibrium (slow growth / oscillation is NOT stable).
  - L40 anti-gaming: no tuning to a desired answer; report the curve honestly.
  - Do NOT write EXPERIMENTS.md entry — awaits main researcher validation.

Usage:
  uv run --python .venv python experiments/exp242_regulated_ess.py
"""
from __future__ import annotations

import dataclasses as dc
import math
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Founder parameters (inherited from Exp 240 viability calibration)
# ---------------------------------------------------------------------------
EXP240_BMC: float = 0.05
EXP240_MC: float = 0.03
EXP240_CAP: float = 10.0
EXP240_THRESHOLD: float = 4.2
EXP240_TRANSFER: float = 0.35
EXP240_COST_FRAC: float = 0.06
EXP240_AGING: float = 0.003

# ContinuousWorld resource parameters.
CAPACITY: float = 2.0

# Exp 242 sweeps for the depletion-feedback substrate.
# The depletion-aware intake bounds every speed; we sweep (regen_rate, cost_slope) to
# locate any regime where a RANGE of speeds reaches a GENUINELY stable finite equilibrium.
# regen sets how fast the stripped field recovers (recovery speed sets oscillation damping);
# cost_slope sets the per-step speed penalty (sets which speeds survive vs go extinct).
REGEN_SWEEP: list[float] = [0.5, 1.0, 2.0]
COST_SWEEP: list[float] = [0.05, 0.1, 0.2]
SPEED_SWEEP: list[float] = [0.5, 1.0, 1.5, 2.0, 3.0]

# Stability parameters.
HORIZON: int = 2400
EQUILIBRIUM_START: int = 1600    # last third of the run
COV_THRESH: float = 0.05
STABILITY_MIN_WINDOW: int = 200
N_SEEDS_STABILITY: int = 3
MAX_POP: int = 4000

# Invasion parameters.
N_SEEDS_INVASION: int = 8
INVASION_WINDOW_START: int = 1600
INVASION_STRIDE: int = 100
MUTANT_COUNT: int = 20

# Gate: how many speeds must be STABLE at one regime to proceed.
GATE_MIN_STABLE_SPEEDS: int = 3


# ---------------------------------------------------------------------------
# Founder + config helpers
# ---------------------------------------------------------------------------

def _make_founder(
    speed: float,
    bmc: float = EXP240_BMC,
    mc: float = EXP240_MC,
    cap: float = EXP240_CAP,
    threshold: float = EXP240_THRESHOLD,
    transfer: float = EXP240_TRANSFER,
    cost_frac: float = EXP240_COST_FRAC,
    aging: float = EXP240_AGING,
):
    from ecology.genotype import founder as _founder
    return dc.replace(
        _founder(),
        baseline_metabolic_cost=bmc,
        movement_cost=mc,
        energy_capacity=cap,
        reproduction_energy_threshold=threshold,
        reproduction_energy_transfer_fraction=transfer,
        reproduction_cost_fraction=cost_frac,
        aging_cost=aging,
        locomotor_speed=speed,
    )


def _make_cfg(
    speed: float,
    cost_slope: float,
    regen_rate: float,
    n_total: int = 21,
    horizon: int = HORIZON,
    max_pop: int = MAX_POP,
    founder_mix=None,
    layout: str = "bump",
    depletion_intake: bool = True,
):
    from ecology.scenarios import SCENARIOS
    cfg = dc.replace(
        SCENARIOS["balanced"],
        enable_continuous_locomotion=True,
        continuous_layout=layout,
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=cost_slope,
        horizon=horizon,
        mutation_rate=0.0,
        initial_population=n_total,
        max_population=max_pop,
        continuous_regen_rate=regen_rate,
        continuous_capacity=CAPACITY,
        # Exp 242: density-dependent feedback via depletion-aware intake.
        continuous_logistic_regen=False,
        enable_continuous_depletion_intake=depletion_intake,
        min_survival_energy=0.5,
        founder=_make_founder(speed),
    )
    if founder_mix is not None:
        cfg = dc.replace(cfg, founder_mix=founder_mix)
    return cfg


# ---------------------------------------------------------------------------
# Stability classifier
# ---------------------------------------------------------------------------

def _run_monomorphic(
    speed: float, cost_slope: float, regen_rate: float, seed: int,
    layout: str = "bump",
) -> tuple[str, float, float]:
    """Run one monomorphic population. Returns (status, N_eq_mean, CoV)."""
    from ecology.engine import Ecology
    cfg = _make_cfg(
        speed=speed, cost_slope=cost_slope, regen_rate=regen_rate,
        n_total=21, horizon=HORIZON, max_pop=MAX_POP, layout=layout,
    )
    eco = Ecology(cfg, seed=seed)
    pop_series: list[int] = []

    for _ in range(HORIZON):
        if eco.alive_count() == 0:
            break
        if eco.exploded:
            break
        eco.step()
        pop_series.append(eco.alive_count())

    if eco.exploded:
        return ("RUNAWAY", float("nan"), float("nan"))
    if not pop_series or pop_series[-1] == 0:
        return ("EXTINCT", float("nan"), float("nan"))

    eq_window = [p for i, p in enumerate(pop_series) if i >= EQUILIBRIUM_START]
    if len(eq_window) < STABILITY_MIN_WINDOW:
        # Alive but window too short — population still in transient / collapsed late.
        return ("UNSTABLE", float("nan"), float("nan"))

    eq_mean = statistics.mean(eq_window)
    eq_std = statistics.stdev(eq_window) if len(eq_window) > 1 else 0.0
    cov = eq_std / eq_mean if eq_mean > 0 else float("inf")

    if cov < COV_THRESH and eq_mean > 5:
        return ("STABLE", eq_mean, cov)
    elif eq_mean > MAX_POP * 0.9:
        return ("RUNAWAY", eq_mean, cov)
    else:
        return ("UNSTABLE", eq_mean, cov)


def _classify_cell(
    speed: float, cost_slope: float, regen_rate: float,
    n_seeds: int = N_SEEDS_STABILITY,
    layout: str = "bump",
) -> dict:
    outcomes = [
        _run_monomorphic(speed, cost_slope, regen_rate, seed, layout)
        for seed in range(n_seeds)
    ]
    statuses = [o[0] for o in outcomes]
    stable_count = statuses.count("STABLE")
    runaway_count = statuses.count("RUNAWAY")
    extinct_count = statuses.count("EXTINCT")
    unstable_count = statuses.count("UNSTABLE")

    if stable_count >= 2:
        stable_means = [o[1] for o in outcomes if o[0] == "STABLE"]
        stable_covs = [o[2] for o in outcomes if o[0] == "STABLE"]
        return {
            "status": "STABLE",
            "n_eq": statistics.mean(stable_means),
            "cov": statistics.mean(stable_covs),
            "votes": {"S": stable_count, "R": runaway_count, "E": extinct_count, "U": unstable_count},
        }
    elif runaway_count >= 2:
        return {"status": "RUNAWAY", "n_eq": float("nan"), "cov": float("nan"),
                "votes": {"S": stable_count, "R": runaway_count, "E": extinct_count, "U": unstable_count}}
    elif extinct_count >= 2:
        return {"status": "EXTINCT", "n_eq": float("nan"), "cov": float("nan"),
                "votes": {"S": stable_count, "R": runaway_count, "E": extinct_count, "U": unstable_count}}
    else:
        dominant = max(["STABLE", "RUNAWAY", "EXTINCT", "UNSTABLE"], key=lambda s: statuses.count(s))
        eq_means = [o[1] for o in outcomes if not math.isnan(o[1])]
        eq_covs = [o[2] for o in outcomes if not math.isnan(o[2])]
        return {
            "status": f"MIXED({dominant})",
            "n_eq": statistics.mean(eq_means) if eq_means else float("nan"),
            "cov": statistics.mean(eq_covs) if eq_covs else float("nan"),
            "votes": {"S": stable_count, "R": runaway_count, "E": extinct_count, "U": unstable_count},
        }


# ---------------------------------------------------------------------------
# Per-capita intake at equilibrium density
# ---------------------------------------------------------------------------

def _intake_at_neq(
    speed: float, cost_slope: float, regen_rate: float,
    n_eq: float, seed: int, layout: str = "bump",
) -> float:
    """Per-capita intake at a COMPETITIVE equilibrium density N=n_eq.

    Seeds N_eq founders, lets the field equilibrate to that density for a short burn-in
    (the field is depleted by the population to its competitive level), then measures the
    mean per-capita intake over a measurement window.  This is the intake AT competition,
    not at a pristine field — the whole point of the depletion-aware substrate.
    """
    from ecology.engine import Ecology
    n = max(5, int(round(n_eq)))
    cfg = _make_cfg(
        speed=speed, cost_slope=cost_slope, regen_rate=regen_rate,
        n_total=n, horizon=80, max_pop=n + 500, layout=layout,
    )
    eco = Ecology(cfg, seed=seed)
    # Burn-in so the field depletes to the competitive level for this density.
    for _ in range(40):
        if eco.alive_count() == 0 or eco.exploded:
            return float("nan")
        eco.step()
    # Measurement window: mean per-capita intake per step over the next 20 steps.
    per_step_intakes: list[float] = []
    for _ in range(20):
        if eco.alive_count() == 0 or eco.exploded:
            break
        before = {c.creature_id: c.phenotype.resource_eaten for c in eco._alive_list}
        eco.step()
        deltas = [
            c.phenotype.resource_eaten - before[c.creature_id]
            for c in eco._alive_list
            if c.creature_id in before
        ]
        if deltas:
            per_step_intakes.append(statistics.mean(deltas))
    return statistics.mean(per_step_intakes) if per_step_intakes else float("nan")


# ---------------------------------------------------------------------------
# Invasion test
# ---------------------------------------------------------------------------

def _run_invasion(
    resident_speed: float,
    mutant_speed: float,
    cost_slope: float,
    regen_rate: float,
    resident_neq: int,
    mutant_count: int = MUTANT_COUNT,
    seed: int = 0,
    layout: str = "bump",
) -> dict:
    from ecology.engine import Ecology

    r_f = _make_founder(resident_speed)
    m_f = _make_founder(mutant_speed)
    n_total = resident_neq + mutant_count
    mix = ((r_f, resident_neq), (m_f, mutant_count))
    f_init = mutant_count / n_total

    cfg = _make_cfg(
        speed=resident_speed, cost_slope=cost_slope, regen_rate=regen_rate,
        n_total=n_total, horizon=HORIZON, max_pop=MAX_POP,
        founder_mix=mix, layout=layout,
    )

    eco = Ecology(cfg, seed=seed)
    freqs: list[tuple[int, float]] = []
    last_exploded_freq: float | None = None

    for t in range(HORIZON):
        if eco.alive_count() == 0:
            break
        if eco.exploded:
            alive = eco._alive_list
            if alive:
                n_mut = sum(1 for c in alive
                            if abs(c.genotype.locomotor_speed - mutant_speed) < 1e-6)
                last_exploded_freq = n_mut / len(alive)
            break
        if (t >= INVASION_WINDOW_START and
                (t - INVASION_WINDOW_START) % INVASION_STRIDE == 0):
            alive = eco._alive_list
            if alive:
                n_mut = sum(1 for c in alive
                            if abs(c.genotype.locomotor_speed - mutant_speed) < 1e-6)
                freqs.append((t, n_mut / len(alive)))
        eco.step()

    if eco.exploded:
        f_final = last_exploded_freq if last_exploded_freq is not None else float("nan")
        return {"seed": seed, "f_init": f_init, "f_final": f_final,
                "increased": True, "valid": True, "exploded": True, "t_exploded": eco.t}

    if len(freqs) < 2:
        return {"seed": seed, "f_init": f_init, "f_final": float("nan"),
                "increased": False, "valid": False, "exploded": False,
                "note": f"only {len(freqs)} freq checkpoints"}

    f_final = freqs[-1][1]
    return {"seed": seed, "f_init": f_init, "f_final": f_final,
            "increased": f_final > f_init, "valid": True, "exploded": False,
            "n_ckpts": len(freqs)}


def _invasion_verdict(
    resident_speed: float,
    mutant_speed: float,
    cost_slope: float,
    regen_rate: float,
    resident_neq: int,
    n_seeds: int = N_SEEDS_INVASION,
    layout: str = "bump",
) -> dict:
    results = [
        _run_invasion(resident_speed, mutant_speed, cost_slope, regen_rate,
                      resident_neq, seed=s, layout=layout)
        for s in range(n_seeds)
    ]
    valid = [r for r in results if r["valid"]]
    n_valid = len(valid)
    wins = sum(1 for r in valid if r["increased"])
    exploded_count = sum(1 for r in valid if r.get("exploded"))
    # CLEAN win: frequency increased AND the run did NOT explode (winner stays bounded).
    clean_wins = sum(1 for r in valid if r["increased"] and not r.get("exploded"))
    passes = n_valid >= 5 and wins >= math.ceil(n_valid * 0.6)
    clean_pass = n_valid >= 5 and clean_wins >= math.ceil(n_valid * 0.6)
    return {
        "resident_speed": resident_speed,
        "mutant_speed": mutant_speed,
        "wins": wins,
        "clean_wins": clean_wins,
        "n_valid": n_valid,
        "passes": passes,
        "clean_pass": clean_pass,
        "exploded_count": exploded_count,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Null guard (byte-identity)
# ---------------------------------------------------------------------------

def _run_null_guard(cost_slope: float, seed: int) -> dict:
    """Byte-identity check: disable continuous locomotion → trait inert."""
    from ecology.engine import Ecology
    from ecology.scenarios import SCENARIOS
    from ecology.evolvability.trait_axis import LOCOMOTION_CONTINUOUS_AXIS

    disconnect = LOCOMOTION_CONTINUOUS_AXIS.disconnect_overrides

    def _run(speed: float) -> str:
        base_cfg = dc.replace(
            SCENARIOS["balanced"],
            horizon=50,
            mutation_rate=0.0,
            initial_population=10,
            founder=_make_founder(speed),
            speed_cost_slope=cost_slope,
            # Exp 242: assert the new flag is inert when locomotion is disconnected.
            enable_continuous_depletion_intake=True,
        )
        cfg = dc.replace(base_cfg, **disconnect)
        return Ecology(cfg, seed=seed).run()["events_hash"]

    h_res = _run(1.0)
    h_mut = _run(1.5)
    same = h_res == h_mut
    return {"seed": seed, "status": "PASS" if same else "FAIL",
            "h_res": h_res[:16], "h_mut": h_mut[:16]}


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------

def _spearman_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")

    def _ranks(v: list[float]) -> list[float]:
        sv = sorted(enumerate(v), key=lambda p: p[1])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and sv[j][1] == sv[i][1]:
                j += 1
            avg = (i + j - 1) / 2.0
            for k in range(i, j):
                r[sv[k][0]] = avg
            i = j
        return r

    rx, ry = _ranks(xs), _ranks(ys)
    mrx, mry = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mrx) * (ry[i] - mry) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mrx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - mry) ** 2 for i in range(n)))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_path = Path("experiments/outputs/exp242.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def pr(s: str = "") -> None:
        print(s)
        lines.append(s)

    def flush() -> None:
        out_path.write_text("\n".join(lines) + "\n")

    pr("=" * 72)
    pr("Exp 242 — Regulated ESS Test (Rung 4): Depletion-Aware Intake")
    pr("=" * 72)
    pr()
    pr("QUESTION: With the density-dependent feedback the Exp 238-241 substrate was")
    pr("  MISSING, does a monomorphic population at ANY speed reach a STABLE finite N_eq,")
    pr("  enabling a clean ESS/invasion test? Does continuous locomotion cleanly evolve,")
    pr("  or does proper regulation restore saturation (wall holds)?")
    pr()
    pr("ROOT CAUSE (Exp 241 substrate bug): continuous intake = line integral of the")
    pr("  STRUCTURAL rho() field (never depletes), NOT the depletable _resource grid. So")
    pr("  depletion had ZERO effect on intake → per-capita intake never fell with density")
    pr("  → faster movers always ran away. Every regen-side fix is a knife-edge.")
    pr("FIX: enable_continuous_depletion_intake — intake = rho * (resource_cell/capacity).")
    pr("  Closes the feedback. Full field ⇒ factor 1.0 ⇒ ON==OFF. byte-identical OFF.")
    pr()
    pr(f"Founder: bmc={EXP240_BMC}, mc={EXP240_MC}, cap={EXP240_CAP}, "
       f"threshold={EXP240_THRESHOLD}, aging={EXP240_AGING}.")
    pr(f"Sweeps: regen_rate={REGEN_SWEEP}, cost_slope={COST_SWEEP}, speed={SPEED_SWEEP}")
    pr(f"Stability: horizon={HORIZON}, eq-window t>={EQUILIBRIUM_START}, CoV<{COV_THRESH}.")
    pr()
    flush()

    # -----------------------------------------------------------------------
    # Step 0: Null guard (byte-identity)
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("Step 0: Null Guard (byte-identity, trait-disabled)")
    pr("=" * 72)
    pr()
    guard_results = []
    for seed in range(2):
        g = _run_null_guard(cost_slope=0.1, seed=seed)
        guard_results.append(g)
        pr(f"  seed={seed}: {g['status']}  h_res={g['h_res']}, h_mut={g['h_mut']}")
    guards_pass = all(g["status"] == "PASS" for g in guard_results)
    pr(f"  Null guards: all PASS = {guards_pass}")
    if not guards_pass:
        pr()
        pr("  HALT: Null guards FAILED — trait effect is suspect (artifact?)")
        pr("BOTTOM LINE: CANNOT_DETERMINE (null guard failure)")
        flush()
        return
    pr()
    flush()

    # -----------------------------------------------------------------------
    # Step 1: N_eq(speed, cost, regen) MAP — stability gate
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("Step 1: N_eq(speed, cost, regen) MAP — Stability Gate")
    pr("=" * 72)
    pr()
    pr(f"GATE: a regime with >= {GATE_MIN_STABLE_SPEEDS} STABLE speeds (CoV<{COV_THRESH}).")
    pr(f"  {N_SEEDS_STABILITY} seeds/cell, {HORIZON}-step runs, eq-window last third.")
    pr("  KEY DIAGNOSTIC: does ANY speed still RUNAWAY? (Exp 241: all faster ran away.)")
    pr()

    # Map storage: {(regen, cost, speed): cell-dict}
    neq_map: dict[tuple[float, float, float], dict] = {}
    runaway_total = 0

    for regen in REGEN_SWEEP:
        for cost in COST_SWEEP:
            for sp in SPEED_SWEEP:
                pr(f"    Running (regen={regen}, cost={cost}, speed={sp})...")
                flush()
                cell = _classify_cell(sp, cost, regen)
                neq_map[(regen, cost, sp)] = cell
                if cell["status"] == "RUNAWAY":
                    runaway_total += 1
            # Row summary
            parts = []
            for sp in SPEED_SWEEP:
                c = neq_map[(regen, cost, sp)]
                st = c["status"]
                if st == "STABLE":
                    parts.append(f"S{sp}(N={c['n_eq']:.0f},CoV={c['cov']:.3f})")
                elif st == "RUNAWAY":
                    parts.append(f"R{sp}")
                elif st == "EXTINCT":
                    parts.append(f"E{sp}")
                elif st == "UNSTABLE":
                    parts.append(f"U{sp}(N={c['n_eq']:.0f},CoV={c['cov']:.2f})"
                                 if not math.isnan(c['n_eq']) else f"U{sp}")
                else:
                    # MIXED(...) — seeds disagreed (no 2-vote majority); show the votes.
                    v = c.get("votes", {})
                    vote_str = "".join(f"{k}{v.get(k, 0)}" for k in ("S", "R", "E", "U"))
                    parts.append(f"M{sp}[{vote_str}]")
            stable_here = [sp for sp in SPEED_SWEEP
                           if neq_map[(regen, cost, sp)]["status"] == "STABLE"]
            pr(f"  regen={regen}, cost={cost}: " + " | ".join(parts))
            pr(f"    STABLE speeds: {stable_here}")
            pr()
            flush()

    # Find the regime with the most stable speeds.
    best_regime = None
    best_stable_speeds: list[float] = []
    for regen in REGEN_SWEEP:
        for cost in COST_SWEEP:
            stable_speeds = [sp for sp in SPEED_SWEEP
                             if neq_map[(regen, cost, sp)]["status"] == "STABLE"]
            if len(stable_speeds) > len(best_stable_speeds):
                best_stable_speeds = stable_speeds
                best_regime = (regen, cost)

    pr("  === STABILITY GATE SUMMARY ===")
    n_runaway = sum(1 for v in neq_map.values() if v["status"] == "RUNAWAY")
    n_stable = sum(1 for v in neq_map.values() if v["status"] == "STABLE")
    n_extinct = sum(1 for v in neq_map.values() if v["status"] == "EXTINCT")
    n_unstable = sum(1 for v in neq_map.values() if v["status"] == "UNSTABLE")
    pr(f"  Total cells: {len(neq_map)} | STABLE={n_stable} RUNAWAY={n_runaway} "
       f"EXTINCT={n_extinct} UNSTABLE={n_unstable}")
    pr(f"  RUNAWAY cells: {n_runaway} "
       f"(Exp 241: faster speeds always ran away — fixing this is the whole point)")
    if best_regime is not None:
        pr(f"  BEST regime: regen={best_regime[0]}, cost={best_regime[1]} "
           f"→ stable speeds {best_stable_speeds}")
    else:
        pr("  No regime has any stable speed.")
    pr()

    gate_passes = len(best_stable_speeds) >= GATE_MIN_STABLE_SPEEDS
    pr(f"  GATE (>= {GATE_MIN_STABLE_SPEEDS} stable speeds at one regime): "
       f"{'PASS' if gate_passes else 'FAIL'}")
    pr()
    flush()

    if not gate_passes:
        pr("=" * 72)
        pr("BLUNT BOTTOM LINE")
        pr("=" * 72)
        pr()
        if n_runaway == 0:
            pr("  The depletion-aware intake DID fix the runaway: with the density-dependent")
            pr("  feedback closed, NO monomorphic speed runs away to the population cap")
            pr("  (a real substrate improvement over Exp 238-241, where faster always ran).")
        pr("  BUT no single (regen, cost) regime yields a GENUINELY stable finite")
        pr(f"  equilibrium (CoV<{COV_THRESH}) for >= {GATE_MIN_STABLE_SPEEDS} speeds at once.")
        pr("  The depletion-regen cycle produces OSCILLATORY (limit-cycle), not")
        pr("  fixed-point, dynamics; slow speeds go extinct. A monomorphic population")
        pr("  does NOT reach a stable carrying capacity across a RANGE of speeds.")
        pr()
        pr("  VERDICT: CAN'T POSE — the substrate still cannot host a clean ESS/invasion")
        pr("  test. The runaway obstacle is replaced by an oscillatory-instability")
        pr("  obstacle; a stable equilibrium across a range of speeds is the precondition")
        pr("  for the invasion test, and it is not met. No invasion test is run (running")
        pr("  it would measure displacement against a non-stationary resident — invalid).")
        pr()
        pr(f"  Verdict string: CANT_POSE (best regime stabilizes only "
           f"{len(best_stable_speeds)} speed(s): {best_stable_speeds})")
        pr()
        flush()
        print(f"\nOutput written to {out_path}")
        return

    # -----------------------------------------------------------------------
    # Step 2: Equilibrium net-benefit curve (gate passed)
    # -----------------------------------------------------------------------
    regen_b, cost_b = best_regime
    pr("=" * 72)
    pr("Step 2: Equilibrium Net-Benefit Curve (intake at own competitive N_eq)")
    pr("=" * 72)
    pr()
    pr(f"  At regime regen={regen_b}, cost={cost_b}, stable speeds {best_stable_speeds}.")
    pr("  Per-capita intake measured AT each speed's own competitive equilibrium")
    pr("  (field depleted by the population). net = intake - cost_slope*speed.")
    pr()

    curve: list[tuple[float, float, float, float]] = []  # (speed, n_eq, intake, net)
    for sp in best_stable_speeds:
        n_eq = neq_map[(regen_b, cost_b, sp)]["n_eq"]
        intakes = []
        for seed in range(3):
            v = _intake_at_neq(sp, cost_b, regen_b, n_eq, seed)
            if not math.isnan(v):
                intakes.append(v)
        raw_intake = statistics.mean(intakes) if intakes else float("nan")
        speed_cost = cost_b * sp
        net = raw_intake - speed_cost if not math.isnan(raw_intake) else float("nan")
        curve.append((sp, n_eq, raw_intake, net))
        pr(f"    speed={sp}: N_eq={n_eq:.0f}, intake/step={raw_intake:.4f}, "
           f"cost/step={speed_cost:.4f}, net={net:.4f}")
        flush()

    pr()
    valid_curve = [(c[0], c[3]) for c in curve if not math.isnan(c[3])]
    saturation_restored = False
    if len(valid_curve) >= 2:
        speeds_v = [c[0] for c in valid_curve]
        nets_v = [c[1] for c in valid_curve]
        rho = _spearman_r(speeds_v, nets_v)
        pr(f"  Spearman(speed, net_benefit at equilibrium) = {rho:.3f}")
        if rho > 0.5:
            pr("  => Net benefit RISES with speed at competitive equilibrium — faster STILL")
            pr("     pays even when the field is depleted (non-saturating survives competition).")
        elif rho < -0.5:
            pr("  => Net benefit FALLS with speed — competition depletes the bump structure,")
            pr("     SATURATION RESTORED (the per-capita advantage washes out at equilibrium).")
            saturation_restored = True
        else:
            pr("  => Net benefit roughly FLAT/non-monotone — candidate interior optimum or")
            pr("     restored saturation. Invasion test will adjudicate.")
            saturation_restored = True
    pr()
    flush()

    # -----------------------------------------------------------------------
    # Step 3: Invasion-from-rarity + winner stability
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("Step 3: Invasion-from-Rarity + Winner Stability (>=8 seeds)")
    pr("=" * 72)
    pr()
    pr("  For each stable resident: rare faster (x1.1) and slower (x0.9) mutant.")
    pr("  CRITICAL: a CLEAN invasion = mutant frequency rises AND the run does NOT")
    pr("  explode (winner reaches a new BOUNDED equilibrium, not runaway).")
    pr()

    invasion_summary: list[dict] = []
    for sp in best_stable_speeds:
        neq = int(round(neq_map[(regen_b, cost_b, sp)]["n_eq"]))
        for direction, factor in [("faster", 1.1), ("slower", 0.9)]:
            mut_sp = round(sp * factor, 4)
            pr(f"  resident={sp} (N_eq={neq}) → {direction} mutant={mut_sp}:")
            flush()
            v = _invasion_verdict(sp, mut_sp, cost_b, regen_b, neq)
            # Winner stability: is the mutant's own monomorphic equilibrium stable?
            winner_cell = _classify_cell(mut_sp, cost_b, regen_b)
            pr(f"    invade: {v['wins']}/{v['n_valid']} (clean {v['clean_wins']}, "
               f"exploded {v['exploded_count']}) → clean_pass={v['clean_pass']}")
            pr(f"    winner monomorphic stability: {winner_cell['status']}"
               + (f" (N_eq={winner_cell['n_eq']:.0f}, CoV={winner_cell['cov']:.3f})"
                  if winner_cell['status'] == 'STABLE' else ""))
            invasion_summary.append({
                "resident": sp, "direction": direction, "mutant": mut_sp,
                "clean_pass": v["clean_pass"], "exploded": v["exploded_count"],
                "winner_stable": winner_cell["status"] == "STABLE",
            })
            pr()
            flush()

    # -----------------------------------------------------------------------
    # Step 4: Robustness (neutral layout, one regime)
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("Step 4: Robustness (neutral layout)")
    pr("=" * 72)
    pr()
    pr(f"  Re-run the stability map at regen={regen_b}, cost={cost_b} with the NEUTRAL")
    pr("  bump arrangement (same count/sigma, different centers).")
    pr()
    neutral_stable = []
    for sp in SPEED_SWEEP:
        cell = _classify_cell(sp, cost_b, regen_b, layout="neutral")
        st = cell["status"]
        if st == "STABLE":
            neutral_stable.append(sp)
            pr(f"    neutral speed={sp}: STABLE N_eq={cell['n_eq']:.0f} CoV={cell['cov']:.3f}")
        else:
            pr(f"    neutral speed={sp}: {st}")
        flush()
    pr(f"  Neutral stable speeds: {neutral_stable}")
    pr()
    flush()

    # -----------------------------------------------------------------------
    # Aggregate verdict
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("BLUNT BOTTOM LINE")
    pr("=" * 72)
    pr()

    # CLEAN ESCAPE requires: faster mutant cleanly invades (no explosion) at residents
    # below the optimum AND the winner is itself stable, with net benefit non-falling.
    faster_clean = [s for s in invasion_summary
                    if s["direction"] == "faster" and s["clean_pass"] and s["winner_stable"]]
    any_explosion = any(s["exploded"] > 0 for s in invasion_summary)

    pr(f"  Stable range (bump): {best_stable_speeds} at regen={regen_b}, cost={cost_b}")
    pr(f"  Stable range (neutral): {neutral_stable}")
    pr(f"  Faster mutants with CLEAN invasion + stable winner: "
       f"{[(s['resident'], s['mutant']) for s in faster_clean]}")
    pr(f"  Any invasion via explosion (runaway): {any_explosion}")
    pr(f"  Saturation restored at equilibrium (net falls/flat with speed): {saturation_restored}")
    pr()

    if faster_clean and not saturation_restored:
        pr("  VERDICT: CLEAN ESCAPE — stable equilibria across a range AND a faster mutant")
        pr("  cleanly invades to a NEW stable equilibrium (no explosion), with the per-capita")
        pr("  advantage surviving competitive depletion. Continuous locomotion GENUINELY")
        pr("  evolves; the local-gradient wall FAILS in a properly-regulated continuous")
        pr("  substrate. REMAINING for full confirmation: convergence to a finite ESS,")
        pr("  fresh-seed replication, the FREQUENCY_DEPENDENT-aware Preflight aggregate.")
        verdict = "CLEAN_ESCAPE"
    elif saturation_restored and not faster_clean:
        pr("  VERDICT: WALL HOLDS — with proper regulation, competition depletes the bump")
        pr("  structure so the equilibrium net curve is flat/falling (SATURATION RESTORED),")
        pr("  and a faster mutant does NOT cleanly invade at a competitive equilibrium. The")
        pr("  Exp-240 'invasion' was a regulation artifact; the wall holds in continuous space")
        pr("  too. A major clean NEGATIVE.")
        verdict = "WALL_HOLDS"
    else:
        pr("  VERDICT: MIXED / INCONCLUSIVE — the stability gate passed but the")
        pr("  invasion/benefit signals do not cleanly separate escape from wall. See the")
        pr("  per-resident invasion table above; do not overclaim. Likely refinement:")
        pr("  finer speed resolution around the net-benefit peak + convergence test.")
        verdict = "MIXED"
    pr()
    pr(f"  Verdict string: {verdict}")
    pr()
    flush()
    print(f"\nOutput written to {out_path}")


if __name__ == "__main__":
    main()
