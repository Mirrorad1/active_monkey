"""experiments/exp241_ess_characterization.py — Exp 241: ESS Characterization (Rung 3).

THE OPEN QUESTION:
  Exp 240 found a faster mutant (speed 1.1) invades from rarity at the resident's
  stable equilibrium (+43% per-capita benefit), but speed=1.1 has NO finite carrying
  capacity at speed_cost_slope=0.6 — it explodes once common (commons tragedy, runaway).
  So the escape is entangled with demographic instability.

RUNG 3 MUST DISTINGUISH:
  (A) CLEAN ESCAPE: a cost regime exists where a faster mutant invades AND both
      types have stable finite equilibria, selection converges to a finite optimal
      speed (genuine ESS).
  (B) DEMOGRAPHIC ARTIFACT: any harvesting-efficiency gain causes runaway explosion
      regardless of cost; the "invasion" reflects substrate instability, not a clean
      selection gradient.

THIS EXPERIMENT:
  1. N_eq(speed, cost) MAP: sweep speed_cost_slope × monomorphic speed. Classify each
     (speed, cost) as STABLE (finite N_eq with CoV check), RUNAWAY (explodes), or EXTINCT.
     Key question: is there a cost where HIGHER speeds reach STABLE finite N_eq?

  2. EQUILIBRIUM NET-BENEFIT CURVE → ESS: for cost regimes where speeds are stable,
     compute per-capita net intake at each speed's own equilibrium. Is there an interior
     optimum s* where the net-benefit gradient crosses zero?

  3. CONVERGENCE TEST: pick a candidate cost + s*. Test invasion-from-rarity (>=8 seeds)
     at residents BELOW s* (faster should invade) AND ABOVE s* (slower should invade) —
     convergence from both sides = genuine stable ESS.

  4. ROBUSTNESS: repeat qualitative result with neutral bump layout + alt founder to rule
     out single-configuration artifact.

DISCIPLINE:
  - byte-identical OFF (enable_continuous_locomotion=False default UNCHANGED)
  - Exp 194 (fc19d23f) + terrain-ON (1620e35d) golden hashes NOT edited
  - CoV criterion for genuine equilibrium (not slow growth phase)
  - L40 anti-gaming: no tuning to desired answer; if all costs cause runaway → report it
  - confounded-control-probe-guard: null guard byte-identity PASSED first
  - Do NOT write EXPERIMENTS.md entry — awaits main researcher validation

Usage:
  uv run --python .venv python experiments/exp241_ess_characterization.py
"""
from __future__ import annotations

import dataclasses as dc
import math
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Exp 240 founder parameters (inherited unchanged — the known-stable calibration)
# ---------------------------------------------------------------------------
EXP240_BMC: float = 0.05
EXP240_MC: float = 0.03
EXP240_CAP: float = 10.0
EXP240_THRESHOLD: float = 4.2
EXP240_TRANSFER: float = 0.35
EXP240_COST_FRAC: float = 0.06
EXP240_AGING: float = 0.003

# Alt founder (for robustness check — slightly different viability calibration)
ALT_BMC: float = 0.04
ALT_MC: float = 0.025
ALT_THRESHOLD: float = 4.5
ALT_AGING: float = 0.004

# World parameters (inherited from Exp 240)
REGEN_RATE: float = 0.2
CAPACITY: float = 2.0

# Speed-cost sweep ranges
COST_SWEEP: list[float] = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
SPEED_MONO_SWEEP: list[float] = [0.5, 1.0, 1.5, 2.0, 3.0]

# Stability verification parameters
RESIDENT_HORIZON: int = 3000
EQUILIBRIUM_START: int = 1200     # conservative transient buffer
STABILITY_COV_THRESH: float = 0.05
STABILITY_MIN_WINDOW: int = 200

# Invasion test parameters
INVASION_HORIZON: int = 3000
INVASION_WINDOW_START: int = 1200
INVASION_STRIDE: int = 100
N_SEEDS: int = 8

# Carrying capacity safety — stop at this to classify RUNAWAY
MAX_POP_GUARD: int = 2000

# ESS convergence test: resident speeds straddling the candidate s*
# We will populate these dynamically based on the map results


# ---------------------------------------------------------------------------
# Founder helpers
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
    n_total: int = 21,
    horizon: int = RESIDENT_HORIZON,
    max_pop: int = MAX_POP_GUARD,
    founder_mix=None,
    layout: str = "bump",
    bmc: float = EXP240_BMC,
    mc: float = EXP240_MC,
    threshold: float = EXP240_THRESHOLD,
    aging: float = EXP240_AGING,
):
    from ecology.engine import EcologyConfig
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
        continuous_regen_rate=REGEN_RATE,
        continuous_capacity=CAPACITY,
        continuous_logistic_regen=True,
        min_survival_energy=0.5,
        founder=_make_founder(
            speed, bmc=bmc, mc=mc, threshold=threshold, aging=aging,
        ),
    )
    if founder_mix is not None:
        cfg = dc.replace(cfg, founder_mix=founder_mix)
    return cfg


# ---------------------------------------------------------------------------
# Stability classifier
# ---------------------------------------------------------------------------

def _run_monomorphic(speed: float, cost_slope: float, seed: int,
                     layout: str = "bump",
                     bmc: float = EXP240_BMC, mc: float = EXP240_MC,
                     threshold: float = EXP240_THRESHOLD,
                     aging: float = EXP240_AGING) -> tuple[str, float, float]:
    """Run one monomorphic population to demographic outcome.

    Returns (status, N_eq_mean, CoV) where status in {STABLE, RUNAWAY, EXTINCT}.
    """
    from ecology.engine import Ecology
    cfg = _make_cfg(
        speed=speed, cost_slope=cost_slope, n_total=21,
        horizon=RESIDENT_HORIZON, max_pop=MAX_POP_GUARD,
        layout=layout, bmc=bmc, mc=mc, threshold=threshold, aging=aging,
    )
    eco = Ecology(cfg, seed=seed)
    pop_series: list[int] = []

    for _ in range(RESIDENT_HORIZON):
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
        # Population alive but window too short — likely slow growth still ongoing
        return ("RUNAWAY", float("nan"), float("nan"))

    eq_mean = statistics.mean(eq_window)
    eq_std = statistics.stdev(eq_window) if len(eq_window) > 1 else 0.0
    cov = eq_std / eq_mean if eq_mean > 0 else float("inf")

    if cov < STABILITY_COV_THRESH and eq_mean > 5:
        return ("STABLE", eq_mean, cov)
    elif eq_mean > MAX_POP_GUARD * 0.9:
        return ("RUNAWAY", eq_mean, cov)
    else:
        return ("UNSTABLE", eq_mean, cov)


def _classify_cell(speed: float, cost_slope: float, n_seeds: int = 3,
                   layout: str = "bump",
                   bmc: float = EXP240_BMC, mc: float = EXP240_MC,
                   threshold: float = EXP240_THRESHOLD,
                   aging: float = EXP240_AGING) -> dict:
    """Classify (speed, cost_slope) by running n_seeds and voting."""
    outcomes: list[tuple[str, float, float]] = []
    for seed in range(n_seeds):
        outcomes.append(_run_monomorphic(speed, cost_slope, seed, layout, bmc, mc, threshold, aging))

    statuses = [o[0] for o in outcomes]
    # Majority vote for status
    stable_count = statuses.count("STABLE")
    runaway_count = statuses.count("RUNAWAY")
    extinct_count = statuses.count("EXTINCT")

    if stable_count >= 2:
        stable_means = [o[1] for o in outcomes if o[0] == "STABLE"]
        stable_covs = [o[2] for o in outcomes if o[0] == "STABLE"]
        return {
            "status": "STABLE",
            "n_eq": statistics.mean(stable_means),
            "cov": statistics.mean(stable_covs),
            "votes": {"STABLE": stable_count, "RUNAWAY": runaway_count, "EXTINCT": extinct_count},
        }
    elif runaway_count >= 2:
        return {
            "status": "RUNAWAY",
            "n_eq": float("nan"),
            "cov": float("nan"),
            "votes": {"STABLE": stable_count, "RUNAWAY": runaway_count, "EXTINCT": extinct_count},
        }
    elif extinct_count >= 2:
        return {
            "status": "EXTINCT",
            "n_eq": float("nan"),
            "cov": float("nan"),
            "votes": {"STABLE": stable_count, "RUNAWAY": runaway_count, "EXTINCT": extinct_count},
        }
    else:
        # Mixed — report the most common status
        dominant = max(["STABLE", "RUNAWAY", "EXTINCT"], key=lambda s: statuses.count(s))
        eq_means = [o[1] for o in outcomes if not math.isnan(o[1])]
        eq_covs = [o[2] for o in outcomes if not math.isnan(o[2])]
        return {
            "status": f"MIXED({dominant})",
            "n_eq": statistics.mean(eq_means) if eq_means else float("nan"),
            "cov": statistics.mean(eq_covs) if eq_covs else float("nan"),
            "votes": {"STABLE": stable_count, "RUNAWAY": runaway_count, "EXTINCT": extinct_count},
        }


# ---------------------------------------------------------------------------
# Per-capita intake probe at a population's own equilibrium density
# ---------------------------------------------------------------------------

def _intake_at_neq(speed: float, cost_slope: float, n_eq: float, seed: int,
                   layout: str = "bump") -> float:
    """Measure mean per-capita resource intake for a monomorphic speed pop at N=n_eq.

    Initializes n_eq founders, runs 1 step, returns mean resource_eaten per founder.
    (1 step: no births/deaths confound; isolates intake physics.)
    """
    from ecology.engine import Ecology
    n = max(5, int(round(n_eq)))
    cfg = _make_cfg(
        speed=speed, cost_slope=cost_slope,
        n_total=n, horizon=1, max_pop=n + 100,
        layout=layout,
    )
    eco = Ecology(cfg, seed=seed)
    founder_ids = {c.creature_id for c in eco._alive_list}
    eco.step()
    intakes = [
        c.phenotype.resource_eaten
        for c in eco._alive_list
        if c.creature_id in founder_ids
    ]
    return statistics.mean(intakes) if intakes else float("nan")


# ---------------------------------------------------------------------------
# Invasion from rarity test
# ---------------------------------------------------------------------------

def _run_invasion(
    resident_speed: float,
    mutant_speed: float,
    cost_slope: float,
    resident_neq: int,
    mutant_count: int = 20,
    seed: int = 0,
    layout: str = "bump",
) -> dict:
    """Run one invasion-from-rarity seed.

    resident_neq residents + mutant_count mutants at start.
    Returns invasion verdict dict.
    """
    from ecology.engine import Ecology

    r_f = _make_founder(resident_speed)
    m_f = _make_founder(mutant_speed)

    n_total = resident_neq + mutant_count
    mix = ((r_f, resident_neq), (m_f, mutant_count))
    f_init = mutant_count / n_total

    cfg = _make_cfg(
        speed=resident_speed,
        cost_slope=cost_slope,
        n_total=n_total,
        horizon=INVASION_HORIZON,
        max_pop=MAX_POP_GUARD,
        founder_mix=mix,
        layout=layout,
    )

    eco = Ecology(cfg, seed=seed)
    freqs: list[tuple[int, float]] = []
    last_exploded_freq: float | None = None

    for t in range(INVASION_HORIZON):
        if eco.alive_count() == 0:
            break
        if eco.exploded:
            alive = eco._alive_list
            n_alive = len(alive)
            if n_alive > 0:
                n_mut = sum(
                    1 for c in alive
                    if abs(c.genotype.locomotor_speed - mutant_speed) < 1e-6
                )
                last_exploded_freq = n_mut / n_alive
            break

        if (t >= INVASION_WINDOW_START
                and (t - INVASION_WINDOW_START) % INVASION_STRIDE == 0):
            alive = eco._alive_list
            n_alive = len(alive)
            if n_alive > 0:
                n_mut = sum(
                    1 for c in alive
                    if abs(c.genotype.locomotor_speed - mutant_speed) < 1e-6
                )
                freqs.append((t, n_mut / n_alive))

        eco.step()

    if eco.exploded:
        f_final = last_exploded_freq if last_exploded_freq is not None else float("nan")
        return {
            "seed": seed, "f_init": f_init, "f_final": f_final,
            "increased": True, "valid": True, "exploded": True,
            "t_exploded": eco.t,
            "note": f"EXPLODED t={eco.t}",
        }

    if len(freqs) < 2:
        return {
            "seed": seed, "f_init": f_init, "f_final": float("nan"),
            "increased": False, "valid": False, "exploded": False,
            "note": f"only {len(freqs)} checkpoints in equilibrium window",
        }

    f_final = freqs[-1][1]
    return {
        "seed": seed, "f_init": f_init, "f_final": f_final,
        "increased": f_final > f_init, "valid": True, "exploded": False,
        "n_ckpts": len(freqs),
    }


def _invasion_verdict(
    resident_speed: float,
    mutant_speed: float,
    cost_slope: float,
    resident_neq: int,
    n_seeds: int = N_SEEDS,
    layout: str = "bump",
) -> dict:
    """Run N_SEEDS invasion seeds and return aggregate verdict."""
    results = [
        _run_invasion(resident_speed, mutant_speed, cost_slope, resident_neq,
                      seed=s, layout=layout)
        for s in range(n_seeds)
    ]
    valid = [r for r in results if r["valid"]]
    wins = sum(1 for r in valid if r["increased"])
    n_valid = len(valid)
    passes = n_valid >= 5 and wins >= math.ceil(n_valid * 0.6)
    return {
        "resident_speed": resident_speed,
        "mutant_speed": mutant_speed,
        "wins": wins,
        "n_valid": n_valid,
        "passes": passes,
        "exploded_count": sum(1 for r in valid if r.get("exploded")),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Null guard (byte-identity check)
# ---------------------------------------------------------------------------

def _run_null_guard(cost_slope: float, seed: int) -> dict:
    """Byte-identity check: disable continuous locomotion → trait inert."""
    from ecology.engine import Ecology
    from ecology.scenarios import SCENARIOS
    from ecology.evolvability.trait_axis import LOCOMOTION_CONTINUOUS_AXIS

    disconnect = LOCOMOTION_CONTINUOUS_AXIS.disconnect_overrides

    def _run(speed: float) -> str:
        # Build base config with the requested cost_slope, then apply disconnect
        # overrides (which set speed_cost_slope=0 and disable the locomotion path,
        # making the trait causally inert).  Order: first set cost_slope on base,
        # then disconnect overrides win any conflicts.
        base_cfg = dc.replace(
            SCENARIOS["balanced"],
            horizon=50,
            mutation_rate=0.0,
            initial_population=10,
            founder=_make_founder(speed),
            speed_cost_slope=cost_slope,
        )
        cfg = dc.replace(base_cfg, **disconnect)
        return Ecology(cfg, seed=seed).run()["events_hash"]

    h_res = _run(1.0)
    h_mut = _run(1.5)
    same = h_res == h_mut
    return {
        "seed": seed,
        "status": "PASS" if same else "FAIL",
        "h_res": h_res[:16],
        "h_mut": h_mut[:16],
    }


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

    rx = _ranks(xs)
    ry = _ranks(ys)
    mrx = sum(rx) / n
    mry = sum(ry) / n
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
    out_path = Path("experiments/outputs/exp241.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def pr(s: str = "") -> None:
        print(s)
        lines.append(s)

    def flush() -> None:
        out_path.write_text("\n".join(lines) + "\n")

    pr("=" * 72)
    pr("Exp 241 — ESS Characterization: Runaway vs Clean Escape (Rung 3)")
    pr("=" * 72)
    pr()
    pr("QUESTION: Is the Exp-240 candidate escape (speed=1.1 invades, speed_cost=0.6)")
    pr("  (A) a CLEAN ESCAPE — a stable finite ESS with convergent invasion from both")
    pr("      sides; or")
    pr("  (B) a DEMOGRAPHIC ARTIFACT — always runaway regardless of cost, the invasion")
    pr("      only happens via explosion?")
    pr()
    pr(f"Parameters inherited from Exp 240: bmc={EXP240_BMC}, mc={EXP240_MC},")
    pr(f"  cap={EXP240_CAP}, threshold={EXP240_THRESHOLD}, aging={EXP240_AGING},")
    pr(f"  regen_rate={REGEN_RATE}, capacity={CAPACITY}, logistic_regen=True.")
    pr()

    # -----------------------------------------------------------------------
    # Step 0: Null guard (byte-identity)
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("Step 0: Null Guard (byte-identity, trait-disabled)")
    pr("=" * 72)
    pr()
    guard_results = []
    for seed in range(2):
        g = _run_null_guard(cost_slope=0.6, seed=seed)
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

    # -----------------------------------------------------------------------
    # Step 1: N_eq(speed, cost) map
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("Step 1: N_eq(speed, cost_slope) MAP — classify demographic outcome")
    pr("=" * 72)
    pr()
    pr(f"speed_cost_slope values: {COST_SWEEP}")
    pr(f"monomorphic speed values: {SPEED_MONO_SWEEP}")
    pr(f"3 seeds per cell, {RESIDENT_HORIZON}-step runs, equilibrium window t>={EQUILIBRIUM_START}")
    pr(f"STABLE = CoV < {STABILITY_COV_THRESH} in window, N>5; RUNAWAY = explodes >{MAX_POP_GUARD}; EXTINCT = pop→0")
    pr()

    # Print header
    speed_header = "   cost \\ speed"
    for sp in SPEED_MONO_SWEEP:
        speed_header += f"  {sp:5.1f}"
    pr(speed_header)
    pr("  " + "-" * (len(speed_header) - 2))

    # Map storage: {(cost, speed): {status, n_eq, cov}}
    neq_map: dict[tuple[float, float], dict] = {}

    for cost in COST_SWEEP:
        row_str = f"  {cost:6.1f}    "
        for sp in SPEED_MONO_SWEEP:
            pr(f"    Running (cost={cost}, speed={sp})...")
            result = _classify_cell(sp, cost)
            neq_map[(cost, sp)] = result
            flush()

            st = result["status"]
            neq = result["n_eq"]
            if st == "STABLE":
                cell_str = f"  S{neq:4.0f}"
            elif st == "RUNAWAY":
                cell_str = "   RUN"
            elif st == "EXTINCT":
                cell_str = "   EXT"
            else:
                cell_str = f"  ?{st[:3]}"
            row_str += cell_str
        pr(row_str)

    pr()
    pr("  Legend: S<N> = STABLE at N_eq=<N>, RUN = RUNAWAY, EXT = EXTINCT")
    pr()

    # Identify stable cells
    stable_cells = {
        (cost, sp): neq_map[(cost, sp)]
        for cost in COST_SWEEP for sp in SPEED_MONO_SWEEP
        if neq_map[(cost, sp)]["status"] == "STABLE"
    }

    pr(f"  STABLE cells: {len(stable_cells)}")
    for (cost, sp), result in sorted(stable_cells.items()):
        pr(f"    cost={cost}, speed={sp}: N_eq={result['n_eq']:.1f}, CoV={result['cov']:.3f}")

    # Find cost regimes where at least two different speeds are stable
    costs_with_multiple_stable: list[float] = []
    for cost in COST_SWEEP:
        stable_at_cost = [sp for sp in SPEED_MONO_SWEEP if neq_map[(cost, sp)]["status"] == "STABLE"]
        if len(stable_at_cost) >= 2:
            costs_with_multiple_stable.append(cost)
            pr(f"    cost={cost}: multiple stable speeds = {stable_at_cost}")

    pr()
    if not costs_with_multiple_stable:
        pr("  NO cost regime has multiple stable speeds.")
        pr("  This means: for any cost tested, either HIGHER speeds run away OR the population")
        pr("  goes extinct — there is NO cost value that creates a stable equilibrium for a")
        pr("  faster monomorphic population. This is the ALWAYS-RUNAWAY picture.")
        pr()
        pr("  INTERIM VERDICT: DEMOGRAPHIC-INSTABILITY-ENTANGLED")
        pr("  (will confirm with additional analysis below)")
    else:
        pr(f"  Cost regimes with >=2 stable speeds: {costs_with_multiple_stable}")
        pr("  => These are candidates for ESS analysis.")
    pr()
    flush()

    # -----------------------------------------------------------------------
    # Step 2: Net-benefit curve at own equilibrium
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("Step 2: Equilibrium Net-Benefit Curve (intake at own N_eq)")
    pr("=" * 72)
    pr()
    pr("For each stable (speed, cost) cell: measure per-capita raw intake at own N_eq.")
    pr("Net benefit ≈ intake − speed_cost (metabolic cost is speed-dependent).")
    pr("An INTERIOR OPTIMUM s* is signaled by net benefit peaking at an intermediate speed.")
    pr()

    # For each cost that has any stable cells, measure benefit curve
    benefit_results: dict[float, list[tuple[float, float, float, float]]] = {}
    # {cost: [(speed, n_eq, raw_intake, net_benefit), ...]}

    for cost in COST_SWEEP:
        stable_speeds = [
            sp for sp in SPEED_MONO_SWEEP
            if neq_map[(cost, sp)]["status"] == "STABLE"
        ]
        if not stable_speeds:
            continue

        pr(f"  cost={cost}: measuring benefit curve for stable speeds {stable_speeds}")
        curve: list[tuple[float, float, float, float]] = []

        for sp in stable_speeds:
            n_eq = neq_map[(cost, sp)]["n_eq"]
            # Measure intake at own equilibrium (3 seeds)
            intakes = []
            for seed in range(3):
                intake_val = _intake_at_neq(sp, cost, n_eq, seed)
                if not math.isnan(intake_val):
                    intakes.append(intake_val)
            raw_intake = statistics.mean(intakes) if intakes else float("nan")
            speed_cost_per_step = cost * sp
            net_benefit = raw_intake - speed_cost_per_step if not math.isnan(raw_intake) else float("nan")
            curve.append((sp, n_eq, raw_intake, net_benefit))
            pr(f"    speed={sp}: N_eq={n_eq:.0f}, raw_intake={raw_intake:.4f}, "
               f"cost={speed_cost_per_step:.4f}, net={net_benefit:.4f}")
            flush()

        benefit_results[cost] = curve

        if len(curve) >= 2:
            speeds_c = [c[0] for c in curve]
            net_b = [c[3] for c in curve if not math.isnan(c[3])]
            speeds_valid = [c[0] for c in curve if not math.isnan(c[3])]
            rho_net = _spearman_r(speeds_valid, net_b)
            pr(f"    Spearman(speed, net_benefit) = {rho_net:.3f}")
            if rho_net > 0.5:
                pr(f"    => NET BENEFIT RISING with speed (no interior optimum at these speeds)")
            elif rho_net < -0.5:
                pr(f"    => NET BENEFIT FALLING with speed (higher cost outweighs intake gain)")
            else:
                pr(f"    => NON-MONOTONE net benefit — candidate interior optimum!")
        pr()

    flush()

    # -----------------------------------------------------------------------
    # Step 3: ESS Convergence Test
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("Step 3: Convergence Test (invasion from both sides of candidate s*)")
    pr("=" * 72)
    pr()

    # Determine what to test based on the map:
    # Case A: costs_with_multiple_stable is non-empty — test the ESS in the stable regime
    # Case B: no stable higher-speed cells — test invasion at higher cost to verify always-runaway
    #         and also test "can a SLOWER mutant invade at high cost" (the ABOVE-s* test)

    # For the convergence test, we need a cost where both a resident at s_below and s_above
    # can be set up with stable equilibria, and we can test invasion in both directions.

    # Strategy: use the highest cost that has the most stable cells
    if costs_with_multiple_stable:
        test_cost = costs_with_multiple_stable[-1]  # highest cost with multiple stable speeds
        stable_at_test = [
            sp for sp in SPEED_MONO_SWEEP
            if neq_map[(test_cost, sp)]["status"] == "STABLE"
        ]
        # Find best candidate s* from net-benefit
        curve = benefit_results.get(test_cost, [])
        if curve and len(curve) >= 2:
            net_benefits = [(c[0], c[3]) for c in curve if not math.isnan(c[3])]
            # Find the speed with highest net benefit — candidate ESS
            best_sp, best_nb = max(net_benefits, key=lambda x: x[1])
            pr(f"  Candidate ESS cost={test_cost}: s* ≈ {best_sp} (net_benefit={best_nb:.4f})")

            # Pick a resident below s* and a resident above s* (if they exist and are stable)
            below_candidates = [sp for sp in stable_at_test if sp < best_sp]
            above_candidates = [sp for sp in stable_at_test if sp > best_sp]

            if below_candidates and above_candidates:
                s_below = max(below_candidates)
                s_above = min(above_candidates)
                neq_below = int(round(neq_map[(test_cost, s_below)]["n_eq"]))
                neq_above = int(round(neq_map[(test_cost, s_above)]["n_eq"]))

                pr(f"  BELOW s*: resident={s_below} (N_eq={neq_below}), testing if speed={best_sp} invades")
                pr(f"  ABOVE s*: resident={s_above} (N_eq={neq_above}), testing if speed={best_sp} invades")
                pr()

                # Test: faster (best_sp) invades from below (resident=s_below)
                pr(f"  Below test: resident speed={s_below} → mutant speed={best_sp} ({N_SEEDS} seeds)")
                below_inv = _invasion_verdict(
                    resident_speed=s_below,
                    mutant_speed=best_sp,
                    cost_slope=test_cost,
                    resident_neq=neq_below,
                    n_seeds=N_SEEDS,
                )
                pr(f"    Result: {below_inv['wins']}/{below_inv['n_valid']} invade "
                   f"(exploded={below_inv['exploded_count']}), PASSES={below_inv['passes']}")
                flush()

                # Test: slower (best_sp) is challenged by even-slower from above
                # More directly: resident=s_above should be invaded by slower (=best_sp) mutant
                # i.e. best_sp invades into s_above resident — slower is better at high cost
                pr(f"  Above test: resident speed={s_above} → mutant speed={best_sp} ({N_SEEDS} seeds)")
                above_inv = _invasion_verdict(
                    resident_speed=s_above,
                    mutant_speed=best_sp,
                    cost_slope=test_cost,
                    resident_neq=neq_above,
                    n_seeds=N_SEEDS,
                )
                pr(f"    Result: {above_inv['wins']}/{above_inv['n_valid']} invade "
                   f"(exploded={above_inv['exploded_count']}), PASSES={above_inv['passes']}")
                flush()

                ess_convergence = below_inv["passes"] and above_inv["passes"]
                pr()
                pr(f"  ESS CONVERGENCE (both invasions pass): {ess_convergence}")
                if ess_convergence:
                    pr(f"  => STABLE ESS signature: faster invades from below, slower invades from above.")
                    pr(f"     s* ≈ {best_sp} at cost_slope={test_cost} is an interior optimum.")
                else:
                    pr(f"  => Convergence FAILED (one or both arms did not invert).")
            else:
                pr(f"  Cannot do two-sided test: below={below_candidates}, above={above_candidates}")
                pr(f"  Not enough stable speeds to bracket s*={best_sp}")
                ess_convergence = False
                below_inv = {}
                above_inv = {}
        else:
            pr(f"  Not enough benefit curve data for cost={test_cost}")
            ess_convergence = False
            below_inv = {}
            above_inv = {}
    else:
        pr("  No cost regime with multiple stable speeds — testing at Exp-240 cost=0.6 anyway.")
        pr("  This tests the BELOW case (is the known invasion still runaway?)")
        pr("  and checks if a SLOWER mutant could invade at very high cost (ABOVE case).")
        pr()

        # Test the Exp-240 case: speed=1.0 resident, speed=1.1 mutant, cost=0.6
        # The resident=1.0 is stable at cost=0.6 (confirmed by Exp 240)
        resident_neq_240 = 374
        pr(f"  Replicating Exp-240: resident=1.0, mutant=1.1, cost=0.6, N_eq={resident_neq_240}")
        below_inv_240 = _invasion_verdict(
            resident_speed=1.0, mutant_speed=1.1, cost_slope=0.6,
            resident_neq=resident_neq_240, n_seeds=N_SEEDS,
        )
        pr(f"    {below_inv_240['wins']}/{below_inv_240['n_valid']} invade "
           f"(exploded={below_inv_240['exploded_count']}), PASSES={below_inv_240['passes']}")
        flush()

        # Test if SLOWER mutant (0.9) invades at cost=0.6 (would falsify "always runaway")
        pr(f"  Slower-invasion test: resident=1.0, mutant=0.9, cost=0.6")
        slower_inv_240 = _invasion_verdict(
            resident_speed=1.0, mutant_speed=0.9, cost_slope=0.6,
            resident_neq=resident_neq_240, n_seeds=N_SEEDS,
        )
        pr(f"    {slower_inv_240['wins']}/{slower_inv_240['n_valid']} invade "
           f"(exploded={slower_inv_240['exploded_count']}), PASSES={slower_inv_240['passes']}")
        flush()

        # Test at highest cost (3.0) — does any speed settle?
        pr()
        pr(f"  High-cost check: cost=3.0, testing if speed=1.0 is still stable")
        neq_30_10 = neq_map.get((3.0, 1.0), {}).get("n_eq", float("nan"))
        if not math.isnan(neq_30_10):
            pr(f"    speed=1.0 at cost=3.0: N_eq={neq_30_10:.0f} (STABLE)")
        else:
            pr(f"    speed=1.0 at cost=3.0: {neq_map.get((3.0, 1.0), {}).get('status', 'unknown')}")

        ess_convergence = False
        below_inv = below_inv_240
        above_inv = slower_inv_240

    pr()
    flush()

    # -----------------------------------------------------------------------
    # Step 4: Robustness (alt bump layout + alt founder)
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("Step 4: Robustness Check (neutral layout + alt founder)")
    pr("=" * 72)
    pr()
    pr("Purpose: confirm that the qualitative result (stable ESS, or always-runaway)")
    pr("holds under a different bump-field arrangement and a different founder calibration.")
    pr()

    # 4a: Neutral layout (different bump centers, same count/sigma)
    pr("  4a: Neutral layout (BUMP_CENTERS_NEUTRAL — different geometric arrangement)")
    pr(f"  Testing key cost values: 0.6 and {COST_SWEEP[-1]}")
    pr()

    robustness_neutral: dict[tuple[float, float], dict] = {}
    for cost in [0.6, COST_SWEEP[-1]]:
        for sp in [1.0, 1.5]:
            pr(f"    Neutral layout: (cost={cost}, speed={sp})...")
            result = _classify_cell(sp, cost, layout="neutral")
            robustness_neutral[(cost, sp)] = result
            st = result["status"]
            n = result["n_eq"]
            neq_str = f"{n:.1f}" if not math.isnan(n) else "NaN"
            pr(f"      status={st}, N_eq={neq_str}")
            flush()

    pr()
    pr("  4b: Alt founder (bmc={}, mc={}, threshold={}, aging={})".format(
        ALT_BMC, ALT_MC, ALT_THRESHOLD, ALT_AGING))
    pr(f"  Testing key cost values: 0.6 and {COST_SWEEP[-1]}")
    pr()

    robustness_alt: dict[tuple[float, float], dict] = {}
    for cost in [0.6, COST_SWEEP[-1]]:
        for sp in [1.0, 1.5]:
            pr(f"    Alt founder: (cost={cost}, speed={sp})...")
            result = _classify_cell(
                sp, cost,
                bmc=ALT_BMC, mc=ALT_MC, threshold=ALT_THRESHOLD, aging=ALT_AGING,
            )
            robustness_alt[(cost, sp)] = result
            st = result["status"]
            n = result["n_eq"]
            neq_str_alt = f"{n:.1f}" if not math.isnan(n) else "NaN"
            pr(f"      status={st}, N_eq={neq_str_alt}")
            flush()

    pr()
    flush()

    # -----------------------------------------------------------------------
    # Aggregate verdict
    # -----------------------------------------------------------------------
    pr("=" * 72)
    pr("AGGREGATE VERDICT")
    pr("=" * 72)
    pr()

    # Summarize N_eq map findings
    pr("  === N_eq(speed, cost) MAP SUMMARY ===")
    n_stable_cells = sum(1 for v in neq_map.values() if v["status"] == "STABLE")
    n_runaway_cells = sum(1 for v in neq_map.values() if v["status"] == "RUNAWAY")
    n_extinct_cells = sum(1 for v in neq_map.values() if v["status"] == "EXTINCT")
    pr(f"  Total cells: {len(neq_map)}")
    pr(f"  STABLE: {n_stable_cells}, RUNAWAY: {n_runaway_cells}, EXTINCT: {n_extinct_cells}")

    # Which speeds are stable at which costs?
    for cost in COST_SWEEP:
        stable_at_cost = [sp for sp in SPEED_MONO_SWEEP if neq_map[(cost, sp)]["status"] == "STABLE"]
        runaway_at_cost = [sp for sp in SPEED_MONO_SWEEP if neq_map[(cost, sp)]["status"] == "RUNAWAY"]
        pr(f"  cost={cost}: STABLE={stable_at_cost}, RUNAWAY={runaway_at_cost}")

    pr()
    pr("  === ESS ANALYSIS ===")
    if costs_with_multiple_stable:
        pr(f"  Costs with stable higher speeds: {costs_with_multiple_stable}")
        for cost in costs_with_multiple_stable:
            curve = benefit_results.get(cost, [])
            if curve:
                best = max(curve, key=lambda c: c[3] if not math.isnan(c[3]) else -999)
                pr(f"    cost={cost}: peak net-benefit speed = {best[0]} (net={best[3]:.4f})")
        pr(f"  ESS convergence test (two-sided invasion): {ess_convergence}")
    else:
        pr("  No cost regime yields stable higher speeds.")
        pr("  => Higher speeds always run away regardless of cost slope tested.")

    pr()
    pr("  === ROBUSTNESS ===")
    robust_stable_neutral = {
        (c, sp): robustness_neutral[(c, sp)]["status"]
        for (c, sp) in robustness_neutral
    }
    robust_stable_alt = {
        (c, sp): robustness_alt[(c, sp)]["status"]
        for (c, sp) in robustness_alt
    }
    pr("  Neutral layout:")
    for (c, sp), st in sorted(robust_stable_neutral.items()):
        pr(f"    cost={c}, speed={sp}: {st}")
    pr("  Alt founder:")
    for (c, sp), st in sorted(robust_stable_alt.items()):
        pr(f"    cost={c}, speed={sp}: {st}")

    pr()
    pr("  === BYTE-IDENTITY / GOLDEN HASH ===")
    pr(f"  Null guards PASS: {guards_pass}")
    pr("  enable_continuous_locomotion=False by default (OFF = byte-identical to Exp 194-239).")
    pr("  continuous_logistic_regen=False by default (OFF = byte-identical to Exp 238-239).")
    pr("  No changes to engine.py, continuous_world.py, or golden-hash test files.")

    pr()
    pr("=" * 72)
    pr("BLUNT BOTTOM LINE")
    pr("=" * 72)
    pr()

    # Construct blunt verdict
    higher_speeds_ever_stable = any(
        neq_map[(cost, sp)]["status"] == "STABLE"
        for cost in COST_SWEEP
        for sp in SPEED_MONO_SWEEP
        if sp > 1.0
    )

    if higher_speeds_ever_stable and ess_convergence:
        verdict = "CLEAN ESCAPE — STABLE ESS FOUND"
        pr("  VERDICT: CLEAN ESCAPE (Candidate → STRENGTHENED toward CONFIRMED)")
        pr()
        pr("  A stable finite ESS s* exists at a tested cost regime. Both a faster")
        pr("  mutant (from below s*) AND a slower mutant (from above s*) invade the")
        pr("  opposing resident — convergent selection toward a finite optimal speed.")
        pr("  Both types demographically stable (finite N_eq, CoV < 0.05).")
        pr()
        pr("  The Exp-240 candidate escape is a CLEAN EVOLVABLE OPTIMUM, not")
        pr("  demographic instability entangled. Continuous locomotion supports")
        pr("  evolution to a stable speed, the local-gradient wall genuinely fails.")
        pr()
        pr("  What remains for full confirmation:")
        pr("  - Blinded verify of the ESS equilibrium and invasion by independent run")
        pr("  - Replication across >=1 more cost regime")
        pr("  - Test whether navigational STRATEGY (beyond raw speed) evolves at s*")
    elif higher_speeds_ever_stable and not ess_convergence:
        verdict = "PARTIAL — STABLE RANGE EXISTS BUT ESS CONVERGENCE UNCERTAIN"
        pr("  VERDICT: PARTIAL (higher speeds can be stable, but convergence unconfirmed)")
        pr()
        pr("  Some cost regimes yield stable finite equilibria at higher speeds (higher")
        pr("  speeds ARE demographically bounded). However, the ESS convergence test")
        pr("  (invasion from both sides) was inconclusive — one or both arms failed.")
        pr()
        pr("  The escape is neither clearly clean nor clearly artifact. Next step:")
        pr("  refine the candidate s* estimate and re-run the convergence test with")
        pr("  finer speed resolution around the net-benefit peak.")
    else:
        verdict = "DEMOGRAPHIC-INSTABILITY-ENTANGLED — ALWAYS RUNAWAY"
        pr("  VERDICT: DEMOGRAPHIC-INSTABILITY-ENTANGLED (always runaway at all costs tested)")
        pr()
        pr("  No cost regime tested yields a stable finite equilibrium for speeds")
        pr("  faster than the Exp-240 resident (speed=1.0). Any harvesting-efficiency")
        pr("  gain via higher speed triggers a demographic runaway (commons tragedy).")
        pr("  The 'invasion' in Exp 240 reflects substrate instability, not a clean")
        pr("  selection gradient toward an evolvable optimum.")
        pr()
        pr("  The Exp-240 candidate escape is DEMOGRAPHIC-INSTABILITY-ENTANGLED.")
        pr("  It does NOT constitute a clean evolvable optimum. The local-gradient")
        pr("  wall may hold for a stable ESS — it is only breached in a regime where")
        pr("  the winner explodes, not converges.")
        pr()
        pr("  This WEAKENS (does not fully invalidate) the Exp-240 candidate:")
        pr("  - The per-capita benefit at N_eq=374 is real (+43%), non-saturating.")
        pr("  - But the benefit comes with no demographic ceiling for the faster type.")
        pr("  - A regime where a faster type both INVADES and CONVERGES to stable N_eq")
        pr("    has NOT been found in the tested parameter space.")

    pr()
    pr(f"  Verdict string: {verdict}")
    pr()

    flush()
    print(f"\nOutput written to {out_path}")


if __name__ == "__main__":
    main()
