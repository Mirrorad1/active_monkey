"""Seed the coalescence mechanism/geometry/boundary library from repo evidence.

Run from the repo root:

    PYTHONPATH=. uv run --python .venv python tools/seed_coalescence_cards.py

It constructs the seed artifacts as real dataclasses (so they are schema-valid by
construction) and writes canonical JSON into mechanisms/, geometry_maps/, and
boundary_notes/. The scientific CONTENT below is authored from committed repo evidence
(EXPERIMENTS.md, docs/research/*, loop/directions/*) and uses conservative language only —
no claim of sentience, consciousness, AGI, or natural-language understanding. Re-running is
idempotent (deterministic output).

Every card's `status` is honest:
  * functional-valence-dyad-v0 = validated (toy scale, frozen-scorer controlled)
  * the sensing / memory walls    = constrained (negative boundaries)
  * communication-scaffold-v0     = scaffold (NOT validated — an existence benchmark only)
"""
from __future__ import annotations

from pathlib import Path

from active_loop.coalescence.schema import (
    AdapterCard,
    BoundaryNote,
    GeometryMap,
    MechanismCard,
    ScorerCard,
    write_json,
)

ROOT = Path(__file__).resolve().parent.parent


# ── functional-valence-dyad-v0 (VALIDATED) ───────────────────────────────────
FUNCTIONAL_VALENCE_DYAD = MechanismCard(
    mechanism_id="functional-valence-dyad-v0",
    mechanism_type="functional-valence-learning",
    status="validated",
    source_experiments=[215, 216, 217, 218, 219, 220, 221, 222, 225],
    claim=(
        "A symbolic dyadic agent (DirectHeadAgent) with a direct "
        "P(valence | intent-like state, response) emission head learns, from functional "
        "valence feedback over a long session, which response earns approval for each "
        "inferred intent-like state. Under a frozen, constant-unfakeable scorer it reaches "
        "reliable genuine discrimination (13/16 seeds at K=4, Exp 220) at toy scale. "
        "Functional valence only — not subjective feeling."
    ),
    works_when=[
        "long session (~300 turns): learning itself is load-bearing (Exp 221)",
        "direct valence head gives a non-vanishing learning gradient (Exp 215)",
        "an honest optimistic POS prior breaks the exploration cold-start (Exp 217)",
        "precision annealed gamma 1->8 over the session fixes the learn-but-don't-exploit "
        "decoupling (Exp 220)",
        "judged by correct_select (a constant policy caps at 2/6 = the 1/3 ceiling)",
    ],
    fails_when=[
        "short session (~100 turns): blocks LEARNING; the precision schedule does NOT "
        "rescue it (Exp 221)",
        "indirect response->valence credit via the intent-transition B never learns until "
        "A/B are already learned (Exp 125/127/214)",
        "low precision (gamma=1) blocks EXPLOITATION even once the map is learned (Exp 219)",
        "fixed high precision (gamma=8) over-commits early (3/16, Exp 220)",
    ],
    required_conditions=[
        "frozen scorer eval/affect_score.py (sha256-pinned in the artifact manifest)",
        "scripted-partner closed loop",
        "K=4 response codes, U=6 utterance codes, CORRECT[c] = c % 4",
        "belief-like posterior carried continuously across turns (never reset per turn)",
    ],
    reusable_interface=(
        "DirectHeadAgent: perceive(code)->posterior over the intent-like factor; "
        "act()->response code; update(valence)->windowed Dirichlet on the A head. "
        "Runnable checkpoint: artifacts/active-monkey-affect-dyad-v0."
    ),
    inputs=["symbolic utterance code (int)", "functional valence feedback (POS/NEU/NEG)"],
    outputs=[
        "response code (int)",
        "belief-like posterior over the intent-like factor",
        "learned P(valence | intent, response) head (Dirichlet counts pA)",
    ],
    state_requirements=[
        "continuous belief across turns",
        "Dirichlet pseudo-counts on the valence head",
    ],
    costs=["~300-turn session is load-bearing", "precision-annealing scaffold"],
    metrics=[
        "mean_last (last-third POS rate)",
        "correct_select (constant-unfakeable discrimination; caps at 2/6)",
        "genuine_fraction",
    ],
    falsifiers=[
        "a constant-reply control must FAIL (genuine_fraction 0, verdict False) — "
        "verified Exp 222",
        "mean_last must beat the 1/3 constant-response ceiling, not merely the 1/5 "
        "uniform-random floor",
    ],
    known_confounds=[
        "session length does much of the work; the schedule was never fully separated "
        "from length at short lengths (Exp 221)",
        "earlier lenient 7/8 counts overstated genuine discrimination before the "
        "constant-unfakeable metric upgrade (Exp 218/219)",
    ],
    next_compositions=[
        "belief-like state -> active-sensing probe (see adapters.json)",
        "dyad -> BeliefBench (active_loop/benchmarks/beliefbench.py)",
        "C1 NEU-aversion as an autopilot-found honest improving move (Exp 225)",
    ],
)

DYAD_TO_ACTIVE_SENSING_ADAPTER = AdapterCard(
    adapter_id="dyad-belief-to-active-sensing-v0",
    from_mechanism="functional-valence-dyad-v0",
    to_mechanism="active-sensing-probe (costed information-gathering)",
    input_contract=(
        "the dyad's belief-like posterior over the intent-like factor (a normalized "
        "distribution) and its action margin |p - 0.5|"
    ),
    output_contract=(
        "an uncertainty signal that could gate a costed probe action (probe when the "
        "margin is ambiguous)"
    ),
    required_state=["continuous belief posterior from the dyad"],
    assumptions=[
        "the intent-like posterior is calibrated enough that margin tracks decision risk",
    ],
    failure_modes=[
        "Exp 211 showed a single cue is often confidently WRONG, invisible to a margin "
        "gate — so margin-gated probing did NOT beat fixed-rate; this adapter is a "
        "composition HYPOTHESIS, not a validated bridge",
    ],
    tests=[
        "does margin-gated probing beat a budget-matched fixed-rate control on decision "
        "quality AND on a benefit ceiling? (open — Exp 211 says benefit ceiling ~0 at the "
        "ecology substrate)",
    ],
)

AFFECT_SCORER_CARD = ScorerCard.from_file(
    "eval/affect_score.py",
    scorer_id="affect-score",
    scorer_version="affect-score-1e",
    repo=ROOT,
    metrics=["mean_last", "improvement", "genuine_fraction", "correct_select"],
    required_controls=[
        "constant-response control (must score genuine_fraction 0 / verdict False)",
    ],
    pass_conditions=[
        "mean_last > 1/3 (beats the constant-response ceiling)",
        "improvement >= 0.10 (learning, not a fixed policy)",
        "genuine_fraction >= 0.5 (>= half of seeds clear correct_select 0.5 AND "
        "last-third > 1/3)",
    ],
    fail_conditions=[
        "mean_last <= 1/3",
        "a constant policy reaching verdict True (would falsify the metric)",
    ],
    limitations=[
        "symbolic utterance codes, not natural language",
        "long-session learning is load-bearing (Exp 221)",
        "functional valence only — not sentience, consciousness, or subjective feeling",
    ],
)


# ── communication-scaffold-v0 (SCAFFOLD — NOT validated) ──────────────────────
COMMUNICATION_SCAFFOLD = MechanismCard(
    mechanism_id="communication-scaffold-v0",
    mechanism_type="costed-signaling",
    status="scaffold",
    source_experiments=[],
    claim=(
        "Communication is NOT yet demonstrated in active_monkey. A v0 sender/receiver "
        "signaling benchmark exists (active_loop/benchmarks/comm_v0.py) as an EXISTENCE "
        "test — costed signaling that beats shuffled/muted controls (~1.9 bits MI) — but "
        "no selection-pressure or ecology result shows proto-communication emerging. "
        "Future tests require a sender/receiver split, message cost, receiver belief "
        "update, and shuffled/muted message controls."
    ),
    works_when=["(unknown — no emergence experiment has been run)"],
    fails_when=[
        "creatures are currently solipsistic (no representation of another agent); "
        "interaction requires NEW provided substrate (a shared world and/or a channel), "
        "each piece declared as a prior",
    ],
    required_conditions=[
        "sender/receiver split",
        "message cost",
        "receiver belief-like update conditioned on the message",
        "shuffled-message and muted-message controls",
    ],
    reusable_interface="active_loop/benchmarks/comm_v0.py (existence test only)",
    inputs=["a costed symbolic message channel (provided)"],
    outputs=["receiver action / belief-like update"],
    falsifiers=[
        "a shuffled or muted message must destroy the receiver's advantage; if it does "
        "not, there is no information transfer",
    ],
    known_confounds=[
        "emergent compositional grammar is a documented OPEN PROBLEM (open_problem.html); "
        "the tractable honest claim is convergence vs divergence of TAUGHT-label maps "
        "under coupling, not language from scratch",
    ],
    next_compositions=[
        "comm channel -> ecology trait (selection for costed signaling) — untested",
    ],
)


# ── GeometryMaps ──────────────────────────────────────────────────────────────
DYAD_SESSION_LENGTH_CURVE = GeometryMap(
    geometry_id="dyad-session-length-curve-v0",
    mechanism_id="functional-valence-dyad-v0",
    source_experiments=[218, 219, 220, 221],
    swept_parameters={
        "session_length": ["~100t (short)", "~300t (long)"],
        "precision_schedule": ["fixed gamma=4", "fixed gamma=8", "annealed gamma 1->8"],
        "capacity_K": [4],
    },
    fixed_parameters={"U": 6, "CORRECT": "c % 4", "scorer": "eval/affect_score.py (frozen)"},
    metrics=["genuine_fraction (constant-unfakeable)", "mean_csel"],
    outcome_regions=[
        {"region": "short session, any schedule", "outcome": "BLOCKED — learning fails; "
         "mean_csel near chance; schedule over-commits before learning (Exp 221)"},
        {"region": "long 300t, fixed gamma=4", "outcome": "7/16 genuine — unreliable"},
        {"region": "long 300t, fixed gamma=8", "outcome": "3/16 — over-commits early"},
        {"region": "long 300t, annealed gamma 1->8", "outcome": "13/16 genuine — "
         "reliable (Exp 220, POSITIVE)"},
    ],
    thresholds={
        "constant_response_ceiling": "1/3 (zero line; a constant policy maps 2/6 codes)",
        "correct_select_gate": ">= 0.5",
    },
    positive_regions=["long 300t with annealed precision gamma 1->8"],
    negative_regions=["short ~100t sessions (learning blocked, schedule does not rescue)"],
    confounds_excluded=[
        "session length was NOT fully separated from the schedule at short lengths — "
        "length is load-bearing (Exp 221)",
    ],
    next_experiment_suggestions=[
        "attack the short-session LEARNING lever (lr / optimism / replay), NOT the "
        "precision schedule (which only optimizes exploitation)",
    ],
)

ACTIVE_SENSING_BENEFIT_WALL_GEOM = GeometryMap(
    geometry_id="active-sensing-benefit-wall-v0",
    mechanism_id="active-sensing-probe",
    source_experiments=[210, 211, 212],
    swept_parameters={
        "probe_rate": "0.0 -> 0.10 (local step)",
        "probe_cost": "0.005 - 0.1 (incl. 0.0 in Exp 211)",
        "carrying_capacity": "cap 50 (drift-dominated) -> cap 250 (drift-suppressed)",
        "gate_threshold": "gain 0.50 -> 0.55 (Exp 211 uncertainty gate)",
    },
    fixed_parameters={"substrate": "enable_hidden_mode + enable_active_sensing"},
    metrics=["selection slope mean_s", "invasion-from-rarity fraction",
             "wrong-cell occupancy", "benefit ceiling (energy/step)"],
    outcome_regions=[
        {"region": "fixed-rate probing (Exp 210)", "outcome": "FAIL_LOCAL_GRADIENT — "
         "info arm ~= pure-cost control at every cost; slope ~0; invasion 0/16; "
         "mechanism LIVE when gifted (wrong-cell 0.40->0.35, ceiling ~0.0345)"},
        {"region": "uncertainty-gated probing (Exp 211)", "outcome": "FAIL_LOCAL_GRADIENT "
         "— gate works as imposed but does NOT beat fixed-rate; gated benefit ceiling ~0; "
         "flat even at probe_cost=0"},
        {"region": "monomorphic N*(rate) landscape (Exp 212)", "outcome": "monotone +1.5% "
         "at full probing — NO valley, NO reachable higher region"},
    ],
    negative_regions=["the entire tested probe-rate x cost surface (slope ~0 everywhere)"],
    positive_regions=[],
    confounds_excluded=[
        "drift = population-size, not cost (raised cap to 250; L29)",
        "cost calibrated to the empirical benefit ceiling (~0.034 energy/step; L30)",
        "staleness was NOT the killer (Theory B confirmed)",
    ],
    next_experiment_suggestions=[
        "the benefit MAGNITUDE is the wall, not the payoff geometry (Exp 213); no named "
        "lever remains without a human word",
    ],
)

COSTLY_SENSING_WALL_GEOM = GeometryMap(
    geometry_id="costly-sensing-wall-v0",
    mechanism_id="costed-sensing-organ",
    source_experiments=[199, 200, 201, 202, 203, 204, 205],
    swept_parameters={
        "sensing_noise": "V1 0.20 -> V4 0.02",
        "organ_efficiency": "standard vs cheap (inefficiency 0.20 / floor 0)",
        "temperature_stress_scale": "1.0 vs 3.0",
        "foraging_band_width": "WIDE 0.15 / NARROW 0.05 / CONTROL uniform",
        "band_staleness_period": "FAST 60 / MEDIUM 120 / SLOW 2400",
        "competition": "COMPETE (depleting + shuffle) vs ABUNDANT",
        "affordance": "smooth graded vs discrete high-stakes residue",
    },
    fixed_parameters={"substrate": "ecology/ homeostatic GridWorld; thermosense organ"},
    metrics=["gene-pool newborn intensity", "N*(h)", "pairwise selection s", "B(h)"],
    outcome_regions=[
        {"region": "avoidance valley (Exp 199)", "outcome": "all depths primitive <0.15; "
         "seeded-functional (0.50) decays or drives extinction — benefit saturates"},
        {"region": "foraging (Exp 200)", "outcome": "all arms primitive 0.04-0.09; "
         "narrower bands give lower mean"},
        {"region": "increasing-returns band-staleness (Exp 201)", "outcome": "FAST shows a "
         "transient climb ~0.18 then decays; not sustained functional (MIXED)"},
        {"region": "interference competition (Exp 202)", "outcome": "decays to ~0.03; "
         "organ selected AGAINST; functional only at collapsed pop (drift)"},
        {"region": "residue / false-positive affordance (Exp 204)", "outcome": "first "
         "functional monomorphic optimum N*(0.60), but local gradient <=0 (un-earnable)"},
        {"region": "survivable-loss sweep (Exp 205)", "outcome": "optimum is functional + "
         "bulk-fitter yet still un-evolvable — the fitness VALLEY is the sole barrier"},
    ],
    negative_regions=["every tested ecology near the resident (g(h_res) <= 0)"],
    positive_regions=[
        "one weak, purely-competitive positive LOCAL gradient in FORAGE/band-staleness "
        "(Exp 203, MIXED, one seed short) — erodes as the trait spreads",
    ],
    confounds_excluded=[
        "resource-memory free-ride, creature-id-ordered competition, free-trait "
        "substitution (Exp 201 audit)",
        "demographic collapse ruled out as the barrier (Exp 205)",
        "CLAMPED-LR controls confirm genuine thermosense, not memory substitution",
    ],
    next_experiment_suggestions=[
        "no costed sense becomes a functional organ at this substrate across seven levers; "
        "the blocker is benefit MAGNITUDE / a fitness valley, not payoff geometry",
    ],
)


# ── BoundaryNotes ─────────────────────────────────────────────────────────────
ACTIVE_SENSING_BENEFIT_WALL = BoundaryNote(
    boundary_id="active-sensing-benefit-wall-v0",
    source_experiments=[210, 211, 212, 213],
    failed_mechanism="costed active information-gathering (probe before acting), incl. "
    "uncertainty-gated probing",
    observed_failure=(
        "A costed probe action does not pay locally near the resident. Fixed-rate probing "
        "(Exp 210) is neutral (slope ~0, invasion 0/16) and loses to a pure-cost "
        "perfect-percept control; uncertainty-gated probing (Exp 211) works as imposed but "
        "does NOT beat fixed-rate and has a benefit ceiling ~0 even at zero cost. The "
        "mechanism is LIVE when gifted (wrong-cell occupancy 0.40->0.35) but un-evolvable."
    ),
    tested_conditions=[
        "probe_rate 0->0.10 local step; probe_cost 0.005-0.1 incl. 0.0",
        "carrying capacity 50 and 250 (drift control)",
        "uncertainty gate on action margin |belief - 0.5|",
        "monomorphic N*(rate) landscape assay (Exp 212); affordance shape audit (Exp 213)",
    ],
    excluded_confounds=[
        "drift (raised cap to 250)",
        "cost mis-calibration (calibrated to benefit ceiling ~0.034 energy/step)",
        "staleness (Theory B: not the killer)",
        "wasted budget (Exp 211 refuted — gated probes were not waste)",
    ],
    implication=(
        "The local-gradient wall spans scalar senses, passive memory/inference, AND active "
        "information-seeking. Useful-when-gifted != locally evolvable. The blocker is "
        "benefit MAGNITUDE, not payoff geometry (Exp 213)."
    ),
    next_safe_region_to_test=(
        "no named lever remains at this substrate; re-opens only on a human word (e.g. a "
        "regime where precision's marginal value is large near the resident)"
    ),
)

COSTLY_SENSING_WALL = BoundaryNote(
    boundary_id="costly-sensing-wall-v0",
    source_experiments=[199, 200, 201, 202, 203, 204, 205, 206, 207],
    failed_mechanism="evolution of a costed sensory organ (thermosense) into a functional "
    "organ under selection",
    observed_failure=(
        "Across seven structurally-distinct levers — avoidance (199), foraging (200), "
        "increasing-returns band-staleness (201), interference competition (202), gradient "
        "audit (203, MIXED), residue/false-positive (204), survivable-loss (205) — a costed "
        "sense never climbs to a functional organ. The gene-pool newborn intensity stays "
        "primitive (<0.15) or decays; seeded-functional organs decay or drive extinction. "
        "Even a functional, survivable, bulk-fitter monomorphic optimum is un-evolvable "
        "because small steps don't pay (Exp 205: the fitness valley is the sole barrier)."
    ),
    tested_conditions=[
        "noise, organ efficiency, temperature stress, foraging band width, band-staleness "
        "period, interference competition, affordance shape",
        "newborn-intensity tracker (survivor-bias-free); CLAMPED-LR control",
        "Evolvability Preflight local-gradient assay (ecology/evolvability/)",
    ],
    excluded_confounds=[
        "survivor bias (de-novo from intensity 0, Exp 198/200)",
        "demographic collapse (Exp 205, survivable losses)",
        "resource-memory substitution (CLAMPED-LR ~= treatment)",
        "engine free-rides (Exp 201 9-agent adversarial audit)",
    ],
    implication=(
        "No costed sense becomes a functional organ at this toy ecology substrate. The "
        "reusable artifact is the BOUNDARY: the marginal payoff of sensory precision is too "
        "small / too noisy to out-breed its cost near any resident the search can reach."
    ),
    next_safe_region_to_test=(
        "barcode-niche or sensor-controller co-adaptation regimes were the last "
        "structurally-distinct escapes (Exp 206-207); re-open on a human word"
    ),
)

HIDDEN_STATE_MEMORY_BOUNDARY = BoundaryNote(
    boundary_id="hidden-state-memory-boundary-v0",
    source_experiments=[208, 209],
    failed_mechanism="passive hidden-state memory / belief-persistence as a locally "
    "evolvable trait",
    observed_failure=(
        "An incremental memory step does not pay locally. Integer memory_horizon 1->2 "
        "(Exp 208) is FLAT_OR_NOISY 6/8 vs a perfect-percept control; continuous "
        "belief_persistence rho 0.5->0.55 (Exp 209) is also 6/8 with the control ALSO 6/8 "
        "(residual is drift, not denoising; denoising-attributable effect ~0.02). Large "
        "gifted jumps DO pay (rho 0->0.85 is 3/3 POSITIVE) — so the mechanism is LIVE but "
        "the local gradient is a coin-flip."
    ),
    tested_conditions=[
        "memory_horizon 1->2 (integer) and belief_persistence rho 0.5->0.55 (continuous)",
        "perfect-percept drift control; invasion-from-rarity; cap-250 re-test (non-drift)",
    ],
    excluded_confounds=[
        "granularity artifact (wall holds at BOTH integer and continuous resolution)",
        "drift (re-tested at cap-250, still flat)",
    ],
    implication=(
        "The local-gradient wall generalises from scalar senses to memory/inference: a "
        "small step in information-processing capacity does not pay near the resident, "
        "while large gifted steps do. Passive memory alone is not robustly adaptive at this "
        "substrate."
    ),
    next_safe_region_to_test=(
        "full evolution (Rung 2) was gated on a POSITIVE local gradient and is therefore "
        "not run; an active (probe) rather than passive memory was tested next (Exp 210, "
        "also a wall)"
    ),
)


SEEDS = [
    ("mechanisms/functional-valence-dyad-v0/mechanism_card.json", FUNCTIONAL_VALENCE_DYAD),
    ("mechanisms/functional-valence-dyad-v0/adapters.json", DYAD_TO_ACTIVE_SENSING_ADAPTER),
    ("mechanisms/functional-valence-dyad-v0/scorer_refs.json", AFFECT_SCORER_CARD),
    ("mechanisms/communication-scaffold-v0/mechanism_card.json", COMMUNICATION_SCAFFOLD),
    ("geometry_maps/dyad-session-length-curve-v0.json", DYAD_SESSION_LENGTH_CURVE),
    ("geometry_maps/active-sensing-benefit-wall-v0.json", ACTIVE_SENSING_BENEFIT_WALL_GEOM),
    ("geometry_maps/costly-sensing-wall-v0.json", COSTLY_SENSING_WALL_GEOM),
    ("boundary_notes/active-sensing-benefit-wall-v0.json", ACTIVE_SENSING_BENEFIT_WALL),
    ("boundary_notes/costly-sensing-wall-v0.json", COSTLY_SENSING_WALL),
    ("boundary_notes/hidden-state-memory-boundary-v0.json", HIDDEN_STATE_MEMORY_BOUNDARY),
]


def main() -> int:
    for rel, card in SEEDS:
        path = ROOT / rel
        write_json(card.to_dict(), path)
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
