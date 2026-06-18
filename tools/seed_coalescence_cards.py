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


# ── recipe-symmetry-breaking-v0 (VALIDATED — the flagship durable finding) ─────
RECIPE_SYMMETRY_BREAKING = MechanismCard(
    mechanism_id="recipe-symmetry-breaking-v0",
    mechanism_type="hidden-state-belief",
    status="validated",
    source_experiments=[26, 27, 28, 31, 34, 35],
    claim=(
        "On the RECIPE — embodiment + grounding + continuous registered experience "
        "(belief never reset) + ONE innate anchor (give the sensory map A OR the motor "
        "model B, not both learned from noise) + a few taught word<->concept labels — a toy "
        "active-inference creature runs the full chain: perceives (place fields self-organize) "
        "-> learns facts -> forms grounded valence (low free energy = comfortable) -> plans "
        "and acts -> forms its OWN individual values -> answers in words what it values. "
        "Functional only, toy scale; the symmetry-breaker is the RECIPE."
    ),
    works_when=[
        "exactly one of A (sensory map) or B (motor model) is provided as the innate anchor",
        "belief is continuous and never reset per episode (registered experience)",
        "word<->concept labels are few-shot taught (not self-generated)",
        "the body is embodied (gridworld) with grounded movement",
    ],
    fails_when=[
        "both A and B are learned from pure noise simultaneously -> collapse (Exp 31)",
        "a pure disembodied symbol stream (see disembodied-stream-collapse-v0)",
        "no anchor at all (fully tabula-rasa) — a documented research frontier, not achieved",
    ],
    required_conditions=[
        "embodiment, grounding, continuous registered experience, ONE innate anchor, "
        "taught labels",
    ],
    reusable_interface=(
        "the continuous-substrate creature spine (creature/, e.g. nira) reuses the recipe in "
        "richer form; same architecture + different history -> different opinion (Exp 26)"
    ),
    inputs=["embodied grounded observations", "a few taught word<->concept label pairs"],
    outputs=[
        "self-organized place fields",
        "grounded valence (= -free energy)",
        "self-formed individual values + a verbal answer about them",
    ],
    state_requirements=["continuous belief across episodes", "one innate generative factor (A or B)"],
    metrics=["held-out surprise (bits/char or nats)", "value-vector divergence across histories"],
    falsifiers=[
        "fully tabula-rasa learning (no innate anything) producing stable distinct structure "
        "— a documented OPEN PROBLEM, not achieved",
        "emergent compositional grammar from scratch — a documented ceiling (open_problem.html)",
    ],
    known_confounds=[
        "the verbal self-report is a TAUGHT template: the labels are taught, the CONTENT "
        "(which feature it values and why) is self-formed (Exp 28) — never conflate the two",
        "toy scale (3-color value space, 5x5-6x6 grids, Dirichlet/categorical models)",
        "the chain works at the PROVIDED richness; scaling is the documented long arc",
    ],
    next_compositions=[
        "re-confirmed on the continuous-substrate creature (Exp 141-151)",
        "feeds belief-like state -> functional-valence-dyad-v0 and active sensing",
    ],
)

DISEMBODIED_STREAM_COLLAPSE = BoundaryNote(
    boundary_id="disembodied-stream-collapse-v0",
    source_experiments=[31, 135],
    failed_mechanism="unsupervised emergence of latent structure from a disembodied symbol "
    "stream (or joint learning of both A and B from pure noise)",
    observed_failure=(
        "Emergence collapses: symmetric saddle / posterior collapse / non-identifiability / "
        "mean-field severs cross-factor inference. Exp 31: learning BOTH the sensory map A and "
        "the motor model B from pure noise produces a degenerate fixed point (all states map to "
        "one) even with unique sensing — embodiment alone with ZERO priors still collapses. "
        "Exp 135 (continuous-conjugate rematch): a mass-linear erosion law delta(n)=n/(kappa_eff+n) "
        "within 0.015, and the substrate-independent twin ratio (~1.52) confirms the collapse "
        "clause is NOT a tabular artifact — it holds for the continuous substrate too."
    ),
    tested_conditions=[
        "both A and B learned from noise (Exp 31)",
        "tabular vs continuous-conjugate (NIW) substrate (Exp 135)",
        "NIW prior strength kappa_0 banking mass against noise",
    ],
    excluded_confounds=[
        "substrate-specificity (collapse holds for tabular AND continuous)",
        "unique-sensing rescue (collapse persists even with unique sensing, Exp 31)",
    ],
    implication=(
        "The collapse is a property of online Bayesian inference without embodied grounding "
        "and an innate anchor, not a quirk of one parameterization. The symmetry-breaker is "
        "the RECIPE (see recipe-symmetry-breaking-v0)."
    ),
    next_safe_region_to_test=(
        "characterize the critical noise fraction at which collapse becomes irreversible as a "
        "function of kappa_0 in the continuous substrate (the erosion law gives the functional "
        "form). Fully tabula-rasa structure remains a genuine research frontier, not a next step."
    ),
)

# ── meta-calibration-n3-v0 (VALIDATED at toy richness) ────────────────────────
META_CALIBRATION_N3 = MechanismCard(
    mechanism_id="meta-calibration-n3-v0",
    mechanism_type="identity-self-modeling",
    status="validated",
    source_experiments=[155, 156, 157, 158, 159, 162, 163, 167, 168],
    claim=(
        "A third-order self-model (N3) — a controller that detects miscalibration in the "
        "creature's own metacognitive diagnoses (N2) and rewrites N2's window parameter — is "
        "constructible and passes the anti-regress test at this world richness: N3 owns a "
        "regime-adaptive control surface over the N2 dial that N2 lacks and no constant window "
        "matches. Agency-over-metacognition is SUPPORTED at toy richness."
    ),
    works_when=[
        "the world has a hidden context alternating slower than N2's fixed window (the "
        "rung-1 blind-spot failure regime)",
        "the label distribution under the correct dial collapses to one class so the "
        "lock-on-consistency controller can fire",
        "the regime is stationary enough for label streams to settle",
    ],
    fails_when=[
        "the RATCHET LAW: a consistency-locked dial is ascent-only — a too-large dial in a "
        "valid world produces consistent (indistinguishable-from-correct) diagnoses, so the "
        "freeze holds it high; valid-segment deficit 0.169 > 0.15 no-harm bar (Exp 168)",
    ],
    required_conditions=[
        "N2 must already exist on the body (Exp 155-159 prereq: per-place expected-uncertainty "
        "channel + OK/NOISE/STRUCTURAL classifier)",
        "the window blind-spot failure regime must be present",
        "K=8 lock length and the dial set are PROVIDED constants",
    ],
    reusable_interface=(
        "forecast-scoring trust monitor (labels make implicit promises scored against the "
        "creature's own next-100-step record) + a lock-on-label-consistency controller over "
        "the N2 window"
    ),
    inputs=["the creature's own N2 diagnoses (OK/NOISE/STRUCTURAL) and forecast record"],
    outputs=["a regime-adaptive N2 window setting (200->400->800->1600) that locks on STRUCTURAL"],
    metrics=["trust gap (1.0000 valid vs 0.6897 broken)", "lock latency", "combined score (1.0 vs 0.7 best constant)"],
    falsifiers=[
        "epiphenomenality (overrides scattered, not concentrated) — NOT fired: overrides "
        "concentrate in broken segments (0.75)",
        "indiscriminate overriding without benefit — NOT fired: helps +0.50 where broken",
    ],
    known_confounds=[
        "all N3 constants (K, dial set, trust thresholds) are PROVIDED design",
        "behavioral consequences (acting on the repaired diagnosis) are UNTESTED — this is "
        "diagnosis-layer authority only",
        "the ratchet residual (no descent driver) is a controller-design gap, not layer "
        "non-existence; Brier-style scoring FAILS (tracks world difficulty, not brokenness)",
    ],
    next_compositions=[
        "homeostatic bias toward the smallest consistent dial (the named ratchet fix)",
        "a behavioral test that acts on the repaired diagnosis",
    ],
)

# ── identity-n4: a real monitor (constrained) + a commitment boundary ─────────
IDENTITY_N4_MONITOR = MechanismCard(
    mechanism_id="identity-n4-monitor-v0",
    mechanism_type="identity-self-modeling",
    status="constrained",
    source_experiments=[176, 177, 178, 179, 180],
    claim=(
        "A read-only linear-drift identity monitor (N4) detects functional identity "
        "displacement — mismatch between self-predicted and actual value-vector evolution — "
        "with median AUROC 0.894 (8/8 >= 0.8) and is SPECIFIC against a value-neutral scramble "
        "(which inverts the signal BELOW quiet baseline). Detection is real; defense is not "
        "(see identity-n4-commitment-v0). 'Identity' = value-vector ordering on a 3-color "
        "simplex; functional, not selfhood."
    ),
    works_when=[
        "snapshot cadence 100 steps, drift window 1000 steps, horizon 100 steps",
        "displacement is a genuine value-vector reordering (not value-neutral noise)",
    ],
    fails_when=[
        "it is read-only: detection without defense — coupling it to commitment control is "
        "CONFIG not agency at this richness (identity-n4-commitment-v0)",
    ],
    required_conditions=["a linear-drift self-prediction over the value vector; per-burst-matched scramble control"],
    reusable_interface="a read-only displacement detector over the creature's value-vector trajectory",
    inputs=["the creature's value-vector trajectory"],
    outputs=["a displacement score (AUROC ~0.894 vs a value-neutral control)"],
    metrics=["AUROC (median 0.894, range 0.859-0.911)", "specificity vs value-neutral scramble (signal inversion)"],
    falsifiers=["if a value-neutral scramble RAISED the signal, the monitor would be non-specific — it does the opposite"],
    known_confounds=[
        "the monitor FORM is provided; contents (values, drift history, mismatches) are self-formed",
        "'identity' = value-vector ordering (functional policy-continuity), not selfhood/biography",
        "settled self is world-determined (occupancy equilibrium), not biography-determined",
    ],
    next_compositions=["feeds identity-n4-commitment-v0 (where commitment control is the boundary)"],
)

IDENTITY_N4_COMMITMENT = BoundaryNote(
    boundary_id="identity-n4-commitment-v0",
    source_experiments=[181, 182, 183, 184, 185, 186, 187, 188, 189, 190],
    failed_mechanism="N4 commitment control AS AGENCY-OVER-IDENTITY (self-predicted drift "
    "regulating value-revision inertia so transient pressure is resisted and sustained "
    "evidence accepted)",
    observed_failure=(
        "Commitment control is CONFIG, not agency, at this richness. Write-gating defends "
        "nothing (Exp 181); an evidence-based concession surrenders mid-attack (E*=600 -> "
        "concession ~703 steps into an 800-step burst, 2/8), while fixed-horizon freeze gates "
        "H1200-H3000 pass BOTH declared bars (Exp 183) — a stopwatch, not agency. The crack "
        "(Exp 186, blind-verified 6/6): nine attack schedules where NO constant both defends "
        "and revises while an oracle defends everywhere. It closed CONSTRUCTIVELY: INT-C2900 "
        "(a stopwatch on continuous pressure, reset by gaps) passes 9/9 at normal tolerance "
        "(Exp 187), and one online regulated controller REG-TB ties it at EXACT defense parity "
        "(Exp 188) — but at fixed-L geometry regulation only matches the right clock, never "
        "beats it (the single-stretch ambiguity bound). At variable-L it uniquely defends "
        "(kappa-reach, Exp 189) yet loses revision to the FLICKER TAX (onset pressure-flicker "
        "resets the continuity clock; de-assert runs 25-2,600 steps overlap attack gaps "
        "525-1,175 steps — no admissible hysteresis), and Exp 190 refuted both named repairs "
        "at design time."
    ),
    tested_conditions=[
        "write-gating (Exp 181), freeze sufficiency (Exp 182), evidence concession + fixed-H "
        "freeze (Exp 183)",
        "the 9-schedule crack map (Exp 184-186); INT-C2900 stopwatch (Exp 187); REG-TB "
        "regulated controller (Exp 188); variable-L escalating trains (Exp 189-190)",
    ],
    excluded_confounds=[
        "blind-verified verdicts (PROTOCOL 4.5); pre-registered bars",
        "the monitor itself is real and specific (identity-n4-monitor-v0) — the failure is the "
        "commitment LAYER, not detection",
    ],
    implication=(
        "The universal-constant law governs at the identity level: regulation is only "
        "necessary where no feasible constant covers all regimes. At fixed-L a constant "
        "stopwatch suffices (regulation ties it); at variable-L regulation uniquely defends but "
        "cannot revise in time, and the flicker tax is STRUCTURAL (the reset that buys train "
        "defense and the one that taxes revision are the same event class). This is the verdict "
        "at THIS body/richness — NOT a universal impossibility of N4 commitment."
    ),
    next_safe_region_to_test=(
        "the tight-tolerance core (no surface covers where tolerance < a single burst) and "
        "post-release retention remain open edges — each re-opens only on a human word"
    ),
)

# ── online-structure-growth-v0 (VALIDATED; the wall was a methodological artifact) ──
ONLINE_STRUCTURE_GROWTH = MechanismCard(
    mechanism_id="online-structure-growth-v0",
    mechanism_type="hidden-state-belief",
    status="validated",
    source_experiments=[152, 153, 154],
    claim=(
        "Detector-triggered, creature-accepted autonomous model growth (detect inadequacy -> "
        "grow latent structure -> quiet) runs end-to-end (24/24 runs, 100% acceptance, zero "
        "detector events in the final 1000 steps across three layouts) once predictive "
        "evaluation uses NORMALIZED mixture densities. The prior five-design 'growth wall' "
        "belonged to the capped-footprint EVALUATION CONVENTION, not to the geometry of online "
        "structure growth."
    ),
    works_when=[
        "all predictive evaluation (surprise, detector, valence, probation) uses normalized "
        "mixture densities — sharp components are louder where they concentrate, so fitted "
        "pieces earn their keep",
        "the growth machinery (EM fitting, probation protocol) is provided",
    ],
    fails_when=[
        "the unnormalized-footprint convention (pre-Exp-154): footprints cap at 1 regardless "
        "of sharpness, so splitting a broad component into K sharp pieces divides predictive "
        "voice by K with nothing gained — the exact arithmetic (p ~0.25->0.067, ~2.7 nats) "
        "that produced the 1.7-2.8 nat probation penalty (Exp 152 autopsy)",
    ],
    required_conditions=[
        "normalized predictive scale; an inadequacy detector calibrated to that scale; the "
        "provided EM+probation growth machinery",
    ],
    reusable_interface="creature growth.py: detect -> grow (batch-jump EM) -> probation -> accept/reject by own lived surprise",
    inputs=["the creature's own predictive surprise on lived experience"],
    outputs=["accepted new latent components (when/what/whether-to-keep self-formed)"],
    metrics=["surprise drop (0.58-1.18 nats)", "ceiling events (0 of 24/24)", "acceptance (53/53 = 100%)"],
    falsifiers=["if growth still failed under normalized evaluation, the wall would be geometric not methodological — it does not"],
    known_confounds=[
        "one growth mechanism (batch-jump) under one acceptance protocol; other strategies "
        "untested at this evaluation convention",
        "the detector threshold (0.7) was reused on the normalized scale where it is conservative",
        "'wall fell' is bounded to this world family and toy scale; the M4 valence-range "
        "narrowing under normalization is a standing PREDICTION, not a confirmed result",
    ],
    next_compositions=["honest normalized evaluation is a prerequisite for any growth/structure-learning composition"],
)


# ── self-other-modeling arc (Exp 228-234) ────────────────────────────────────
FUNCTIONAL_GOAL_INFERENCE = MechanismCard(
    mechanism_id="functional-goal-inference-v0",
    mechanism_type="hidden-state-belief",
    status="constrained",
    source_experiments=[228, 229, 230, 231, 232, 233, 234],
    claim=(
        "A clade-mate can functionally infer ANOTHER agent's latent goal from its trajectory "
        "(a provided goal-directed-policy model; the goal is inferred, never provided) and "
        "predict the other better than learning the other's transition dynamics -- but ONLY at "
        "TRANSITIONS (cold-start and goal-changes), in proportion to the other's non-stationarity; "
        "in steady state an adaptive transition model matches it. The first crossing of the "
        "solipsism gap social-emergence named at closure, at toy scale and LIGHT (policy provided)."
    ),
    works_when=[
        "cold-start: goal-inference is near-oracle within a few steps; the learned-transition "
        "baseline needs hundreds (Exp 229, early edge +0.10, 8/8)",
        "goal-changes: re-tracks a switched goal far faster than re-learning transitions "
        "(Exp 230, post-switch +0.13)",
        "the edge GROWS monotonically with the other's change-frequency (Exp 231)",
    ],
    fails_when=[
        "steady state: an adaptive learned-transition model catches up (Exp 230/231 overall edge ~0)",
        "behaviorally: reactive stigmergy already near-optimally coordinates, leaving no headroom; "
        "naive proactive yielding HURTS (Exp 234, joint 0.46 vs 0.77)",
    ],
    required_conditions=[
        "a PROVIDED goal-directed-policy model (eps-greedy BFS toward a target); only the goal is inferred",
        "a NON-STATIONARY other for a sustained benefit (the benefit is transition-localised)",
        "the binding control is a LEARNED adaptive-transition model, not a naive random walk",
    ],
    inputs=["the other's observed position/trajectory"],
    outputs=["a posterior over the other's latent goal (q_goal/q_target); a next-cell prediction"],
    metrics=["next-cell prediction accuracy", "post-switch re-tracking edge", "goal-posterior concentration"],
    falsifiers=[
        "must beat a LEARNED adaptive-transition baseline (the binding control), not just a naive "
        "random-walk tracker (Exp 228 caveat -> Exp 229+)",
        "predeclare a manipulation check that the inference is ACTUALLY hard (L38) -- a high goal "
        "posterior means the test was easy",
    ],
    known_confounds=[
        "LIGHT: the policy is provided and the goal is often trivially inferable (q->0.999 at 3-way, "
        "0.93 at 25-way separable targets)",
        "inference difficulty is set by behavioral SEPARABILITY, not goal-space SIZE (Exp 232)",
        "on this gridworld goal-directed BFS is intrinsically LEGIBLE -- target geometry cannot make "
        "inference hard (Exp 233); see self-other-substrate-legibility-wall-v0",
    ],
    next_compositions=[
        "a substrate where the simple baselines FAIL -- a less-legible/partially-observed other, or a "
        "coordination task stigmergy cannot solve -- is required to test whether ToM pays beyond transitions",
    ],
)

SELF_OTHER_LEGIBILITY_WALL = BoundaryNote(
    boundary_id="self-other-substrate-legibility-wall-v0",
    source_experiments=[232, 233, 234],
    failed_mechanism="demonstrating that structured self-other modeling (theory-of-mind) genuinely "
    "beats SIMPLE baselines under a HARD inference or in BEHAVIOR, on the toy gridworld substrate",
    observed_failure=(
        "Three consecutive degenerate-baseline results, each caught by a predeclared manipulation/sanity "
        "gate (L38). (Exp 232) inference cannot be made hard by goal-space SIZE -- 4 corners in a 25-way "
        "space stay trivially separable (q->0.93); difficulty = behavioral separability, not cardinality. "
        "(Exp 233) target geometry CANNOT create inference-ambiguity -- even adjacent cells are separable "
        "(q->0.89) because goal-directed BFS broadcasts the goal. (Exp 234) reactive stigmergy already "
        "near-optimally coordinates (1.6% crowding), leaving no behavioral headroom for ToM; naive "
        "proactive yielding HURTS (joint 0.46 vs 0.77)."
    ),
    tested_conditions=[
        "25-way separable corner targets (Exp 232)",
        "ambiguous center-adjacent cluster targets (Exp 233)",
        "shared-resource coordination: proactive model-based yield vs reactive comfort-gated stigmergy (Exp 234)",
    ],
    excluded_confounds=[
        "the L38 manipulation check was predeclared and gated each result (the goal posterior must stay "
        "low if inference is hard; the baseline must crowd) -- so the degeneracy was caught BEFORE being "
        "mis-read as a positive",
    ],
    implication=(
        "On this toy gridworld the SIMPLE baselines (adaptive transition learning; reactive comfort-gated "
        "stigmergy) are robustly near-optimal, so structured self-other modeling has little room to beat "
        "them -- predictively in steady state OR behaviorally. The functional-goal-inference benefit "
        "(functional-goal-inference-v0) is REAL but confined to TRANSITIONS."
    ),
    next_safe_region_to_test=(
        "a LESS-LEGIBLE other (a noisy/high-temperature policy, or partial/intermittent observation so "
        "the goal is not broadcast) AND/OR a coordination task stigmergy cannot solve (anti-coordination "
        "requiring commitment, or a non-depleting shared good) -- a substrate where the simple baseline FAILS"
    ),
)


SEEDS = [
    ("mechanisms/functional-valence-dyad-v0/mechanism_card.json", FUNCTIONAL_VALENCE_DYAD),
    ("mechanisms/functional-valence-dyad-v0/adapters.json", DYAD_TO_ACTIVE_SENSING_ADAPTER),
    ("mechanisms/functional-valence-dyad-v0/scorer_refs.json", AFFECT_SCORER_CARD),
    ("mechanisms/communication-scaffold-v0/mechanism_card.json", COMMUNICATION_SCAFFOLD),
    ("mechanisms/recipe-symmetry-breaking-v0/mechanism_card.json", RECIPE_SYMMETRY_BREAKING),
    ("mechanisms/meta-calibration-n3-v0/mechanism_card.json", META_CALIBRATION_N3),
    ("mechanisms/identity-n4-monitor-v0/mechanism_card.json", IDENTITY_N4_MONITOR),
    ("mechanisms/online-structure-growth-v0/mechanism_card.json", ONLINE_STRUCTURE_GROWTH),
    ("geometry_maps/dyad-session-length-curve-v0.json", DYAD_SESSION_LENGTH_CURVE),
    ("geometry_maps/active-sensing-benefit-wall-v0.json", ACTIVE_SENSING_BENEFIT_WALL_GEOM),
    ("geometry_maps/costly-sensing-wall-v0.json", COSTLY_SENSING_WALL_GEOM),
    ("boundary_notes/active-sensing-benefit-wall-v0.json", ACTIVE_SENSING_BENEFIT_WALL),
    ("boundary_notes/costly-sensing-wall-v0.json", COSTLY_SENSING_WALL),
    ("boundary_notes/hidden-state-memory-boundary-v0.json", HIDDEN_STATE_MEMORY_BOUNDARY),
    ("boundary_notes/disembodied-stream-collapse-v0.json", DISEMBODIED_STREAM_COLLAPSE),
    ("boundary_notes/identity-n4-commitment-v0.json", IDENTITY_N4_COMMITMENT),
    ("mechanisms/functional-goal-inference-v0/mechanism_card.json", FUNCTIONAL_GOAL_INFERENCE),
    ("boundary_notes/self-other-substrate-legibility-wall-v0.json", SELF_OTHER_LEGIBILITY_WALL),
]


def main() -> int:
    for rel, card in SEEDS:
        path = ROOT / rel
        write_json(card.to_dict(), path)
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
