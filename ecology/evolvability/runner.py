"""
ecology.evolvability.runner — orchestration layer for the Evolvability Preflight.

`run_preflight(cfg)` dispatches all requested gates, aggregates verdicts,
writes all artefacts, and returns a PreflightResult.

Scientific honesty contract
---------------------------
- A negative scientific verdict does NOT raise.  Only infra errors propagate.
- The runner does not reinterpret or soften gate verdicts.
- gate raw rows are written verbatim; no post-hoc filtering.
"""
from __future__ import annotations

import dataclasses as D
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import io
from . import verdicts as _v
from .config import PreflightConfig
from .gates import (
    ControllerSpec,
    GateOutcome,
    build_base_cfg,
    run_gifted_benefit,
    run_monomorphic_sweep,
    run_local_pairwise_gradient,
    run_invasion_from_rarity,
    run_density_independent_growth,
    run_cost_sensitivity,
    run_null_guards,
    run_controller_cross_partial,
)


# ---------------------------------------------------------------------------
# PreflightResult
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    """Full result of a preflight run.

    Fields
    ------
    slug              : config slug
    run_id            : directory name (run_id or timestamp)
    aggregate_verdict : AggregateVerdict.value string
    failure_reason    : human-readable reason from aggregate_verdict()
    gates             : list of gate summary dicts (one per gate that ran)
    trait             : trait axis dict
    controller        : controller dict or None
    config_hash       : SHA-256 of the config
    git_commit        : HEAD commit or "unknown"
    artifact_paths    : dict of important output paths (strings)
    repro             : reproducibility dict
    guard_statuses    : list of guard dicts from null_guards (name/status/reason)
    """
    slug: str
    run_id: str
    aggregate_verdict: str
    failure_reason: str
    gates: list
    trait: dict
    controller: Optional[dict]
    config_hash: str
    git_commit: str
    artifact_paths: dict
    repro: dict
    guard_statuses: list

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        return {
            "slug": self.slug,
            "run_id": self.run_id,
            "aggregate_verdict": self.aggregate_verdict,
            "failure_reason": self.failure_reason,
            "gates": self.gates,
            "trait": self.trait,
            "controller": self.controller,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "artifact_paths": self.artifact_paths,
            "repro": self.repro,
            "guard_statuses": self.guard_statuses,
        }


# ---------------------------------------------------------------------------
# CSV flattening helper
# ---------------------------------------------------------------------------

def _gate_csv_row(name: str, outcome: GateOutcome) -> dict:
    """Flatten a GateOutcome aggregate to a flat CSV row.  Missing keys -> ''."""
    agg = outcome.aggregate

    def _get(key, default=""):
        v = agg.get(key, default)
        return "" if v is None else v

    base = {"name": name, "verdict": outcome.verdict}

    if name == "local_pairwise_gradient":
        base.update({
            "wins": _get("wins"),
            "n_valid": _get("n_valid"),
            "mean_effect": _get("mean_effect"),
            "mean_s": _get("mean_s"),
        })
    elif name == "monomorphic_sweep":
        base.update({
            "optimum_h": _get("optimum_h"),
            "above_resident": _get("above_resident"),
            "survivable": _get("survivable"),
        })
    elif name == "gifted_benefit":
        base.update({"mean_delta": _get("mean_delta")})
    elif name == "invasion_from_rarity":
        base.update({
            "increase_count": _get("increase_count"),
            "n_valid": _get("n_valid"),
        })
    elif name == "density_independent_growth":
        base.update({"mean_delta_r": _get("mean_delta_r")})
    elif name == "cost_sensitivity":
        base.update({
            "sign_change_cost": _get("sign_change_cost"),
            "unaffordable_cost": _get("unaffordable_cost"),
        })
    elif name == "controller_cross_partial":
        eff = agg.get("corner_effects") or {}
        base.update({
            "cross_partial": eff.get("cross_partial", ""),
            "B_ll": _get("B_ll"),
            "B_hh": _get("B_hh"),
        })
    elif name == "null_guards":
        base.update({
            "all_pass": _get("all_pass"),
            "n_guards": len(agg.get("guards", [])),
        })

    return base


# ---------------------------------------------------------------------------
# run_preflight
# ---------------------------------------------------------------------------

def run_preflight(
    cfg: PreflightConfig,
    *,
    run_id: Optional[str] = None,
) -> PreflightResult:
    """Run all requested gates and return a PreflightResult.

    Scientific failures (negative verdicts) are NOT raised — they are returned
    as part of the result.  Only genuine infrastructure errors propagate.

    Parameters
    ----------
    cfg    : full preflight configuration
    run_id : optional explicit run ID; uses a timestamp if None
    """
    # ------------------------------------------------------------------
    # 1. Build base EcologyConfig
    # ------------------------------------------------------------------
    base_cfg = build_base_cfg(cfg.base_scenario, cfg.horizon, cfg.base_overrides)
    if cfg.founder_overrides:
        base_cfg = D.replace(
            base_cfg,
            founder=D.replace(base_cfg.founder, **cfg.founder_overrides),
        )

    win, lose = cfg.effective_thresholds()
    min_valid = cfg.min_valid_seeds
    seeds = list(cfg.seeds)
    window = tuple(cfg.measurement_window)
    min_pop = cfg.min_population
    axis = cfg.trait

    # ------------------------------------------------------------------
    # 2. Create run directory
    # ------------------------------------------------------------------
    run_dir = io.new_run_dir(cfg.output_dir, cfg.slug, run_id)
    raw_dir = run_dir / "raw"
    effective_run_id = run_dir.name

    # ------------------------------------------------------------------
    # 3. Write config artefacts
    # ------------------------------------------------------------------
    cfg_hash = cfg.config_hash()
    git_sha = io.git_commit() or "unknown"

    io.write_json(run_dir / "config.json", cfg.to_dict())
    io.write_text(run_dir / "config_hash.txt", cfg_hash + "\n")
    io.write_text(run_dir / "git_commit.txt", git_sha + "\n")

    # ------------------------------------------------------------------
    # 4. Gate dispatch — in canonical order, skipping those not in cfg.gates
    # ------------------------------------------------------------------
    gate_order = [
        "gifted_benefit",
        "monomorphic_sweep",
        "local_pairwise_gradient",
        "invasion_from_rarity",
        "density_independent_growth",
        "cost_sensitivity",
        "controller_cross_partial",
        "null_guards",
    ]
    requested = set(cfg.gates)
    outcomes: dict[str, GateOutcome] = {}
    pairwise_ext: Optional[float] = None

    for gate_name in gate_order:
        if gate_name not in requested:
            continue

        gp = cfg.gate_params.get(gate_name, {})

        if gate_name == "gifted_benefit":
            outcome = run_gifted_benefit(base_cfg, axis, seeds, **gp)

        elif gate_name == "monomorphic_sweep":
            outcome = run_monomorphic_sweep(
                base_cfg, axis, seeds, list(cfg.monomorphic_grid),
                min_pop=min_pop, **gp,
            )

        elif gate_name == "local_pairwise_gradient":
            outcome = run_local_pairwise_gradient(
                base_cfg, axis, seeds,
                win_threshold=win,
                lose_threshold=lose,
                min_valid=min_valid,
                window=window,
                min_pop=min_pop,
                **gp,
            )
            pairwise_ext = outcome.validity_flags.get("extinct_fraction")

        elif gate_name == "invasion_from_rarity":
            outcome = run_invasion_from_rarity(
                base_cfg, axis, seeds,
                win_threshold=win,
                lose_threshold=lose,
                min_valid=min_valid,
                window=window,
                min_pop=min_pop,
                **gp,
            )

        elif gate_name == "density_independent_growth":
            outcome = run_density_independent_growth(base_cfg, axis, seeds, **gp)

        elif gate_name == "cost_sensitivity":
            outcome = run_cost_sensitivity(
                base_cfg, axis, seeds, list(cfg.cost_values),
                win_threshold=win,
                lose_threshold=lose,
                min_valid=min_valid,
                window=window,
                min_pop=min_pop,
                **gp,
            )

        elif gate_name == "controller_cross_partial":
            if cfg.controller is None:
                # Record a skipped note without crashing
                outcomes[gate_name] = GateOutcome(
                    name="controller_cross_partial",
                    question="N/A — no controller configured",
                    metric="",
                    raw_rows=[],
                    per_seed=[],
                    aggregate={"note": "skipped: cfg.controller is None"},
                    verdict="",
                    validity_flags={},
                    interpretation="Gate skipped because cfg.controller is None.",
                )
                continue
            spec = ControllerSpec(
                config_field=cfg.controller.config_field,
                low_value=cfg.controller.low_value,
                high_value=cfg.controller.high_value,
            )
            outcome = run_controller_cross_partial(base_cfg, axis, spec, seeds, **gp)

        elif gate_name == "null_guards":
            outcome = run_null_guards(
                base_cfg, axis, seeds,
                min_pop=min_pop,
                pairwise_extinct_fraction=pairwise_ext,
                **gp,
            )

        else:
            continue

        # Write raw JSONL — each row augmented with slug / run_id / config_hash
        raw_path = raw_dir / f"{gate_name}.jsonl"
        augmented_rows = [
            {**row, "slug": cfg.slug, "run_id": effective_run_id, "config_hash": cfg_hash}
            for row in outcome.raw_rows
        ]
        io.write_jsonl(raw_path, augmented_rows)

        outcomes[gate_name] = outcome

    # ------------------------------------------------------------------
    # 5. Aggregate verdict
    # ------------------------------------------------------------------
    local_outcome = outcomes.get("local_pairwise_gradient")
    if local_outcome is not None and local_outcome.verdict:
        gradient = _v.GradientVerdict(local_outcome.verdict)
    else:
        gradient = _v.GradientVerdict.NO_VERDICT

    gifted_outcome = outcomes.get("gifted_benefit")
    benefit: Optional[_v.BenefitVerdict] = (
        _v.BenefitVerdict(gifted_outcome.verdict)
        if gifted_outcome is not None and gifted_outcome.verdict
        else None
    )

    mono_outcome = outcomes.get("monomorphic_sweep")
    monomorphic_above: Optional[bool] = (
        mono_outcome.aggregate.get("above_resident")
        if mono_outcome is not None else None
    )
    monomorphic_surv: Optional[bool] = (
        mono_outcome.aggregate.get("survivable")
        if mono_outcome is not None else None
    )

    ctrl_outcome = outcomes.get("controller_cross_partial")
    crosspartial: Optional[_v.CrossPartialVerdict] = None
    if ctrl_outcome is not None and ctrl_outcome.verdict:
        try:
            crosspartial = _v.CrossPartialVerdict(ctrl_outcome.verdict)
        except ValueError:
            crosspartial = None

    null_outcome = outcomes.get("null_guards")
    guards_all_pass: bool = (
        null_outcome.aggregate.get("all_pass", True)
        if null_outcome is not None else True
    )

    agg_verdict, reason = _v.aggregate_verdict(
        gradient=gradient,
        benefit=benefit,
        monomorphic_above_resident=monomorphic_above,
        monomorphic_survivable=monomorphic_surv,
        crosspartial=crosspartial,
        guards_all_pass=guards_all_pass,
    )

    # ------------------------------------------------------------------
    # 6. Guard statuses
    # ------------------------------------------------------------------
    guard_statuses: list = (
        null_outcome.aggregate.get("guards", [])
        if null_outcome is not None else []
    )

    # ------------------------------------------------------------------
    # 7. Build gate summary list
    # ------------------------------------------------------------------
    gate_summaries = []
    for gate_name in gate_order:
        if gate_name not in outcomes:
            continue
        o = outcomes[gate_name]
        raw_rel = (
            f"raw/{gate_name}.jsonl"
            if (raw_dir / f"{gate_name}.jsonl").exists()
            else ""
        )
        gate_summaries.append({
            "name": o.name,
            "question": o.question,
            "metric": o.metric,
            "verdict": o.verdict,
            "aggregate": o.aggregate,
            "validity_flags": o.validity_flags,
            "interpretation": o.interpretation,
            "raw_path": raw_rel,
        })

    # ------------------------------------------------------------------
    # 8. Build PreflightResult
    # ------------------------------------------------------------------
    repro = {
        "python": sys.version.split()[0],
        "deterministic": cfg.deterministic,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
    }

    artifact_paths = {
        "run_dir": str(run_dir),
        "config_json": str(run_dir / "config.json"),
        "summary_json": str(run_dir / "summary.json"),
        "summary_csv": str(run_dir / "summary.csv"),
        "report_md": str(run_dir / "report.md"),
        "raw_dir": str(raw_dir),
    }

    result = PreflightResult(
        slug=cfg.slug,
        run_id=effective_run_id,
        aggregate_verdict=agg_verdict.value,
        failure_reason=reason,
        gates=gate_summaries,
        trait=axis.to_dict(),
        controller=cfg.controller.to_dict() if cfg.controller is not None else None,
        config_hash=cfg_hash,
        git_commit=git_sha,
        artifact_paths=artifact_paths,
        repro=repro,
        guard_statuses=guard_statuses,
    )

    # ------------------------------------------------------------------
    # 9. Write summary artefacts
    # ------------------------------------------------------------------
    io.write_json(run_dir / "summary.json", result.to_dict())

    # CSV: one row per gate
    csv_rows = [
        _gate_csv_row(g, outcomes[g])
        for g in gate_order
        if g in outcomes
    ]
    csv_fieldnames = [
        "name", "verdict",
        "wins", "n_valid", "mean_effect", "mean_s",
        "optimum_h", "above_resident", "survivable",
        "mean_delta", "mean_delta_r",
        "increase_count",
        "sign_change_cost", "unaffordable_cost",
        "cross_partial", "B_ll", "B_hh",
        "all_pass", "n_guards",
    ]
    io.write_csv(run_dir / "summary.csv", csv_rows, fieldnames=csv_fieldnames)

    # Report markdown
    from . import report as _report
    report_md = _report.build_report(result, cfg)
    io.write_text(run_dir / "report.md", report_md)

    return result
