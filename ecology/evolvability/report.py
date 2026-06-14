"""
ecology.evolvability.report — Markdown report builder for the Evolvability Preflight.

`build_report(result, cfg)` returns a Markdown string with a fixed section layout:
  # Evolvability Preflight Report: <slug>
  ## Verdict
  ## Executive Summary
  ## Trait Axis
  ## Gate Results
  ## Anti-Cheat / Null Checks
  ## What This Does NOT Prove
  ## Recommended Next Action
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import PreflightResult
    from .config import PreflightConfig

from .verdicts import AggregateVerdict


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _verdict_section(result: "PreflightResult") -> str:
    return f"## Verdict\n\n{result.aggregate_verdict}  —  {result.failure_reason}\n"


def _executive_summary(result: "PreflightResult") -> str:
    """Plain-English summary: which gate decided it, the key numbers, and what it means."""
    lines = ["## Executive Summary", ""]

    # Find the deciding gate
    local_gate = next(
        (g for g in result.gates if g["name"] == "local_pairwise_gradient"), None
    )
    mono_gate = next(
        (g for g in result.gates if g["name"] == "monomorphic_sweep"), None
    )
    gifted_gate = next(
        (g for g in result.gates if g["name"] == "gifted_benefit"), None
    )
    ctrl_gate = next(
        (g for g in result.gates if g["name"] == "controller_cross_partial"), None
    )

    agg = result.aggregate_verdict

    if agg == AggregateVerdict.PASS_LOCAL_GRADIENT:
        if local_gate:
            agg_d = local_gate.get("aggregate", {})
            lines.append(
                f"The local pairwise gradient gate PASSED. "
                f"The single-step mutant (h={result.trait.get('mutant_value')}) "
                f"outperformed the resident (h={result.trait.get('resident_value')}) "
                f"in {agg_d.get('wins', '?')}/{agg_d.get('n_valid', '?')} valid seeds "
                f"(mean_effect={agg_d.get('mean_effect', '?'):.4g}, "
                f"mean_s={agg_d.get('mean_s', '?'):.4g}). "
                f"All null/anti-cheat guards passed. "
                f"This is the binding criterion for local evolvability."
            )

    elif agg == AggregateVerdict.FAIL_LOCAL_GRADIENT:
        if local_gate:
            agg_d = local_gate.get("aggregate", {})
            lines.append(
                f"The local pairwise gradient gate FAILED. "
                f"The mutant (h={result.trait.get('mutant_value')}) "
                f"did not consistently outperform the resident "
                f"(h={result.trait.get('resident_value')}): "
                f"{agg_d.get('wins', '?')}/{agg_d.get('n_valid', '?')} valid seeds favoured the mutant "
                f"(mean_effect={agg_d.get('mean_effect', '?'):.4g}). "
                f"The local selection gradient is not positive at the current resident value."
            )

    elif agg == AggregateVerdict.GLOBAL_BENEFIT_ONLY:
        detail = ""
        if gifted_gate and gifted_gate.get("verdict") == "BENEFIT":
            agg_d = gifted_gate.get("aggregate", {})
            detail = (
                f" A gifted/cost-off benefit was detected "
                f"(mean_delta={agg_d.get('mean_delta', '?'):.4g}),"
                f" but the local pairwise gradient is not positive."
            )
        elif mono_gate:
            agg_d = mono_gate.get("aggregate", {})
            detail = (
                f" The monomorphic optimum (h*={agg_d.get('optimum_h', '?'):.3g}) "
                f"is above the resident, but the local gradient does not support invasion."
            )
        lines.append(
            f"A global benefit was found but the LOCAL selection gradient is not positive.{detail} "
            f"This means the trait cannot evolve by small mutational steps from the resident — "
            f"there is no uphill path from h={result.trait.get('resident_value')} to the "
            f"global optimum."
        )

    elif agg == AggregateVerdict.CONTROLLER_PAYS_ALONE:
        if ctrl_gate:
            agg_d = ctrl_gate.get("aggregate", {})
            eff = agg_d.get("corner_effects", {}) or {}
            lines.append(
                f"The controller cross-partial gate found that the controller pays alone "
                f"(verdict={ctrl_gate.get('verdict')}). "
                f"cross_partial={eff.get('cross_partial', '?'):.4g}, "
                f"dB_dtheta_lo_h={eff.get('dB_dtheta_lo_h', '?'):.4g}. "
                f"The trait h does not contribute independently to fitness; "
                f"the controller alone drives the benefit."
            )
        else:
            lines.append(
                "The controller cross-partial gate found that the controller pays alone. "
                "The trait h does not contribute independently to fitness."
            )

    elif agg == AggregateVerdict.NO_EFFECT:
        lines.append(
            f"No benefit was detected and the monomorphic optimum is not above the resident "
            f"(h={result.trait.get('resident_value')}). "
            f"The trait appears to have no evolutionary effect in this configuration. "
            f"Consider revising the payoff geometry or trait definition."
        )

    elif agg == AggregateVerdict.NO_VERDICT:
        lines.append(
            f"The preflight could not produce a definitive verdict. "
            f"Reason: {result.failure_reason}. "
            f"Common causes: population collapse, too few valid seeds, or a null guard failure "
            f"indicating a possible artifact."
        )

    else:
        lines.append(f"Aggregate verdict: {agg}. {result.failure_reason}.")

    return "\n".join(lines)


def _trait_section(result: "PreflightResult", cfg: "PreflightConfig") -> str:
    t = result.trait
    lines = [
        "## Trait Axis",
        "",
        f"- **name**: {t.get('name', '?')}",
        f"- **resident_value**: {t.get('resident_value', '?')}",
        f"- **mutant_value**: {t.get('mutant_value', '?')}",
        f"- **low_value**: {t.get('low_value', '?')}",
        f"- **high_value**: {t.get('high_value', '?')}",
        f"- **cost_enabled**: {t.get('cost_enabled', '?')}",
        f"- **cost_floor**: {t.get('cost_floor', '?')}",
        f"- **cost_inefficiency**: {t.get('cost_inefficiency', '?')}",
        f"- **h_trait**: {t.get('h_trait', '?')}",
        f"- **enable_flag**: {t.get('enable_flag', '?')}",
        f"- **backend**: {t.get('backend', '?')}",
    ]
    if t.get("freeze_flag"):
        lines.append(f"- **freeze_flag**: {t.get('freeze_flag')}")
    if t.get("inefficiency_trait"):
        lines.append(f"- **inefficiency_trait**: {t.get('inefficiency_trait')} = {t.get('inefficiency_value', '?')}")
    return "\n".join(lines)


def _gate_results_section(result: "PreflightResult") -> str:
    lines = ["## Gate Results", ""]
    for gate in result.gates:
        name = gate.get("name", "?")
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"**Question**: {gate.get('question', '')}")
        lines.append("")
        lines.append(f"**Metric**: {gate.get('metric', '')}")
        lines.append("")
        verdict = gate.get("verdict", "")
        if verdict:
            lines.append(f"**Verdict**: `{verdict}`")
            lines.append("")

        # Key numbers from aggregate
        agg = gate.get("aggregate", {}) or {}
        if name == "local_pairwise_gradient":
            wins = agg.get("wins", "?")
            n_valid = agg.get("n_valid", "?")
            me = agg.get("mean_effect")
            ms = agg.get("mean_s")
            me_str = f"{me:.4g}" if isinstance(me, float) else str(me)
            ms_str = f"{ms:.4g}" if isinstance(ms, float) else str(ms)
            lines.append(
                f"**Aggregate**: wins {wins}/{n_valid}, "
                f"mean_effect={me_str}, mean_s={ms_str}"
            )
        elif name == "monomorphic_sweep":
            opt_h = agg.get("optimum_h")
            opt_v = agg.get("optimum_value")
            above = agg.get("above_resident")
            surv = agg.get("survivable")
            opt_h_str = f"{opt_h:.3g}" if isinstance(opt_h, float) else str(opt_h)
            opt_v_str = f"{opt_v:.3g}" if isinstance(opt_v, float) else str(opt_v)
            lines.append(
                f"**Aggregate**: optimum_h={opt_h_str}, optimum_value={opt_v_str}, "
                f"above_resident={above}, survivable={surv}"
            )
        elif name == "gifted_benefit":
            md = agg.get("mean_delta")
            md_str = f"{md:.4g}" if isinstance(md, float) else str(md)
            lines.append(f"**Aggregate**: mean_delta={md_str}")
        elif name == "invasion_from_rarity":
            inc = agg.get("increase_count", "?")
            n_valid = agg.get("n_valid", "?")
            lines.append(f"**Aggregate**: increase_count {inc}/{n_valid}")
        elif name == "density_independent_growth":
            mdr = agg.get("mean_delta_r")
            mdr_str = f"{mdr:.4g}" if isinstance(mdr, float) else str(mdr)
            lines.append(f"**Aggregate**: mean_delta_r={mdr_str}")
        elif name == "cost_sensitivity":
            scc = agg.get("sign_change_cost", "none")
            uac = agg.get("unaffordable_cost", "none")
            lines.append(
                f"**Aggregate**: sign_change_cost={scc}, unaffordable_cost={uac}"
            )
            per_cost = agg.get("per_cost", [])
            if per_cost:
                lines.append("")
                lines.append("| cost | wins | n_valid | mean_effect | verdict |")
                lines.append("|------|------|---------|-------------|---------|")
                for pc in per_cost:
                    me = pc.get("mean_effect")
                    me_str = f"{me:.4g}" if isinstance(me, float) else str(me)
                    lines.append(
                        f"| {pc.get('cost')} | {pc.get('wins', '?')} "
                        f"| {pc.get('n_valid', '?')} | {me_str} "
                        f"| {pc.get('verdict', '')} |"
                    )
        elif name == "controller_cross_partial":
            eff = agg.get("corner_effects") or {}
            cp = eff.get("cross_partial")
            cp_str = f"{cp:.4g}" if isinstance(cp, float) else str(cp)
            lines.append(
                f"**Aggregate**: cross_partial={cp_str}, "
                f"B_ll={agg.get('B_ll', '?'):.4g}, "
                f"B_hh={agg.get('B_hh', '?'):.4g}"
            )
        elif name == "null_guards":
            all_pass = agg.get("all_pass", "?")
            n_g = len(agg.get("guards", []))
            lines.append(f"**Aggregate**: all_pass={all_pass}, n_guards={n_g}")
        else:
            # Generic: show note if present
            note = agg.get("note")
            if note:
                lines.append(f"**Note**: {note}")

        # Validity flags
        vf = gate.get("validity_flags", {}) or {}
        if vf:
            vf_str = ", ".join(f"{k}={v}" for k, v in vf.items())
            lines.append("")
            lines.append(f"**Validity flags**: {vf_str}")

        lines.append("")
        interp = gate.get("interpretation", "")
        if interp:
            lines.append(f"**Interpretation**: {interp}")
        lines.append("")

    return "\n".join(lines)


def _guards_section(result: "PreflightResult") -> str:
    lines = ["## Anti-Cheat / Null Checks", ""]
    if not result.guard_statuses:
        lines.append("*(null_guards gate did not run)*")
        return "\n".join(lines)

    lines.append("| Guard | Status | Reason |")
    lines.append("|-------|--------|--------|")
    for g in result.guard_statuses:
        name = g.get("name", "?")
        status = g.get("status", "?")
        reason = g.get("reason", "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | **{status}** | {reason} |")

    return "\n".join(lines)


def _not_prove_section() -> str:
    return """\
## What This Does NOT Prove

- Gifted benefit does not prove evolvability.
- A monomorphic/global optimum does not prove local evolvability.
- A failed preflight does not prove the trait can never evolve in any possible world.
- A positive local gradient does not prove stable long-run organ evolution."""


_NEXT_ACTION_MAP = {
    AggregateVerdict.PASS_LOCAL_GRADIENT: (
        "Run a full evolution candidate (local gradient + guards passed)."
    ),
    AggregateVerdict.FAIL_LOCAL_GRADIENT: (
        "Do not run full evolution; the local gradient failed."
    ),
    AggregateVerdict.GLOBAL_BENEFIT_ONLY: (
        "Redesign payoff geometry; benefit exists but the local gradient is not positive."
    ),
    AggregateVerdict.CONTROLLER_PAYS_ALONE: (
        "Audit the controller confound; the controller pays alone."
    ),
    AggregateVerdict.NO_EFFECT: (
        "No benefit and no optimum above resident; reconsider the trait."
    ),
    AggregateVerdict.NO_VERDICT: (
        "Population invalid/collapsed or insufficient data; fix validity and re-run."
    ),
}


def _next_action_section(result: "PreflightResult") -> str:
    try:
        av = AggregateVerdict(result.aggregate_verdict)
        action = _NEXT_ACTION_MAP.get(av, "See aggregate_verdict for details.")
    except ValueError:
        action = "See aggregate_verdict for details."
    return f"## Recommended Next Action\n\n{action}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_report(result: "PreflightResult", cfg: "PreflightConfig") -> str:
    """Build and return a Markdown report string for a PreflightResult."""
    sections = [
        f"# Evolvability Preflight Report: {result.slug}",
        "",
        _verdict_section(result),
        "",
        _executive_summary(result),
        "",
        _trait_section(result, cfg),
        "",
        _gate_results_section(result),
        "",
        _guards_section(result),
        "",
        _not_prove_section(),
        "",
        _next_action_section(result),
        "",
    ]
    return "\n".join(sections)
