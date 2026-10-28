"""Independent arithmetic and nonperturbation guards for Exp283's observer."""

from dataclasses import replace
from math import sqrt, pi

import numpy as np
import pytest


def test_price_includes_survivor_and_newborn_of_dead_parent():
    from ecology.transmission import price_step

    # Parent 1 survives unchanged; parent 2 dies after producing child 3.
    # Before mean .5; descendant-parent mean .5; after mean .55.
    got = price_step({1: .2, 2: .8}, {1: .2, 3: .9}, {3: 2})
    assert got["n_before"] == got["n_after"] == 2
    assert got["births"] == got["deaths"] == 1
    assert got["mean_before"] == pytest.approx(.5)
    assert got["mean_after"] == pytest.approx(.55)
    assert got["selection"] == pytest.approx(0)
    assert got["transmission"] == pytest.approx(.05)
    assert got["delta"] == pytest.approx(.05)
    assert abs(got["residual"]) < 1e-15


def test_price_weights_multiple_descendants_including_self():
    from ecology.transmission import price_step

    # Mean before .5; parents of descendants [0,0,0]; after [0,.1,.2].
    got = price_step({1: 0., 2: 1.}, {1: 0., 3: .1, 4: .2}, {3: 1, 4: 1})
    assert got["selection"] == pytest.approx(-.5)
    assert got["transmission"] == pytest.approx(.1)
    assert got["delta"] == pytest.approx(-.4)
    assert abs(got["residual"]) < 1e-15


@pytest.mark.parametrize("ancestry", [{}, {3: 99}, {1: 1, 3: 1}, {3: 1, 4: 1}])
def test_price_rejects_missing_or_extra_newborn_ancestry(ancestry):
    from ecology.transmission import price_step

    with pytest.raises(ValueError, match="parent|ancestry"):
        price_step({1: .2}, {1: .2, 3: .3}, ancestry)


@pytest.mark.parametrize("trait", [float("nan"), float("inf"), -.01, 1.01])
def test_price_rejects_nonfinite_or_out_of_bounds_traits(trait):
    from ecology.transmission import price_step

    with pytest.raises(ValueError, match="finite|bound"):
        price_step({1: trait}, {1: .2}, {})


def test_extinction_has_undefined_mean_and_terms_not_a_false_zero():
    from ecology.transmission import price_step

    got = price_step({1: .3}, {}, {})
    assert got["mean_before"] == .3
    assert got["mean_after"] is None
    assert got["selection"] is got["transmission"] is got["residual"] is None
    assert got["deaths"] == 1


def test_clipped_gaussian_at_zero_midpoint_and_symmetric_boundary():
    from ecology.transmission import clipped_gaussian_drift

    got = clipped_gaussian_drift(np.array([0., .5, 1.]), .05)
    # At 20 standard deviations, upper-tail correction is below 1e-88.
    assert got == pytest.approx([.05 / sqrt(2 * pi), 0., -.05 / sqrt(2 * pi)], abs=1e-15)
    x = np.array([.02, .17, .37])
    assert clipped_gaussian_drift(x, .3) == pytest.approx(
        -clipped_gaussian_drift(1 - x, .3), abs=1e-15
    )


def test_clipped_gaussian_agrees_with_independent_quadrature():
    from scipy.integrate import quad
    from scipy.stats import norm
    from ecology.transmission import clipped_gaussian_drift

    # Integrate the *unclipped Gaussian proposal*, splitting at both clip points.
    x, sigma, lo, hi = .1, .4, -.5, 1.5
    density = lambda proposal: norm.pdf(proposal, loc=x, scale=sigma)
    below = (lo - x) * quad(density, -np.inf, lo)[0]
    middle = quad(lambda proposal: (proposal - x) * density(proposal), lo, hi)[0]
    above = (hi - x) * quad(density, hi, np.inf)[0]
    assert clipped_gaussian_drift(x, sigma, lo, hi) == pytest.approx(below + middle + above, abs=1e-12)


def test_zero_sigma_has_no_mutational_drift():
    from ecology.transmission import clipped_gaussian_drift

    assert clipped_gaussian_drift([0., .3, 1.], 0.) == pytest.approx([0., 0., 0.])


@pytest.mark.parametrize("args", [(0., -.1), (0., np.nan), (np.inf, .05), (1.1, .05), (.5, .05, 1., 0.)])
def test_clipping_rejects_invalid_inputs(args):
    from ecology.transmission import clipped_gaussian_drift

    with pytest.raises(ValueError):
        clipped_gaussian_drift(*args)


@pytest.mark.parametrize("arm", ["NEUTRAL", "COST_ONLY", "THERMAL"])
def test_observer_leaves_event_hash_unchanged_and_accounts_every_step(arm):
    from experiments.exp283_boundary_transmission import run_one

    observed = run_one(arm, 9499, horizon=120)
    unobserved = run_one(arm, 9499, horizon=120, observe=False)
    summary = observed["summary"]
    assert summary["events_hash"] == unobserved["summary"]["events_hash"]
    assert summary["accounting_valid"]
    assert len(observed["trajectory"]) == summary["steps_run"]
    assert summary["births"] > 0
    assert summary["cumulative_selection"] + summary["cumulative_transmission"] == pytest.approx(
        summary["final_living_mean"] - summary["initial_living_mean"], abs=1e-12
    )
    assert summary["max_abs_identity_residual"] < 1e-12
    first = observed["trajectory"][0]
    assert first["selection"] == pytest.approx(0.)  # every founder is zero
    assert first["expected_transmission"] == pytest.approx(
        first["births"] / first["n_after"] * .05 / sqrt(2 * pi), abs=1e-15
    )


def test_extinct_run_retains_undefined_final_mean(monkeypatch):
    from experiments import exp283_boundary_transmission as experiment

    original = experiment.make_config
    monkeypatch.setattr(experiment, "make_config", lambda arm, horizon=3000: replace(
        original(arm, horizon), initial_population=0
    ))
    got = experiment.run_one("COST_ONLY", 9499, horizon=5)
    assert got["summary"]["extinct"]
    assert not got["summary"]["reached_horizon"]
    assert got["summary"]["final_living_mean"] is None
    assert got["summary"]["final_newborn_mean_last500"] is None
    assert got["summary"]["accounting_valid"]


def test_neutral_threshold_disables_cost_even_at_upper_bound():
    from ecology.genotype import thermosense_active, thermosense_upkeep
    from experiments.exp283_boundary_transmission import make_config

    cfg = make_config("NEUTRAL")
    largest = replace(cfg.founder, thermosense_intensity=1.)
    assert not thermosense_active(largest, cfg.thermosense_active_threshold)
    assert thermosense_upkeep(largest, cfg.thermosense_upkeep_floor, cfg.thermosense_active_threshold) == 0.
    assert not cfg.enable_temperature


@pytest.mark.parametrize("passing, invalid, want", [(12, False, "POSITIVE"), (11, False, "NEGATIVE"), (16, True, "INVALID")])
def test_verdict_counts_joint_completion_and_trait_bars_without_exclusions(passing, invalid, want):
    from experiments.exp283_boundary_transmission import evaluate_results

    summaries = []
    for arm in ("NEUTRAL", "COST_ONLY", "THERMAL"):
        for offset, seed in enumerate(range(9500, 9516)):
            summaries.append({
                "arm": arm, "seed": seed, "accounting_valid": not invalid,
                "reached_horizon": offset < passing,
                "final_living_mean": .03, "cumulative_selection": -.03,
            })
    result = evaluate_results(summaries)
    assert result["verdict"] == want
    assert result["cost_only_joint_count"] == passing
    assert result["neutral_joint_count"] == passing


def test_incomplete_batch_cannot_earn_a_positive_verdict():
    from experiments.exp283_boundary_transmission import evaluate_results

    assert evaluate_results([])["verdict"] == "INVALID"


def test_expired_batch_deadline_is_retained_as_failed_completion():
    from experiments.exp283_boundary_transmission import run_one

    got = run_one("COST_ONLY", 9499, horizon=10, deadline=0.)["summary"]
    assert got["budget_exceeded"]
    assert not got["reached_horizon"]
    assert got["steps_run"] == 0
    assert got["accounting_valid"]


def test_smoke_persists_full_precision_trajectories_and_excludes_pilot(tmp_path):
    import gzip
    import json
    from experiments.exp283_boundary_transmission import main

    prefix = tmp_path / "smoke"
    main(["--smoke", "--smoke-horizon", "3", "--output-prefix", str(prefix)])
    stored = json.loads(prefix.with_suffix(".json").read_text())
    assert stored["criteria"]["verdict"] == "SMOKE_ONLY"
    assert stored["seeds"] == [9499]
    assert stored["pilot"]["excluded_from_scientific_cohort"]
    assert len(stored["summaries"]) == 3
    for summary, filename in zip(stored["summaries"], stored["trajectory_files"]):
        with gzip.open(tmp_path / filename, "rt") as handle:
            trajectory = json.load(handle)
        rows = [dict(zip(trajectory["columns"], values)) for values in trajectory["trajectory"]]
        assert len(rows) == summary["steps_run"] == 3
        assert rows[-1]["mean_after"] == summary["final_living_mean"]
        assert trajectory["source_hashes"] == stored["source_hashes"]
    stdout_copy = prefix.with_suffix(".txt").read_text()
    assert stdout_copy.count("\nRESULT ") == 3
    assert "SOURCE_HASHES " in stdout_copy


def test_observer_runtime_margin_blocks_before_launching_full_cohort(tmp_path, monkeypatch):
    import json
    from experiments import exp283_boundary_transmission as experiment

    # A 1200-second unobserved batch is safe at preflight but exceeds 1800s
    # after the fixed observer factor of two. The costly probe is replaced,
    # but the production gate and persisted abort are exercised directly.
    monkeypatch.setattr(experiment, "preflight", lambda *args, **kwargs: {
        "safe": True, "configs": [{"proj_per_job_s": 100.}],
        "n_jobs": 48, "recommended_workers": 4,
    })
    prefix = tmp_path / "blocked"
    with pytest.raises(RuntimeError, match="observer-adjusted projection"):
        experiment.main(["--full", "--output-prefix", str(prefix)])
    result = json.loads(prefix.with_suffix(".json").read_text())
    assert result["observer_runtime_projection"]["projected_batch_s"] > 2400.
    assert result["summaries"] == []
    assert result["trajectory_files"] == []
    assert "observer-adjusted projection" in result["aborted"]


@pytest.mark.parametrize("existing", ["transcript", "metadata", "trajectories"])
def test_existing_output_rejected_before_any_write(tmp_path, existing):
    from experiments.exp283_boundary_transmission import main

    prefix = tmp_path / "preserved"
    transcript = prefix.with_suffix(".txt")
    metadata = prefix.with_suffix(".json")
    trajectories = tmp_path / "preserved_trajectories"
    sentinel = "original research evidence\n"
    if existing == "transcript":
        transcript.write_text(sentinel)
    elif existing == "metadata":
        metadata.write_text(sentinel)
    else:
        trajectories.mkdir()
        (trajectories / "sentinel").write_text(sentinel)

    with pytest.raises(FileExistsError, match="fresh --output-prefix"):
        main(["--smoke", "--smoke-horizon", "1", "--output-prefix", str(prefix)])

    if existing == "transcript":
        assert transcript.read_text() == sentinel
        assert not metadata.exists()
        assert not trajectories.exists()
    elif existing == "metadata":
        assert metadata.read_text() == sentinel
        assert not transcript.exists()
        assert not trajectories.exists()
    else:
        assert (trajectories / "sentinel").read_text() == sentinel
        assert list(trajectories.iterdir()) == [trajectories / "sentinel"]
        assert not transcript.exists()
        assert not metadata.exists()
