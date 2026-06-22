"""Prospective Causal Calibration helpers.

These utilities score self-improvement patches by both observed task delta and
the patch's precommitted forecast of where the delta should occur.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


EFFECT_LABELS = ("improve", "same", "worse")
EPS = 1e-9
WRONG_CONFIDENT_NLL = -math.log(0.5)


@dataclass(frozen=True)
class Trial:
    """One task outcome under a particular scaffold."""

    task_id: str
    tags: tuple[str, ...]
    correct: bool


@dataclass(frozen=True)
class Forecast:
    """Precommitted categorical forecast over effect labels per task tag."""

    per_tag: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class PatchSummary:
    """Observed and forecast-calibrated score for one candidate patch."""

    patch_id: str
    baseline_accuracy: float
    patched_accuracy: float
    delta_accuracy: float
    mean_forecast_nll: float
    wrong_confident_rate: float
    pcc_score: float
    n_tasks: int


def _effect_label(before: bool, after: bool) -> str:
    if after and not before:
        return "improve"
    if before and not after:
        return "worse"
    return "same"


def _normalize_probs(raw: Mapping[str, float]) -> dict[str, float]:
    clipped = {label: max(float(raw.get(label, 0.0)), 0.0) for label in EFFECT_LABELS}
    total = sum(clipped.values())
    if total <= 0.0:
        return {label: 1.0 / len(EFFECT_LABELS) for label in EFFECT_LABELS}
    return {label: value / total for label, value in clipped.items()}


def _label_probability(forecast: Forecast, tags: Sequence[str], label: str) -> float:
    if not tags:
        return 1.0 / len(EFFECT_LABELS)
    probs = []
    for tag in tags:
        dist = _normalize_probs(forecast.per_tag.get(tag, {}))
        probs.append(dist[label])
    return max(sum(probs) / len(probs), EPS)


def _align_trials(baseline: Sequence[Trial], patched: Sequence[Trial]) -> list[tuple[Trial, Trial]]:
    base_by_id = {trial.task_id: trial for trial in baseline}
    patch_by_id = {trial.task_id: trial for trial in patched}
    if set(base_by_id) != set(patch_by_id):
        missing_after = sorted(set(base_by_id) - set(patch_by_id))
        missing_before = sorted(set(patch_by_id) - set(base_by_id))
        raise ValueError(
            "baseline and patched trials must contain identical task ids "
            f"(missing_after={missing_after}, missing_before={missing_before})"
        )
    return [(base_by_id[task_id], patch_by_id[task_id]) for task_id in sorted(base_by_id)]


def summarize_patch(
    patch_id: str,
    forecast: Forecast,
    baseline: Sequence[Trial],
    patched: Sequence[Trial],
    *,
    alpha: float = 0.1,
    complexity_penalty: float = 0.0,
) -> PatchSummary:
    """Score a patch by task improvement and effect-forecast calibration."""

    pairs = _align_trials(baseline, patched)
    if not pairs:
        raise ValueError("at least one trial is required")

    baseline_accuracy = sum(base.correct for base, _ in pairs) / len(pairs)
    patched_accuracy = sum(after.correct for _, after in pairs) / len(pairs)
    nlls: list[float] = []
    wrong_confident = 0
    for base, after in pairs:
        label = _effect_label(base.correct, after.correct)
        probability = _label_probability(forecast, base.tags, label)
        nll = -math.log(probability)
        nlls.append(nll)
        if nll > WRONG_CONFIDENT_NLL:
            wrong_confident += 1

    mean_nll = sum(nlls) / len(nlls)
    wrong_confident_rate = wrong_confident / len(nlls)
    wrong_confident_penalty = max(0.0, mean_nll - WRONG_CONFIDENT_NLL)
    delta_accuracy = patched_accuracy - baseline_accuracy
    pcc_score = delta_accuracy - alpha * mean_nll - wrong_confident_penalty - complexity_penalty

    return PatchSummary(
        patch_id=patch_id,
        baseline_accuracy=baseline_accuracy,
        patched_accuracy=patched_accuracy,
        delta_accuracy=delta_accuracy,
        mean_forecast_nll=mean_nll,
        wrong_confident_rate=wrong_confident_rate,
        pcc_score=pcc_score,
        n_tasks=len(pairs),
    )


def choose_patch(summaries: Sequence[PatchSummary]) -> PatchSummary:
    """Choose the highest PCC-scoring patch with deterministic tie-breaking."""

    if not summaries:
        raise ValueError("at least one patch summary is required")
    return max(summaries, key=lambda summary: (summary.pcc_score, summary.patch_id))
