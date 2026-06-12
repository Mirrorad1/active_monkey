"""Summarize collected passive Meta Monkey episode records."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

from meta_monkey.schemas import MetaEpisode


def load_episodes(root: pathlib.Path) -> list[MetaEpisode]:
    episode_dir = root / "meta" / "episodes"
    if not episode_dir.exists():
        return []
    return [
        MetaEpisode.read_json(path)
        for path in sorted(episode_dir.glob("exp*.json"), key=lambda item: item.name)
    ]


def _count_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["  (none)"]
    return [f"  {key}: {counter[key]}" for key in sorted(counter)]


def _ranked_count_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["  (none)"]
    return [
        f"  {key}: {count}"
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_report(episodes: list[MetaEpisode]) -> str:
    if not episodes:
        return (
            "Meta Monkey report\n"
            "total episodes: 0\n"
            "No episode records found in meta/episodes.\n"
        )

    verdict_counts: Counter[str] = Counter(
        episode.entry.claimed_verdict for episode in episodes if episode.entry.claimed_verdict
    )
    insight_counts: Counter[str] = Counter(
        episode.entry.insight_tag for episode in episodes if episode.entry.insight_tag
    )
    risk_counts: Counter[str] = Counter(
        risk for episode in episodes for risk in episode.process.likely_risks
    )
    hint_counts: Counter[str] = Counter(episode.future_policy_hint for episode in episodes)

    process_failures = sum(1 for episode in episodes if episode.process.process_failure)
    verifier_disagreements = sum(
        1 for episode in episodes if episode.entry.verifier_status == "disagreed"
    )
    check_warnings = sum(1 for episode in episodes if episode.checks.warnings)
    check_failures = sum(1 for episode in episodes if not episode.checks.passed)

    lines = [
        "Meta Monkey report",
        f"total episodes: {len(episodes)}",
        "verdict counts:",
    ]
    lines.extend(_count_lines(verdict_counts))
    lines.append("insight tag counts:")
    lines.extend(_count_lines(insight_counts))
    lines.append(f"process failures: {process_failures}")
    lines.append("most common likely risks:")
    lines.extend(_ranked_count_lines(risk_counts))
    lines.append(f"verifier disagreements: {verifier_disagreements}")
    lines.append(f"check warnings: {check_warnings}")
    lines.append(f"check failures: {check_failures}")
    lines.append("future_policy_hint counts:")
    lines.extend(_ranked_count_lines(hint_counts))
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Report on passive Meta Monkey episodes.")


def main(argv: list[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    try:
        print(build_report(load_episodes(pathlib.Path.cwd())), end="")
        return 0
    except Exception as exc:
        print(f"report error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
