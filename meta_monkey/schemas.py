"""Deterministic schema objects for passive Meta Monkey episode records."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactStatus:
    script_path: str | None
    output_path: str | None
    script_exists: bool
    output_exists: bool
    site_data_references_script: bool
    site_data_references_output: bool


@dataclass(frozen=True)
class EntryStatus:
    entry_exists: bool
    has_plain: bool
    has_verdict: bool
    has_honest_caveat: bool
    has_verifier: bool
    claimed_verdict: str | None
    insight_tag: str | None
    verifier_status: str | None


@dataclass(frozen=True)
class CheckStatus:
    hard_failures: list[str]
    warnings: list[str]
    passed: bool


@dataclass(frozen=True)
class ProcessStatus:
    likely_risks: list[str]
    process_failure: bool
    notes: list[str]


@dataclass(frozen=True)
class MetaEpisode:
    schema_version: int
    exp: int
    collected_at_utc: str
    commit_sha: str | None
    artifacts: ArtifactStatus
    entry: EntryStatus
    checks: CheckStatus
    process: ProcessStatus
    future_policy_hint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetaEpisode":
        return cls(
            schema_version=int(data["schema_version"]),
            exp=int(data["exp"]),
            collected_at_utc=str(data["collected_at_utc"]),
            commit_sha=str(data["commit_sha"]) if data["commit_sha"] is not None else None,
            artifacts=ArtifactStatus(**data["artifacts"]),
            entry=EntryStatus(**data["entry"]),
            checks=CheckStatus(**data["checks"]),
            process=ProcessStatus(**data["process"]),
            future_policy_hint=str(data["future_policy_hint"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "MetaEpisode":
        return cls.from_dict(json.loads(text))

    def write_json(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read_json(cls, path: pathlib.Path) -> "MetaEpisode":
        return cls.from_json(path.read_text(encoding="utf-8"))


def episode_to_json(episode: MetaEpisode) -> str:
    return episode.to_json()


def episode_from_json(text: str) -> MetaEpisode:
    return MetaEpisode.from_json(text)
