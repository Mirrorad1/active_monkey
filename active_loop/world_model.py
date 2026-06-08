"""Persistent, growing world-model store: file-per-belief markdown + append-only journal.

Grows or sharpens only — beliefs are never deleted (contradicted ones are kept with
lowered confidence). INDEX.md is a derived view, safe to regenerate each write.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Belief:
    name: str
    claim: str
    evidence_for: int
    evidence_against: int

    @property
    def confidence(self) -> float:
        return (self.evidence_for + 1) / (self.evidence_for + self.evidence_against + 2)


_FM = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


class WorldModel:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        (self.root / "beliefs").mkdir(parents=True, exist_ok=True)
        (self.root / "findings").mkdir(parents=True, exist_ok=True)
        (self.root / "evidence").mkdir(parents=True, exist_ok=True)

    def _belief_path(self, name: str) -> Path:
        return self.root / "beliefs" / f"{name}.md"

    def get_belief(self, name: str) -> Belief | None:
        path = self._belief_path(name)
        if not path.exists():
            return None
        m = _FM.match(path.read_text())
        meta = {}
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return Belief(
            name=name,
            claim=m.group(2).strip(),
            evidence_for=int(meta.get("evidence_for", 0)),
            evidence_against=int(meta.get("evidence_against", 0)),
        )

    def record_belief(self, name: str, claim: str, supported: bool) -> Belief:
        existing = self.get_belief(name)
        ef = (existing.evidence_for if existing else 0) + (1 if supported else 0)
        ea = (existing.evidence_against if existing else 0) + (0 if supported else 1)
        belief = Belief(name=name, claim=claim, evidence_for=ef, evidence_against=ea)
        self._write_belief(self._belief_path(name), belief)
        self.rebuild_index()
        return belief

    def _write_belief(self, path: Path, belief: Belief) -> None:
        path.write_text(
            "---\n"
            f"name: {belief.name}\n"
            f"confidence: {belief.confidence:.3f}\n"
            f"evidence_for: {belief.evidence_for}\n"
            f"evidence_against: {belief.evidence_against}\n"
            "---\n"
            f"{belief.claim}\n"
        )

    def all_beliefs(self) -> list[Belief]:
        out = []
        for p in sorted((self.root / "beliefs").glob("*.md")):
            b = self.get_belief(p.stem)
            if b:
                out.append(b)
        return out

    def append_evidence(self, record: dict) -> None:
        with (self.root / "evidence" / "journal.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def promote_findings(self, threshold: float = 0.8) -> list[str]:
        promoted = []
        for b in self.all_beliefs():
            if b.confidence >= threshold:
                self._write_belief(self.root / "findings" / f"{b.name}.md", b)
                promoted.append(b.name)
        return promoted

    def rebuild_index(self) -> None:
        lines = ["# World Model Index", ""]
        for b in self.all_beliefs():
            lines.append(f"- **{b.name}** (conf {b.confidence:.2f}, +{b.evidence_for}/-{b.evidence_against}): {b.claim}")
        (self.root / "INDEX.md").write_text("\n".join(lines) + "\n")
