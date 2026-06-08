import json
from pathlib import Path

from active_loop.world_model import WorldModel


def test_record_belief_creates_file_and_index(tmp_path):
    wm = WorldModel(tmp_path)
    wm.record_belief("horizon-3-best", "Horizon 3 minimizes free energy.", supported=True)
    f = tmp_path / "beliefs" / "horizon-3-best.md"
    assert f.exists()
    assert "Horizon 3 minimizes free energy." in f.read_text()
    assert (tmp_path / "INDEX.md").exists()
    assert "horizon-3-best" in (tmp_path / "INDEX.md").read_text()


def test_confidence_grows_with_supporting_evidence(tmp_path):
    wm = WorldModel(tmp_path)
    wm.record_belief("b", "claim", supported=True)
    c1 = wm.get_belief("b").confidence
    wm.record_belief("b", "claim", supported=True)
    wm.record_belief("b", "claim", supported=True)
    c2 = wm.get_belief("b").confidence
    assert c2 > c1
    assert wm.get_belief("b").evidence_for == 3


def test_contradicting_evidence_lowers_confidence_but_keeps_belief(tmp_path):
    wm = WorldModel(tmp_path)
    wm.record_belief("b", "claim", supported=True)
    high = wm.get_belief("b").confidence
    wm.record_belief("b", "claim", supported=False)
    wm.record_belief("b", "claim", supported=False)
    low = wm.get_belief("b").confidence
    assert low < high
    assert (tmp_path / "beliefs" / "b.md").exists()


def test_append_evidence_is_append_only(tmp_path):
    wm = WorldModel(tmp_path)
    wm.append_evidence({"iter": 1, "metric": -1.5, "kept": True})
    wm.append_evidence({"iter": 2, "metric": -1.6, "kept": True})
    lines = (tmp_path / "evidence" / "journal.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["iter"] == 2


def test_promote_findings_copies_high_confidence(tmp_path):
    wm = WorldModel(tmp_path)
    for _ in range(5):
        wm.record_belief("strong", "a robust finding", supported=True)
    wm.record_belief("weak", "shaky", supported=False)
    wm.promote_findings(threshold=0.8)
    assert (tmp_path / "findings" / "strong.md").exists()
    assert not (tmp_path / "findings" / "weak.md").exists()


def test_store_persists_across_instances(tmp_path):
    WorldModel(tmp_path).record_belief("b", "claim", supported=True)
    wm2 = WorldModel(tmp_path)
    assert wm2.get_belief("b").evidence_for == 1
