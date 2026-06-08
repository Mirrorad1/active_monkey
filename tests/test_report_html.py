from pathlib import Path

from active_loop.world_model import WorldModel
from active_loop.report_html import render_report


def test_render_writes_self_contained_html(tmp_path):
    wm = WorldModel(tmp_path / "wm")
    wm.record_belief("b1", "Asking helps under uncertainty.", supported=True)
    score = {"metric": -1.70, "success_rate": 0.54, "ask_rate": 0.54,
             "guardrails": {"success_floor": True, "ask_rate_band": True}, "verdict": True}
    out = tmp_path / "reports" / "index.html"
    render_report(out, score=score, world_model=wm, history=[-1.5, -1.6, -1.70])

    assert out.exists()
    html = out.read_text()
    assert "<html" in html.lower()
    assert "Asking helps under uncertainty." in html
    assert "-1.70" in html or "-1.7" in html
    assert "verdict" in html.lower()
    assert "http://" not in html and "https://" not in html


def test_render_handles_empty_world_model(tmp_path):
    wm = WorldModel(tmp_path / "wm")
    score = {"metric": 0.0, "success_rate": 0.0, "ask_rate": 0.0,
             "guardrails": {"success_floor": False, "ask_rate_band": False}, "verdict": False}
    out = tmp_path / "reports" / "index.html"
    render_report(out, score=score, world_model=wm, history=[])
    assert out.exists()
    assert "<html" in out.read_text().lower()
