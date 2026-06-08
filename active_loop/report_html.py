"""Render a self-contained HTML showcase of the loop's progress and world model."""
from __future__ import annotations

import html
from pathlib import Path

from active_loop.world_model import WorldModel

_CSS = (
    "body{font-family:system-ui,sans-serif;margin:2rem;background:#0f1115;color:#e6e6e6}"
    "h1,h2{color:#8ab4f8}table{border-collapse:collapse;width:100%;margin:1rem 0}"
    "td,th{border:1px solid #333;padding:.4rem .6rem;text-align:left}"
    ".pass{color:#7ee787}.fail{color:#ff7b72}.metric{font-size:1.5rem}"
)


def _guardrail_rows(guardrails: dict) -> str:
    rows = ""
    for name, ok in guardrails.items():
        cls = "pass" if ok else "fail"
        label = "PASS" if ok else "FAIL"
        rows += f"<tr><td>{html.escape(name)}</td><td class='{cls}'>{label}</td></tr>"
    return rows


def _belief_rows(world_model: WorldModel) -> str:
    rows = ""
    for b in world_model.all_beliefs():
        rows += (
            f"<tr><td>{html.escape(b.name)}</td>"
            f"<td>{b.confidence:.2f}</td>"
            f"<td>+{b.evidence_for}/-{b.evidence_against}</td>"
            f"<td>{html.escape(b.claim)}</td></tr>"
        )
    return rows or "<tr><td colspan='4'><em>no beliefs yet</em></td></tr>"


def _history_cells(history: list[float]) -> str:
    if not history:
        return "<tr><td><em>no history yet</em></td></tr>"
    return "<tr>" + "".join(f"<td>{m:.3f}</td>" for m in history) + "</tr>"


def render_report(out_path: Path | str, *, score: dict, world_model: WorldModel,
                  history: list[float]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_cls = "pass" if score["verdict"] else "fail"
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>active-loop showcase</title><style>{_CSS}</style></head><body>"
        "<h1>active-loop — active inference research showcase</h1>"
        f"<p class='metric'>free-energy metric: <b>{score['metric']:.3f}</b> "
        "(lower is better)</p>"
        f"<p>success rate: {score['success_rate']:.3f} &nbsp; "
        f"ask rate: {score['ask_rate']:.3f} &nbsp; "
        f"verdict: <b class='{verdict_cls}'>{'PASS' if score['verdict'] else 'FAIL'}</b></p>"
        "<h2>guardrails</h2><table><tr><th>name</th><th>status</th></tr>"
        f"{_guardrail_rows(score['guardrails'])}</table>"
        "<h2>metric history</h2><table>"
        f"{_history_cells(history)}</table>"
        "<h2>world model — beliefs</h2>"
        "<table><tr><th>belief</th><th>confidence</th><th>evidence</th><th>claim</th></tr>"
        f"{_belief_rows(world_model)}</table>"
        "</body></html>"
    )
    out_path.write_text(doc)
