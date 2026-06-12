"""Guards for the Claude /loop routing and bootstrap docs."""
import pathlib
import re

from loop.compose import compose


ROOT = pathlib.Path(__file__).parent.parent


def test_compose_includes_model_routing_card():
    prompt = compose(direction="identity-n4", persona="default", idea=None)

    assert "loop/ROUTING.md" in prompt
    assert "highest-thinking" in prompt
    assert "Sonnet" in prompt
    assert "Haiku" in prompt
    assert "blinded verifier" in prompt


def test_agents_doc_uses_builtin_loop_not_ruflo():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "/loop" in text
    assert "ruflo" not in text.lower()


def test_resume_mentions_recent_logged_experiment():
    experiments = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    resume = (ROOT / "RESUME.md").read_text(encoding="utf-8")

    logged = [int(m) for m in re.findall(r"^## Exp (\d+) ", experiments, re.MULTILINE)]
    mentioned = [int(m) for m in re.findall(r"\bExp (\d+)\b", resume)]

    assert logged, "EXPERIMENTS.md has no experiment headings"
    assert mentioned, "RESUME.md mentions no experiment numbers"
    assert max(mentioned) >= max(logged) - 2, (
        f"RESUME.md is stale: highest mentioned Exp {max(mentioned)}, "
        f"latest logged Exp {max(logged)}"
    )
