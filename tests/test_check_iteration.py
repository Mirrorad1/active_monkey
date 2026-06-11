"""Fast, stdlib-only tests for loop/check_iteration.py.

The loop/ directory is not a package; the module is loaded via importlib so
that no __init__.py is needed there.
"""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Load the module under test
# ---------------------------------------------------------------------------

def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_iteration",
        ROOT / "loop" / "check_iteration.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
check_entry = _mod.check_entry
find_entries = _mod.find_entries
VERIFIER_FLOOR = _mod.VERIFIER_FLOOR


# ---------------------------------------------------------------------------
# Helpers to build a fake repo layout
# ---------------------------------------------------------------------------

_GOOD_DOCSTRING = '''\
"""
Hypothesis: the widget expands under heat.
Prediction: diameter > 1.05 at 100 C.
Falsifier: diameter <= 1.05 refutes.
Predeclared thresholds fixed before run.
"""
'''

_GOOD_ENTRY = """\
## Exp 7 — widget expansion test (POSITIVE; NEW INSIGHT)
- Plain: We heated the widget and measured it.
- Setup: heat to 100 C, measure diameter.
- Result: diameter = 9.999, fraction = 4/5.
- Implication: expands as predicted.
- Honest caveat: single trial only.
- Verdict: POSITIVE / NEW INSIGHT. Self-grade: POSITIVE-SINGLE.
- Next: replicate at 200 C.
"""


def _make_root(tmp_path, entry_text=_GOOD_ENTRY, script_src=_GOOD_DOCSTRING,
               include_output=True, output_text="diameter = 9.999\n4/5 pass"):
    """Build a minimal fake repo tree under tmp_path."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    out_dir = exp_dir / "outputs"
    out_dir.mkdir()

    experiments_md = tmp_path / "EXPERIMENTS.md"
    experiments_md.write_text(entry_text, encoding="utf-8")

    script = exp_dir / "exp7_widget.py"
    script.write_text(script_src, encoding="utf-8")

    if include_output:
        (out_dir / "exp7.txt").write_text(output_text, encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: fully compliant entry passes
# ---------------------------------------------------------------------------

def test_compliant_entry_passes(tmp_path):
    root = _make_root(tmp_path)
    experiments_text = (root / "EXPERIMENTS.md").read_text(encoding="utf-8")
    hard, warn = check_entry(7, experiments_text=experiments_text, root=root)
    assert hard == [], f"Expected no hard failures, got: {hard}"


# ---------------------------------------------------------------------------
# Test 2: each missing field independently produces a hard failure
# ---------------------------------------------------------------------------

def _check_produces_hard(entry_text, script_src=_GOOD_DOCSTRING,
                         include_output=True, output_text="diameter = 9.999\n4/5",
                         tmp_path=None):
    root = _make_root(
        tmp_path,
        entry_text=entry_text,
        script_src=script_src,
        include_output=include_output,
        output_text=output_text,
    )
    experiments_text = (root / "EXPERIMENTS.md").read_text(encoding="utf-8")
    hard, _warn = check_entry(7, experiments_text=experiments_text, root=root)
    return hard


def test_missing_plain_fails(tmp_path):
    entry = _GOOD_ENTRY.replace("- Plain: We heated the widget and measured it.\n", "")
    hard = _check_produces_hard(entry, tmp_path=tmp_path)
    assert any("Plain" in f for f in hard), f"Expected Plain failure, got: {hard}"


def test_missing_verdict_fails(tmp_path):
    entry = _GOOD_ENTRY.replace("- Verdict: POSITIVE / NEW INSIGHT. Self-grade: POSITIVE-SINGLE.\n", "")
    hard = _check_produces_hard(entry, tmp_path=tmp_path)
    assert any("Verdict" in f for f in hard), f"Expected Verdict failure, got: {hard}"


def test_missing_consolidation_new_insight_fails(tmp_path):
    # Remove both from the entry body
    entry = _GOOD_ENTRY.replace("NEW INSIGHT", "NOTEWORTHY").replace("CONSOLIDATION", "NOTED")
    hard = _check_produces_hard(entry, tmp_path=tmp_path)
    assert any("CONSOLIDATION" in f or "NEW INSIGHT" in f for f in hard), (
        f"Expected CONSOLIDATION/NEW INSIGHT failure, got: {hard}"
    )


def test_missing_honest_caveat_fails(tmp_path):
    entry = _GOOD_ENTRY.replace("- Honest caveat: single trial only.\n", "")
    hard = _check_produces_hard(entry, tmp_path=tmp_path)
    assert any("Honest caveat" in f for f in hard), f"Expected caveat failure, got: {hard}"


def test_positive_without_self_grade_fails(tmp_path):
    entry = _GOOD_ENTRY.replace("Self-grade: POSITIVE-SINGLE.", "")
    hard = _check_produces_hard(entry, tmp_path=tmp_path)
    assert any("self-grade" in f.lower() or "BREAKTHROUGH" in f or "POSITIVE-SINGLE" in f
               for f in hard), f"Expected self-grade failure, got: {hard}"


def test_missing_script_fails(tmp_path):
    root = _make_root(tmp_path)
    # Remove the script
    (root / "experiments" / "exp7_widget.py").unlink()
    experiments_text = (root / "EXPERIMENTS.md").read_text(encoding="utf-8")
    hard, _warn = check_entry(7, experiments_text=experiments_text, root=root)
    assert any("script" in f.lower() for f in hard), f"Expected script failure, got: {hard}"


def test_missing_output_fails(tmp_path):
    root = _make_root(tmp_path, include_output=False)
    experiments_text = (root / "EXPERIMENTS.md").read_text(encoding="utf-8")
    hard, _warn = check_entry(7, experiments_text=experiments_text, root=root)
    assert any("output" in f.lower() for f in hard), f"Expected output failure, got: {hard}"


def test_docstring_lacking_falsifier_fails(tmp_path):
    bad_src = '"""Hypothesis: something. Prediction: it works. No F-word here."""\n'
    hard = _check_produces_hard(_GOOD_ENTRY, script_src=bad_src, tmp_path=tmp_path)
    assert any("falsifier" in f.lower() for f in hard), f"Expected falsifier failure, got: {hard}"


# ---------------------------------------------------------------------------
# Test 3: NEGATIVE verdict without self-grade passes check 6
# ---------------------------------------------------------------------------

def test_negative_verdict_no_self_grade_passes_check6(tmp_path):
    entry = """\
## Exp 7 — widget contraction test (NEGATIVE; NEW INSIGHT)
- Plain: We cooled the widget and it shrank.
- Setup: cool to 0 C.
- Result: diameter = 0.95.
- Implication: contracts when cold.
- Honest caveat: one trial.
- Verdict: NEGATIVE / NEW INSIGHT.
- Next: replicate.
"""
    root = _make_root(tmp_path, entry_text=entry, output_text="diameter = 0.95")
    experiments_text = (root / "EXPERIMENTS.md").read_text(encoding="utf-8")
    hard, _warn = check_entry(7, experiments_text=experiments_text, root=root)
    self_grade_failures = [
        f for f in hard
        if "self-grade" in f.lower() or "BREAKTHROUGH" in f or "POSITIVE-SINGLE" in f
    ]
    assert not self_grade_failures, (
        f"NEGATIVE verdict should not require self-grade, but got: {self_grade_failures}"
    )


# ---------------------------------------------------------------------------
# Test 4: Verifier floor behaviour
# ---------------------------------------------------------------------------

def _make_root_for_n(tmp_path, n, include_verifier=True):
    verifier_line = "- Verifier: independent re-run confirmed.\n" if include_verifier else ""
    entry = f"""\
## Exp {n} — test entry (POSITIVE; NEW INSIGHT)
- Plain: Synthetic test entry.
- Setup: nothing.
- Result: 1.000.
- Implication: works.
- Honest caveat: synthetic.
{verifier_line}- Verdict: POSITIVE. Self-grade: POSITIVE-SINGLE.
- Next: nothing.
"""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(exist_ok=True)
    out_dir = exp_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    (tmp_path / "EXPERIMENTS.md").write_text(entry, encoding="utf-8")
    (exp_dir / f"exp{n}_foo.py").write_text(_GOOD_DOCSTRING, encoding="utf-8")
    (out_dir / f"exp{n}.txt").write_text("result = 1.000", encoding="utf-8")
    return tmp_path


def test_verifier_required_at_floor(tmp_path):
    root = _make_root_for_n(tmp_path, VERIFIER_FLOOR, include_verifier=False)
    experiments_text = (root / "EXPERIMENTS.md").read_text(encoding="utf-8")
    hard, _warn = check_entry(VERIFIER_FLOOR, experiments_text=experiments_text, root=root)
    assert any("Verifier" in f for f in hard), (
        f"Expected Verifier failure for n={VERIFIER_FLOOR}, got: {hard}"
    )


def test_verifier_not_required_below_floor(tmp_path):
    n = VERIFIER_FLOOR - 1
    root = _make_root_for_n(tmp_path, n, include_verifier=False)
    experiments_text = (root / "EXPERIMENTS.md").read_text(encoding="utf-8")
    hard, _warn = check_entry(n, experiments_text=experiments_text, root=root)
    verifier_failures = [f for f in hard if "Verifier" in f]
    assert not verifier_failures, (
        f"Verifier should not be required for n={n}, but got: {verifier_failures}"
    )


# ---------------------------------------------------------------------------
# Test 5: soft re-quote check
# ---------------------------------------------------------------------------

def test_requote_absent_number_warns(tmp_path):
    # 9.999 is in the entry but not in the output
    entry = """\
## Exp 7 — requote test (POSITIVE; NEW INSIGHT)
- Plain: Requote test.
- Setup: nothing.
- Result: score = 9.999.
- Implication: notable.
- Honest caveat: synthetic.
- Verdict: POSITIVE. Self-grade: POSITIVE-SINGLE.
- Next: nothing.
"""
    output_text = "score = 1.000"  # 9.999 absent
    root = _make_root(tmp_path, entry_text=entry, output_text=output_text)
    experiments_text = (root / "EXPERIMENTS.md").read_text(encoding="utf-8")
    _hard, warn = check_entry(7, experiments_text=experiments_text, root=root)
    assert any("9.999" in w for w in warn), (
        f"Expected warning about 9.999, got warnings: {warn}"
    )


def test_requote_present_number_no_warn(tmp_path):
    output_text = "diameter = 9.999\n4/5 pass"
    root = _make_root(tmp_path, output_text=output_text)
    experiments_text = (root / "EXPERIMENTS.md").read_text(encoding="utf-8")
    _hard, warn = check_entry(7, experiments_text=experiments_text, root=root)
    requote_warns = [w for w in warn if "9.999" in w or "4/5" in w]
    assert not requote_warns, (
        f"Numbers present in output should not warn, but got: {requote_warns}"
    )


# ---------------------------------------------------------------------------
# Test 6: forward guard — real EXPERIMENTS.md, entries >= VERIFIER_FLOOR
# ---------------------------------------------------------------------------

def test_real_experiments_at_and_above_verifier_floor():
    """Parse the real EXPERIMENTS.md; every entry with n >= VERIFIER_FLOOR passes
    all hard checks.

    Currently vacuous — the latest entry is Exp 151 and VERIFIER_FLOOR is 152,
    so no entries qualify yet — but this test arms automatically the moment
    Exp 152 is committed, enforcing the Verifier requirement from day one.
    """
    experiments_text = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    entries = find_entries(experiments_text)
    qualifying = {n: e for n, e in entries.items() if n >= VERIFIER_FLOOR}

    failures = {}
    for n in sorted(qualifying):
        hard, _warn = check_entry(n, experiments_text=experiments_text, root=ROOT)
        if hard:
            failures[n] = hard

    assert not failures, (
        "Real entries at or above VERIFIER_FLOOR have hard failures:\n"
        + "\n".join(f"  Exp {n}: {fs}" for n, fs in sorted(failures.items()))
    )


# ---------------------------------------------------------------------------
# Test 7: find_entries on the real EXPERIMENTS.md
# ---------------------------------------------------------------------------

def test_find_entries_real_file():
    """find_entries on the real EXPERIMENTS.md returns a dict with max key >= 151
    and the full block for that entry."""
    experiments_text = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    entries = find_entries(experiments_text)

    assert entries, "find_entries returned an empty dict on the real EXPERIMENTS.md"
    assert max(entries) >= 151, (
        f"Expected max entry >= 151, got {max(entries)}"
    )

    block = entries[max(entries)]
    # The block should start with the heading
    assert block.startswith("## Exp "), (
        f"Entry block for Exp {max(entries)} does not start with '## Exp '"
    )
    # The block should be non-trivial
    assert len(block) > 200, (
        f"Entry block for Exp {max(entries)} suspiciously short ({len(block)} chars)"
    )
