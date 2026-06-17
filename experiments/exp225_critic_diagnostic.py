"""Exp 225 (diagnostic) — WHY does the M4b critic reject? Capture the AffectClaudeCritic's reasons.

Exp 224 was NEGATIVE but CRITIC-GATED: both proposer prior-changes were rejected by the
AffectClaudeCritic before the FROZEN scorer ever measured them, and the critic_reason was not
recorded (the clone was deleted). This cheap diagnostic (NO 10-min scoring) runs N propose->critic
cycles on an isolated clone and captures, per cycle: the proposed diff to active_loop/affect_spec.py
+ the critic's APPROVE/REJECT verdict AND its reason. The question it answers: is the critic
rejecting UNSOUND prior-hacks (correctly conservative) or is it MIS-CALIBRATED (too strict on small
prior changes)? That decides whether M4b needs the critic loosened before a full re-run.

PREDECLARATION (diagnostic read-criterion; this is NOT a falsifier-bound experiment): classify the
critic SOUND if it rejects gaming-style prior-hacks (e.g. baking the code->intent map into A0) and
approves legitimate preference-only tweaks (MISCALIBRATED otherwise); a critic-APPROVED change then
IMPROVES iff its FROZEN-scored mean_last > the 0.4225 baseline (measured in Exp 225b). hypothesis: the
Exp 224 critic-gating is SOUND and a critic-approved direction beats baseline.

Functional valence only; no sentience claim.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
N_CYCLES = 3


def _run(cmd, cwd, timeout):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def main() -> None:
    lines: list[str] = []
    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("=" * 78)
    log("EXP 225 (diagnostic) — WHY does the M4b AffectClaudeCritic reject? (N propose->critic cycles)")
    log("No scoring; captures the proposed diff + the critic verdict+reason per cycle.")
    log("=" * 78)
    log("")

    clone = Path(tempfile.mkdtemp(prefix="exp225_critic_clone_"))
    clone_repo = clone / "repo"
    try:
        _run(["git", "clone", "--local", "--no-hardlinks", str(_REPO), str(clone_repo)], _REPO, 300)
        _run(["git", "config", "user.email", "exp225@loop"], clone_repo, 30)
        _run(["git", "config", "user.name", "exp225"], clone_repo, 30)
        (clone_repo / ".venv").symlink_to(_REPO / ".venv")
        sys.path.insert(0, str(clone_repo))
        from active_loop.affect_pr_loop import AffectClaudeProposer, AffectClaudeCritic

        proposer = AffectClaudeProposer(timeout_s=600)
        critic = AffectClaudeCritic(timeout_s=600)
        approvals = 0
        for c in range(1, N_CYCLES + 1):
            log(f"--- CYCLE {c}/{N_CYCLES} ---")
            # reset affect_spec.py to baseline so each proposal's diff is clean
            _run(["git", "checkout", "--", "active_loop/affect_spec.py"], clone_repo, 30)
            try:
                hyp = proposer.propose(clone_repo)
            except Exception as e:  # noqa: BLE001
                log(f"  PROPOSER FAILED: {type(e).__name__}: {str(e)[:300]}")
                log("")
                continue
            diff = _run(["git", "diff", "--", "active_loop/affect_spec.py"], clone_repo, 30).stdout
            # diff summary: only the changed (+/-) lines, capped
            changed = [ln for ln in diff.splitlines()
                       if (ln.startswith("+") or ln.startswith("-"))
                       and not ln.startswith("+++") and not ln.startswith("---")]
            log(f"  proposer hypothesis: {hyp}")
            log(f"  diff: {len(changed)} changed lines; sample:")
            for ln in changed[:18]:
                log(f"    {ln}")
            if len(changed) > 18:
                log(f"    ... (+{len(changed) - 18} more changed lines)")
            try:
                verdict = critic.review(diff, clone_repo)
            except Exception as e:  # noqa: BLE001
                log(f"  CRITIC FAILED: {type(e).__name__}: {str(e)[:300]}")
                log("")
                continue
            approvals += int(bool(verdict.approved))
            log(f"  CRITIC: approved={verdict.approved}")
            log(f"  CRITIC REASON: {verdict.reason}")
            log("")

        log("--- SUMMARY ---")
        log(f"cycles={N_CYCLES}  approvals={approvals}  rejections={N_CYCLES - approvals}")
        log("READ: if rejections cite genuine unsoundness/gaming -> critic is correctly conservative;")
        log("      if they cite trivial/over-strict reasons on small prior tweaks -> miscalibrated, loosen it.")
        log(f"MACHINE SUMMARY: cycles={N_CYCLES} approvals={approvals} rejections={N_CYCLES - approvals}")

    finally:
        out = _REPO / "experiments" / "outputs" / "exp225_critic_diag.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(f"\n[saved {out}]")
        import shutil
        shutil.rmtree(clone, ignore_errors=True)


if __name__ == "__main__":
    main()
