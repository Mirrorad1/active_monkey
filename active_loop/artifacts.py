"""Local, Hugging-Face-style artifact system for the active_monkey affective dyad.

An *artifact* is a self-contained, copyable directory describing one agent: its
init/learned numeric checkpoints (safetensors), a manifest pinning the frozen scorer
by hash, a model card, config, bundled eval results, and runnable examples.  Nothing
here uploads anywhere or requires the network; everything is local-first.

Honest framing (see docs/ARTIFACTS.md):
  - "weights" == probability tables / Dirichlet pseudo-count tensors / generative-model
    arrays, NOT neural-net weights.
  - "belief" == a posterior over a hidden state, NOT subjective belief.
  - This is functional valence only; no sentience / feeling / AGI claim.

The save / load / inspect / hash paths are kept free of any agent-runtime import
(pymdp / JAX): a fresh clone can build, hash, and inspect the init tensors without the
inference engine installed.  The *learned* checkpoint, scoring, and conversing paths
do need the engine and import it lazily, raising a clear error if it is absent.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from active_loop.state import (
    SCHEMA_VERSION,
    AgentCheckpoint,
    AgentProvenance,
    AgentState,
    ArtifactManifest,
    ScorerCompatibility,
    canonical_json,
)

# Paths relative to repo root.
FROZEN_SCORER_PATH = "eval/affect_score.py"
DEFAULT_ARTIFACT_ID = "active-monkey-affect-dyad-v0"
SOURCE_EXPERIMENTS = [222, 225]
KNOWN_LIMITATIONS = [
    "symbolic utterance codes, not natural language",
    "long-session learning is load-bearing",
    "not sentience, consciousness, AGI, or subjective feeling",
    "constant-response and shuffled controls required",
]


# ── Hashing helpers ──────────────────────────────────────────────────────────

def hash_file(path: str | Path) -> str:
    """sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(directory: str | Path, exclude: tuple[str, ...] = ("manifest.json",)) -> dict:
    """Return {relpath: sha256} for every file under directory (sorted), plus a combined hash.

    `exclude` skips files by basename (e.g. the manifest, which itself stores hashes).
    """
    directory = Path(directory)
    files: dict[str, str] = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.name not in exclude:
            files[str(p.relative_to(directory))] = hash_file(p)
    combined = hashlib.sha256(canonical_json(files).encode()).hexdigest()
    return {"files": files, "combined": combined}


def repo_commit(repo: str | Path = ".") -> str:
    """Best-effort git SHA of the repo; 'unknown' if not a git repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def scorer_hash(repo: str | Path = ".") -> str:
    """sha256 of the FROZEN scorer file (provenance pin)."""
    return hash_file(Path(repo) / FROZEN_SCORER_PATH)


# ── Tensor extraction (no pymdp needed to build the init model) ───────────────

_MODEL_KEYS = [
    ("A", 0, "a0"), ("A", 1, "a1"),
    ("pA", 0, "pa0"), ("pA", 1, "pa1"),
    ("B", 0, "b0"), ("B", 1, "b1"),
    ("pB", 0, "pb0"), ("pB", 1, "pb1"),
    ("C", 0, "c0"), ("C", 1, "c1"),
    ("D", 0, "d0"), ("D", 1, "d1"),
]


def _model_tensors(model_dict: dict) -> dict[str, np.ndarray]:
    """Flatten a build_direct_head_model dict (lists of batch-first arrays) to named numpy."""
    out: dict[str, np.ndarray] = {}
    for key, idx, name in _MODEL_KEYS:
        out[name] = np.asarray(model_dict[key][idx], dtype=np.float32)
    return out


def _tensors_to_model_dict(tensors: dict[str, np.ndarray]) -> dict:
    """Inverse of _model_tensors: rebuild a model_dict of JAX arrays (lazy jax import)."""
    import jax.numpy as jnp  # noqa: PLC0415 (lazy: artifact build/inspect must not need JAX)

    def g(name: str):
        return jnp.asarray(tensors[name])

    return dict(
        A=[g("a0"), g("a1")],
        pA=[g("pa0"), g("pa1")],
        B=[g("b0"), g("b1")],
        pB=[g("pb0"), g("pb1")],
        C=[g("c0"), g("c1")],
        D=[g("d0"), g("d1")],
    )


# ── Construction config for the affect-dyad preset (the frozen winning config) ─

@dataclass(frozen=True)
class AffectDyadConfig:
    """The DirectHeadAgent construction config used by the affect-dyad-v0 preset.

    Mirrors eval/affect_score._direct_head_factory / cli.converse._make_agent (the
    validated Exp 220 sched_full config).  Stored verbatim in the artifact config.json
    so a loaded agent reconstructs deterministically from seed.
    """

    k: int = 4
    optimism: float = 2.0
    lr_pA: float = 4.0
    gamma: float = 1.0
    alpha: float = 1.0
    lv: float = 0.999
    canonical_turns: int = 300
    canonical_seeds: tuple[int, ...] = tuple(range(20, 28))

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "optimism": self.optimism,
            "lr_pA": self.lr_pA,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "lv": self.lv,
            "canonical_turns": self.canonical_turns,
            "canonical_seeds": list(self.canonical_seeds),
            "architecture": "M4a affective dyad (DirectHeadAgent, direct response->valence head)",
        }


# ── init / learned checkpoint builders ───────────────────────────────────────

def build_init_state(seed: int, cfg: AffectDyadConfig, repo: str | Path = ".") -> AgentState:
    """Build the INIT AgentState (untrained generative-model tensors). No pymdp needed."""
    from active_loop.affect_spec import build_direct_head_model  # numpy/jax only, no pymdp

    model = build_direct_head_model(seed, k=cfg.k)
    tensors = _model_tensors(model)
    prov = AgentProvenance(
        repo_commit=repo_commit(repo),
        source_experiments=list(SOURCE_EXPERIMENTS),
        created_at=AgentProvenance.now(),
        source_repo="active_monkey",
        notes="init (untrained) generative-model tensors for the affective dyad",
    )
    sc = ScorerCompatibility(
        scorer_path=FROZEN_SCORER_PATH,
        scorer_hash=scorer_hash(repo),
    )
    return AgentState(
        architecture_id="M4a-affect-dyad-DirectHeadAgent",
        agent_class="DirectHeadAgent",
        tensors=tensors,
        belief_state=None,
        history_hashes={},
        rng_state=[0, int(seed)],  # JAX PRNGKey(seed) form: [0, seed]
        provenance=prov,
        scorer_compat=sc,
        metadata={
            "checkpoint": "init",
            "seed": int(seed),
            "construction": cfg.as_dict(),
            "runtime_claim": "functional valence only; no subjective feeling claim",
        },
    )


def build_learned_state(
    seed: int, cfg: AffectDyadConfig, turns: int = 60, repo: str | Path = "."
) -> AgentState:
    """Run one scripted-partner session and snapshot the LEARNED tensors (needs pymdp).

    The learned tensors (A/pA changed; B/pB structural) plus the final belief posterior
    are captured.  History is stored as hashes only (observation/action sequences), not
    raw logs.
    """
    agent, hist = _run_learning_session(seed, cfg, turns)
    learned_model = dict(
        A=list(agent.agent.A), pA=list(agent.agent.pA),
        B=list(agent.agent.B), pB=list(agent.agent.pB),
        C=list(agent.agent.C), D=list(agent.agent.D),
    )
    tensors = _model_tensors(learned_model)
    belief = None
    if agent._last_qs is not None:
        belief = {f"q{f}": np.asarray(agent._last_qs[f], dtype=np.float32)
                  for f in range(len(agent._last_qs))}

    prov = AgentProvenance(
        repo_commit=repo_commit(repo),
        source_experiments=list(SOURCE_EXPERIMENTS),
        created_at=AgentProvenance.now(),
        source_repo="active_monkey",
        notes=f"learned example after a {turns}-turn scripted-partner session (seed {seed})",
    )
    sc = ScorerCompatibility(scorer_path=FROZEN_SCORER_PATH, scorer_hash=scorer_hash(repo))
    return AgentState(
        architecture_id="M4a-affect-dyad-DirectHeadAgent",
        agent_class="DirectHeadAgent",
        tensors=tensors,
        belief_state=belief,
        history_hashes={
            "observation_codes": hashlib.sha256(np.asarray(hist["codes"]).tobytes()).hexdigest(),
            "actions": hashlib.sha256(np.asarray(hist["actions"]).tobytes()).hexdigest(),
            "valences": hashlib.sha256(np.asarray(hist["valences"]).tobytes()).hexdigest(),
        },
        rng_state=None,
        provenance=prov,
        scorer_compat=sc,
        metadata={
            "checkpoint": "learned_example",
            "seed": int(seed),
            "turns": int(turns),
            "construction": cfg.as_dict(),
            "session_pos_rate": hist["pos_rate"],
            "runtime_claim": "functional valence only; no subjective feeling claim",
        },
    )


def _run_learning_session(seed: int, cfg: AffectDyadConfig, turns: int):
    """Drive a DirectHeadAgent through a scripted-partner session (lazy pymdp import)."""
    from active_loop.affect_agent import DirectHeadAgent  # lazy: imports pymdp
    from active_loop.affect_spec import build_direct_head_model, U, POS, NEU, ASK

    correct = {c: c % 4 for c in range(U)}
    agent = DirectHeadAgent(
        build_direct_head_model(seed, k=cfg.k), seed=seed,
        gamma=cfg.gamma, alpha=cfg.alpha, lr_pA=cfg.lr_pA, lv=cfg.lv,
        optimism=cfg.optimism, gamma_schedule=(1.0, 8.0, turns),
    )
    rng = np.random.default_rng(seed)
    pool: list[int] = []

    def nxt() -> int:
        nonlocal pool
        if not pool:
            b = list(range(U)); rng.shuffle(b); pool += b
        return pool.pop(0)

    codes, actions, valences = [], [], []
    pos = 0
    for _ in range(turns):
        code = nxt()
        agent.perceive(code)
        r = agent.act()
        val = POS if r == correct[code] else (NEU if r == ASK else 0)
        agent.observe_feedback(code, val)
        codes.append(code); actions.append(int(r)); valences.append(int(val))
        pos += (val == POS)
    return agent, {
        "codes": codes, "actions": actions, "valences": valences,
        "pos_rate": pos / turns if turns else 0.0,
    }


# ── checkpoint save/load passthroughs ────────────────────────────────────────

def save_checkpoint(state: AgentState, directory: str | Path, name: str) -> AgentCheckpoint:
    """Persist an AgentState as a named checkpoint under directory."""
    return AgentCheckpoint(name=name, state=state).save(directory)


def load_checkpoint(directory: str | Path, name: str, allow_schema_mismatch: bool = False) -> AgentCheckpoint:
    """Load a named checkpoint written by save_checkpoint."""
    return AgentCheckpoint.load(directory, name, allow_schema_mismatch=allow_schema_mismatch)


# ── Manifest load ────────────────────────────────────────────────────────────

def load_manifest(artifact_dir: str | Path, allow_schema_mismatch: bool = False) -> dict:
    """Load + validate manifest.json; raises on missing/corrupt/incompatible."""
    m = ArtifactManifest.from_path(Path(artifact_dir) / "manifest.json",
                                   allow_schema_mismatch=allow_schema_mismatch)
    return m.data


# ── Export ───────────────────────────────────────────────────────────────────

def export_affect_dyad_artifact(
    out_dir: str | Path,
    seed: int = 20,
    learn_turns: int = 60,
    run_learned: bool = True,
    run_eval: bool = True,
    eval_seeds: tuple[int, ...] = (20, 21),
    eval_turns: int = 60,
    repo: str | Path = ".",
    artifact_id: str = DEFAULT_ARTIFACT_ID,
) -> dict:
    """Write a complete affect-dyad artifact directory and return the manifest dict.

    run_learned=False / run_eval=False skip the pymdp-dependent steps (fast path used
    by tests and by clones without the inference engine).  The bundled eval_results use
    an abbreviated config for speed; the manifest records the *canonical* frozen config
    (300 turns x seeds 20..27) that `active-monkey score` uses.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results").mkdir(exist_ok=True)
    (out_dir / "examples").mkdir(exist_ok=True)
    cfg = AffectDyadConfig()

    # ── init checkpoint (always; no pymdp) ──
    init_state = build_init_state(seed, cfg, repo=repo)
    init_ckpt = save_checkpoint(init_state, out_dir, "init")
    init_hash = init_state.content_hash()

    # ── learned checkpoint (optional; needs pymdp) ──
    learned_hash = None
    learned_unavailable = None
    if run_learned:
        try:
            learned_state = build_learned_state(seed, cfg, turns=learn_turns, repo=repo)
            save_checkpoint(learned_state, out_dir, "learned_example")
            learned_hash = learned_state.content_hash()
        except Exception as e:  # pymdp/JAX missing or runtime error -> honest skip
            learned_unavailable = f"{type(e).__name__}: {e}"

    # ── bundled eval results (optional; needs pymdp) ──
    eval_status = "skipped"
    if run_eval:
        eval_status = _write_eval_results(out_dir / "eval_results", eval_seeds, eval_turns, cfg)

    # ── config.json ──
    config = {
        "artifact_id": artifact_id,
        "schema_version": SCHEMA_VERSION,
        "agent_class": "DirectHeadAgent",
        "architecture": cfg.as_dict()["architecture"],
        "construction": cfg.as_dict(),
        "seed": int(seed),
        "runtime_claim": "functional valence only; no subjective feeling claim",
    }
    (out_dir / "config.json").write_text(canonical_json(config) + "\n")

    # ── manifest.json ──
    manifest = {
        "artifact_id": artifact_id,
        "schema_version": SCHEMA_VERSION,
        "repo_commit": repo_commit(repo),
        "source_experiments": list(SOURCE_EXPERIMENTS),
        "agent_class": "DirectHeadAgent",
        "architecture": "M4a/M4b affective dyad (DirectHeadAgent direct response->valence head)",
        "runtime_claim": "functional valence only; no subjective feeling claim",
        "frozen_scorer": FROZEN_SCORER_PATH,
        "scorer_hash": scorer_hash(repo),
        "scorer_version": "affect-score-1e",
        "init_checkpoint_hash": init_hash,
        "learned_checkpoint_hash": learned_hash,
        "learned_unavailable": learned_unavailable,
        "seeds": list(cfg.canonical_seeds),
        "turns": cfg.canonical_turns,
        "metric_name": "mean_last_third_pos",
        "eval_results_status": eval_status,
        "known_limitations": list(KNOWN_LIMITATIONS),
    }
    ArtifactManifest(manifest).validate()
    (out_dir / "manifest.json").write_text(canonical_json(manifest) + "\n")

    # ── model card + README + examples ──
    _write_model_card(out_dir, manifest, cfg)
    _write_readme(out_dir, manifest, cfg)
    _write_examples(out_dir, artifact_id)

    return manifest


def _write_eval_results(eval_dir: Path, seeds: tuple[int, ...], turns: int, cfg: AffectDyadConfig) -> str:
    """Run an abbreviated learner vs constant-control score; write two JSON files.

    Returns a status string ('ok' / 'skipped:<reason>').
    """
    try:
        from eval.affect_score import score_affect, _constant_factory  # lazy: imports pymdp
    except Exception as e:
        (eval_dir / "affect_score_baseline.json").write_text(
            canonical_json({"status": "skipped", "reason": f"{type(e).__name__}: {e}"}) + "\n")
        return f"skipped:{type(e).__name__}"

    from dataclasses import asdict
    baseline = score_affect(seeds=seeds, turns=turns, agent_factory=_constant_factory(0))
    candidate = score_affect(seeds=seeds, turns=turns)
    note = {"note": "ABBREVIATED demo config; canonical scoring is `active-monkey score` "
                    "(300 turns x seeds 20..27).", "seeds": list(seeds), "turns": turns}
    (eval_dir / "affect_score_baseline.json").write_text(
        canonical_json({**asdict(baseline), **note, "agent": "constant-control"}) + "\n")
    (eval_dir / "affect_score_candidate.json").write_text(
        canonical_json({**asdict(candidate), **note, "agent": "DirectHeadAgent-learner"}) + "\n")
    return "ok"


# ── Load agent ───────────────────────────────────────────────────────────────

def load_agent_from_artifact(artifact_dir: str | Path, which: str = "init", seed: int | None = None):
    """Reconstruct a runnable DirectHeadAgent from an artifact checkpoint (lazy pymdp).

    which="init"  -> a fresh learner from the init tensors (learns live during a session).
    which="learned_example" -> resumes from the learned Dirichlet counts (optimism already
       baked into the counts, so it is not re-applied).
    """
    from active_loop.affect_agent import DirectHeadAgent  # lazy: imports pymdp

    artifact_dir = Path(artifact_dir)
    manifest = load_manifest(artifact_dir)
    cfg_data = json.loads((artifact_dir / "config.json").read_text())
    cons = cfg_data.get("construction", {})
    use_seed = seed if seed is not None else int(cfg_data.get("seed", 20))

    ckpt = load_checkpoint(artifact_dir, which)
    model_dict = _tensors_to_model_dict(ckpt.state.tensors)
    # optimism is re-applied only for a fresh 'init' agent; the learned counts already
    # include it, so resuming from 'learned_example' must NOT double-count it.
    optimism = float(cons.get("optimism", 2.0)) if which == "init" else 0.0
    agent = DirectHeadAgent(
        model_dict, seed=use_seed,
        gamma=float(cons.get("gamma", 1.0)), alpha=float(cons.get("alpha", 1.0)),
        lr_pA=float(cons.get("lr_pA", 4.0)), lv=float(cons.get("lv", 0.999)),
        optimism=optimism,
    )
    return agent


# ── Inspect ──────────────────────────────────────────────────────────────────

def inspect_artifact(artifact_dir: str | Path, verify: bool = True) -> dict:
    """Return a structured summary of an artifact; optionally verify recorded hashes.

    Raises (via load_manifest) on a missing/corrupt/incompatible manifest.  Hash
    verification mismatches are reported in the returned dict (not raised) so callers
    can decide severity.
    """
    artifact_dir = Path(artifact_dir)
    manifest = load_manifest(artifact_dir)
    summary: dict[str, Any] = {
        "artifact_id": manifest.get("artifact_id"),
        "schema_version": manifest.get("schema_version"),
        "agent_class": manifest.get("agent_class"),
        "architecture": manifest.get("architecture"),
        "repo_commit": manifest.get("repo_commit"),
        "source_experiments": manifest.get("source_experiments"),
        "runtime_claim": manifest.get("runtime_claim"),
        "frozen_scorer": manifest.get("frozen_scorer"),
        "scorer_hash": manifest.get("scorer_hash"),
        "init_checkpoint_hash": manifest.get("init_checkpoint_hash"),
        "learned_checkpoint_hash": manifest.get("learned_checkpoint_hash"),
        "known_limitations": manifest.get("known_limitations"),
        "files": sorted(str(p.relative_to(artifact_dir)) for p in artifact_dir.rglob("*") if p.is_file()),
        "checks": {},
    }
    if verify:
        checks: dict[str, Any] = {}
        # init checkpoint content hash
        try:
            init = load_checkpoint(artifact_dir, "init")
            checks["init_hash_ok"] = (init.content_hash() == manifest.get("init_checkpoint_hash"))
        except Exception as e:
            checks["init_hash_ok"] = False
            checks["init_error"] = f"{type(e).__name__}: {e}"
        # learned checkpoint, if present
        if (artifact_dir / "learned_example.safetensors").exists() and manifest.get("learned_checkpoint_hash"):
            try:
                learned = load_checkpoint(artifact_dir, "learned_example")
                checks["learned_hash_ok"] = (learned.content_hash() == manifest.get("learned_checkpoint_hash"))
            except Exception as e:
                checks["learned_hash_ok"] = False
                checks["learned_error"] = f"{type(e).__name__}: {e}"
        # scorer hash still matches the on-disk frozen scorer (verified against the repo copy)
        repo_scorer = Path(FROZEN_SCORER_PATH)
        if repo_scorer.exists():
            checks["scorer_hash_ok"] = (hash_file(repo_scorer) == manifest.get("scorer_hash"))
        else:
            checks["scorer_hash_ok"] = None  # cannot verify outside the repo
        summary["checks"] = checks
    return summary


# ── Score ────────────────────────────────────────────────────────────────────

def score_artifact(
    artifact_dir: str | Path,
    seeds: tuple[int, ...] | None = None,
    turns: int | None = None,
    repo: str | Path = ".",
) -> dict:
    """Score an artifact with the FROZEN scorer; verify the scorer hash matches the manifest.

    Returns a dict: either the AffectScoreReport fields (on success) or a structured
    failure {"status": "error", ...} if the engine is missing or the scorer hash drifted.
    """
    from dataclasses import asdict

    artifact_dir = Path(artifact_dir)
    manifest = load_manifest(artifact_dir)

    # provenance gate: the on-disk frozen scorer must match the pinned hash.
    try:
        current = scorer_hash(repo)
    except FileNotFoundError:
        return {"status": "error", "reason": "frozen scorer file not found", "scorable": False}
    if current != manifest.get("scorer_hash"):
        return {
            "status": "error",
            "reason": "scorer hash drift: the frozen scorer changed but the manifest was not "
                      "reissued as a new scorer version",
            "manifest_scorer_hash": manifest.get("scorer_hash"),
            "current_scorer_hash": current,
            "scorable": False,
        }

    try:
        from eval.affect_score import score_affect, SEEDS_DEFAULT, TURNS_DEFAULT  # lazy: pymdp
    except Exception as e:
        return {
            "status": "error",
            "reason": f"scoring engine unavailable ({type(e).__name__}: {e}); "
                      f"install pymdp to score this artifact",
            "scorable": False,
        }

    report = score_affect(
        seeds=seeds or SEEDS_DEFAULT,
        turns=turns or TURNS_DEFAULT,
    )
    out = asdict(report)
    out["status"] = "ok"
    out["scorer_hash"] = current
    out["scorer_path"] = FROZEN_SCORER_PATH
    out["artifact_id"] = manifest.get("artifact_id")
    return out


# ── Card / README / examples writers ─────────────────────────────────────────

def _write_model_card(out_dir: Path, manifest: dict, cfg: AffectDyadConfig) -> None:
    card = _MODEL_CARD_TEMPLATE.format(
        artifact_id=manifest["artifact_id"],
        scorer_hash=manifest["scorer_hash"],
        repo_commit=manifest["repo_commit"],
        init_hash=manifest["init_checkpoint_hash"],
        learned_hash=manifest.get("learned_checkpoint_hash"),
        turns=cfg.canonical_turns,
        seeds=list(cfg.canonical_seeds),
        limitations="\n".join(f"    - {x}" for x in KNOWN_LIMITATIONS),
    )
    (out_dir / "model_card.yaml").write_text(card)


def _write_readme(out_dir: Path, manifest: dict, cfg: AffectDyadConfig) -> None:
    (out_dir / "README.md").write_text(_README_TEMPLATE.format(
        artifact_id=manifest["artifact_id"],
        scorer_hash=manifest["scorer_hash"],
        scorer_path=manifest["frozen_scorer"],
    ))


def _write_examples(out_dir: Path, artifact_id: str) -> None:
    (out_dir / "examples" / "converse_demo.py").write_text(_CONVERSE_DEMO_EXAMPLE)
    (out_dir / "examples" / "score_model.py").write_text(_SCORE_EXAMPLE)


_MODEL_CARD_TEMPLATE = """\
# Model card (Hugging-Face-style) for an active_monkey artifact.
model_card_version: "0.1.0"
artifact_id: "{artifact_id}"
license: "see repository"
tags:
  - active-inference
  - functional-valence
  - affective-dyad
  - toy
summary: >
  This artifact is a toy affective-dyad agent. It receives symbolic utterance codes,
  infers a hidden intent-like state, chooses response codes, receives functional valence
  feedback, and updates learned tables during a session. It does not use natural language,
  does not claim subjective feeling, and should be evaluated only through the included
  frozen scorer and controls.
what_is_saved:
  init.safetensors: "untrained generative-model tensors (probability tables + Dirichlet counts)"
  learned_example.safetensors: "tensors after a scripted-partner session (A/pA learned; B structural)"
weights_meaning: >
  'weights' here means probability tables, learned Dirichlet counts, and generative-model
  tensors -- NOT neural-network weights.
belief_meaning: >
  'belief' means a posterior distribution over a hidden state -- NOT subjective belief.
provenance:
  repo_commit: "{repo_commit}"
  frozen_scorer_sha256: "{scorer_hash}"
  init_checkpoint_sha256: "{init_hash}"
  learned_checkpoint_sha256: "{learned_hash}"
evaluation:
  canonical_turns: {turns}
  canonical_seeds: {seeds}
  metric: "mean last-third POSITIVE-feedback rate; genuine only if it clears the 1/3
    constant-response ceiling AND correct_select >= 0.5 (constant-unfakeable)."
known_limitations:
{limitations}
not_a_claim_of: >
  sentience, consciousness, AGI, subjective feeling, or natural-language understanding.
citation: >
  Fork the repository and cite the artifact_id, repo_commit, and frozen_scorer sha256.
"""

_README_TEMPLATE = """\
# {artifact_id}

A local, copyable **active_monkey** artifact: a toy *affective-dyad* agent.

It receives symbolic utterance **codes**, infers a hidden **intent-like** state, chooses a
response code, receives **functional valence** feedback, and updates learned tables during
a session. It does **not** use natural language and makes **no** claim of subjective
feeling. Evaluate it only through the bundled frozen scorer and its controls.

## Files
- `manifest.json` — provenance, pinned frozen-scorer hash, checkpoint hashes, limitations.
- `config.json` — construction config (deterministic from seed).
- `model_card.yaml` — Hugging-Face-style model card (conservative language).
- `init.safetensors` — untrained generative-model tensors.
- `learned_example.safetensors` — tensors after a short scripted session (if present).
- `eval_results/` — bundled (abbreviated) learner-vs-constant scores.
- `examples/` — runnable `converse_demo.py` and `score_model.py`.

## Use
```bash
uv run active-monkey artifact inspect <this-dir>
uv run active-monkey score <this-dir>            # full frozen config (300 turns x 8 seeds)
uv run active-monkey converse <this-dir> --demo
```

## Honesty
- "weights" = probability tables / Dirichlet counts, not neural-net weights.
- "belief" = posterior over a hidden state, not subjective belief.
- Frozen scorer: `{scorer_path}` (sha256 `{scorer_hash}`). If the scorer ever changes it
  must become a NEW scorer version with a new hash and docs — never edited silently.
"""

_CONVERSE_DEMO_EXAMPLE = '''\
"""Runnable example: load this artifact and run a scripted-partner demo session.

    uv run python examples/converse_demo.py <artifact_dir>
"""
import sys
from active_loop.artifacts import load_agent_from_artifact
from active_loop.affect_spec import U, POS, NEU, ASK
import numpy as np

def main(artifact_dir):
    agent = load_agent_from_artifact(artifact_dir, which="init")
    correct = {c: c % 4 for c in range(U)}
    rng = np.random.default_rng(0)
    for t in range(30):
        code = int(rng.integers(0, U))
        agent.perceive(code)
        r = agent.act()
        val = POS if r == correct[code] else (NEU if r == ASK else 0)
        agent.observe_feedback(code, val)
        print(f"t={t:2d} code={code} response={r} valence={val}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
'''

_SCORE_EXAMPLE = '''\
"""Runnable example: score this artifact with the FROZEN scorer.

    uv run python examples/score_model.py <artifact_dir>
"""
import json, sys
from active_loop.artifacts import score_artifact

def main(artifact_dir):
    # Use a short config here for speed; drop the kwargs for the full frozen config.
    print(json.dumps(score_artifact(artifact_dir, seeds=(20, 21), turns=60), indent=2))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
'''
