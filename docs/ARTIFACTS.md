# active_monkey artifacts

An **artifact** is a self-contained, copyable directory that captures one agent: its
numeric checkpoints, provenance, the frozen scorer it should be judged by, a model card,
and runnable examples. It is local-first — exporting, loading, inspecting, scoring, and
conversing never upload anything and never require the network.

The first artifact is **`active-monkey-affect-dyad-v0`**: the toy affective-dyad agent
(`DirectHeadAgent`) from the M4a/M4b chapter (Exp 214–225).

## What this is (and is not)

This is **functional valence only**. The agent receives symbolic utterance **codes**,
infers a hidden **intent-like** state, chooses a response code, receives functional
valence feedback (a scalar, not a feeling), and updates learned tables during a session.

It does **not** claim sentience, consciousness, AGI, subjective feeling, or natural-language
understanding. "Talking to it" means feeding it integer codes and `+`/`-` feedback.

## Vocabulary (read this before "weights" or "belief" mislead you)

- **"weights"** here means **probability tables, learned Dirichlet pseudo-counts, and
  generative-model tensors** — *not* neural-network weights. The affect dyad has no neural
  net; its "parameters" are `A`/`B` emission/transition tensors and their Dirichlet
  concentrations `pA`/`pB`.
- **"belief"** means a **posterior distribution over a hidden state** (operational,
  Bayesian) — *not* subjective belief.
- **"learning"** means windowed Dirichlet count updates during a session.

## Layout

```
active-monkey-affect-dyad-v0/
  README.md                  # human overview
  model_card.yaml            # Hugging-Face-style model card (conservative language)
  config.json                # construction config (deterministic from seed)
  manifest.json              # provenance + pinned hashes + limitations
  init.safetensors           # untrained generative-model tensors
  init.config.json           # sidecar metadata for the init checkpoint
  learned_example.safetensors  # tensors after a short scripted session (if exported)
  learned_example.config.json
  eval_results/
    affect_score_baseline.json   # constant-control score (abbreviated demo config)
    affect_score_candidate.json  # learner score (abbreviated demo config)
  examples/
    converse_demo.py
    score_model.py
```

### What is in `init.safetensors` vs `learned_example.safetensors`

Both store the same named tensors of the `DirectHeadAgent` generative model:

| name | meaning |
|------|---------|
| `a0` | `P(utterance \| intent, last_response)` emission |
| `a1` | `P(valence \| intent, last_response)` — the direct valence head (learned) |
| `b0` | `P(intent' \| intent)` — identity, uncontrolled |
| `b1` | `P(last_response' \| last_response, response)` — deterministic action-set |
| `pa0`, `pa1` | Dirichlet concentrations for `a0`, `a1` (the learned counts) |
| `pb0`, `pb1` | Dirichlet concentrations for `b0`, `b1` (structural) |
| `c0`, `c1` | preferences over utterance / valence (`c1 = [-2, 0, 3]` favors POS) |
| `d0`, `d1` | priors over the two hidden factors |

- **`init`** = the untrained model straight from `build_direct_head_model(seed)`.
- **`learned_example`** = the same tensors after a scripted-partner session; `a0/a1` and
  their `pa0/pa1` counts have moved, `b*` stays structural. It also stores the final
  belief posterior under `belief::q0`, `belief::q1`, and **history hashes** (sha256 of the
  observation/action/valence sequences — not the raw logs).

## AgentState / checkpoint abstraction

`active_loop/state.py` defines a substrate-independent state layer intended to later
carry ecology agents, belief agents, and communication agents too:

- `AgentState` — architecture id, agent class, tensor dict, belief state, history hashes,
  RNG state, provenance, scorer compatibility, schema version, metadata.
- `AgentProvenance` — repo commit, source experiment ids, created-at, source repo.
- `ScorerCompatibility` — the frozen scorer path + sha256 + version + metric name.
- `AgentCheckpoint` — a named `AgentState` + its on-disk files.
- `ArtifactManifest` — the validated `manifest.json`.

It supports deterministic JSON metadata, safetensors tensor payloads, save/load
roundtrips, a **stable content hash** (excludes volatile `created_at`/`repo_commit` so
identical agents hash identically), and **graceful schema-mismatch refusal**
(`SchemaMismatch` unless `allow_schema_mismatch=True`). Public artifacts never use pickle.

## CLI

```bash
# Export the preset artifact (init tensors need no pymdp; learned/eval need it)
uv run active-monkey artifact export --preset affect-dyad-v0 --out artifacts/active-monkey-affect-dyad-v0

# Inspect + verify recorded hashes (nonzero exit if a check fails)
uv run active-monkey artifact inspect artifacts/active-monkey-affect-dyad-v0 [--json]

# Score with the FROZEN scorer (full config = 300 turns x seeds 20..27)
uv run active-monkey score artifacts/active-monkey-affect-dyad-v0 [--json] [--quick]

# Converse: scripted demo or interactive REPL
uv run active-monkey converse artifacts/active-monkey-affect-dyad-v0 --demo
uv run active-monkey converse artifacts/active-monkey-affect-dyad-v0 --interactive
```

`--no-learned` / `--no-eval` skip the pymdp/JAX-dependent export steps (fast path for
clones without the inference engine, and for the fast test suite).

## Reproducing the scorer

The frozen scorer is `eval/affect_score.py`. Its sha256 is recorded in every manifest as
`scorer_hash`. `active-monkey score` recomputes that hash and **refuses to score** (clear
structured error, nonzero exit) if the on-disk scorer no longer matches — so the scorer
cannot drift silently. If the scorer must change it becomes a **new scorer version with a
new hash and docs**; the old artifacts keep pointing at the old hash.

```bash
uv run python -c "import hashlib;print(hashlib.sha256(open('eval/affect_score.py','rb').read()).hexdigest())"
```

The metric is the **mean last-third POSITIVE-feedback rate**; a seed counts as *genuine*
only if it clears the **1/3 constant-response ceiling** AND `correct_select >= 0.5`
(constant-unfakeable — a constant policy maps at most 2/6 codes correctly).

## Known limitations

- symbolic utterance codes, not natural language;
- long-session learning is load-bearing (Exp 221: short sessions block learning);
- not sentience, consciousness, AGI, or subjective feeling;
- constant-response and shuffled controls are required to read any result honestly.

## Citing / forking

Fork the repository and cite the `artifact_id`, the `repo_commit`, and the frozen-scorer
sha256 from `manifest.json`. There is no required cloud upload; an artifact directory is
the citable object.
