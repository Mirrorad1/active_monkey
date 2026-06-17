# active-monkey-affect-dyad-v0

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
- Frozen scorer: `eval/affect_score.py` (sha256 `68064f6980c570fefced2c9e918d07424f76229be9360c0d43860d0f987ef22a`). If the scorer ever changes it
  must become a NEW scorer version with a new hash and docs — never edited silently.
