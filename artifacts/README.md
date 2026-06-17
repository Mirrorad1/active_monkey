# artifacts/

Local, copyable **active_monkey** agent artifacts (Hugging-Face-style layout, but
local-first — no upload required, no network).

Export the first artifact here:

```bash
uv run active-monkey artifact export --preset affect-dyad-v0 --out artifacts/active-monkey-affect-dyad-v0
uv run active-monkey artifact inspect artifacts/active-monkey-affect-dyad-v0
uv run active-monkey score   artifacts/active-monkey-affect-dyad-v0
uv run active-monkey converse artifacts/active-monkey-affect-dyad-v0 --demo
```

Exported artifact directories are **not committed** by default (they are reproducible from
seed + config, and the learned/eval payloads are large-ish). See `docs/ARTIFACTS.md` for
the full layout, vocabulary ("weights" = probability tables; "belief" = posterior over a
hidden state), scorer reproduction, and the conservative-claims policy.

This is functional valence only — no sentience, consciousness, AGI, or subjective-feeling
claim.
