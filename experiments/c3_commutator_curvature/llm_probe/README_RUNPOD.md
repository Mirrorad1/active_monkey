# C3 LLM-probe on RunPod (pull → run)

Tests whether the **second-order residue signal survives a real, noisy LLM
loss** (the deterministic experiment only proved the *mechanism* on a noiseless
oracle). It replaces the binary oracle with the **NLL of the gold answer** under
a small causal LM.

**The headline is DETECTABILITY + COVERAGE, not AUC alone.** ROC-AUC is computed
only over true pairs that actually reach it, so a high AUC over a handful of
survivable pairs can overstate viability. The verdict therefore requires *both*:

- **Q1a (detectability / signal quality):** is `sigma_ij` for the true dangerous
  pair (the two redundant mentions) separated from `sigma_ij` of random safe
  pairs? Reported as median split + **test-only ROC-AUC with a bootstrap CI**
  (plus all-split AUC). AUC ≈ 0.5 ⇒ no usable residue ⇒ clean falsification.
- **Q1b (coverage / signal reachability):** does the true pair even reach the
  AUC? It only does if both endpoints pass the safe gate (`delta ≤ tau`) AND
  survive the pair cap. Reported as `true_pair_tested_rate` + a breakdown of why
  pairs go missing (gated vs capped). A low tested-rate ⇒ the signal exists but
  is not reliably reachable by the mechanism.
- **Q2 (utility):** does NLL-thresholded C3 beat `solo_delta_greedy` on
  greedy-decode answer accuracy under tight budgets (threshold calibrated on a
  dev split only), with a **bootstrap CI on the per-budget delta**?

It is self-contained — one file, standard `transformers`, ~minutes on any
modern GPU. Tiny job: do **not** rent an H200/B200.

## 1. Pick the pod
- **Image:** official **`runpod/pytorch` CUDA 12.x** template (torch already
  installed; this probe installs only `transformers`). A `cu124`/`cu128` tag is
  fine. Avoid bare `ubuntu`/ComfyUI/SD templates.
- **GPU:** anything mature — RTX 4090 / L40S / A100 / H100. ~10 GB is plenty.
  Blackwell (RTX 5090 / RTX PRO 6000, sm_120) needs a **CUDA 12.8+** image; the
  setup script's kernel check will fail loudly if the arch/image mismatch.
- Disk ~20 GB. (Qwen2.5-0.5B is ~1 GB.)

## 2. Clone this repo on the pod
> ⚠️ This probe currently lives on branch **`sparse-llm`** (commit `301239c` or
> later), **not** on `main`. You MUST check out that branch or commit, or the
> `experiments/c3_commutator_curvature/llm_probe/` path will not exist.

Public repo (branch):
```bash
git clone --depth 1 --branch sparse-llm \
  https://github.com/<owner>/<repo>.git /workspace/active-loop
```
Private repo — fine-grained PAT (Contents: Read-only), injected into the URL
(not written to disk):
```bash
export GITHUB_TOKEN=github_pat_xxx
git clone --depth 1 --branch sparse-llm \
  "https://${GITHUB_TOKEN}@github.com/<owner>/<repo>.git" /workspace/active-loop
```
Pin an exact commit instead of the branch tip (reproducible):
```bash
git clone "https://${GITHUB_TOKEN}@github.com/<owner>/<repo>.git" /workspace/active-loop
cd /workspace/active-loop && git checkout 301239c   # or the later commit you want
```
(Once this is merged to `main`, drop `--branch sparse-llm`.)

## 3. Run
```bash
cd /workspace/active-loop/experiments/c3_commutator_curvature/llm_probe
bash setup_runpod.sh                 # installs transformers, CUDA check, runs n=60
# bigger run / different model:
bash setup_runpod.sh --n 200
C3_MODEL=Qwen/Qwen2.5-1.5B-Instruct bash setup_runpod.sh --n 120
```
Expect silence during the residue sweep (no per-step printer until every 10th
instance). It prints `Q1`/`Q2` and a `VERDICT HINT` at the end.

## 4. Pull results back to your Mac
```bash
# on the pod:
runpodctl send results/llm_probe_summary.json     # prints a one-time code
# on the Mac (install runpodctl once, see below):
runpodctl receive <code>
```
Install `runpodctl` on Apple-Silicon without sudo:
```bash
curl -fsSL -o "$HOME/.local/bin/runpodctl" \
  https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-darwin-arm64
chmod +x "$HOME/.local/bin/runpodctl"
```
**Terminate the pod once you have the file — it meters per hour.**

## 5. Reading the result
`results/llm_probe_summary.json` → `detectability`:
- `test_only_roc_auc_true_vs_random` (+ `test_only_roc_auc_ci95`) — **signal
  quality.** Held-out AUC with a 95% bootstrap CI. `all_split_roc_auc_true_vs_random`
  is also reported for reference; trust the test-only number.
- `true_pair_tested_rate` — **signal coverage.** Fraction of instances whose true
  dangerous pair actually reached the AUC. If this is low, AUC alone is
  misleading — the residue exists but is not reliably reachable.
- `true_pair_safe_gate_rate`, `true_pair_missing_due_to_gate`,
  `true_pair_missing_due_to_pair_cap` — *why* pairs go missing (a true pair is
  dropped if an endpoint fails `delta ≤ tau`, or if the pair cap subsamples it
  out).
- `utility_c3_vs_solo[budget].delta` (+ `.delta_ci95`) — accuracy gain of C3 over
  solo per budget, with a bootstrap CI (CI straddling 0 ⇒ not significant).
- `verdict_hint` — **PASS only when BOTH** `test_only_roc_auc ≥ 0.75` AND
  `true_pair_tested_rate ≥ 0.8`. Otherwise it states which condition failed
  (signal weak vs. signal not reliably reachable).

> If `true_pair_tested_rate` is low, raise `--tau` (more spans pass the safe
> gate) and/or `--max_safe_pairs` (less capping), then re-run — but report the
> coverage you actually achieved, don't tune until the verdict flips.

## Knobs
| flag | default | meaning |
|---|---|---|
| `--n` | 60 | instances (¼ used as dev for threshold calibration) |
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | any HF causal LM |
| `--tau` | 0.05 | NLL `delta` below which a span is "safe" |
| `--budgets` | `0.5,0.35,0.25` | compression ratios for Q2 |
| `--max_safe_pairs` | 60 | cap on sigma pair-tests per instance |

## Notes / honesty
- This task is deliberately a clean retrieval-style 2-cover (the answer word
  appears in both mentions). It measures whether the model's NLL **registers the
  loss of redundant evidence** — the minimal precondition for residue-guided
  compression. A positive here is necessary, not sufficient, for real-world C3.
- The detectability threshold is calibrated on the dev split (dev-median
  midpoint) and applied to test — do not re-tune on test.
- GPU NLL is not bit-reproducible across hardware; compare runs on one machine.
