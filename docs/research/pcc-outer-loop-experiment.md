# PCC outer-loop experiment

This is the cheap first test for Prospective Causal Calibration (PCC) in an LLM
RL/RSI setting.

The target hypothesis is narrow:

> A scaffold patch should be selected not only because it improves a dev score,
> but because its precommitted forecast correctly predicts where the improvement,
> non-effect, or harm occurs.

## What this harness tests

The harness compares two frozen-model scaffold-selection rules:

- `score_only`: choose the patch with the largest observed dev-score delta.
- `pcc`: choose the patch with the largest calibrated score:
  `delta_accuracy - alpha * forecast_nll - wrong_confident_forecast_penalty - complexity_penalty`.

The wrong-confident penalty is deliberate. Without it, a lucky patch that improves
the dev slice while confidently predicting the wrong causal effect can still win,
which collapses the method back into ordinary score-only scaffold search.

## Smoke command

This dependency-free command exercises the PCC bookkeeping with deterministic
synthetic outcomes:

```bash
/Users/mirro/Projects/active-loop/.venv/bin/python experiments/pcc_outer_loop.py \
  --backend smoke \
  --iterations 1
```

It prints a pretty JSON report followed by a one-line JSON report. The last line
is intended for automated capture.

## Qwen/GSM8K command

The real open-source model path uses `Qwen/Qwen2.5-0.5B-Instruct` on GSM8K:

```bash
/Users/mirro/Projects/active-loop/.venv/bin/python experiments/pcc_outer_loop.py \
  --backend hf \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset openai/gsm8k \
  --dataset-config main \
  --split test \
  --limit 48
```

This path requires optional packages that are intentionally not added to the base
project dependency set by the harness:

- `torch`
- `transformers`
- `datasets`

It may also require network access to download the model and dataset if they are
not already cached locally.

For a single-GPU RunPod setup, use `runpod/PCC_OUTER_LOOP.md` and
`runpod/setup_pcc_outer_loop.sh`.

## Honest interpretation

This is not yet GRPO/LoRA weight training. It is the pre-RL gate: if PCC does not
beat or de-risk score-only scaffold selection in a frozen outer loop, there is no
good reason to spend GPU on a weight-update RL version.

A useful positive result would be:

- PCC chooses a different patch than score-only, and
- that patch has better held-out transfer, lower dev-test gap, or fewer harmful
  regressions.

A useful negative result would be:

- forecast accuracy is uncorrelated with held-out transfer after controlling for
  dev-score delta, or
- the open model cannot produce stable enough answers for patch-effect forecasts
  to be measured cheaply.

In either case the result is informative: PCC either provides a distinct
self-improvement signal or collapses into ordinary scaffold search plus overhead.

## First plumbing check

On 2026-06-22, the optional packages were installed into a local ignored
`.venv-pcc` environment:

```bash
uv venv --python 3.12 .venv-pcc
uv pip install --python .venv-pcc torch transformers datasets accelerate
```

Then the smallest real open-model check was run:

```bash
.venv-pcc/bin/python experiments/pcc_outer_loop.py \
  --backend hf \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset openai/gsm8k \
  --dataset-config main \
  --split test \
  --limit 2 \
  --max-new-tokens 96
```

Result: the model/dataset path works, but this was not a scientific PCC verdict.
Baseline and all three candidate scaffolds scored 0/2, so there was no observed
patch-effect signal. Both `score_only` and `pcc` selected `lucky_overclaim` only
because all deltas were tied at zero and that patch forecasted "same" most
strongly on the two sampled tasks.

The follow-up cached 8-example pilot used the same model and command with
`--limit 8`. It produced one nonzero patch delta, but still no positive evidence
for PCC over score-only:

- baseline: `0/8`
- `arith_check`: `0/8`, PCC score `-0.535`, wrong-confident rate `0.625`
- `concise`: `0/8`, PCC score `-0.051`, wrong-confident rate `0.000`
- `lucky_overclaim`: `1/8`, PCC score `0.078`, wrong-confident rate `0.125`

Both `score_only` and `pcc` selected `lucky_overclaim`. This is a NEGATIVE pilot
for the current fixed forecasts/scaffolds, not for the broader PCC hypothesis:
the patch set is weak, the forecasts are hand-authored rather than model-authored,
and `n=8` is far below a verdict batch. The next useful iteration is to log
per-task responses and load the model once per run, then use either a larger
GSM8K slice or an easier arithmetic benchmark where patch deltas are dense enough
to measure calibration.
