"""Cheap PCC outer-loop experiment for scaffold selection.

This is not a weight-update RL run. It is the cheaper first gate: compare
score-only scaffold selection with Prospective Causal Calibration (PCC) before
spending GPU on GRPO/LoRA.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from active_loop.pcc import Forecast, PatchSummary, Trial, choose_patch, summarize_patch


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET = "openai/gsm8k"


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    question: str
    answer: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CandidatePatch:
    patch_id: str
    instruction: str
    forecast: Forecast
    complexity_penalty: float = 0.0


def _candidate_patches() -> list[CandidatePatch]:
    return [
        CandidatePatch(
            patch_id="arith_check",
            instruction=(
                "Solve step by step. Before the final answer, verify each arithmetic "
                "operation once. End with '#### <number>'."
            ),
            forecast=Forecast(
                {
                    "multi_step": {"improve": 0.75, "same": 0.20, "worse": 0.05},
                    "unit": {"improve": 0.65, "same": 0.25, "worse": 0.10},
                    "single_step": {"improve": 0.10, "same": 0.80, "worse": 0.10},
                }
            ),
            complexity_penalty=0.02,
        ),
        CandidatePatch(
            patch_id="concise",
            instruction="Answer directly. End with '#### <number>'.",
            forecast=Forecast(
                {
                    "multi_step": {"improve": 0.10, "same": 0.55, "worse": 0.35},
                    "unit": {"improve": 0.10, "same": 0.65, "worse": 0.25},
                    "single_step": {"improve": 0.20, "same": 0.75, "worse": 0.05},
                }
            ),
            complexity_penalty=0.005,
        ),
        CandidatePatch(
            patch_id="lucky_overclaim",
            instruction=(
                "Use any shortcut that seems likely. End with '#### <number>'."
            ),
            forecast=Forecast(
                {
                    "multi_step": {"improve": 0.05, "same": 0.90, "worse": 0.05},
                    "unit": {"improve": 0.05, "same": 0.90, "worse": 0.05},
                    "single_step": {"improve": 0.05, "same": 0.90, "worse": 0.05},
                }
            ),
            complexity_penalty=0.0,
        ),
    ]


def _smoke_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            "smoke_multi",
            "A store sold 12 apples on Monday and twice as many on Tuesday. How many total apples?",
            "36",
            ("multi_step",),
        ),
        BenchmarkTask(
            "smoke_unit",
            "A runner travels 3 miles each hour for 4 hours. How many miles?",
            "12",
            ("unit",),
        ),
        BenchmarkTask(
            "smoke_single",
            "What is 7 plus 5?",
            "12",
            ("single_step",),
        ),
        BenchmarkTask(
            "smoke_stable",
            "What is 9 minus 2?",
            "7",
            ("single_step",),
        ),
    ]


def _smoke_trials(tasks: Sequence[BenchmarkTask], patch_id: str | None) -> list[Trial]:
    outcomes = {
        None: {
            "smoke_multi": False,
            "smoke_unit": False,
            "smoke_single": True,
            "smoke_stable": True,
        },
        "arith_check": {
            "smoke_multi": True,
            "smoke_unit": True,
            "smoke_single": True,
            "smoke_stable": True,
        },
        "concise": {
            "smoke_multi": False,
            "smoke_unit": False,
            "smoke_single": True,
            "smoke_stable": True,
        },
        "lucky_overclaim": {
            "smoke_multi": True,
            "smoke_unit": True,
            "smoke_single": True,
            "smoke_stable": True,
        },
    }
    selected = outcomes[patch_id]
    return [Trial(task.task_id, task.tags, selected[task.task_id]) for task in tasks]


def _numeric_answer(text: str) -> str | None:
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).rstrip("0").rstrip(".")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not numbers:
        return None
    return numbers[-1].rstrip("0").rstrip(".")


def _is_correct(response: str, answer: str) -> bool:
    expected = _numeric_answer(answer)
    observed = _numeric_answer(response)
    return expected is not None and observed == expected


def _task_tags(question: str) -> tuple[str, ...]:
    lowered = question.lower()
    tags = []
    numbers = re.findall(r"\d+", question)
    if len(numbers) >= 3:
        tags.append("multi_step")
    else:
        tags.append("single_step")
    if any(unit in lowered for unit in ("mile", "hour", "dollar", "$", "cent", "minute", "kg", "pound")):
        tags.append("unit")
    return tuple(dict.fromkeys(tags))


def _prompt(question: str, instruction: str) -> str:
    return f"{instruction}\n\nProblem:\n{question}\n\nSolution:"


def _generate_response_hf(model, tokenizer, question: str, instruction: str, max_new_tokens: int) -> str:
    prompt = _prompt(question, instruction)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You solve grade-school math problems."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _load_hf_tasks(dataset_name: str, config: str, split: str, limit: int) -> list[BenchmarkTask]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "HF backend requires the optional 'datasets' package. "
            "Install transformers, datasets, and torch before running --backend hf."
        ) from exc

    dataset = load_dataset(dataset_name, config, split=split)
    rows = list(dataset.select(range(min(limit, len(dataset)))))
    return [
        BenchmarkTask(
            task_id=f"{split}_{idx}",
            question=str(row["question"]),
            answer=str(row["answer"]),
            tags=_task_tags(str(row["question"])),
        )
        for idx, row in enumerate(rows)
    ]


def _hf_trials(
    tasks: Sequence[BenchmarkTask],
    model_name: str,
    instruction: str,
    max_new_tokens: int,
) -> list[Trial]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "HF backend requires optional packages: torch, transformers, datasets. "
            "Install them before running --backend hf."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()

    trials = []
    for task in tasks:
        response = _generate_response_hf(model, tokenizer, task.question, instruction, max_new_tokens)
        trials.append(Trial(task.task_id, task.tags, _is_correct(response, task.answer)))
    return trials


def _summaries_for_trials(
    baseline: Sequence[Trial],
    candidates: Iterable[tuple[CandidatePatch, Sequence[Trial]]],
    alpha: float,
) -> list[PatchSummary]:
    return [
        summarize_patch(
            patch.patch_id,
            patch.forecast,
            baseline,
            patched,
            alpha=alpha,
            complexity_penalty=patch.complexity_penalty,
        )
        for patch, patched in candidates
    ]


def _score_only(summaries: Sequence[PatchSummary]) -> PatchSummary:
    return max(summaries, key=lambda summary: (summary.delta_accuracy, summary.patch_id))


def run_smoke(iterations: int, alpha: float) -> dict:
    tasks = _smoke_tasks()
    baseline = _smoke_trials(tasks, None)
    summaries = []
    for patch in _candidate_patches():
        patched = _smoke_trials(tasks, patch.patch_id)
        summaries.append(
            summarize_patch(
                patch.patch_id,
                patch.forecast,
                baseline,
                patched,
                alpha=alpha,
                complexity_penalty=patch.complexity_penalty,
            )
        )

    pcc_choice = choose_patch(summaries)
    score_choice = _score_only(summaries)
    return {
        "backend": "smoke",
        "iterations": iterations,
        "baseline_accuracy": sum(trial.correct for trial in baseline) / len(baseline),
        "summaries": [asdict(summary) for summary in summaries],
        "methods": {
            "pcc": asdict(pcc_choice),
            "score_only": asdict(score_choice),
        },
        "chosen": {"method": "pcc", **asdict(pcc_choice)},
    }


def run_hf(args: argparse.Namespace) -> dict:
    tasks = _load_hf_tasks(args.dataset, args.dataset_config, args.split, args.limit)
    baseline_instruction = "Solve the problem. End with '#### <number>'."
    baseline = _hf_trials(tasks, args.model, baseline_instruction, args.max_new_tokens)

    patched_trials = []
    for patch in _candidate_patches():
        trials = _hf_trials(tasks, args.model, patch.instruction, args.max_new_tokens)
        patched_trials.append((patch, trials))
    summaries = _summaries_for_trials(baseline, patched_trials, args.alpha)
    pcc_choice = choose_patch(summaries)
    score_choice = _score_only(summaries)

    return {
        "backend": "hf",
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "limit": args.limit,
        "baseline_accuracy": sum(trial.correct for trial in baseline) / len(baseline),
        "summaries": [asdict(summary) for summary in summaries],
        "methods": {
            "pcc": asdict(pcc_choice),
            "score_only": asdict(score_choice),
        },
        "chosen": {"method": "pcc", **asdict(pcc_choice)},
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("smoke", "hf"), default="smoke")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output", help="Optional path to write the final JSON report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    if args.backend == "smoke":
        result = run_smoke(args.iterations, args.alpha)
    else:
        result = run_hf(args)

    pretty = json.dumps(result, indent=2, sort_keys=True)
    compact = json.dumps(result, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(pretty + "\n", encoding="utf-8")
    print(pretty)
    print(compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
