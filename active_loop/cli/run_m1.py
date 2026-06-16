"""M1 smoke run: score the suite once and print the report."""
from eval.score import score_suite


def main() -> None:
    report = score_suite()
    print("=== active-loop M1 score report ===")
    print(f"metric (mean chosen-action EFE, lower=better): {report.metric:.4f}")
    print(f"success_rate: {report.success_rate:.3f}")
    print(f"ask_rate:     {report.ask_rate:.3f}")
    print("guardrails:")
    for name, ok in report.guardrails.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"verdict: {'PASS' if report.verdict else 'FAIL'}")


if __name__ == "__main__":
    main()
