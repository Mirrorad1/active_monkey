"""active_loop/meta_metrics.py — scorers for the N3 diagnostic/repair workstream.

Small, dependency-light scorers used to grade the failure-mode diagnostic
(``n3_diagnostics``) against ground-truth regime labels:

  confusion_matrix(y_true, y_pred, labels)  -- integer count matrix
  per_class_prf(cm)                          -- precision/recall/F1 per class
  macro_f1(cm)                               -- unweighted mean F1 over classes
  confusion_rate(cm, labels, t, p)           -- P(pred=p | true=t), the gate metric
  format_confusion(cm, labels)               -- pretty ASCII table for outputs/*.txt

These are deliberately pure functions over plain lists / numpy arrays so they can
be unit-tested in isolation and reused by N3a (shadow) and N3b (control) alike.

Design doc: docs/specs/n3-open-world.md (§5 metrics, §7 N3a gate).
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> np.ndarray:
    """Return the ``(n_labels, n_labels)`` integer confusion matrix.

    ``cm[i, j]`` = number of items whose true label is ``labels[i]`` and whose
    predicted label is ``labels[j]``.  Rows = truth, columns = prediction.

    Args:
        y_true: ground-truth label per item.
        y_pred: predicted label per item (same length as ``y_true``).
        labels: ordered label vocabulary; every label in the inputs must appear.

    Raises:
        ValueError: if the inputs differ in length or contain an unknown label.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred differ in length: {len(y_true)} != {len(y_pred)}"
        )
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t not in index:
            raise ValueError(f"unknown true label {t!r}; labels={labels}")
        if p not in index:
            raise ValueError(f"unknown predicted label {p!r}; labels={labels}")
        cm[index[t], index[p]] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> list[dict[str, float]]:
    """Return per-class precision/recall/F1 from a confusion matrix.

    For class ``i``: TP = cm[i,i], FP = column-sum − TP, FN = row-sum − TP.
    A class with no support (row-sum 0) and no predictions reports 0.0 for all
    three (it cannot contribute evidence).  Precision/recall default to 0.0 when
    their denominator is 0, matching the sklearn ``zero_division=0`` convention.
    """
    n = cm.shape[0]
    out: list[dict[str, float]] = []
    col_sums = cm.sum(axis=0)
    row_sums = cm.sum(axis=1)
    for i in range(n):
        tp = int(cm[i, i])
        fp = int(col_sums[i] - tp)
        fn = int(row_sums[i] - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        out.append({"precision": precision, "recall": recall, "f1": f1})
    return out


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean of per-class F1 (macro-F1).

    Macro (not micro) is deliberate: each regime counts equally regardless of how
    many runs instantiate it, so a diagnostic that nails the common regimes but
    fails a rare one cannot hide behind support imbalance.
    """
    prf = per_class_prf(cm)
    if not prf:
        return 0.0
    return float(np.mean([c["f1"] for c in prf]))


def confusion_rate(
    cm: np.ndarray,
    labels: list[str],
    true_label: str,
    pred_label: str,
) -> float:
    """Return ``P(pred = pred_label | true = true_label)``.

    This is the **gate metric** for the two load-bearing confusions
    (noise↔structural, nonstationarity↔structural): the fraction of items whose
    true regime is ``true_label`` that were misread as ``pred_label``.  Returns
    0.0 when ``true_label`` has no support (no items to confuse).
    """
    index = {lab: i for i, lab in enumerate(labels)}
    i = index[true_label]
    j = index[pred_label]
    row_sum = int(cm[i].sum())
    if row_sum == 0:
        return 0.0
    return float(cm[i, j]) / row_sum


def format_confusion(cm: np.ndarray, labels: list[str]) -> str:
    """Render an ASCII confusion table (rows = truth, cols = prediction).

    Labels are abbreviated to their first 8 chars for column headers so the
    table stays readable in a committed ``experiments/outputs/*.txt``.
    """
    abbrev = [lab[:8] for lab in labels]
    col_w = max(8, max(len(a) for a in abbrev))
    row_w = max(len(lab) for lab in labels)
    header = " " * (row_w + 3) + "".join(f"{a:>{col_w + 1}}" for a in abbrev)
    lines = [header, " " * (row_w + 3) + "-" * (len(header) - row_w - 3)]
    for i, lab in enumerate(labels):
        cells = "".join(f"{int(cm[i, j]):>{col_w + 1}}" for j in range(len(labels)))
        lines.append(f"{lab:>{row_w}} | {cells}")
    return "\n".join(lines)
