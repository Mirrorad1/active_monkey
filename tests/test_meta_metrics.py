"""Unit tests for active_loop/meta_metrics.py (N3 diagnostic scorers)."""
from __future__ import annotations

import numpy as np
import pytest

from active_loop.meta_metrics import (
    confusion_matrix,
    confusion_rate,
    format_confusion,
    macro_f1,
    per_class_prf,
)

LABELS = ["a", "b", "c"]


def test_confusion_matrix_counts():
    y_true = ["a", "a", "b", "c", "c"]
    y_pred = ["a", "b", "b", "c", "a"]
    cm = confusion_matrix(y_true, y_pred, LABELS)
    # row a: 1 correct (a), 1 -> b ; row b: 1 correct ; row c: 1 correct, 1 -> a
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1 and cm[2, 0] == 1
    assert cm.sum() == 5


def test_confusion_matrix_length_mismatch():
    with pytest.raises(ValueError):
        confusion_matrix(["a"], ["a", "b"], LABELS)


def test_confusion_matrix_unknown_label():
    with pytest.raises(ValueError):
        confusion_matrix(["z"], ["a"], LABELS)


def test_macro_f1_perfect():
    y = ["a", "b", "c", "a"]
    cm = confusion_matrix(y, y, LABELS)
    assert macro_f1(cm) == pytest.approx(1.0)


def test_macro_f1_chance_is_low():
    # All predicted "a": only class a has any recall, b/c F1 = 0 -> macro pulled down.
    y_true = ["a", "b", "c"]
    y_pred = ["a", "a", "a"]
    cm = confusion_matrix(y_true, y_pred, LABELS)
    assert macro_f1(cm) < 0.5


def test_per_class_prf_zero_division_safe():
    # Class c has no support and no predictions -> all zeros, no exception.
    y_true = ["a", "a", "b"]
    y_pred = ["a", "a", "b"]
    cm = confusion_matrix(y_true, y_pred, LABELS)
    prf = per_class_prf(cm)
    assert prf[2]["precision"] == 0.0
    assert prf[2]["recall"] == 0.0
    assert prf[2]["f1"] == 0.0


def test_confusion_rate():
    # 3 true-"a": 1 stays a, 2 -> b  => rate(a->b) = 2/3
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "b", "b", "b"]
    cm = confusion_matrix(y_true, y_pred, LABELS)
    assert confusion_rate(cm, LABELS, "a", "b") == pytest.approx(2 / 3)
    assert confusion_rate(cm, LABELS, "a", "a") == pytest.approx(1 / 3)


def test_confusion_rate_no_support():
    cm = np.zeros((3, 3), dtype=int)
    assert confusion_rate(cm, LABELS, "a", "b") == 0.0


def test_format_confusion_runs():
    cm = confusion_matrix(["a", "b"], ["a", "b"], LABELS)
    out = format_confusion(cm, LABELS)
    assert isinstance(out, str)
    assert "a" in out and "b" in out
