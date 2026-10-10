"""Scoring helpers for labelled handwriting lines."""
from __future__ import annotations

import pytest

from cdi_adapter.recognition import scoring as sc


def test_levenshtein_and_error_rates():
    assert sc.levenshtein("kitten", "sitting") == 3 and sc.levenshtein("", "abc") == 3
    assert sc.levenshtein(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    s = sc.score(["telma 40", "hba1c 7.8"], ["telma 40", "hba1c 7.9"])
    assert s["exact_match"] == 0.5 and s["wer"] == round(1 / 4, 4) and s["cer"] == round(1 / 17, 4)


def test_critical_value_accuracy_judges_the_literal_digits():
    s = sc.score(["Telma 40 1-0-1", "no numbers here", "HbA1c 7.8"], ["Telma 4O 1-0-1", "no numbers here", "HbA1c 7.8"])
    assert s["lines_with_numbers"] == 2 and s["critical_value_accuracy"] == 0.5       # 4O is not 40


def test_case_and_spacing_do_not_count_as_errors():
    assert sc.score(["Telma  40"], ["telma 40"])["exact_match"] == 1.0


def test_wilson_lower_bound():
    assert sc.wilson_lower(0, 0) == 0.0
    assert sc.wilson_lower(100, 100) == pytest.approx(0.963, abs=0.002)
    assert sc.wilson_lower(95, 100) < 0.95 < sc.wilson_lower(950, 1000) + 0.02
    assert sc.wilson_lower(950, 1000) > sc.wilson_lower(95, 100)                  # more data, tighter bound
