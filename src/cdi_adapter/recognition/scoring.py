"""Scoring helpers for labelled handwriting lines: character and word error rates, exact match, critical-value accuracy
(every number on a line exactly right, judged on the literal digits) and the Wilson lower bound of a proportion.

Used by the evaluation scorer and the measurement scripts. A number is never separated from the model that produced it:
the callers record the model version next to every score.
"""
from __future__ import annotations

import math
import re
from typing import Any


def levenshtein(a: str | list[str], b: str | list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().casefold())


def numbers(s: str) -> list[str]:
    """The literal numbers on a line (no look-alike repair: a misread digit is a wrong number)."""
    return re.findall(r"\d+(?:\.\d+)?", s)


def wilson_lower(correct: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the 95 % Wilson interval for a proportion (0.0 for no data)."""
    if n <= 0:
        return 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def score(truths: list[str], preds: list[str]) -> dict[str, Any]:
    """CER / WER over the whole set (total edits / total reference length), exact match and
    critical-value accuracy (lines that carry numbers: all numbers right)."""
    n = len(truths)
    ch_edits = sum(levenshtein(_norm(t), _norm(p)) for t, p in zip(truths, preds))
    ch_len = sum(len(_norm(t)) for t in truths)
    wd_edits = sum(levenshtein(_norm(t).split(), _norm(p).split()) for t, p in zip(truths, preds))
    wd_len = sum(len(_norm(t).split()) for t in truths)
    exact = sum(_norm(t) == _norm(p) for t, p in zip(truths, preds))
    with_numbers = [(t, p) for t, p in zip(truths, preds) if numbers(t)]
    crit = sum(numbers(t) == numbers(p) for t, p in with_numbers)
    return {"lines": n, "cer": round(ch_edits / ch_len, 4) if ch_len else None,
            "wer": round(wd_edits / wd_len, 4) if wd_len else None,
            "exact_match": round(exact / n, 4) if n else None,
            "lines_with_numbers": len(with_numbers),
            "critical_value_accuracy": round(crit / len(with_numbers), 4) if with_numbers else None}
