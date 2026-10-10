"""Disagreement-rate drift alarm (RD-S3).

The share of handwritten lines on which two readings of the line disagree is a cheap health signal: a new
scanner, a new doctor, a model update or a broken host moves it. This compares each document's rate
with the rates of the runs before it and says when the move is larger than normal. It only
measures and warns: it never changes a reading or a decision.

Pure functions plus one small query; nothing here needs a GPU.
"""
from __future__ import annotations

import statistics
from typing import Any

from sqlalchemy import text

from ..config import settings

_HANDWRITTEN_STATES = ("agree", "disagree", "single_engine", "no_reading")


def disagreement_rate(states: dict[str, int]) -> float | None:
    """Disagreeing lines / handwritten lines, or ``None`` when the document had none."""
    total = sum(int(states.get(s, 0)) for s in _HANDWRITTEN_STATES)
    return round(int(states.get("disagree", 0)) / total, 4) if total else None


def drift_alarm(history: list[float], current: float, *, min_runs: int | None = None,
                sigma: float | None = None, abs_tol: float | None = None) -> dict[str, Any]:
    """``{"alarm": bool, ...}``: alarm when ``current`` is further from the usual rate than
    ``max(sigma x the usual spread, abs_tol)``. With fewer than ``min_runs`` earlier runs there is
    no usual rate yet, so no alarm (and the reason says so)."""
    n_min = settings.drift_min_runs if min_runs is None else min_runs
    k = settings.drift_sigma if sigma is None else sigma
    tol = settings.drift_abs_tolerance if abs_tol is None else abs_tol
    runs = len(history)
    if runs < n_min:
        return {"alarm": False, "reason": f"only {runs} earlier runs; {n_min} are needed", "runs": runs,
                "current": current}
    mean = statistics.fmean(history)
    band = max(k * statistics.pstdev(history), tol)
    delta = current - mean
    return {"alarm": abs(delta) > band, "runs": runs, "current": current, "mean": round(mean, 4),
            "band": round(band, 4), "direction": "up" if delta > 0 else "down" if delta < 0 else "none"}


def recent_rates(sess: Any, exclude_run_id: Any, limit: int = 200) -> list[float]:
    """Disagreement rates of the most recent finished recognition runs (from their stored metrics)."""
    rows = sess.execute(text(
        "SELECT metrics->'states' FROM pipeline_run WHERE stage = 'ocr' AND model_name = 'recognition-v2' "
        "AND status = 'ok' AND id <> :rid AND metrics->'states' IS NOT NULL ORDER BY started_at DESC LIMIT :n"),
        {"rid": str(exclude_run_id), "n": limit}).all()
    rates = [disagreement_rate(r[0]) for r in rows if isinstance(r[0], dict)]
    return [x for x in rates if x is not None]
